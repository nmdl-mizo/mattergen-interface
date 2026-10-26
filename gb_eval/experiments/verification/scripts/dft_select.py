#!/usr/bin/env python3
"""dft_select.py -- selection part of the DFT stage (deterministic, fixed before running VASP). Standard library only. VASP inputs are not written (next step).

Plan: see the experiment description (selection and acceptance condition D0).
  1. Population: structures of the main experiment (main layer) with stage=post that are converged AND G2 (summarize_g2_and_e.classify). m structures per (arm, target).
  2. For each (arm, target), ascending gamma_relaxed (ties by ascending (chunk, cif_index)). lo = floor(0.1(m-1)+0.5), hi = floor(0.9(m-1)+0.5) (exact with integer arithmetic
     lo = (m-1+5)//10, hi = (9(m-1)+5)//10). If m < 2 or lo == hi, that (arm, target) is excluded (the exclusion is recorded).
  3. Expectation: proposed 7 pairs (m=50 -> lo=5, hi=44); for baseline, from the per-target m=[3,8,11,18,22,1,11], target 5 with m=1 is removed, giving 6 pairs -> 13 pairs, 26 structures.
     Only pairs with |d gamma_MACE| >= 0.04 are used for the sign test -> n_used. Expected n_used=13.
     It is recomputed from the m of the run; if it differs from the expectation, selection.json is not written, the script exits non-zero (exit 3) and prints a diagnosis to standard error (stop and report).
  4. On success, it is saved to DIR/selection.json (if it exists, FileExistsError -> exit 2).
Usage: python dft_select.py --out-dir DIR [--symfit-csv CSV]    (the default CSV is gb_symfit_main.csv of RUN; it should be identical to the output of S0)
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import summarize_g2_and_e as SG  # noqa: E402

DESIGN = _gbrel.WORK + '/an internal design note that is not included'
LAYER = "main"
W = "2.0"
DELTA_MIN = 0.04
EXPECT_PROPOSED_M = [50] * 7
EXPECT_BASELINE_M = [3, 8, 11, 18, 22, 1, 11]
EXPECT_PAIRS = {"proposed": 7, "baseline": 6}
EXPECT_N_STRUCT = 26
EXPECT_N_USED = 13
DEFAULT_CSV = f"{K.P(K.RUN)}/results/gb_equivalence/gb_symfit_main.csv"


def lo_hi(m):
    """floor(0.1(m-1)+0.5) and floor(0.9(m-1)+0.5) with integers only (independent of floating-point rounding)."""
    return (m - 1 + 5) // 10, (9 * (m - 1) + 5) // 10


def select(csv_rows, manifest_rows):
    """csv_rows: the CSV of gb_symfit (layer=main). manifest_rows: inputs_manifest_main.csv (the source of target_id)."""
    tid = {}
    for r in manifest_rows:
        tid[(r["arm"], int(r["chunk"]), int(r["cif_index"]))] = int(r["target_id"])
    pop = {}
    for r in csv_rows:
        if r["layer"] != LAYER or r["stage"] != "post" or f"{float(r['w']):.1f}" != W:
            continue
        g2, _e = SG.classify(r["status"], r["gb0_status"], SG.to_finite(r["gb_symfit_max_A"]))
        gam = SG.to_finite(r["gamma_relaxed"])
        if not g2 or gam is None:
            continue
        key = (r["arm"], int(r["chunk"]), int(r["cif_index"]))
        t = tid[key]
        if t != key[1] // K.CHUNKS_PER_TARGET:
            raise K.InputError(f"target_id {t} != chunk//2 for {key}")
        pop.setdefault((r["arm"], t), []).append({"arm": r["arm"], "target_id": t, "chunk": key[1], "cif_index": key[2], "gamma_relaxed": gam,
                                                  "natoms": int(r["natoms"]), "cell_a_A": float(r["cell_a_A"]), "cell_b_A": float(r["cell_b_A"]),
                                                  "cell_c_A": float(r["cell_c_A"]), "axes": r["axes"], "op_name": r["op_name"]})
    cells, pairs, excluded = [], [], []
    for arm in ("proposed", "baseline"):
        for t in range(K.N_TARGETS):
            mem = sorted(pop.get((arm, t), []), key=lambda x: (x["gamma_relaxed"], x["chunk"], x["cif_index"]))
            m = len(mem)
            lo, hi = lo_hi(m) if m >= 1 else (None, None)
            cells.append({"arm": arm, "target_id": t, "m": m, "lo_index": lo, "hi_index": hi})
            if m < 2 or lo == hi:
                excluded.append({"arm": arm, "target_id": t, "m": m, "reason": "m<2" if m < 2 else "lo==hi"})
                continue
            a, b = mem[lo], mem[hi]
            d = b["gamma_relaxed"] - a["gamma_relaxed"]
            pairs.append({"arm": arm, "target_id": t, "m": m, "lo_index": lo, "hi_index": hi, "lo": a, "hi": b,
                          "delta_gamma_mace": d, "used_in_sign_test": abs(d) >= DELTA_MIN})
    return cells, pairs, excluded


def add_frame_info(pairs):
    """Adds the path (compute-node notation) of relaxed.extxyz of the selected structure and the frame_index (used for the next generation of inputs)."""
    for p in pairs:
        for role in ("lo", "hi"):
            s = p[role]
            rdir = f"{K.norm_compute(K.relax_root_for(LAYER))}/{K.armw(s['arm'], float(W))}/{K.chunk_name(s['chunk'])}"
            rr = {int(r["cif_index"]): r for r in K.read_csv_rows(K.P(rdir) + "/relax_rows.csv")}[s["cif_index"]]
            if rr["status"] != "ok_converged":
                raise K.InputError(f"selected structure not converged: {s['arm']} chunk{s['chunk']} cif{s['cif_index']}: {rr['status']}")
            if abs(float(rr["gamma_relaxed"]) - s["gamma_relaxed"]) > 1e-9:
                raise K.InputError(f"gamma_relaxed mismatch relax_rows vs gb_symfit csv: {s['arm']} chunk{s['chunk']} cif{s['cif_index']}")
            s["relaxed_extxyz"] = f"{rdir}/relaxed.extxyz"
            s["frame_index"] = int(rr["frame_index"])


def check_expectations(cells, pairs, excluded):
    """Acceptance condition D0. Returns: [(item, expected, measured, ok)]."""
    ms = {a: [c["m"] for c in cells if c["arm"] == a] for a in ("proposed", "baseline")}
    pc = {a: sum(1 for p in pairs if p["arm"] == a) for a in ("proposed", "baseline")}
    n_struct = 2 * len(pairs)
    n_used = sum(1 for p in pairs if p["used_in_sign_test"])
    out = [("proposed m per target", EXPECT_PROPOSED_M, ms["proposed"], ms["proposed"] == EXPECT_PROPOSED_M),
           ("baseline m per target", EXPECT_BASELINE_M, ms["baseline"], ms["baseline"] == EXPECT_BASELINE_M),
           ("number of pairs proposed / baseline", EXPECT_PAIRS, pc, pc == EXPECT_PAIRS),
           ("number of pairs (total)", 13, len(pairs), len(pairs) == 13),
           ("number of structures", EXPECT_N_STRUCT, n_struct, n_struct == EXPECT_N_STRUCT),
           ("n_used(|Δγ_MACE| ≥ 0.04)", EXPECT_N_USED, n_used, n_used == EXPECT_N_USED),
           ("excluded (arm,target)", [{"arm": "baseline", "target_id": 5}], [{"arm": e["arm"], "target_id": e["target_id"]} for e in excluded],
            [(e["arm"], e["target_id"]) for e in excluded] == [("baseline", 5)])]
    proposed_lohi = {(c["lo_index"], c["hi_index"]) for c in cells if c["arm"] == "proposed"}
    out.append(("(lo, hi) indices of proposed (m=50)", [(5, 44)], sorted(proposed_lohi), proposed_lohi == {(5, 44)}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--symfit-csv", default=DEFAULT_CSV)
    args = ap.parse_args()
    out = Path(K.P(args.out_dir)) / "selection.json"
    try:
        if out.exists():
            raise FileExistsError(f"refusing to overwrite: {out}")
        csv_rows = K.read_csv_rows(args.symfit_csv)
        man = K.read_csv_rows(K.manifest_path_for(LAYER))
        cells, pairs, excluded = select(csv_rows, man)
        chk = check_expectations(cells, pairs, excluded)
        for name, e, a, ok in chk:
            print(f"[D0] {'PASS' if ok else 'FAIL'} {name}: expected {e} actual {a}", file=sys.stderr)
        if not all(ok for *_x, ok in chk):
            print("STOP: acceptance condition D0 not satisfied. selection.json is not written because the values differ from the expectation (stop and report)", file=sys.stderr)
            sys.exit(3)
        add_frame_info(pairs)
    except (K.InputError, FileExistsError, FileNotFoundError, KeyError) as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
    doc = {"design": DESIGN, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "source_csv": args.symfit_csv, "layer": LAYER, "w": W, "stage": "post",
           "rule": "population = converged ∧ G2 per (arm, target); sort by (gamma_relaxed, chunk, cif_index); lo=floor(0.1(m-1)+0.5), hi=floor(0.9(m-1)+0.5); "
                   "exclude m<2 or lo==hi; sign test uses |Δγ_MACE| >= 0.04",
           "delta_min_J_m2": DELTA_MIN, "cells": cells, "excluded": excluded, "pairs": pairs,
           "n_pairs": len(pairs), "n_structures": 2 * len(pairs), "n_used": sum(1 for p in pairs if p["used_in_sign_test"]),
           "check_D0": [{"item": n, "expected": e, "actual": a, "pass": ok} for n, e, a, ok in chk]}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {out}: pairs={len(pairs)} structures={2 * len(pairs)} n_used={doc['n_used']}", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
