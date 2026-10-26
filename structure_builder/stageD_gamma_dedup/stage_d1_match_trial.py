#!/usr/bin/env python3
"""Data rebuild, stage D1-4: matches all pairs of the check group (k=2, dx=0.0, rows with converged=True) without a window and measures checks 4 and 5.

  stage_d1_match_trial.py plan      [--results-dir DIR]   # print the number of tasks of the array
  stage_d1_match_trial.py chunk ID  [--results-dir DIR]   # 1 task = 20 pairs -> match_trial/chunk_<ID>.csv
  stage_d1_match_trial.py aggregate [--results-dir DIR]   # -> stage_d1_match_trial.json

Content of a task = all pairs of the check group (kind=pair) + the validation: 5 self-matchings (kind=self; the same POSCAR_relaxed is read twice.
fit=True is expected) and 1 pair with different numbers of atoms (kind=natoms_mismatch; records whether pymatgen returns False or raises).
The matching is StructureMatcher(stol=0.025, primitive_cell=False, scale=False, attempt_supercell=False). The mean s_bar and the 95th percentile are given for kind=pair only.
For pairs with fit=True the RMS is also recorded with get_rms_dist (not included in the time). Exceptions are recorded in the error column and not treated as "non-duplicate".
r2: expected CPU time = s_bar x P_all / 3600 (default without window). Only when --window-ev W is given explicitly, s_bar x P_W / 3600 and the decision of check 5 (as before).
Without a window, Delta E_max is only kept as a reference value; check 5 is "reference" and is not a condition for submitting D2.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from importlib.metadata import version

import numpy as np

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "lib"))  # structure_builder/lib
import builder_common as C
import stage_d_common as D

CS = 20
COLS = ["kind", "row_i", "row_j", "fit", "rms_norm", "max_norm", "sec", "error"]


def build_plan(rows):
    g = D.groups_of(rows)
    pk = D.pilot_group_key()
    if pk not in g:
        raise ValueError(f"pilot group {pk} not in gamma csv (converged rows only)")
    prow = g[pk]
    if len(prow) < 6:
        raise ValueError(f"pilot group too small: {len(prow)}")
    pairs = [("pair", i, j) for (i, j) in D.all_pairs(prow)]
    rng = random.Random(20261002)
    selfs = [("self", r["row"], r["row"]) for r in sorted(rng.sample(prow, 5), key=lambda r: r["row"])]
    k0 = [r for r in g.get(f"{C.dx_tag(D.PILOT_DX)}_n{C.N_ATOMS}", [])]
    if not k0:
        raise ValueError("no k=0 row at dx=0.0 for the natoms-mismatch validation")
    mism = [("natoms_mismatch", prow[0]["row"], k0[0]["row"])]
    return pairs + selfs + mism, len(pairs), prow


def n_tasks(plan):
    return (len(plan) + CS - 1) // CS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "chunk", "aggregate"])
    ap.add_argument("id", nargs="?", type=int)
    ap.add_argument("--results-dir", default=None)
    D.add_window_arg(ap)
    args = ap.parse_args()
    res = D.results_dir(args.results_dir)
    W = D.window_ev(args.window_ev)
    rows = D.read_gamma(res / D.GAMMA_ALL)
    plan, n_pair, prow = build_plan(rows)
    byrow = {r["row"]: r for r in rows}
    out_dir = res / "match_trial"

    if args.cmd == "plan":
        print(n_tasks(plan))
        return

    if args.cmd == "chunk":
        from pymatgen.analysis.structure_matcher import StructureMatcher
        tid = args.id
        assert tid is not None and 0 <= tid < n_tasks(plan), (tid, n_tasks(plan))
        out_dir.mkdir(parents=True, exist_ok=True)
        fn = out_dir / f"chunk_{tid:04d}.csv"
        if fn.exists():
            print("exists, skip", fn)
            return
        sm0 = StructureMatcher(**D.UNI)
        print(f"pymatgen {version('pymatgen')} matcher: stol={sm0.stol} ltol={sm0.ltol} angle_tol={sm0.angle_tol} "
              f"prim={sm0._primitive_cell} scale={sm0._scale} supercell={sm0._supercell}", flush=True)
        cache, out = {}, []
        for (kind, i, j) in plan[tid * CS:(tid + 1) * CS]:
            fit, rms, mx, err, sec = "", "", "", "", ""
            try:
                if kind == "self":
                    si, sj = D.load_relaxed(byrow[i]), D.load_relaxed(byrow[i])   # reads the same file twice
                else:
                    for r in (i, j):
                        if r not in cache:
                            cache[r] = D.load_relaxed(byrow[r])
                    si, sj = cache[i], cache[j]
                sm = StructureMatcher(**D.UNI)
                t1 = time.time()
                f_ = bool(sm.fit(si, sj))
                sec = f"{time.time() - t1:.4f}"
                fit = int(f_)
                if f_:
                    rr = sm.get_rms_dist(si, sj)
                    rms, mx = repr(float(rr[0])), repr(float(rr[1]))
            except Exception as e:  # noqa: BLE001
                err = repr(e)
            out.append([kind, i, j, fit, rms, mx, sec, err])
        tmp = fn.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            w.writerows(out)
        os.replace(tmp, fn)
        print(f"done {fn.name} rows={len(out)} fits={sum(1 for r in out if r[3] == 1)} "
              f"errors={sum(1 for r in out if r[7])}", flush=True)
        return

    # aggregate
    nt = n_tasks(plan)
    missing = [t for t in range(nt) if not (out_dir / f"chunk_{t:04d}.csv").exists()]
    got = D.read_chunks(out_dir)
    want = {(k, i, j) for (k, i, j) in plan}
    have = [(r["kind"], int(r["row_i"]), int(r["row_j"])) for r in got]
    cover_ok = (not missing) and len(have) == len(want) and set(have) == want
    pairs = [r for r in got if r["kind"] == "pair"]
    p_err = [r for r in pairs if r["error"]]
    secs = np.array([float(r["sec"]) for r in pairs if not r["error"] and r["sec"] != ""])
    s_bar = float(secs.mean()) if len(secs) else None
    p95 = float(np.percentile(secs, 95)) if len(secs) else None
    g = D.groups_of(rows)
    p_all = sum(len(v) * (len(v) - 1) // 2 for v in g.values())
    p_w = sum(len(D.windowed_pairs(v, W)) for v in g.values()) if W is not None else None
    p_target = p_all if W is None else p_w          # without a window: estimated with all pairs P_all
    est = (s_bar * p_target / 3600.0) if s_bar is not None else None
    fits = [r for r in pairs if r["fit"] == "1"]
    fit_list = []
    for r in fits:
        a, b = byrow[int(r["row_i"])], byrow[int(r["row_j"])]
        fit_list.append({"cnid_i": a["cnid"], "seed_i": a["seed"], "cnid_j": b["cnid"], "seed_j": b["seed"],
                         "dE_eV": abs(a["energy_eV"] - b["energy_eV"]), "rms_norm": float(r["rms_norm"]),
                         "max_norm": float(r["max_norm"])})
    de_max = max((x["dE_eV"] for x in fit_list), default=None)
    if W is None:
        g5 = "REFERENCE"                            # without a window: reference value only (no decision; not a condition for submitting D2)
    elif not fits:
        g5 = "FLAG"
    else:
        g5 = "PASS" if de_max <= D.LIMIT_C5_EV else "FAIL"
    g4 = None if est is None else ("PASS" if est <= D.LIMIT_C4_HOURS else "FAIL")
    selfs = [r for r in got if r["kind"] == "self"]
    mm = [r for r in got if r["kind"] == "natoms_mismatch"]
    mm_outcome = None
    if mm:
        mm_outcome = f"exception: {mm[0]['error']}" if mm[0]["error"] else ("False" if mm[0]["fit"] == "0" else "True")
    summary = {
        "group_key": D.pilot_group_key(), "n_group": len(prow), "n_pairs_planned": n_pair, "n_tasks": nt,
        "n_tasks_missing": len(missing), "coverage_ok": cover_ok, "window_eV": W,
        "s_bar_sec": s_bar, "p95_sec": p95, "n_timed_pairs": int(len(secs)),
        "P_all": p_all, "P_W": p_w, "P_for_estimate": p_target, "est_cpu_hours": est,
        "C4": {"threshold_hours": D.LIMIT_C4_HOURS, "verdict": g4},
        "n_fit_true_pairs": len(fits), "n_fit_false_pairs": sum(1 for r in pairs if r["fit"] == "0"),
        "dE_max_fit_true_eV": de_max,
        "C5": {"threshold_eV": D.LIMIT_C5_EV, "verdict": g5,
                    "note": "reference (no decision without a window; not a condition for D2)" if W is None else "decision with a window"},
        "fit_true_pairs": fit_list, "n_pair_errors": len(p_err),
        "pair_error_sample": [r["error"] for r in p_err[:5]],
        "validation_self_fit_true": f"{sum(1 for r in selfs if r['fit'] == '1')}/{len(selfs)}",
        "validation_fit_false_observed": any(r["fit"] == "0" for r in pairs),
        "validation_natoms_mismatch_outcome": mm_outcome,
        "gamma_csv_sha256": D.sha256_file(res / D.GAMMA_ALL),
    }
    with open(res / "stage_d1_match_trial.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("fit_true_pairs",)}, indent=1))
    if W is None:
        print(f"C5 (reference): Delta E_max(fit=True)={de_max} eV, reference value only (no decision). C4: estimate s_bar*P_all/3600={est} CPU h -> {g4}")
    bad = (not cover_ok) or p_err or len(selfs) != 5 or any(r["fit"] != "1" for r in selfs)
    if bad:
        print("FAIL: coverage / pair errors / self-validation (see above)")
        sys.exit(2)


if __name__ == "__main__":
    main()
