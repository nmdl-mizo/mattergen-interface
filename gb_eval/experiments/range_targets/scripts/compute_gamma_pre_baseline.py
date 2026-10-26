#!/usr/bin/env python3
"""compute_gamma_pre_baseline.py -- gamma before relaxation (gamma_pre) of the baseline (a thin wrapper).

The plan defines the gamma calculation of the baseline as "compute_gamma_unrelaxed_baseline.py (area is |L2xL3| with the axes sorted in descending order of length; error=angle_fail if an angle deviates from 90 deg +- 1.0 deg;
non_al_count is recorded; as in the source)". It also requires w, chunk, target_id, is_inner, ckpt_sha256 and min_dist_A in the output CSV
(for the baseline angle_max_dev_deg, perm and len_L1 of the source are also kept) and requires non-finite values to be left in the row as error=nonfinite.
run() of the source cannot add columns, so this file reuses the functions of the source (load_chunk, process_structure, OUT_FIELDS) read-only with importlib
and holds only the assembly of rows and the CSV writing [author's choice; a file that is not in the list of products of the plan]. The formula of gamma, the canonicalization and the angle decision are those of the source.

  Source: ${GBREL_CODE}/gb_eval/baseline_generation/compute_gamma_unrelaxed_baseline.py (read only)
  Added: cross-check with run_meta.json (17 keys) (arm, chunk, w, gamma_target, seed, ckpt_sha256), min_dist_A and nonfinite rows.

Execution environment: the evaluation environment (compute node. Not executed on the login node). PYTHONDONTWRITEBYTECODE=1.
Usage:
  python compute_gamma_pre_baseline.py --gen-root <gen/baseline/w<W>> --arm-label baseline_w<W> --w <W> \
      --ckpt-resolved <ckpt_resolved_baseline.json> --out <gamma_pre.csv> [--dry-run]
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_gamma_pre_proposed as H  # noqa: E402  (common part of run_meta, min_dist, nonfinite and CSV writing)

_VAN_CANDIDATES = [
    Path(_gbrel.CODE + '/gb_eval/baseline_generation/compute_gamma_unrelaxed_baseline.py'),
    Path(_gbrel.CODE + '/gb_eval/baseline_generation/compute_gamma_unrelaxed_baseline.py'),
    Path(_gbrel.CODE + '/gb_eval/baseline_generation/compute_gamma_unrelaxed_baseline.py'),
]
VAN_PATH = next((p for p in _VAN_CANDIDATES if p.is_file()), _VAN_CANDIDATES[0])


def _load_baseline():
    if not VAN_PATH.is_file():
        raise RuntimeError(f"baseline compute script not found: {VAN_PATH}")
    spec = importlib.util.spec_from_file_location("compute_gamma_unrelaxed_baseline", str(VAN_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compute_gamma_unrelaxed_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


VAN = _load_baseline()
V2 = VAN.V2
BaselineError = VAN.BaselineError
# columns = columns of the source (V2.OUT_FIELDS + non_al_count, angle_max_dev_deg, perm, len_L1) + additional columns (non_al_count is duplicated, so once)
OUT_FIELDS = list(VAN.OUT_FIELDS) + [c for c in H.EXTRA_FIELDS if c not in VAN.OUT_FIELDS]
EXPECT_CHUNKS = 14
EXPECT_PER_CHUNK = 25


def build_rows(gen_root: Path, arm_label: str, *, w: float, ckpt_sha256: str, calc, angle_tol_deg: float, dry_run: bool) -> list[dict]:
    out_rows: list[dict] = []
    t0 = time.time()
    n_ok = n_err = 0
    for ci in range(EXPECT_CHUNKS):
        cdir = gen_root / f"chunk{ci:02d}"
        if not cdir.is_dir():
            raise BaselineError(f"chunk dir missing: {cdir}")
        meta, atoms_list = VAN.load_chunk(cdir, EXPECT_PER_CHUNK)
        try:
            rm = H.load_run_meta(cdir, arm="baseline", w=w, ckpt_sha256=ckpt_sha256)
        except H.ComputeError as e:
            raise BaselineError(str(e))
        if float(rm["gamma_target"]) != float(meta["gamma_target_J_m2"]):
            raise BaselineError(f"{cdir}: run_meta.gamma_target={rm['gamma_target']!r} != chunk_meta.gamma_target_J_m2={meta['gamma_target_J_m2']!r}")
        if int(rm["seed"]) != int(meta["seed"]):
            raise BaselineError(f"{cdir}: run_meta.seed={rm['seed']} != chunk_meta.seed={meta['seed']}")
        tid = int(rm["target_id"])
        for i, atoms in enumerate(atoms_list):
            row = {k: "" for k in OUT_FIELDS}
            row.update({
                "arm": arm_label, "gamma_target": meta["gamma_target_J_m2"], "seed": meta["seed"],
                "struct_idx": i, "chunk_dir": str(cdir), "cif_index": i, "checkpoint_path": meta["checkpoint_path"],
                "w": float(rm["w"]), "chunk": int(rm["chunk"]), "target_id": tid,
                "is_inner": 1 if tid <= H.INNER_MAX_TARGET_ID else 0, "ckpt_sha256": rm["ckpt_sha256"],
                "non_al_count": int((np.asarray(atoms.numbers) != H.AL_Z).sum()),
                "natoms": len(atoms),
                "min_dist_A": H.min_dist_A(atoms),
            })
            if not H.is_finite_structure(atoms):
                row["error"] = "nonfinite"
            else:
                res = VAN.process_structure(atoms, calc, angle_tol_deg=angle_tol_deg, dry_run=dry_run)
                row.update(res)
                if not row["error"] and not all(H._finite(row[k]) for k in ("gamma_unrelaxed", "area_A2", "energy_eV")):
                    row["error"] = "nonfinite"
                    for k in ("gamma_unrelaxed", "energy_eV"):
                        row[k] = ""
            out_rows.append(row)
            if row["error"] and row["error"] != "DRY_RUN_SKIPPED":
                n_err += 1
                print(f"[ERROR] {cdir} struct_idx={i}: {row['error']}", flush=True)
            else:
                n_ok += 1
        print(f"[progress] chunk{ci:02d} done rows={len(out_rows)} n_ok={n_ok} n_err={n_err} elapsed={time.time() - t0:.1f}s", flush=True)
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-root", required=True, type=Path, help="directory that directly contains chunkNN/ (chunk_meta.json + run_meta.json + generated_crystals.extxyz)")
    ap.add_argument("--arm-label", required=True)
    ap.add_argument("--w", required=True, type=float)
    ap.add_argument("--ckpt-resolved", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--angle-tol-deg", type=float, default=1.0)
    ap.add_argument("--allow-overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        if args.out.exists() and not args.allow_overwrite:
            raise BaselineError(f"--out {args.out} already exists (no overwrite)")
        if not args.gen_root.is_dir():
            raise BaselineError(f"--gen-root not found: {args.gen_root}")
        try:
            sha = H.read_ckpt_sha(args.ckpt_resolved, "baseline")
        except H.ComputeError as e:
            raise BaselineError(str(e))
        calc = None
        if not args.dry_run:
            print("[GUARD] vram_precheck ...", flush=True)
            free_gib = V2.RG.vram_precheck(min_free_gib=8.0)
            print(f"[GUARD] VRAM free={free_gib:.1f} GiB", flush=True)
            calc, model_path = V2.build_calc()
            print(f"MACE model: {model_path}", flush=True)
        else:
            print("--dry-run: skipping MACE model load and single-point evaluation", flush=True)
        t0 = time.time()
        rows = build_rows(args.gen_root, args.arm_label, w=args.w, ckpt_sha256=sha, calc=calc,
                          angle_tol_deg=args.angle_tol_deg, dry_run=args.dry_run)
        if len(rows) != EXPECT_CHUNKS * EXPECT_PER_CHUNK:
            raise BaselineError(f"row count {len(rows)} != {EXPECT_CHUNKS * EXPECT_PER_CHUNK}")
        H.OUT_FIELDS = OUT_FIELDS          # replace the columns used by write_csv for the baseline
        H.write_csv(args.out, rows)
        n_angle_fail = sum(1 for r in rows if r["error"] == "angle_fail")
        n_nonfinite = sum(1 for r in rows if r["error"] == "nonfinite")
        n_non_al = sum(1 for r in rows if int(r["non_al_count"]) > 0)
        n_err = sum(1 for r in rows if r["error"] and r["error"] != "DRY_RUN_SKIPPED")
        mode = "DRY_RUN" if args.dry_run else "MAIN"
        print(f"{mode}_DONE arm={args.arm_label} n_rows={len(rows)} n_err_or_fail={n_err} n_angle_fail={n_angle_fail} "
              f"n_nonfinite={n_nonfinite} n_non_al={n_non_al} elapsed={time.time() - t0:.1f}s -> {args.out}", flush=True)
    except BaselineError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
