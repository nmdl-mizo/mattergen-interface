#!/usr/bin/env python3
"""compute_gamma_unrelaxed.py -- from the CIF just after generation (before relaxation) it computes
gamma_unrelaxed with a MACE single-point energy (called once for each of the 3 arms N/A/R, all paths given as arguments).

Specification: dx as a diffused variable.
  Section on the calculation of gamma before relaxation (the existing compute_gamma_unrelaxed.py must not be run as it is).

Differences from the existing compute_gamma_unrelaxed.py of the earlier gamma-distribution review:
  - The input and output paths are fully given as arguments (the gamma_eval.csv of the earlier stage, the generated/ directory and
    the output destination of the anchor CSV are not hard-coded; overwriting the anchor CSV is prohibited)
  - It does not presuppose the gamma_eval.csv of the earlier stage (batch/cif_index already in CSV form).
    The arms N/A/R do not yet have a CSV equivalent to gamma_eval.csv, so each generation invocation directory
    (under --output-root, with generation_metrics.json + cif/gen_i.cif) is scanned directly and
    (dx_A, gamma_target, seed, struct_idx, natoms) is restored from generation_metrics.json.
  - The restoration logic is based on the implementation of the generator (struct_idx_per_sample = list(range(
    num_samples)) filtered by the keep_mask of the padding reduction) and gamma_target is looked up with
    targets_used[struct_idx] (not by direct reference with the positional index i
    -- cmd_assemble of the earlier run_step_d_relax.py was a simplified version that depended on the implicit assumption that
    the padding reduction dropped no structure. This v2 adopts this general form so that the correspondence is correct
    even if some samples are dropped by the padding reduction).

Origin of constants and formulas (the 3 points stated explicitly in the plan, the same formulas as the config.py of the data production
and the compute_gamma_unrelaxed.py of the earlier stages are reused as they are; the formulas themselves are not rewritten):
(see the next two lines)
  E_BULK_PER_ATOM, EV_PER_A2_TO_J_PER_M2 -> imported from scripts/config/config.py
  single_point_gamma / in_plane_area_bc -> the same formulas are copied here (A=|b x c|, GB normal = a axis index 0)

Output CSV columns (the same basic columns as the existing gamma_unrelaxed.csv + 4 new columns. Existing CSV readers downstream
refer to column names, so the additional columns do not break them):
  arm, dx_A, gamma_target, seed, struct_idx, chunk_dir, cif_index,
  gamma_unrelaxed, natoms, area_A2, energy_eV, error,
  gb_dx_realized_A, dx_mode, check5_abs_diff_from_gcm, checkpoint_path

Computing resources: MACE requires a GPU (do not run an MLIP on CPU). The execution environment is the same as that of the earlier
compute_gamma_unrelaxed.py (the evaluation environment on a GPU node,
or on a GPU partition with a record of running MACE).

--dry-run: does not load the MACE model and leaves the gamma_unrelaxed column blank,
  and runs only the pipeline of scanning, joining and writing the CSV of the manifest (generation_metrics.json)
  (a smoke test without GPU/MACE. It verifies the route that enters through the same entrance as production [directory scan]
  and exits through the same exit [CSV writing]).
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_CONFIG_DIR_CANDIDATES = [
    Path(_gbrel.CODE + '/gb_eval/scripts/config'),
    Path(_gbrel.CODE + '/gb_eval/scripts/config'),
    Path(_gbrel.CODE + '/gb_eval/scripts/config'),
]
CONFIG_DIR_DIR = next((p for p in _CONFIG_DIR_CANDIDATES if p.exists()), _CONFIG_DIR_CANDIDATES[0])
sys.path.insert(0, str(CONFIG_DIR_DIR))
import config as C  # noqa: E402

_RG_CANDIDATES = [_gbrel.CODE + '/gb_eval/vendor/bin', _gbrel.CODE + '/gb_eval/vendor/bin']
for _p in _RG_CANDIDATES:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import runner_guards as RG  # noqa: E402

OUT_FIELDS = [
    "arm", "dx_A", "gamma_target", "seed", "struct_idx", "chunk_dir", "cif_index",
    "gamma_unrelaxed", "natoms", "area_A2", "energy_eV", "error",
    "gb_dx_realized_A", "dx_mode", "check5_abs_diff_from_gcm", "checkpoint_path",
]


class ComputeError(RuntimeError):
    pass


def in_plane_area_bc(atoms) -> float:
    cell = np.array(atoms.get_cell())
    return float(np.linalg.norm(np.cross(cell[1], cell[2])))


def build_calc():
    from mace.calculators import mace_mp

    model_path = C.resolve(C.MACE_MODEL)
    print(f"MACE model loaded: {model_path} head={C.MACE_HEAD}", flush=True)
    t0 = time.time()
    calc = mace_mp(model=model_path, device="cuda", default_dtype="float64", head=C.MACE_HEAD)
    print(f"model load: {time.time()-t0:.1f}s", flush=True)
    return calc, model_path


def single_point_gamma(atoms, calc) -> dict:
    natoms = len(atoms)
    area = in_plane_area_bc(atoms)
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())  # single-point evaluation only. No relaxation or optimization is performed
    gamma = (energy - natoms * C.E_BULK_PER_ATOM) / (2.0 * area) * C.EV_PER_A2_TO_J_PER_M2
    return {"natoms": natoms, "area_A2": area, "energy_eV": energy, "gamma_unrelaxed": gamma}


def discover_chunk_dirs(output_root: Path) -> list[Path]:
    """Lists, in sorted order (ascending directory name = chunk/batch number order), the directories directly under --output-root or one level below that have a generation_metrics.json.
    The naming of arm R and the arms N/A may differ:
    (results/arm_r_generated/batchNN/) and arms N/A (results/arm_{n,a}_generated/chunkNN/);
    only one level is scanned so that it works with both namings."""
    if not output_root.is_dir():
        raise ComputeError(f"--output-root not found or not a directory: {output_root}")
    candidates = sorted(
        p for p in output_root.iterdir() if p.is_dir() and (p / "generation_metrics.json").is_file()
    )
    if not candidates:
        raise ComputeError(
            f"--output-root {output_root}: no subdirectory with generation_metrics.json found "
            "(no silent fallback: refusing to write an empty CSV silently)"
        )
    return candidates


def build_manifest_rows(chunk_dirs: list[Path], arm_label: str) -> list[dict]:
    """Scans generation_metrics.json + cif/gen_{i}.cif of each chunk directory and
    makes manifest rows with 1 structure = 1 row (no MACE calculation; pure restoration of metadata)."""
    rows: list[dict] = []
    for cdir in chunk_dirs:
        gcm_path = cdir / "generation_metrics.json"
        with open(gcm_path) as f:
            gcm = json.load(f)

        targets_used = gcm["targets_used"]
        dx_A_per_sample = gcm["dx_A_per_sample"]
        struct_idx_per_sample = gcm["struct_idx_per_sample"]
        gb_dx_realized = gcm.get("gb_dx_realized_A_per_sample")
        n_atoms_gcm = gcm.get("n_atoms_per_sample")
        seed = gcm["seed"]
        dx_mode = gcm.get("dx_mode")
        checkpoint_path = gcm.get("checkpoint_path")
        check5_diff = gcm.get("gb_dx_vs_cell_derived_max_abs_diff_A")
        n_kept = gcm.get("num_samples_after_padding_reduction")

        if n_kept is None:
            n_kept = len(dx_A_per_sample)
        lengths = {
            "dx_A_per_sample": len(dx_A_per_sample),
            "struct_idx_per_sample": len(struct_idx_per_sample),
        }
        if gb_dx_realized is not None:
            lengths["gb_dx_realized_A_per_sample"] = len(gb_dx_realized)
        if n_atoms_gcm is not None:
            lengths["n_atoms_per_sample"] = len(n_atoms_gcm)
        if not all(v == n_kept for v in lengths.values()):
            raise ComputeError(
                f"{gcm_path}: filtered-list length mismatch (n_kept={n_kept}): {lengths} "
                "(refusing to silently zip mismatched-length lists)"
            )

        cif_dir = cdir / "cif"
        for i in range(n_kept):
            struct_idx = struct_idx_per_sample[i]
            if struct_idx < 0 or struct_idx >= len(targets_used):
                raise ComputeError(
                    f"{gcm_path}: struct_idx_per_sample[{i}]={struct_idx} out of range for "
                    f"targets_used (len={len(targets_used)})"
                )
            gamma_target = targets_used[struct_idx]
            dx_A = dx_A_per_sample[i]
            cif_path = cif_dir / f"gen_{i}.cif"
            rows.append(
                {
                    "arm": arm_label,
                    "dx_A": dx_A,
                    "gamma_target": gamma_target,
                    "seed": seed,
                    "struct_idx": struct_idx,
                    "chunk_dir": str(cdir),
                    "cif_index": i,
                    "cif_path": str(cif_path),
                    "gb_dx_realized_A": gb_dx_realized[i] if gb_dx_realized is not None else "",
                    "dx_mode": dx_mode if dx_mode is not None else "",
                    "check5_abs_diff_from_gcm": check5_diff if check5_diff is not None else "",
                    "checkpoint_path": checkpoint_path if checkpoint_path is not None else "",
                }
            )
    return rows


def cmd_run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    out_csv = Path(args.out)
    if out_csv.exists() and not args.allow_overwrite:
        raise ComputeError(
            f"--out {out_csv} already exists. Refusing to overwrite silently "
            "(pass --allow-overwrite to explicitly permit; this script never overwrites the "
            " anchor CSV because that path is never passed as --out by convention)."
        )

    chunk_dirs = discover_chunk_dirs(output_root)
    print(f"discovered {len(chunk_dirs)} chunk directories under {output_root}", flush=True)
    manifest_rows = build_manifest_rows(chunk_dirs, args.arm_label)
    print(f"manifest rows (1 struct each): {len(manifest_rows)}", flush=True)

    calc = None
    model_path = None
    if not args.dry_run:
        print("[GUARD] vram_precheck ...", flush=True)
        free_gib = RG.vram_precheck(min_free_gib=8.0)
        print(f"[GUARD] VRAM free={free_gib:.1f} GiB", flush=True)
        calc, model_path = build_calc()
    else:
        print("--dry-run: skipping MACE model load and single-point evaluation", flush=True)

    from ase.io import read as ase_read

    t0 = time.time()
    n_ok = 0
    n_err = 0
    out_rows = []
    for n_done, r in enumerate(manifest_rows, 1):
        out_row = {k: r.get(k, "") for k in OUT_FIELDS}
        out_row["gamma_unrelaxed"] = ""
        out_row["natoms"] = ""
        out_row["area_A2"] = ""
        out_row["energy_eV"] = ""
        out_row["error"] = ""
        if args.dry_run:
            out_row["error"] = "DRY_RUN_SKIPPED"
        else:
            cif_path = Path(r["cif_path"])
            try:
                if not cif_path.exists():
                    raise FileNotFoundError(str(cif_path))
                atoms = ase_read(str(cif_path))
                out = single_point_gamma(atoms, calc)
                out_row["gamma_unrelaxed"] = out["gamma_unrelaxed"]
                out_row["natoms"] = out["natoms"]
                out_row["area_A2"] = out["area_A2"]
                out_row["energy_eV"] = out["energy_eV"]
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                out_row["error"] = f"{type(e).__name__}: {e}"
                n_err += 1
                print(f"[ERROR] {r['chunk_dir']} cif_index={r['cif_index']}: {out_row['error']}", flush=True)
        out_rows.append(out_row)
        if n_done % 200 == 0 or n_done == len(manifest_rows):
            elapsed = time.time() - t0
            print(
                f"[progress] {n_done}/{len(manifest_rows)} n_ok={n_ok} n_err={n_err} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    elapsed = time.time() - t0
    mode = "DRY_RUN" if args.dry_run else "MAIN"
    print(
        f"{mode}_DONE arm={args.arm_label} n_rows={len(out_rows)} n_ok={n_ok} n_err={n_err} "
        f"elapsed={elapsed:.1f}s model_path={model_path} -> {out_csv}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-root", required=True,
        help="parent directory of --output_path of the generator "
        "(chunk/batch directories that directly contain generation_metrics.json are lined up directly below)",
    )
    ap.add_argument("--arm-label", required=True, help="value to write in the arm column of the output CSV (N / A / R etc.)")
    ap.add_argument("--out", required=True, help="output CSV path (never give the existing gamma_unrelaxed.csv anchor)")
    ap.add_argument("--allow-overwrite", action="store_true", help="allow overwriting even if --out is an existing file")
    ap.add_argument("--dry-run", action="store_true", help="skip the MACE calculation and only build the manifest and write the CSV (no GPU needed)")
    args = ap.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()
