#!/usr/bin/env python3
"""compute_gamma_pre_proposed.py -- gamma before relaxation (gamma_pre) of the proposed model.

Transcribed from: ${GBREL_CODE}/gb_eval/generation_hooks/compute_gamma_pre_proposed.py (read only; copied and modified)
  Reads v2 (compute_gamma_unrelaxed.py) with importlib and uses single_point_gamma, in_plane_area_bc, build_calc, discover_chunk_dirs and
  OUT_FIELDS of v2 as they are (the formula of gamma is not copied).

[Differences from the source]
  D-1: The processing that raises ComputeError and stops the whole run when a cell is non-orthogonal (source :116-120) was changed so that the row gets error=nonorthogonal (gb_dx and gamma empty) and
       the processing continues. The decision is not the component decision of the source (|cell[0,1]|,|cell[0,2]|>1e-6; always False for CIF input) but
       cell_orthogonality.is_nonorthogonal (the alpha, beta, gamma of cellpar deviate from 90 deg by more than 0.01 deg; shared with check_chunk0.py).
       The other ComputeError (missing key, count mismatch, run_meta mismatch, etc., i.e. a breakdown of the pipeline) are raised as before.
  D-2: columns added to the output CSV: w, chunk, target_id, is_inner (1 if target_id<=4), ckpt_sha256, non_al_count, min_dist_A.
  D-3: non-finite values of coordinates, energy and gamma leave the row with error=nonfinite (no special workaround is added).
  D-4: gamma_target_J_m2 and seed of generation_metrics.json are cross-checked with run_meta.json (17 keys); a mismatch exits non-zero.
       run_meta.ckpt_sha256 is cross-checked with the sha256 of --ckpt-resolved.
  D-5: v2.cmd_run is not called; the same flow (refusal of existing outputs, vram_precheck, build_calc, single-point calculation, CSV writing) is held in cmd_run of this file
       [author's choice]. Because D-1 to D-3 need per-row error and additional columns (v2.cmd_run fills the error column only with exceptions of reading the CIF).
       The reading of the CIF is gathered in one place, read_cif() (the validation case is inserted here).

--dry-run: does not load MACE and only builds the manifest and writes the CSV (no GPU needed). A real calculation requires a GPU.
Execution environment: the evaluation environment (compute node. Not executed on the login node). PYTHONDONTWRITEBYTECODE=1.
Usage:
  python compute_gamma_pre_proposed.py --output-root <gen/proposed/w<W>> --arm-label proposed_w<W> --w <W> \
      --ckpt-resolved <ckpt_resolved_proposed.json> --out <gamma_pre.csv> --template-a 54.15601152562611 [--dry-run]
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_orthogonality as CO  # noqa: E402  (decision function of non-orthogonality; shared with check_chunk0.py)

_V2_CANDIDATES = [
    Path(_gbrel.CODE + '/gb_eval/experiments/dx_gamma_unrelaxed/scripts/compute_gamma_unrelaxed.py'),
    Path(_gbrel.CODE + '/gb_eval/experiments/dx_gamma_unrelaxed/scripts/compute_gamma_unrelaxed.py'),
    Path(_gbrel.CODE + '/gb_eval/experiments/dx_gamma_unrelaxed/scripts/compute_gamma_unrelaxed.py'),
]
V2_PATH = next((p for p in _V2_CANDIDATES if p.is_file()), _V2_CANDIDATES[0])


def _load_v2():
    if not V2_PATH.is_file():
        raise RuntimeError(f"v2 not found: {V2_PATH}")
    spec = importlib.util.spec_from_file_location("compute_gamma_unrelaxed", str(V2_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compute_gamma_unrelaxed"] = mod
    spec.loader.exec_module(mod)
    return mod


V2 = _load_v2()
ComputeError = V2.ComputeError
single_point_gamma = V2.single_point_gamma  # the formula is that of v2 as it is (not copied)

EXTRA_FIELDS = ["w", "chunk", "target_id", "is_inner", "ckpt_sha256", "non_al_count", "min_dist_A"]
OUT_FIELDS = list(V2.OUT_FIELDS) + EXTRA_FIELDS
REQUIRED_GCM_KEYS = ["gamma_target_J_m2", "n_atoms_per_sample", "num_samples_requested", "seed", "checkpoint_path"]
RUN_META_KEYS = [
    "arm", "w", "chunk", "target_id", "gamma_target", "seed", "num_samples", "ckpt_sha256", "ckpt_epoch",
    "num_atoms_key", "host_z", "solute_zs", "card_hash12", "gamma_csv_path", "bulk_membership_csv_path",
    "manifest_dir", "reference_dir",
]
AL_Z = 13
INNER_MAX_TARGET_ID = 4
_CIF_RE = re.compile(r"^gen_(\d+)\.cif$")
_CHUNK_RE = re.compile(r"^chunk(\d{2})$")


def read_cif(path):
    """The only entrance of reading the CIF (the validation case is inserted here)."""
    from ase.io import read as ase_read

    return ase_read(str(path))


def _require(gcm: dict, key: str, gcm_path: Path):
    if key not in gcm:
        raise ComputeError(f"{gcm_path}: required key {key!r} missing (no silent fallback)")
    return gcm[key]


def min_dist_A(atoms) -> "float | str":
    """Minimum interatomic distance [A] over off-diagonal pairs (MIC). An empty string if it cannot be read or is not finite."""
    n = len(atoms)
    if n < 2:
        return ""
    try:
        d = atoms.get_all_distances(mic=True)
        v = float(d[np.triu_indices(n, 1)].min())
    except Exception:  # noqa: BLE001  for an error row, give it if the CIF can be read, empty otherwise
        return ""
    return v if math.isfinite(v) else ""


def is_finite_structure(atoms) -> bool:
    return bool(np.isfinite(np.asarray(atoms.positions, dtype=float)).all()
                and np.isfinite(np.asarray(atoms.cell[:], dtype=float)).all())


def load_run_meta(chunk_dir: Path, *, arm: str, w: float, ckpt_sha256: str) -> dict:
    """Reads run_meta.json (17 keys) and checks the chunk number, arm, w and ckpt_sha256. A mismatch is ComputeError."""
    p = chunk_dir / "run_meta.json"
    if not p.is_file():
        raise ComputeError(f"{p}: run_meta.json missing")
    rm = json.loads(p.read_text())
    missing = [k for k in RUN_META_KEYS if k not in rm]
    if missing:
        raise ComputeError(f"{p}: run_meta missing keys {missing}")
    m = _CHUNK_RE.match(chunk_dir.name)
    if m is None or int(m.group(1)) != int(rm["chunk"]):
        raise ComputeError(f"{p}: run_meta.chunk={rm['chunk']} does not match dir name {chunk_dir.name}")
    if rm["arm"] != arm:
        raise ComputeError(f"{p}: run_meta.arm={rm['arm']!r} != {arm!r}")
    if float(rm["w"]) != float(w):
        raise ComputeError(f"{p}: run_meta.w={rm['w']} != --w {w}")
    if rm["ckpt_sha256"] != ckpt_sha256:
        raise ComputeError(f"{p}: run_meta.ckpt_sha256={rm['ckpt_sha256']} != ckpt_resolved sha256 {ckpt_sha256}")
    return rm


def read_ckpt_sha(ckpt_resolved: Path, arm: str) -> str:
    j = json.loads(Path(ckpt_resolved).read_text())
    if j.get("arm") != arm:
        raise ComputeError(f"{ckpt_resolved}: arm={j.get('arm')!r} != {arm!r}")
    return j["sha256"]


def build_manifest_rows(chunk_dirs, arm_label: str, *, template_a: float, w: float, ckpt_sha256: str) -> list[dict]:
    """Makes a manifest row with 1 structure = 1 row from generation_metrics.json, run_meta.json and cif/gen_i.cif of each chunk (MACE is not called).
    A non-orthogonal cell sets error=nonorthogonal of the row and continues (D-1)."""
    rows: list[dict] = []
    for cdir in chunk_dirs:
        cdir = Path(cdir)
        gcm_path = cdir / "generation_metrics.json"
        with open(gcm_path) as f:
            gcm = json.load(f)
        gamma_t = _require(gcm, "gamma_target_J_m2", gcm_path)
        n_atoms = _require(gcm, "n_atoms_per_sample", gcm_path)
        n_req = _require(gcm, "num_samples_requested", gcm_path)
        seed = _require(gcm, "seed", gcm_path)
        ckpt = _require(gcm, "checkpoint_path", gcm_path)
        rm = load_run_meta(cdir, arm="proposed", w=w, ckpt_sha256=ckpt_sha256)
        if float(rm["gamma_target"]) != float(gamma_t):
            raise ComputeError(f"{cdir}: run_meta.gamma_target={rm['gamma_target']!r} != generation_metrics.gamma_target_J_m2={gamma_t!r}")
        if int(rm["seed"]) != int(seed):
            raise ComputeError(f"{cdir}: run_meta.seed={rm['seed']} != generation_metrics.seed={seed}")

        cif_dir = cdir / "cif"
        if not cif_dir.is_dir():
            raise ComputeError(f"{cdir}: cif/ not found (expand the zip with prep_and_verify_unrelaxed.py extract before running)")
        named = []
        for p in cif_dir.glob("*.cif"):
            m = _CIF_RE.match(p.name)
            if m is None:
                raise ComputeError(f"{cdir}: unexpected cif file name {p.name!r} (expected gen_<int>.cif)")
            named.append((int(m.group(1)), p))
        cifs = [p for _, p in sorted(named)]  # numerical order
        if not (len(n_atoms) == n_req == len(cifs)):
            raise ComputeError(
                f"{cdir}: length mismatch: len(n_atoms_per_sample)={len(n_atoms)} "
                f"num_samples_requested={n_req} n_cif={len(cifs)} "
                "(refusing to silently zip mismatched-length lists)"
            )
        for i in range(n_req):
            atoms = read_cif(cifs[i])
            if len(atoms) != n_atoms[i]:
                raise ComputeError(f"{cifs[i]}: len(atoms)={len(atoms)} != n_atoms_per_sample[{i}]={n_atoms[i]}")
            cell = atoms.cell[:]
            nonortho = CO.is_nonorthogonal(atoms)   # angle decision
            dx = "" if nonortho else (float(cell[0, 0]) - template_a) / 2.0
            tid = int(rm["target_id"])
            rows.append({
                "arm": arm_label,
                "dx_A": dx,
                "gamma_target": gamma_t,
                "seed": seed,
                "struct_idx": 0,
                "chunk_dir": str(cdir),
                "cif_index": i,
                "cif_path": str(cifs[i]),
                "gb_dx_realized_A": dx,
                "dx_mode": "variable_length_generated",
                "check5_abs_diff_from_gcm": "",
                "checkpoint_path": ckpt,
                "w": float(rm["w"]),
                "chunk": int(rm["chunk"]),
                "target_id": tid,
                "is_inner": 1 if tid <= INNER_MAX_TARGET_ID else 0,
                "ckpt_sha256": rm["ckpt_sha256"],
                "_atoms": atoms,
                "_nonortho": bool(nonortho),
            })
    return rows


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def compute_row(r: dict, calc, dry_run: bool) -> dict:
    """Manifest row -> output row. error is nonorthogonal / nonfinite / <exception> / empty."""
    atoms = r["_atoms"]
    out = {k: r.get(k, "") for k in OUT_FIELDS}
    for k in ("gamma_unrelaxed", "natoms", "area_A2", "energy_eV", "error"):
        out[k] = ""
    out["natoms"] = len(atoms)
    out["non_al_count"] = int((np.asarray(atoms.numbers) != AL_Z).sum())
    out["min_dist_A"] = min_dist_A(atoms)
    if r["_nonortho"]:
        out["error"] = "nonorthogonal"
        return out
    if not is_finite_structure(atoms):
        out["error"] = "nonfinite"
        return out
    if dry_run:
        out["error"] = "DRY_RUN_SKIPPED"
        return out
    try:
        res = single_point_gamma(atoms, calc)
    except Exception as e:  # noqa: BLE001  failures on the MACE side are kept in the row (handled as in v2)
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    if not (_finite(res["gamma_unrelaxed"]) and _finite(res["energy_eV"]) and _finite(res["area_A2"])):
        out["error"] = "nonfinite"
        return out
    out["gamma_unrelaxed"] = res["gamma_unrelaxed"]
    out["natoms"] = res["natoms"]
    out["area_A2"] = res["area_A2"]
    out["energy_eV"] = res["energy_eV"]
    return out


def write_csv(path: Path, out_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        wr.writeheader()
        for row in out_rows:
            wr.writerow({k: row.get(k, "") for k in OUT_FIELDS})


def cmd_run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    out_csv = Path(args.out)
    if out_csv.exists() and not args.allow_overwrite:
        raise ComputeError(f"--out {out_csv} already exists (no overwrite; pass --allow-overwrite to permit)")
    ckpt_sha = read_ckpt_sha(args.ckpt_resolved, "proposed")
    chunk_dirs = V2.discover_chunk_dirs(output_root)
    print(f"discovered {len(chunk_dirs)} chunk directories under {output_root}", flush=True)
    manifest = build_manifest_rows(chunk_dirs, args.arm_label, template_a=args.template_a, w=args.w, ckpt_sha256=ckpt_sha)
    print(f"manifest rows (1 struct each): {len(manifest)}", flush=True)

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
    out_rows = []
    n_ok = n_err = 0
    for n_done, r in enumerate(manifest, 1):
        row = compute_row(r, calc, args.dry_run)
        out_rows.append(row)
        if row["error"] and row["error"] != "DRY_RUN_SKIPPED":
            n_err += 1
            print(f"[ERROR] {r['chunk_dir']} cif_index={r['cif_index']}: {row['error']}", flush=True)
        else:
            n_ok += 1
        if n_done % 50 == 0 or n_done == len(manifest):
            print(f"[progress] {n_done}/{len(manifest)} n_ok={n_ok} n_err={n_err} elapsed={time.time() - t0:.1f}s", flush=True)
    write_csv(out_csv, out_rows)
    n_nonortho = sum(1 for r in out_rows if r["error"] == "nonorthogonal")
    n_nonfinite = sum(1 for r in out_rows if r["error"] == "nonfinite")
    n_non_al = sum(1 for r in out_rows if int(r["non_al_count"]) > 0)
    mode = "DRY_RUN" if args.dry_run else "MAIN"
    print(f"{mode}_DONE arm={args.arm_label} n_rows={len(out_rows)} n_ok={n_ok} n_err={n_err} n_nonorthogonal={n_nonortho} "
          f"n_nonfinite={n_nonfinite} n_non_al={n_non_al} elapsed={time.time() - t0:.1f}s -> {out_csv}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", required=True, help="directory that directly contains chunkNN/ (generation_metrics.json + run_meta.json + cif/gen_i.cif)")
    ap.add_argument("--arm-label", required=True, help="value to write in the arm column of the output CSV")
    ap.add_argument("--w", required=True, type=float, help="guidance strength (cross-checked with run_meta.w)")
    ap.add_argument("--ckpt-resolved", required=True, type=Path, help="ckpt_resolved_proposed.json (cross-checked with run_meta.ckpt_sha256)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--template-a", required=True, type=float, help="cell a of the template [A] (required; no default). 54.15601152562611 for Al")
    ap.add_argument("--allow-overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="skip MACE and only build the manifest and write the CSV (no GPU needed)")
    args = ap.parse_args()
    try:
        cmd_run(args)
    except ComputeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
