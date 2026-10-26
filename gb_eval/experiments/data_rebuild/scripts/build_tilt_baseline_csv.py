#!/usr/bin/env python
"""Data rebuild, stage E-6: makes the csv-to-dataset input CSV for the baseline training from gamma_dedup.csv and split.json (train/val/test).

Transcribed from (read only): GC/runs/baseline_generation/scripts/build_tilt_baseline_dedup_csv.py. The differences are only what the plan requires:
  (1) Input: <results-dir>/gamma_dedup.csv (all rows converged; position = the same space as the split of the proposed model) and <results-dir>/split.json.
  (2) Split: SPLIT_KEYS = train/val/test = train_indices / val_indices / test_indices. The assertion of the counts is replaced by agreement with n_* of split.json
      (the earlier one was fixed at 3878). The union of the three must agree with all rows of the CSV and they must be mutually disjoint.
  (3) ALLOWED_NATOMS = {112,114,116,118,120} (same as before).
  (4) Default output: MN/datasets/csv/scm_al_baseline/{train,val,test}.csv. Records go to baseline_csv_split_manifest.json and
      baseline_csv_conversion_check.log in <results-dir>. An exception if the output destination has existing files (no --force).
  material_id = r{row_idx}_{cnid}_k{k}_s{seed}_dx{dx_A:+.3f} (same as before. row_idx is the row position in gamma_dedup.csv).
The GB mechanism (manifest / sidecar / template_path) is not referenced (definition of the baseline). The main mattergen package is not imported (pymatgen only).

Run (do not start a plain python on the login node):
  ssh <LOGIN_NODE> "srun -p <CPU_PARTITION> -n1 -c2 --immediate=60 ${MACE_PY} <this file> [--results-dir DIR] [--output-csv-dir DIR]"
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

csv.field_size_limit(sys.maxsize)

GC = Path(_gbrel.WORK + '/workspace')
RUN = GC / "runs" / "data_rebuild"
RESULTS_DEFAULT = Path(os.environ.get("DATA_RESULTS_DIR") or RUN / "results")
OUTPUT_CSV_DIR_DEFAULT = Path(_gbrel.WORK + '/scm-work/datasets/csv/scm_al_baseline')
GAMMA_NAME = "gamma_dedup.csv"
SPLIT_NAME = "split.json"

ALLOWED_NATOMS = {112, 114, 116, 118, 120}
INPUT_GAMMA_COL = "gamma_Jm2"
PROPERTY_COL = "gb_gamma_J_m2"   # existing keys already registered in PROPERTY_SOURCE_IDS (same as before)
SPLIT_KEYS = {"train": "train_indices", "val": "val_indices", "test": "test_indices"}
N_KEYS = {"train": "n_train", "val": "n_val", "test": "n_test"}


def resolve_view(path: str) -> str:
    """Absorbs the difference of the mount views of the login and compute nodes (<CLUSTER_MOUNT> <-> <CLUSTER_MOUNT>). FileNotFoundError if absent."""
    candidates = [path]
    if path.startswith('/mnt/CLUSTER_ROOT_A'):
        candidates.append(path.replace('/mnt/CLUSTER_ROOT_A', '/CLUSTER_ROOT_B', 1))
    elif path.startswith('/CLUSTER_ROOT_B'):
        candidates.append("/mnt" + path)
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"path not found in any view: {path} (tried {candidates})")


def material_id_of(row_idx: int, row: dict) -> str:
    return f"r{row_idx}_{row['cnid']}_k{row['k']}_s{row['seed']}_dx{float(row['dx_A']):+.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DEFAULT))
    ap.add_argument("--output-csv-dir", default=str(OUTPUT_CSV_DIR_DEFAULT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    res, out_dir = Path(args.results_dir), Path(args.output_csv_dir)
    gamma_csv, split_json = res / GAMMA_NAME, res / SPLIT_NAME
    rec_manifest = res / "baseline_csv_split_manifest.json"
    rec_log = res / "baseline_csv_conversion_check.log"
    outs = [out_dir / f"{s}.csv" for s in SPLIT_KEYS] + [rec_manifest, rec_log]
    for p in outs:
        if p.exists() and not args.force:
            raise SystemExit(f"FATAL: {p} exists (overwriting is prohibited; use --force to overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(gamma_csv, newline="") as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for r in all_rows if r["converged"] == "True"]
    n_total = len(rows)
    assert len(all_rows) == n_total, "there are non-converged rows (stopping because the position space would deviate from that of the proposed model)"
    with open(split_json) as f:
        idx = json.load(f)
    expect = {s: int(idx[N_KEYS[s]]) for s in SPLIT_KEYS}
    assert n_total == sum(expect.values()), (n_total, expect)
    for s, key in SPLIT_KEYS.items():
        assert len(idx[key]) == expect[s], (s, len(idx[key]), expect[s])

    manifest = {"gamma_csv": str(gamma_csv), "split_json": str(split_json), "splits": {}}
    seen_positions: set[int] = set()
    density_train: list[float] = []
    gamma_all: list[float] = []
    natoms_by_split: dict[str, dict[int, int]] = {}

    for split, key in SPLIT_KEYS.items():
        positions = idx[key]
        assert not (set(positions) & seen_positions), f"split {split} overlaps another split"
        assert all(isinstance(p, int) and 0 <= p < n_total for p in positions), f"{split}: position out of range"
        seen_positions |= set(positions)
        out_rows, mids = [], []
        natoms_hist: dict[int, int] = {}
        for i in positions:
            row = rows[i]
            structure = Structure.from_file(resolve_view(row["dir"] + "/POSCAR_relaxed"))
            n_atoms = len(structure)
            assert n_atoms in ALLOWED_NATOMS, f"row[{i}] natoms={n_atoms} not in {sorted(ALLOWED_NATOMS)}"
            assert n_atoms == int(row["natoms"]), f"row[{i}] POSCAR natoms {n_atoms} != CSV natoms {row['natoms']}"
            natoms_hist[n_atoms] = natoms_hist.get(n_atoms, 0) + 1
            mid = material_id_of(i, row)
            mids.append(mid)
            gamma_val = float(row[INPUT_GAMMA_COL])
            gamma_all.append(gamma_val)
            if split == "train":
                density_train.append(n_atoms / structure.volume)
            out_rows.append({"material_id": mid, PROPERTY_COL: repr(gamma_val), "cif": CifWriter(structure).__str__()})
        assert len(set(mids)) == len(mids), f"{split}: duplicate material_id"
        out_path = out_dir / f"{split}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["material_id", PROPERTY_COL, "cif"])
            w.writeheader()
            w.writerows(out_rows)
        with open(out_path, newline="") as f:  # read-back cross-check (number of rows, order, material_id, gamma)
            reread = list(csv.DictReader(f))
        assert len(reread) == len(out_rows) == expect[split], (split, len(reread), expect[split])
        for pos, wr in zip(positions, reread):
            assert wr["material_id"] == material_id_of(pos, rows[pos])
            assert float(wr[PROPERTY_COL]) == float(rows[pos][INPUT_GAMMA_COL])
        natoms_by_split[split] = dict(sorted(natoms_hist.items()))
        manifest["splits"][split] = {"n": len(positions), "positions": positions, "material_ids": mids, "source_indices_key": key}
        print(f"{split}: {len(out_rows)} rows -> {out_path}", flush=True)

    assert len(seen_positions) == n_total, (len(seen_positions), n_total)
    rec_manifest.write_text(json.dumps(manifest))
    avg = sum(density_train) / len(density_train)
    ad = res / "average_density.json"
    cmp_line = "average_density.json: not present"
    if ad.exists():
        ref = json.loads(ad.read_text())["average_density_train_split"]
        assert abs(ref - avg) <= 1e-12, (ref, avg)
        cmp_line = f"average_density.json agrees within 1e-12 ({ref!r})"
    with open(rec_log, "w") as f:
        f.write("=== Data rebuild, stage E-6: check of the CSV conversion for the baseline ===\n")
        f.write(f"input CSV: {gamma_csv}\nsplit: {split_json}\n")
        f.write(f"n_total (converged rows) = {n_total}\n")
        for s in SPLIT_KEYS:
            f.write(f"  {s}: {expect[s]} rows (source key {SPLIT_KEYS[s]}) natoms={natoms_by_split[s]}\n")
        f.write(f"union of positions = {len(seen_positions)} (= n_total)  mutually disjoint\n")
        f.write(f"gamma_Jm2 min={min(gamma_all)!r} max={max(gamma_all)!r}\n")
        f.write(f"train average density (n/volume simple mean) = {avg!r}\n{cmp_line}\n")
    print(f"DONE build_tilt_baseline_csv: counts={expect} avg_density_train={avg!r}", flush=True)


if __name__ == "__main__":
    main()
