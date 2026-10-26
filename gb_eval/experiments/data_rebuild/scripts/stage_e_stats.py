#!/usr/bin/env python3
"""Data rebuild, stage E-3: statistics of the train rows only (mu and sigma2 of dx, distribution of the number of atoms, average density, evaluation targets and generation seeds).

  <python> stage_e_stats.py [--results-dir DIR] [--force]

Input: <results-dir>/gamma_dedup.csv, <results-dir>/split.json (only the rows of train_indices are used),
      dir/POSCAR_relaxed of each train row (for the average density only: simple mean of n/volume. Source of the definition:
      GC/runs/baseline_generation/scripts/build_tilt_baseline_dedup_csv.py:144).
Output (<results-dir>. An exception if it exists; overwrite with --force):
  dxvpsde_stats.json      mu_uncond = mean(dx_A), sigma2 = population variance (ddof=0) (the same definition as compute_dxvpsde_stats_al_dedup.py)
  num_atoms_dist_train.json  the counts of natoms and the dictionary row to paste into num_atoms_distribution.py (not pasted this time)
  average_density.json    simple mean of n/volume of the train structures (recomputed value. Same as the value of the model specification. This value is also used for the training and settings)
  eval_targets.json       q10, q90, min, max, the 5 inner points, delta, the 2 extrapolation points, the targets of chunks c=0..13 and seed=20301000+c
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

import stage_e_common as E


def dx_stats(train: list[dict]) -> dict:
    dx = np.asarray([r["dx_A"] for r in train], dtype=np.float64)
    mu = float(np.mean(dx))
    sigma2 = float(np.var(dx, ddof=0))
    if not (np.isfinite(mu) and np.isfinite(sigma2) and sigma2 > 0):
        raise SystemExit(f"FATAL: degenerate dx stats mu={mu} sigma2={sigma2}")
    levels = sorted(set(dx.tolist()))
    return {"mu_uncond": mu, "sigma2": sigma2, "n_rows": int(dx.size), "n_unique_levels": len(levels),
            "level_counts": {str(lv): int((dx == lv).sum()) for lv in levels}}


def natoms_dist(train: list[dict]) -> dict:
    cnt = collections.Counter(r["natoms"] for r in train)
    n = len(train)
    assert sum(cnt.values()) == n
    entries = [{"p": cnt[k] / n, "n": k, "count": cnt[k]} for k in sorted(cnt)]
    lines = [f'        {{"p": {e["p"]!r}, "n": {e["n"]}}},  # count {e["count"]}' for e in entries]
    block = (f'    "{E.NUM_ATOMS_KEY}": [\n' + "\n".join(lines) + "\n    ],")
    outside = {k: c for k, c in cnt.items() if k not in E.ALLOWED_NATOMS}
    return {"n_train": n, "entries": entries, "support": sorted(cnt), "natoms_outside_allowed": outside,
            "num_atoms_distribution_py_block": block,
            "note": "The key is not added to num_atoms_distribution.py in stage E of this plan (the values are fixed by the production data). Paste the block above."}


def eval_targets(train: list[dict]) -> dict:
    g = np.asarray([r["gamma_Jm2"] for r in train], dtype=np.float64)
    q10 = float(np.percentile(g, 10))
    q90 = float(np.percentile(g, 90))
    gmin, gmax = float(g.min()), float(g.max())
    inner = [float(x) for x in np.linspace(q10, q90, 5)]
    delta = (q90 - q10) / 4.0
    lo, hi = gmin - delta, gmax + delta
    targets = inner + [lo, hi]
    labels = ["inner0", "inner1", "inner2", "inner3", "inner4", "extrap_low", "extrap_high"]
    chunks = [{"chunk": c, "target_index": c // 2, "label": labels[c // 2], "target_gamma_Jm2": targets[c // 2],
               "seed": E.CHUNK_SEED_BASE + c} for c in range(E.N_CHUNKS)]
    return {"gamma_col": "gamma_Jm2", "n_train_rows": int(g.size), "percentile_method": "numpy.percentile default (linear)",
            "q10": q10, "q90": q90, "min": gmin, "max": gmax, "delta": delta,
            "targets": targets, "labels": labels,
            "order": "[inner0..4 (ascending), extrap_low = min - delta, extrap_high = max + delta]",
            "chunk_seed_rule": "chunk c = 0..13: target = targets[c // 2], seed = 20301000 + c (same value for proposed and baseline)",
            "chunks": chunks}


def average_density(train: list[dict]) -> dict:
    from pymatgen.core import Structure
    dens = []
    for r in train:
        s = Structure.from_file(str(Path(r["dir"]) / "POSCAR_relaxed"))
        if len(s) != r["natoms"]:
            raise SystemExit(f"FATAL: POSCAR natoms {len(s)} != CSV natoms {r['natoms']} for {r['dir']}")
        dens.append(len(s) / s.volume)
    return {"average_density_train_split": sum(dens) / len(dens), "n_train": len(dens), "unit": "atoms/A^3",
            "method": "n_atoms(from Structure)/structure.volume, simple mean over the train split only "
                      "(the same definition as build_tilt_baseline_dedup_csv.py:144)",
            "card_value_used_in_configs": E.CARD_AVERAGE_DENSITY,
            "note": "The value of the model specification is used for the training and settings. This recomputed value is for the record."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-density", action="store_true", help="skip the average density that uses pymatgen (normally not used)")
    a = ap.parse_args()
    res = E.results_dir(a.results_dir)
    rows = E.read_dedup(res)
    split = E.load_split(res)
    train = E.train_rows(rows, split)
    if len(train) != split["n_train"]:
        raise SystemExit(f"FATAL: len(train)={len(train)} != n_train={split['n_train']}")
    meta = {"design": "data rebuild stage E-3", "gamma_csv": str(res / E.C_GAMMA_DEDUP),
            "split_json": str(res / E.SPLIT_JSON), "n_train": len(train)}
    E.write_json(res / "dxvpsde_stats.json", {**meta, **dx_stats(train),
                 "recipe": "mu_uncond=mean(dx_A), sigma2=population variance (ddof=0), train rows only"}, a.force)
    E.write_json(res / "num_atoms_dist_train.json", {**meta, **natoms_dist(train)}, a.force)
    E.write_json(res / "eval_targets.json", {**meta, **eval_targets(train)}, a.force)
    if not a.skip_density:
        E.write_json(res / "average_density.json", {**meta, **average_density(train)}, a.force)
    print(f"WROTE dxvpsde_stats.json num_atoms_dist_train.json eval_targets.json "
          f"{'' if a.skip_density else 'average_density.json '}in {res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
