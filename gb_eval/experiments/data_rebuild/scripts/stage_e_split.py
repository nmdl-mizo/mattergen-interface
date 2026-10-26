#!/usr/bin/env python3
"""Data rebuild, stage E-1: split by template (permutation with seed 42, 25:3:3).

  <python> stage_e_split.py [--results-dir DIR] [--template-map CSV | --templates-from-csv]
                            [--n-val N --n-test N] [--force]

- The template names (cnid) are sorted lexicographically, rearranged with numpy.random.default_rng(42).permutation(n),
  and the first n_train are train, the next n_val are val and the rest are test. n_val, n_test and n_train come from stage_e_common.split_counts
  (25/3/3 for n=31). The counts can be overridden explicitly with --n-val/--n-test (for small tests).
- The default set of templates is all 31 of template_map.csv (--template-map, default RUN/results/template_map.csv).
  With --templates-from-csv, the set of cnid appearing in gamma_dedup.csv is used (for small tests. Not all 31 are required).
- Each row of gamma_dedup.csv is assigned to the side of its cnid. The *_indices of the output split.json are the
  0-based row positions of gamma_dedup.csv (ascending). val_dx_indices is an empty list.
Results directory: --results-dir or the environment variable DATA_RESULTS_DIR (the default is the production RUN/results).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import stage_e_common as E


def build_split(cnids_all: list[str], rows: list[dict], n_val: int | None = None, n_test: int | None = None) -> dict:
    names = sorted(set(cnids_all))
    if len(names) != len(cnids_all):
        raise SystemExit("FATAL: duplicate template names")
    n = len(names)
    if n_val is None and n_test is None:
        n_train, n_val, n_test = E.split_counts(n)
    else:
        if n_val is None or n_test is None:
            raise SystemExit("FATAL: --n-val and --n-test must be given together")
        n_train = n - n_val - n_test
        if n_train < 1 or n_val < 0 or n_test < 0:
            raise SystemExit(f"FATAL: bad explicit counts n={n} val={n_val} test={n_test}")
    perm = np.random.default_rng(E.SPLIT_SEED).permutation(n)
    order = [names[int(i)] for i in perm]
    t_train, t_val, t_test = order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]
    assert len(t_test) == n_test
    side = {c: "train" for c in t_train} | {c: "val" for c in t_val} | {c: "test" for c in t_test}
    idx = {"train": [], "val": [], "test": []}
    for r in rows:
        if r["cnid"] not in side:
            raise SystemExit(f"FATAL: row {r['row']} cnid={r['cnid']} is not in the template set")
        idx[side[r["cnid"]]].append(r["row"])
    return {
        "train_indices": idx["train"], "val_indices": idx["val"], "test_indices": idx["test"],
        "val_dx_indices": [],
        "n_train": len(idx["train"]), "n_val": len(idx["val"]), "n_test": len(idx["test"]), "n_val_dx": 0,
        "templates_train": sorted(t_train), "templates_val": sorted(t_val), "templates_test": sorted(t_test),
        "seed": E.SPLIT_SEED,
        "n_templates": n, "template_counts": [n_train, n_val, n_test],
        "split_rule": "names sorted; default_rng(42).permutation(n); first n_train / next n_val / rest n_test; "
                      "counts = stage_e_common.split_counts (n_val=n_test=max(1, n*1//10); 31 -> 25/3/3) unless explicit",
        "permutation": [int(i) for i in perm],
        "cnid_0_0_side": side.get("cnid_0_0"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--template-map", default=None, help="default: RUN/results/template_map.csv (production)")
    ap.add_argument("--templates-from-csv", action="store_true", help="build the set of templates from the cnid of gamma_dedup (for small tests)")
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    res = E.results_dir(a.results_dir)
    rows = E.read_dedup(res)
    if a.templates_from_csv:
        cnids = sorted({r["cnid"] for r in rows})
        tm_src = "gamma_dedup_cnids"
    else:
        tm = Path(a.template_map) if a.template_map else E.C.RESULTS / "template_map.csv"
        with open(tm, newline="") as f:
            cnids = [r["cnid"] for r in csv.DictReader(f)]
        if len(cnids) != E.C.N_TEMPLATES:
            raise SystemExit(f"FATAL: template_map has {len(cnids)} templates (expected {E.C.N_TEMPLATES}); "
                             "use --templates-from-csv for small tests")
        tm_src = str(tm)
    out = build_split(cnids, rows, a.n_val, a.n_test)
    if len(cnids) == E.C.N_TEMPLATES and a.n_val is None and out["template_counts"] != [25, 3, 3]:
        raise SystemExit(f"FATAL: 31 templates must give 25/3/3, got {out['template_counts']}")
    out["gamma_csv"] = str(res / E.C_GAMMA_DEDUP)
    out["gamma_csv_sha256"] = E.D.sha256_file(res / E.C_GAMMA_DEDUP)
    out["n_rows"] = len(rows)
    out["template_source"] = tm_src
    E.write_json(res / E.SPLIT_JSON, out, a.force)
    print(f"template_counts={out['template_counts']} n_rows={len(rows)} "
          f"n_train/val/test={out['n_train']}/{out['n_val']}/{out['n_test']} cnid_0_0_side={out['cnid_0_0_side']}")
    print(f"templates_train={out['templates_train']}\ntemplates_val={out['templates_val']}\ntemplates_test={out['templates_test']}")
    print(f"WROTE {res / E.SPLIT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
