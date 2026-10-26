#!/usr/bin/env python3
"""Data rebuild, r2, stage D2-1: matching for the removal of duplicates (chunk for an array). The default of r2 is no window = all pairs within a group are matched. Only when --window-ev W (or the environment variable DATA_WINDOW_EV) is given explicitly are the pairs narrowed to |Delta E| <= W.

  stage_d2_match_chunk.py plan       [--csv NAME] [--match-subdir NAME] [--no-reuse-pilot] [--results-dir DIR] [--window-ev W] [--pairs-per-task N]
  stage_d2_match_chunk.py chunk ID   [same as above]      # 1 task = N pairs (default 1000; also possible with the environment variable DATA_D2_PAIRS_PER_TASK) -> <results>/<match-subdir>/chunk_<ID>.csv

Default: csv=gamma_all.csv, match-subdir=match. The check group (k=2, dx=0.0) reuses the all-pair matching of D1
(match_trial/), so it is not matched here.
Re-run check (the "re-run on gamma_dedup" of the acceptance condition): --csv gamma_dedup.csv --match-subdir match_verify --no-reuse-pilot.
Result columns: group_key,row_i,row_j,fit,sec,error (row is the 0-based row position of the input CSV). Exceptions are recorded in error and fit is left empty (not treated as non-duplicate).
The result is a separate file per task. If it already exists, nothing is done.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from importlib.metadata import version

import stage_d_common as D

COLS = ["group_key", "row_i", "row_j", "fit", "sec", "error"]


def add_args(ap):
    ap.add_argument("--csv", default=D.GAMMA_ALL)
    ap.add_argument("--match-subdir", default="match")
    ap.add_argument("--no-reuse-pilot", action="store_true")
    ap.add_argument("--results-dir", default=None)
    D.add_window_arg(ap)
    ap.add_argument("--pairs-per-task", type=int, default=None,
                    help=f"number of pairs per task (default {D.PAIRS_PER_TASK}. Also possible with the environment variable DATA_D2_PAIRS_PER_TASK)")


def n_tasks(pairs, cs):
    return (len(pairs) + cs - 1) // cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "chunk"])
    ap.add_argument("id", nargs="?", type=int)
    add_args(ap)
    args = ap.parse_args()
    res = D.results_dir(args.results_dir)
    rows = D.read_gamma(res / args.csv)
    W, CS = D.window_ev(args.window_ev), D.pairs_per_task(args.pairs_per_task)
    pairs, sizes, pilot_all = D.d2_plan(rows, reuse_pilot=not args.no_reuse_pilot, w=W)
    if args.cmd == "plan":
        print(n_tasks(pairs, CS))
        return

    from pymatgen.analysis.structure_matcher import StructureMatcher
    tid = args.id
    assert tid is not None and 0 <= tid < n_tasks(pairs, CS), (tid, n_tasks(pairs, CS))
    out_dir = res / args.match_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"chunk_{tid:04d}.csv"
    if fn.exists():
        print("exists, skip", fn)
        return
    byrow = {r["row"]: r for r in rows}
    sm0 = StructureMatcher(**D.UNI)
    print(f"window_eV={W} pairs_per_task={CS} pymatgen {version('pymatgen')} matcher: stol={sm0.stol} ltol={sm0.ltol} angle_tol={sm0.angle_tol} "
          f"prim={sm0._primitive_cell} scale={sm0._scale} supercell={sm0._supercell}", flush=True)
    cache, out = {}, []
    t0 = time.time()
    for (gk, i, j) in pairs[tid * CS:(tid + 1) * CS]:
        fit, err, sec = "", "", ""
        try:
            for r in (i, j):
                if r not in cache:
                    cache[r] = D.load_relaxed(byrow[r])
            t1 = time.time()
            fit = int(bool(StructureMatcher(**D.UNI).fit(cache[i], cache[j])))
            sec = f"{time.time() - t1:.4f}"
        except Exception as e:  # noqa: BLE001  exceptions are recorded in error (not regarded as non-duplicate)
            err = repr(e)
        out.append([gk, i, j, fit, sec, err])
    tmp = fn.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(out)
    os.replace(tmp, fn)
    print(f"done {fn.name} rows={len(out)} fits={sum(1 for r in out if r[3] == 1)} "
          f"errors={sum(1 for r in out if r[5])} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
