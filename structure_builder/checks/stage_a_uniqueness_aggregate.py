#!/usr/bin/env python3
"""Data rebuild, stage A-5 aggregation: collects results/stage_a_uniqueness/chunk_*.csv and writes stage_a_dx_uniqueness.csv.

  stage_a_uniqueness_aggregate.py [--chunk-dir DIR] [--out-dir DIR]

- Requires that the result files of all tasks are present, the total number of rows = 4,650 and 0 exceptions (error column) (otherwise non-zero exit = 2).
- If there is a pair with fit=True (match), "REPORT" is printed. The processing is not stopped (exit code 0; duplicates are removed in stage D).
- Output: stage_a_dx_uniqueness.csv (columns: dx_A, dx_idx, cnid_a, cnid_b. Only matching pairs. Header only if 0 pairs),
  stage_a_uniqueness_summary.json。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "lib"))  # structure_builder/lib
import builder_common as C
import stage_a_uniqueness_chunk as U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", default=str(U.OUT_DIR))
    ap.add_argument("--out-dir", default=str(C.RESULTS))
    args = ap.parse_args()
    chunk_dir, out_dir = Path(args.chunk_dir), Path(args.out_dir)

    nt = U.n_tasks()
    total = U.n_pairs_total()
    missing = [t for t in range(nt) if not (chunk_dir / f"chunk_{t:04d}.csv").exists()]
    rows = []
    for t in range(nt):
        fn = chunk_dir / f"chunk_{t:04d}.csv"
        if fn.exists():
            with open(fn, newline="") as f:
                rows.extend(csv.DictReader(f))
    errors = [r for r in rows if r["error"]]
    unresolved = [r for r in rows if not r["error"] and r["fit"] not in ("0", "1")]
    matches = [r for r in rows if r["fit"] == "1"]

    summary = {
        "n_tasks_expected": nt, "n_tasks_missing": len(missing), "n_pairs_expected": total,
        "n_pairs_read": len(rows), "n_errors": len(errors), "n_unresolved": len(unresolved),
        "n_match_pairs": len(matches),
        "matches": [{k: r[k] for k in ("dx_A", "cnid_a", "cnid_b")} for r in matches],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "stage_a_dx_uniqueness.csv.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dx_A", "dx_idx", "cnid_a", "cnid_b"])
        for r in matches:
            w.writerow([r["dx_A"], r["dx_idx"], r["cnid_a"], r["cnid_b"]])
    os.replace(tmp, out_dir / "stage_a_dx_uniqueness.csv")
    with open(out_dir / "stage_a_uniqueness_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "matches"}))

    bad = missing or errors or unresolved or len(rows) != total
    if bad:
        print(f"FAIL: missing_tasks={len(missing)} rows={len(rows)}/{total} errors={len(errors)} "
              f"unresolved={len(unresolved)}")
        raise SystemExit(2)
    if matches:
        print(f"REPORT: {len(matches)} matching pair(s) after dx (report to parent; not stopping):")
        for r in matches:
            print(f"  dx={r['dx_A']} {r['cnid_a']} == {r['cnid_b']}")
    else:
        print("OK: no matching pair in any dx (all 31 structures unique per dx)")


if __name__ == "__main__":
    main()
