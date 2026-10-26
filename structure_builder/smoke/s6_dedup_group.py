#!/usr/bin/env python3
"""S6: rerun the stage D2 matching and the connected-component selection for one (dx, natoms) group and compare with the released dedup.

  SB_WORK=<work dir of S4 (holds results/gamma_all.csv)> python s6_dedup_group.py --released $GBREL_DATA/training \
          --group dx_p0.000_n120 --out s6_compare.txt

Writes the converged rows of the group to $SB_WORK/s6/results/gamma_all.csv, runs stage_d2_match_chunk.py (all pairs) and
stage_d2_aggregate.py (--no-reuse-pilot --skip-reference-check), and compares the kept (cnid, k, seed) set with the rows of the released
gamma_dedup.csv that belong to the same group. Only numbers are written; no verdict.
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

SB_WORK = Path(os.environ["SB_WORK"])
SB = Path(__file__).resolve().parents[1]


def group_of(row):
    dx = float(row["dx_A"])
    return "dx_%s%.3f_n%d" % ("p" if dx >= 0 else "m", abs(dx), int(row["natoms"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", required=True)
    ap.add_argument("--group", default="dx_p0.000_n120")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    allp = SB_WORK / "results" / "gamma_all.csv"
    work6 = SB_WORK / "s6"
    res6 = work6 / "results"
    res6.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(allp, newline="")))
    sub = [r for r in rows if group_of(r) == a.group and r["converged"] == "True"]
    cols = ["cnid", "k", "seed", "dir", "converged", "natoms", "energy_eV", "area_A2", "gamma_Jm2", "dx_A"]
    with open(res6 / "gamma_all.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sub:
            w.writerow([r[c] for c in cols])
    print("S6 group=%s converged rows=%d (of %d rows in the group, converged or not)" % (a.group, len(sub), sum(1 for r in rows if group_of(r) == a.group)), flush=True)
    env = dict(os.environ, SB_WORK=str(work6))
    d2 = SB / "stageD_gamma_dedup"
    common = ["--results-dir", str(res6), "--no-reuse-pilot"]
    n = int(subprocess.run([sys.executable, str(d2 / "stage_d2_match_chunk.py"), "plan"] + common, env=env, capture_output=True, text=True, check=True).stdout.strip())
    print("S6 D2 tasks:", n, flush=True)
    for t in range(n):
        rc = subprocess.run([sys.executable, str(d2 / "stage_d2_match_chunk.py"), "chunk", str(t)] + common, env=env).returncode
        print("S6 chunk", t, "exit", rc, flush=True)
    rc = subprocess.run([sys.executable, str(d2 / "stage_d2_aggregate.py"), "--skip-reference-check"] + common, env=env).returncode
    print("S6 aggregate exit code:", rc)
    kept = {(r["cnid"], r["k"], r["seed"]) for r in csv.DictReader(open(res6 / "gamma_dedup.csv", newline=""))}
    rel = [r for r in csv.DictReader(open(Path(a.released) / "gamma_dedup.csv", newline="")) if group_of(r) == a.group]
    relset = {(r["cnid"], r["k"], r["seed"]) for r in rel}
    lines = ["group %s" % a.group,
             "converged rows in the group (input to the matching): %d" % len(sub),
             "kept by the rerun: %d    released rows of the group: %d" % (len(kept), len(relset)),
             "kept by the rerun only: %s" % sorted(kept - relset),
             "released only: %s" % sorted(relset - kept),
             "identical sets: %s" % (kept == relset)]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join("S6 " + l for l in lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
