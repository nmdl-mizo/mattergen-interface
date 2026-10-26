#!/usr/bin/env python3
"""S4: recompute gamma for every released structure (POSCAR_relaxed + energy.json) with stage D1 and compare with the released CSV.

  SB_WORK=<new work dir> python s4_gamma_recompute.py --released $GBREL_DATA/training --out s4_compare.tsv

1. writes $SB_WORK/relax_inputs/<cnid>.json (the stage-B output format) from the released bulk_membership.csv, pointing at the released
   structure directories, and $SB_WORK/results/template_map.csv (names only);
2. runs stageD_gamma_dedup/stage_d1_build_csv.py (unchanged stage code) -> $SB_WORK/results/gamma_all.csv;
3. compares gamma / energy / area of the rows that exist in the released gamma_dedup.csv. Only numbers are written; no verdict.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

SB_WORK = Path(os.environ["SB_WORK"])
SB = Path(__file__).resolve().parents[1]


def dx_of(tag):
    return (1.0 if tag.startswith("dx_p") else -1.0) * float(tag[4:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", required=True, help="$GBREL_DATA/training")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.released)
    res = SB_WORK / "results"
    inp = SB_WORK / "relax_inputs"
    res.mkdir(parents=True, exist_ok=True)
    inp.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(rel / "bulk_membership.csv", newline="")))
    by_cnid = {}
    for r in rows:
        dxi = (int(r["seed"]) - 10000) // 1000
        # directory of the released structure: find the dx tag from the seed's dx index (10 levels, A_BASE grid)
        by_cnid.setdefault(r["cnid"], []).append((r, dxi))
    dx10 = [-0.75, -0.60, -0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45, 0.60]   # DX10 of lib/builder_common.py
    names = sorted(by_cnid)
    with open(res / "template_map.csv", "w", newline="") as f:
        f.write("cnid\n" + "\n".join(names) + "\n")
    for cnid in names:
        d = {}
        for r, dxi in by_cnid[cnid]:
            dx = dx10[dxi]
            tag = "dx_%s%.3f" % ("p" if dx >= 0 else "m", abs(dx))
            k, seed = int(r["k"]), int(r["seed"])
            poscar = rel / "structures" / cnid / tag / ("k%d_s%d" % (k, seed)) / "POSCAR"
            if not poscar.exists():
                raise FileNotFoundError(poscar)
            fr = [int(x) for x in r["frozen_indices"].split(";") if x != ""]
            d["%s|%s|k%d|s%d" % (cnid, tag, k, seed)] = {"poscar_path": str(poscar), "frozen_indices": fr, "natoms": int(r["natoms"]),
                                                          "cnid": cnid, "k": k, "seed": seed, "dx_A": dx, "dx_idx": dxi}
        json.dump(d, open(inp / (cnid + ".json"), "w"))
    env = dict(os.environ, SB_WORK=str(SB_WORK))
    cmd = [sys.executable, str(SB / "stageD_gamma_dedup" / "stage_d1_build_csv.py"), "--inputs-dir", str(inp), "--results-dir", str(res)]
    print("S4 run:", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    print("S4 stage_d1_build_csv exit code:", rc)
    allp = res / "gamma_all.csv"
    new = {}
    for r in csv.DictReader(open(allp, newline="")):
        new[(r["cnid"], int(r["k"]), int(r["seed"]))] = r
    relrows = list(csv.DictReader(open(rel / "gamma_dedup.csv", newline="")))
    out, dg, de, da, miss = [], 0.0, 0.0, 0.0, 0
    for r in relrows:
        key = (r["cnid"], int(r["k"]), int(r["seed"]))
        n = new.get(key)
        if n is None:
            miss += 1
            out.append((key, "missing in recomputed", "", "", ""))
            continue
        g, e, ar = abs(float(n["gamma_Jm2"]) - float(r["gamma_Jm2"])), abs(float(n["energy_eV"]) - float(r["energy_eV"])), abs(float(n["area_A2"]) - float(r["area_A2"]))
        dg, de, da = max(dg, g), max(de, e), max(da, ar)
        out.append((key, "ok", "%.6e" % g, "%.6e" % e, "%.6e" % ar))
    with open(a.out, "w") as f:
        f.write("cnid_k_seed\tstatus\tabs_dgamma_Jm2\tabs_denergy_eV\tabs_darea_A2\n")
        for o in out:
            f.write("%s\t%s\t%s\t%s\t%s\n" % ("|".join(map(str, o[0])), *o[1:]))
    n_conv = sum(1 for v in new.values() if v["converged"] == "True")
    print("S4 recomputed rows=%d converged=%d released_dedup_rows=%d missing=%d" % (len(new), n_conv, len(relrows), miss))
    print("S4 max_abs_dgamma_Jm2=%.6e max_abs_denergy_eV=%.6e max_abs_darea_A2=%.6e (over the %d released rows)" % (dg, de, da, len(relrows) - miss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
