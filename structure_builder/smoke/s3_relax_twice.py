#!/usr/bin/env python3
"""S3 (GPU): relax four released structures twice with stage C and compare the energies with the released energy.json.

  SB_WORK=<new work dir> MACE_MODEL_PATH=<mace-mh-1.model> python s3_relax_twice.py --released $GBREL_DATA/training --out s3_compare.tsv

The four structures: cnid_0_0 | dx_m0.750 | k0_s10000, k1_s10100 and two further structures of cnid_0_0 at dx_m0.750 (the first k2 and k3
directories in sort order). Each is relaxed in two independent output directories; the difference between the two runs is the noise floor.
The relaxation function and the calculator construction are those of stageC_relax/stage_c_relax_chunk.py. Only numbers are written.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

SB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SB / "stageC_relax"))
sys.path.insert(0, str(SB / "lib"))
SB_WORK = Path(os.environ["SB_WORK"])

import config as CFG  # noqa: E402
import stage_c_relax_chunk as SC  # noqa: E402


def pick(rel, cnid="cnid_0_0", dx="dx_m0.750"):
    base = rel / "structures" / cnid / dx
    ks = {}
    for p in sorted(base.glob("k*_s*")):
        ks.setdefault(int(p.name[1:].split("_s")[0]), []).append(p)
    return [base / "k0_s10000", base / "k1_s10100", ks[2][0], ks[3][0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.released)
    mem = {}
    for r in csv.DictReader(open(rel / "bulk_membership.csv", newline="")):
        mem[(r["cnid"], int(r["k"]), int(r["seed"]))] = r
    from mace.calculators import mace_mp
    model_path = CFG.mace_model_path()
    t0 = time.time()
    calc = mace_mp(model=model_path, device="cuda", default_dtype="float64", head=CFG.MACE_HEAD)
    print("S3 model load %.1fs" % (time.time() - t0), flush=True)
    rows = []
    for sd in pick(rel):
        k, seed = int(sd.name[1:].split("_s")[0]), int(sd.name.split("_s")[1])
        cnid = sd.parent.parent.name
        frozen = [int(x) for x in mem[(cnid, k, seed)]["frozen_indices"].split(";") if x != ""]
        e_rel = json.load(open(sd / "energy.json"))
        es = []
        for rep in (1, 2):
            out_dir = SB_WORK / "s3" / ("%s_%s_%s" % (sd.parent.parent.name, sd.parent.name, sd.name)) / ("rep%d" % rep)
            res = SC.relax_one(sd / "POSCAR", out_dir, calc, frozen)
            es.append(res)
        rows.append((str(sd.relative_to(rel / "structures")), e_rel["energy_eV"], es[0]["energy_eV"], es[1]["energy_eV"], es[0]["n_steps"], es[1]["n_steps"],
                     es[0]["converged"], es[1]["converged"]))
    with open(a.out, "w") as f:
        f.write("structure\treleased_energy_eV\trun1_energy_eV\trun2_energy_eV\trun1_steps\trun2_steps\trun1_converged\trun2_converged\t|run1-released|\t|run2-released|\t|run1-run2|\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\t%.6e\t%.6e\t%.6e\n" % (abs(r[2] - r[1]), abs(r[3] - r[1]), abs(r[2] - r[3])))
    for r in rows:
        print("S3", r[0], "released=%.8f run1=%.8f run2=%.8f |run1-run2|=%.3e |run1-released|=%.3e" % (r[1], r[2], r[3], abs(r[2] - r[3]), abs(r[2] - r[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
