#!/usr/bin/env python3
"""S2: compare the stage-B output in $SB_WORK (removal structures and bulk_membership.csv) with the released ones.

  python s2_compare_removal.py --released-structures $GBREL_DATA/training/structures \
         --released-membership $GBREL_DATA/training/bulk_membership.csv --out s2_compare.tsv [--dx dx_p0.000]

Per (template, dx, k>=1): the set of seeds present in each side (a seed that exists on one side only is a gap and is listed, not
compared), and for the seeds present on both sides the largest position difference of the POSCAR (file order, minimum image) and
whether the frozen_indices of bulk_membership.csv are identical. Only numbers are written; there is no verdict.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

SB_WORK = Path(os.environ["SB_WORK"])


def read_membership(p):
    out = {}
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            out[(r["cnid"], int(r["k"]), int(r["seed"]))] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released-structures", required=True)
    ap.add_argument("--released-membership", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dx", default="dx_p0.000", help="restrict to this dx tag (default dx_p0.000); 'all' = every dx")
    a = ap.parse_args()
    rel_root, new_root = Path(a.released_structures), SB_WORK / "structures"
    rel_m = read_membership(a.released_membership)
    new_m = read_membership(SB_WORK / "results" / "bulk_membership.csv")
    rel_dirs = sorted(p for p in rel_root.glob("cnid_*/dx_*/k[1-4]_s*") if a.dx in ("all", p.parent.name))
    new_dirs = sorted(p for p in new_root.glob("cnid_*/dx_*/k[1-4]_s*") if a.dx in ("all", p.parent.name))
    def idx(dirs, root):
        d = {}
        for p in dirs:
            rel = p.relative_to(root)
            cnid, dx, ks = rel.parts
            k, s = ks[1:].split("_s")
            d.setdefault((cnid, dx, int(k)), set()).add(int(s))
        return d
    R, N = idx(rel_dirs, rel_root), idx(new_dirs, new_root)
    keys = sorted(set(R) | set(N))
    rows = []
    max_diff, n_cmp, n_same_frozen, n_diff_frozen = 0.0, 0, 0, 0
    gaps_released_only, gaps_new_only = 0, 0
    for key in keys:
        cnid, dx, k = key
        sr, sn = R.get(key, set()), N.get(key, set())
        both = sorted(sr & sn)
        ro, no = sorted(sr - sn), sorted(sn - sr)
        gaps_released_only += len(ro)
        gaps_new_only += len(no)
        for s in both:
            A = ase_read(str(rel_root / cnid / dx / ("k%d_s%d" % (k, s)) / "POSCAR"), format="vasp")
            B = ase_read(str(new_root / cnid / dx / ("k%d_s%d" % (k, s)) / "POSCAR"), format="vasp")
            if len(A) != len(B):
                rows.append((cnid, dx, k, s, "natoms differ", len(A), len(B), "", ""))
                continue
            cell = np.array(A.get_cell())
            d = A.get_scaled_positions(wrap=False) - B.get_scaled_positions(wrap=False)
            d -= np.round(d)
            md = float(np.linalg.norm(d @ cell, axis=1).max())
            max_diff = max(max_diff, md)
            n_cmp += 1
            fr = rel_m.get((cnid, k, s), {}).get("frozen_indices")
            fn = new_m.get((cnid, k, s), {}).get("frozen_indices")
            same = (fr is not None and fr == fn)
            n_same_frozen += int(same)
            n_diff_frozen += int(not same)
            rows.append((cnid, dx, k, s, "ok", len(A), "%.6e" % md, "same" if same else "DIFFERENT", ""))
        if ro or no:
            rows.append((cnid, dx, k, "", "seed gap", "released_only=%s" % ro, "rebuilt_only=%s" % no, "", ""))
    with open(a.out, "w") as f:
        f.write("cnid\tdx\tk\tseed\tstatus\tnatoms_or_info\tmax_pos_diff_A_or_info\tfrozen_indices\tnote\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print("S2 dx=%s released_dirs=%d rebuilt_dirs=%d compared=%d" % (a.dx, len(rel_dirs), len(new_dirs), n_cmp))
    print("S2 max_pos_diff_A=%.6e frozen_indices same=%d different=%d" % (max_diff, n_same_frozen, n_diff_frozen))
    print("S2 seed gaps: released-only=%d rebuilt-only=%d (listed in %s)" % (gaps_released_only, gaps_new_only, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
