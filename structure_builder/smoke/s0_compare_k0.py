#!/usr/bin/env python3
"""S0: compare the k0 structures rebuilt by stage 0 -> 1 -> A (in $SB_WORK/structures) with the released k0 POSCAR files.

  python s0_compare_k0.py --released $GBREL_DATA/training/structures --out s0_compare.tsv [--tol 1e-6]

For every released (template, dx) directory `<cnid>/<dx_tag>/k0_s<seed>/POSCAR` the rebuilt file `$SB_WORK/structures/...` is read.
Columns: atom count, cell difference, largest position difference in file order (minimum image, Angstrom) and -- if that exceeds
the tolerance -- the largest difference after an optimal atom assignment (so a pure re-ordering of atoms can be told from a
different structure). Only numbers are written; there is no pass/fail verdict.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read as ase_read
from scipy.optimize import linear_sum_assignment

SB_WORK = Path(os.environ["SB_WORK"])


def mic_cart(cell, fa, fb):
    d = fa[:, None, :] - fb[None, :, :]
    d -= np.round(d)
    return np.linalg.norm(d @ cell, axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    rel_root, new_root = Path(a.released), SB_WORK / "structures"
    rel = sorted(rel_root.glob("cnid_*/dx_*/k0_s*/POSCAR"))
    rows = []
    n_missing = 0
    for p in rel:
        key = p.relative_to(rel_root).parent
        q = new_root / key / "POSCAR"
        if not q.exists():
            rows.append((str(key), "missing", "", "", "", ""))
            n_missing += 1
            continue
        A, B = ase_read(str(p), format="vasp"), ase_read(str(q), format="vasp")
        if len(A) != len(B) or A.get_chemical_symbols() != B.get_chemical_symbols():
            rows.append((str(key), "natoms/species differ", len(A), len(B), "", ""))
            continue
        cell = np.array(A.get_cell())
        dcell = float(np.abs(cell - np.array(B.get_cell())).max())
        fa, fb = A.get_scaled_positions(wrap=False), B.get_scaled_positions(wrap=False)
        d = fa - fb
        d -= np.round(d)
        ordered = float(np.linalg.norm(d @ cell, axis=1).max())
        assigned = ""
        if ordered > a.tol:
            r, c = linear_sum_assignment(mic_cart(cell, fa, fb))
            assigned = "%.6e" % float(mic_cart(cell, fa, fb)[r, c].max())
        rows.append((str(key), "ok", len(A), "%.6e" % dcell, "%.6e" % ordered, assigned))
    with open(a.out, "w") as f:
        f.write("released_dir\tstatus\tnatoms\tmax_cell_diff_A\tmax_pos_diff_ordered_A\tmax_pos_diff_assigned_A\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    ok = [r for r in rows if r[1] == "ok"]
    ords = [float(r[4]) for r in ok]
    over = [r for r in ok if float(r[4]) > a.tol]
    print("S0 pairs=%d missing=%d compared=%d" % (len(rel), n_missing, len(ok)))
    print("S0 max_pos_diff_ordered_A=%s  n_over_tol(%g)=%d  max_cell_diff_A=%s" % (
        ("%.6e" % max(ords)) if ords else "n/a", a.tol, len(over), ("%.6e" % max(float(r[3]) for r in ok)) if ok else "n/a"))
    if over:
        asg = [float(r[5]) for r in over if r[5] != ""]
        print("S0 over-tolerance structures:", len(over), " max assigned diff:", ("%.6e" % max(asg)) if asg else "n/a")
        for r in over[:40]:
            print("S0 over", "\t".join(str(x) for x in r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
