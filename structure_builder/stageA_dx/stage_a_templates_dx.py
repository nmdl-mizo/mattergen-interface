#!/usr/bin/env python3
"""Data rebuild, stage A (1 to 4): import of the 31 templates, structures at 10 levels of dx, refit of the symmetry operation and the bulk.

  python stage_a_templates_dx.py [--cnids cnid_0_0,...]
    --cnids: narrows the templates for which dx structures are made and refitted (for smoke tests). template_map.csv always has 31 rows.

Exit code: 0 = all passed / 2 = check 1 FAIL (there is a (cnid,dx) with residual >= 1e-6 A).
A violation of the checks (not 31 rows, cell mismatch, etc.) exits non-zero with an exception.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from ase.io import read as ase_read, write as ase_write

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "lib"))  # structure_builder/lib
import builder_common as C
import dx_lib
import removal_builder as RB
from sym_common import audit_symmetry


def load_templates():
    """Stage A-1: exactly 31 rows with is_rep==1; makes the names and POSCAR paths. Sorted in lexicographic order of (i,j)."""
    with open(C.GROUPS_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["is_rep"] == "1"]
    if len(rows) != C.N_TEMPLATES:
        raise ValueError(f"is_rep==1 rows = {len(rows)} (expected {C.N_TEMPLATES})")
    out = []
    for r in rows:
        i, j = int(r["i"]), int(r["j"])
        if not (0 <= i <= 9 and 0 <= j <= 9):
            raise ValueError(f"(i,j) out of 1-digit range: {i},{j}")
        src = C.STRUCT_G10 / f"p_{i:02d}_{j:02d}" / "POSCAR"
        out.append({"cnid": f"cnid_{i}_{j}", "i": i, "j": j, "group_id": r["group_id"],
                    "group_size": r["group_size"], "matches_existing": r["matches_existing"],
                    "src_poscar": src})
    out.sort(key=lambda d: (d["i"], d["j"]))
    if len({d["cnid"] for d in out}) != C.N_TEMPLATES:
        raise ValueError("cnid not unique")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnids", default=os.environ.get("DATA_CNIDS", ""))
    args = ap.parse_args()
    t_start = time.time()

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    templates = load_templates()

    # stage A-1 check: 120 atoms, all Al, the cells of the 31 are identical
    ref_cell = None
    for t in templates:
        if not t["src_poscar"].exists():
            raise FileNotFoundError(t["src_poscar"])
        at = ase_read(str(t["src_poscar"]), format="vasp")
        t["atoms"] = at
        if len(at) != C.N_ATOMS:
            raise ValueError(f"{t['cnid']}: natoms={len(at)}")
        if set(at.get_chemical_symbols()) != {"Al"}:
            raise ValueError(f"{t['cnid']}: not all Al")
        cell = np.array(at.get_cell())
        if ref_cell is None:
            ref_cell = cell
        dmax = float(np.abs(cell - ref_cell).max())
        if dmax > C.CELL_TOL:
            raise ValueError(f"{t['cnid']}: cell differs from reference by {dmax:.3e} A")

    # stage A-2: template_map.csv (always writes 31 rows)
    c00 = [t for t in templates if t["cnid"] == "cnid_0_0"]
    if len(c00) != 1 or (c00[0]["i"], c00[0]["j"]) != (0, 0):
        raise ValueError("cnid_0_0 with (i,j)=(0,0) missing")
    tm_path = C.RESULTS / "template_map.csv"
    tmp = tm_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cnid", "i", "j", "group_id", "group_size", "matches_existing", "src_poscar",
                    "a_A", "b_A", "c_A"])
        for t in templates:
            a_, b_, c_ = t["atoms"].cell.lengths()
            w.writerow([t["cnid"], t["i"], t["j"], t["group_id"], t["group_size"], t["matches_existing"],
                        str(t["src_poscar"]), repr(float(a_)), repr(float(b_)), repr(float(c_))])
    os.replace(tmp, tm_path)
    print(f"wrote {tm_path} rows={len(templates)}", flush=True)

    all_cnids = [t["cnid"] for t in templates]
    sel = C.select_cnids(all_cnids, args.cnids)
    print(f"selected templates: {len(sel)} {sel if len(sel) < 6 else sel[:3] + ['...']}", flush=True)

    (C.RESULTS / "template_bulk").mkdir(exist_ok=True)
    failures = []
    max_res = 0.0
    per_dx_max = {}
    n_struct = 0
    n_fit_ok = 0
    for t in templates:
        if t["cnid"] not in sel:
            continue
        for dx_idx, dx in enumerate(C.DX10):
            tag = C.dx_tag(dx)
            atoms_dx, n_g2, n_g1 = dx_lib.apply_dx_transform(t["atoms"], dx)   # the same function is used also for dx=0
            a_expect = C.A_BASE + 2.0 * dx
            if abs(float(atoms_dx.cell[0, 0]) - a_expect) > C.CELL_TOL:
                raise ValueError(f"{t['cnid']} {tag}: a={atoms_dx.cell[0, 0]!r} expected {a_expect!r}")
            sd = C.struct_dir(t["cnid"], dx, 0, C.seed_of(dx_idx, 0, 0))
            sd.mkdir(parents=True, exist_ok=True)
            poscar = sd / "POSCAR"
            ase_write(str(poscar), atoms_dx, format="vasp")
            n_struct += 1
            # read back from what was written, check and refit
            atoms_rb = ase_read(str(poscar), format="vasp")
            if abs(float(atoms_rb.cell[0, 0]) - a_expect) > C.CELL_TOL:
                raise ValueError(f"{t['cnid']} {tag}: read-back a={atoms_rb.cell[0, 0]!r} expected {a_expect!r}")
            # stage A-4: refit of the symmetry operation
            try:
                R, tt, res = RB.fit_symmetry_op(atoms_rb)
            except RB.RemovalSkip as e:
                r2 = audit_symmetry(atoms_rb, normal_axis=0, use_v2_candidates=True,
                                    force_r_name=RB.FORCE_R_NAME)["max_residual_A"]
                failures.append({"cnid": t["cnid"], "dx_A": dx, "residual_A": float(r2), "msg": str(e)})
                max_res = max(max_res, float(r2))
                per_dx_max[tag] = max(per_dx_max.get(tag, 0.0), float(r2))
                print(f"C1 FAIL {t['cnid']} {tag} residual={r2:.3e}", flush=True)
                continue
            tb = dx_lib.fit_template_bulk(atoms_rb)
            max_res = max(max_res, float(res))
            per_dx_max[tag] = max(per_dx_max.get(tag, 0.0), float(res))
            rec = {
                "cnid": t["cnid"], "dx_A": dx, "dx_idx": dx_idx, "dx_tag": tag,
                "natoms": len(atoms_rb),
                "cell": np.array(atoms_rb.get_cell()).tolist(),
                "bulkA_idx": [int(i) for i in tb["bulkA_idx"]],
                "bulkB_idx": [int(i) for i in tb["bulkB_idx"]],
                "bulkA_frac": np.asarray(tb["bulkA_frac"]).tolist(),
                "bulkB_frac": np.asarray(tb["bulkB_frac"]).tolist(),
                "op_residual_A": float(res),
                "template_bulk_op_residual_A": float(tb["op_residual_A"]),
                "R": np.asarray(R).tolist(), "t": [float(x) for x in tt],
                "bulkB_duplicate_at_build": bool(tb["bulkB_duplicate_at_build"]),
                "bulkB_image_dist_max_A": float(tb["bulkB_image_dist_max_A"]),
            }
            out = C.RESULTS / "template_bulk" / f"{t['cnid']}_{tag}.json"
            tmpj = out.with_suffix(".json.tmp")
            with open(tmpj, "w") as f:
                json.dump(rec, f)
            os.replace(tmpj, out)
            n_fit_ok += 1
        print(f"done {t['cnid']} elapsed={time.time() - t_start:.1f}s", flush=True)

    summary = {
        "cnids": sel, "n_templates_selected": len(sel), "n_dx_structures": n_struct,
        "n_fit_ok": n_fit_ok, "n_fit_fail": len(failures),
        "max_op_residual_A": max_res, "per_dx_max_residual_A": per_dx_max,
        "failures": failures, "elapsed_s": time.time() - t_start,
    }
    with open(C.RESULTS / "stage_a_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"C1: structures={n_struct} fit_ok={n_fit_ok} fail={len(failures)} "
          f"max_residual={max_res:.3e} A (tol 1e-6)", flush=True)
    raise SystemExit(2 if failures else 0)


if __name__ == "__main__":
    main()
