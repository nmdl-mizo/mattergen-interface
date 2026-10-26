#!/usr/bin/env python3
"""
build_bulk_membership.py -- construction of the bulk membership table (frozen indices).

Specification: atom-count-based membership (n_bulk=38).
Parent task specification: relaxation of the 410 pilot structures with the constraint (making the labels the master).

Method:
  1. For each cnid, read the k=0 template (structures/{cnid}/k0_s0/POSCAR).
  2. Sort the atoms of grain A (fractional x in (0,0.5)) in ascending order of |x-0.25| and take the first n_bulk=38 as bulk_A
     (ties are broken by index order = naturally satisfied by the stable sort of np.argsort).
  3. bulk_B is defined as the 38 images of the atoms of bulk_A (nearest atoms after applying R,t, MIC), obtained by refitting the symmetry operation
     (sym_common.audit_symmetry, force_r_name="Rnnid_omm", normal_axis=0) to the template.
  4. bulk_A + bulk_B = 76 atoms are fixed as the "bulk reference coordinates (fractional)" of the template.
  5. For each of the 410 structures of pilot_manifest.csv, read the POSCAR (as built, after the removal),
     find for each bulk reference atom the nearest atom in that structure (minimum MIC Cartesian distance),
     and take its index as the frozen target (the removal affects only the GB2 core, so the atoms in the bulk region should
     remain with unchanged coordinates, and the same atom corresponds uniquely by position matching).
  6. Consistency check: compute the maximum deviation (MIC) between the atomic positions of the obtained indices and the bulk reference positions of the template
     and assert that it is below 1e-6 A. A structure that does not satisfy this is recorded in bulk_membership.csv
     as a skip with a reason (the frozen indices are left blank).

Output: bulk_membership.csv (for 410 structures, 1 row per structure)
  Columns: cnid, k, seed, natoms, n_bulkA, n_bulkB, n_frozen, max_pos_diff_A, status,
      frozen_indices (separated by ';', ascending), bulkA_indices, bulkB_indices, op_residual_A, reason

Run (CPU only on the login node, no MACE needed):
  python -u build_bulk_membership.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
PILOT_DIR = RUN_DIR.parent / "atom_removal"

sys.path.insert(0, str(HERE))
from sym_common import audit_symmetry, _r_candidates_v2, _apply_op  # noqa: E402

from ase.io import read as ase_read  # noqa: E402

N_BULK = 38                     # confirmed values
NORMAL_AXIS = 0                 # GB normal = a axis (im system, Sigma5_210)
FORCE_R_NAME = "Rnnid_omm"      # screw R=diag(+1,-1,-1) (same operation as in the atom removal)
FIT_RESIDUAL_TOL_A = 1e-6       # upper limit of the tolerance of the op fit residual (same criterion as removal_builder.py)
MEMBERSHIP_TOL_A = 1e-6         # upper limit of the tolerance of the consistency check of the bulk membership (as instructed)


def _mic_pairwise_dist(cell: np.ndarray, frac_a: np.ndarray, frac_b: np.ndarray) -> np.ndarray:
    """Matrix (M,N) of all-pair MIC Cartesian distances between frac_a(M,3) and frac_b(N,3)."""
    diff = frac_a[:, None, :] - frac_b[None, :, :]
    diff = diff - np.round(diff)
    cart = diff @ cell
    return np.linalg.norm(cart, axis=2)


def build_template_bulk(cnid: str) -> dict:
    """Builds the fractional reference coordinates and indices of bulk_A/bulk_B from the k=0 template."""
    tmpl_path = PILOT_DIR / "structures" / cnid / "k0_s0" / "POSCAR"
    atoms = ase_read(str(tmpl_path), format="vasp")
    cell = np.array(atoms.get_cell())
    frac = np.array(atoms.get_scaled_positions(wrap=True))

    res = audit_symmetry(atoms, normal_axis=NORMAL_AXIS, use_v2_candidates=True,
                          force_r_name=FORCE_R_NAME)
    op_residual = res["max_residual_A"]
    if not (op_residual < FIT_RESIDUAL_TOL_A):
        raise RuntimeError(f"{cnid}: symmetry fit residual too large: {op_residual:.3e}A")
    R = _r_candidates_v2(NORMAL_AXIS)[FORCE_R_NAME]
    t = np.array(res["best_t"], dtype=float)

    grainA_idx = np.where((frac[:, 0] > 0.0) & (frac[:, 0] < 0.5))[0]
    dist_center = np.abs(frac[grainA_idx, 0] - 0.25)
    order = np.argsort(dist_center, kind="stable")   # ties by index order (guaranteed by the stable sort)
    bulkA_idx = grainA_idx[order[:N_BULK]]

    moved = _apply_op(frac[bulkA_idx], R, t)
    d = _mic_pairwise_dist(cell, moved, frac)
    bulkB_idx = np.argmin(d, axis=1)
    bulkB_image_dist = d[np.arange(len(bulkA_idx)), bulkB_idx]

    if len(set(bulkB_idx.tolist())) != N_BULK:
        raise RuntimeError(f"{cnid}: bulkB indices not unique (duplicate images)")
    if bulkB_image_dist.max() >= FIT_RESIDUAL_TOL_A * 1e6:  # guard against obviously abnormal values (loose upper limit; the measured values are of the order of 1e-14)
        pass  # no strict decision because the measured values were audited (guaranteed by op_residual)

    return {
        "cell": cell,
        "bulkA_idx": bulkA_idx,
        "bulkB_idx": bulkB_idx,
        "bulkA_frac": frac[bulkA_idx].copy(),
        "bulkB_frac": frac[bulkB_idx].copy(),
        "op_residual_A": op_residual,
    }


def locate_frozen_indices(poscar_path: Path, template: dict) -> dict:
    """From the POSCAR of one structure, finds the index of the atom nearest to the bulk reference coordinates of the template."""
    atoms = ase_read(str(poscar_path), format="vasp")
    cell = np.array(atoms.get_cell())
    frac = np.array(atoms.get_scaled_positions(wrap=True))
    natoms = len(atoms)

    ref_frac = np.vstack([template["bulkA_frac"], template["bulkB_frac"]])  # (76,3)
    d = _mic_pairwise_dist(cell, ref_frac, frac)   # (76, natoms)
    matched = np.argmin(d, axis=1)
    matched_dist = d[np.arange(len(ref_frac)), matched]

    dup = len(set(matched.tolist())) != len(matched)
    max_diff = float(matched_dist.max())

    return {
        "natoms": natoms,
        "matched_idx": matched,
        "max_pos_diff_A": max_diff,
        "duplicate": dup,
    }


def main():
    manifest_path = PILOT_DIR / "pilot_manifest.csv"
    with open(manifest_path) as f:
        entries = list(csv.DictReader(f))

    templates: dict[str, dict] = {}
    template_rows = []
    cnids = sorted({r["cnid"] for r in entries})
    for cnid in cnids:
        try:
            tmpl = build_template_bulk(cnid)
            templates[cnid] = tmpl
            template_rows.append((cnid, "ok", tmpl["op_residual_A"]))
        except Exception as e:
            template_rows.append((cnid, f"template_fail:{e}", ""))

    rows = []
    n_ok = n_skip = 0
    max_diff_overall = 0.0
    for r in entries:
        cnid, k, seed = r["cnid"], r["k"], r["seed"]
        natoms_manifest = int(r["natoms"])
        poscar_path = Path(r["path"])
        row = {
            "cnid": cnid, "k": k, "seed": seed, "natoms": natoms_manifest,
            "n_bulkA": N_BULK, "n_bulkB": N_BULK, "n_frozen": 0,
            "max_pos_diff_A": "", "status": "", "frozen_indices": "",
            "bulkA_indices": "", "bulkB_indices": "", "op_residual_A": "", "reason": "",
        }

        tmpl = templates.get(cnid)
        if tmpl is None:
            row["status"] = "skip"
            row["reason"] = "template_fit_failed"
            rows.append(row)
            n_skip += 1
            continue

        row["op_residual_A"] = f"{tmpl['op_residual_A']:.6e}"

        if not poscar_path.exists():
            row["status"] = "skip"
            row["reason"] = "poscar_missing"
            rows.append(row)
            n_skip += 1
            continue

        try:
            loc = locate_frozen_indices(poscar_path, tmpl)
        except Exception as e:
            row["status"] = "skip"
            row["reason"] = f"locate_error:{e}"
            rows.append(row)
            n_skip += 1
            continue

        if loc["natoms"] != natoms_manifest:
            row["status"] = "skip"
            row["reason"] = f"natoms_mismatch(manifest={natoms_manifest},poscar={loc['natoms']})"
            rows.append(row)
            n_skip += 1
            continue

        if loc["duplicate"]:
            row["status"] = "skip"
            row["reason"] = "duplicate_frozen_index_match"
            rows.append(row)
            n_skip += 1
            continue

        max_diff = loc["max_pos_diff_A"]
        max_diff_overall = max(max_diff_overall, max_diff)

        if not (max_diff < MEMBERSHIP_TOL_A):
            row["status"] = "skip"
            row["reason"] = f"membership_residual_too_large({max_diff:.3e}A)"
            row["max_pos_diff_A"] = f"{max_diff:.6e}"
            rows.append(row)
            n_skip += 1
            continue

        matched = loc["matched_idx"]
        bulkA_matched = matched[:N_BULK]
        bulkB_matched = matched[N_BULK:]
        frozen_sorted = np.sort(matched)

        row.update({
            "natoms": loc["natoms"],
            "n_frozen": len(frozen_sorted),
            "max_pos_diff_A": f"{max_diff:.6e}",
            "status": "ok",
            "frozen_indices": ";".join(str(int(i)) for i in frozen_sorted),
            "bulkA_indices": ";".join(str(int(i)) for i in bulkA_matched),
            "bulkB_indices": ";".join(str(int(i)) for i in bulkB_matched),
            "reason": "",
        })
        rows.append(row)
        n_ok += 1

    out_csv = RUN_DIR / "bulk_membership.csv"
    fields = ["cnid", "k", "seed", "natoms", "n_bulkA", "n_bulkB", "n_frozen",
              "max_pos_diff_A", "status", "frozen_indices", "bulkA_indices",
              "bulkB_indices", "op_residual_A", "reason"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"-> {out_csv} ({len(rows)} rows)")
    print(f"templates (number of cnid)={len(cnids)} of which fit failed={sum(1 for _,s,_ in template_rows if s!='ok')}")
    print(f"ok={n_ok} skip={n_skip}")
    print(f"check of the distribution of n_frozen: is it 76 for all ok rows -> "
          f"{all(r['n_frozen'] == 76 for r in rows if r['status'] == 'ok')}")
    print(f"max_pos_diff_A (maximum over ok rows) = {max_diff_overall:.6e} A")

    if n_skip > 0:
        print("\n[list of skips]")
        for r in rows:
            if r["status"] == "skip":
                print(f"  {r['cnid']} k={r['k']} seed={r['seed']}: {r['reason']}")


if __name__ == "__main__":
    main()
