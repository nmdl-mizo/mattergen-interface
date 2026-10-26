#!/usr/bin/env python3
"""Data rebuild, stage E-4: makes the manifests ({cnid}.json) from the dx=0, k=0 structures of the 31 templates.

Transcribed from (read only): MN/runs/scm_manifest/scripts/build_manifests.py (role assignment, projection fit, residual of reconstruction, keys of the JSON).
Differences (only what the plan requires):
  (1) Input: structure = RUN/structures/{cnid}/dx_p0.000/k0_s15000/POSCAR (the dx=0 as-built of stage A),
      FIXED = frozen_indices of the row (cnid, k=0, seed=15000) of bulk_membership.csv. The old PILOT_DIR and the old bulk_membership are not touched.
  (2) The list of templates and (i,j) are from template_map.csv (the old CNID_LABEL_TO_IJ uses the names of the old 4x4 grid, so it is not used).
  (3) The reference template (the basis of gb_t_effective) is cnid_0_0 as before (the new cnid_0_0 has the same structure as the old cnid_0_0).
      The templates to output can be narrowed with --cnids (for small tests. The reference is always made from the structure of cnid_0_0).
  (4) gb_t_dydz_seed_frac_yz (previously the argument value of the 4x4 grid of interface_master. A reference value for traceability that the projection, the checks and the proposed model do not use) is
      not computed on the 10x10 grid (the source of the computation is not used), so null is written. The key is the same as before.
The key set is the same as the old manifests/cnid_0_0.json (cnid, natoms, cnid_ij, cell, gb_t_effective_frac_yz, gb_t_dydz_seed_frac_yz,
fixed_indices, master_indices, slave_indices, master_sub_{lo,hi}_indices, slave_sub_{lo,hi}_indices, recon, t0_inplane,
reference_cnid, recon_residual_A). What the proposed model uses is only cell (dataset.py:521-549).

  <python> stage_e_manifests.py [--results-dir DIR] [--cnids a,b] [--structures-dir DIR] [--bulk-membership CSV]
                                [--template-map CSV] [--force]
python: mattergen.symmetry_constrained.manifest does not use torch. Set PYTHONPATH=${GBREL_CODE}.
Output: <results-dir>/manifests/{cnid}.json and stage_e_manifests_summary.json
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# constants.py reads the environment variables at import time (Al: host Z=13, no solute). Set them explicitly before the import.
for _k, _v in (("SCM_HOST_Z", "13"), ("SCM_SOLUTE_ZS", "")):
    if _k in os.environ and os.environ[_k] != _v:
        raise SystemExit(f"FATAL: {_k}={os.environ[_k]!r} (this processing requires {_v!r})")
    os.environ[_k] = _v

sys.path.insert(0, _gbrel.CODE)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from ase.io import read as ase_read  # noqa: E402

import stage_e_common as E  # noqa: E402
from mattergen.symmetry_constrained.manifest import (  # noqa: E402
    SymmetricTiltGeometryError,
    build_tilt_template_manifest,
    read_gb_t_inplane,
    reconstruct_slave,
)

C = E.C
REFERENCE_CNID = "cnid_0_0"
GATE_TOL_A = 1e-3
FIT_TOL_A = 1e-6
DX0_IDX = C.DX10.index(0.0)
K0_SEED = C.seed_of(DX0_IDX, 0, 0)          # 15000
MANIFEST_KEYS = [
    "cnid", "natoms", "cnid_ij", "cell", "gb_t_effective_frac_yz", "gb_t_dydz_seed_frac_yz", "fixed_indices",
    "master_indices", "slave_indices", "master_sub_lo_indices", "master_sub_hi_indices", "slave_sub_lo_indices",
    "slave_sub_hi_indices", "recon", "t0_inplane", "reference_cnid", "recon_residual_A",
]


def k0_poscar(structures: Path, cnid: str) -> Path:
    return C.struct_dir(cnid, 0.0, 0, K0_SEED, root=structures) / "POSCAR"


def load_frozen_k0(bm_csv: Path, cnids: set[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    with open(bm_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["cnid"] in cnids and r["k"] == "0" and int(r["seed"]) == K0_SEED:
                if r["status"] != "ok":
                    raise SystemExit(f"FATAL: {r['cnid']} k=0 seed={K0_SEED} status={r['status']}")
                if r["cnid"] in out:
                    raise SystemExit(f"FATAL: duplicate bulk_membership row for {r['cnid']} k=0 seed={K0_SEED}")
                out[r["cnid"]] = sorted(int(i) for i in r["frozen_indices"].split(";") if i != "")
    missing = cnids - set(out)
    if missing:
        raise SystemExit(f"FATAL: no bulk_membership k=0 seed={K0_SEED} row for {sorted(missing)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--cnids", default="")
    ap.add_argument("--structures-dir", default=str(C.STRUCTURES))
    ap.add_argument("--bulk-membership", default=str(C.RESULTS / "bulk_membership.csv"))
    ap.add_argument("--template-map", default=str(C.RESULTS / "template_map.csv"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    res = E.results_dir(a.results_dir)
    out_dir = res / "manifests"
    structures = Path(a.structures_dir)
    with open(a.template_map, newline="") as f:
        tm = {r["cnid"]: r for r in csv.DictReader(f)}
    if REFERENCE_CNID not in tm:
        raise SystemExit(f"FATAL: {REFERENCE_CNID} not in template_map")
    targets = C.select_cnids(list(tm), a.cnids)
    frozen = load_frozen_k0(Path(a.bulk_membership), set(targets) | {REFERENCE_CNID})

    ref_atoms = ase_read(str(k0_poscar(structures, REFERENCE_CNID)), format="vasp")
    ref_cell = np.array(ref_atoms.get_cell())
    ref_frac = np.array(ref_atoms.get_scaled_positions(wrap=True))
    ref_m = build_tilt_template_manifest(
        ref_frac, ref_cell, np.array(frozen[REFERENCE_CNID], dtype=np.int64), REFERENCE_CNID, np.zeros(2),
        (int(tm[REFERENCE_CNID]["i"]), int(tm[REFERENCE_CNID]["j"])), symmetry_tolerance=FIT_TOL_A,
    )
    print(f"[reference] {REFERENCE_CNID}: fit_residual={ref_m.fit_residual_A:.3e} A", flush=True)

    summary = {"design": "data rebuild stage E-4", "reference_cnid": REFERENCE_CNID, "k0_seed": K0_SEED,
               "structures_dir": str(structures), "bulk_membership": a.bulk_membership, "templates": {}}
    n_ok = 0
    for cnid in targets:
        i, j = int(tm[cnid]["i"]), int(tm[cnid]["j"])
        poscar = k0_poscar(structures, cnid)
        atoms = ase_read(str(poscar), format="vasp")
        cell = np.array(atoms.get_cell())
        if not np.allclose(cell, ref_cell, atol=1e-6):
            raise SystemExit(f"FATAL: {cnid}: cell differs from the reference template cell")
        frac = np.array(atoms.get_scaled_positions(wrap=True))
        fixed_idx = np.array(frozen[cnid], dtype=np.int64)
        gb_t_eff = read_gb_t_inplane(frac, ref_frac, ref_m.roles.master_sub_hi_idx)
        try:
            m = build_tilt_template_manifest(frac, cell, fixed_idx, cnid, gb_t_eff, (i, j), symmetry_tolerance=FIT_TOL_A)
        except SymmetricTiltGeometryError as e:
            raise SystemExit(f"FATAL: {cnid}: manifest fit failed: {e}")
        recon_slave = reconstruct_slave(frac, ref_m.roles, ref_m.recon, gb_t_eff, t0_inplane=np.zeros(2))
        diff = recon_slave - frac[ref_m.recon.slave_atom_indices]
        diff = diff - np.round(diff)
        resid = float(np.linalg.norm(diff @ cell, axis=1).max())
        if resid >= GATE_TOL_A:
            raise SystemExit(f"FATAL: {cnid}: Gate G-a reconstruction residual {resid:.3e} A >= {GATE_TOL_A}")
        payload = {
            "cnid": m.cnid, "natoms": m.natoms, "cnid_ij": list(m.gb_t_cnid_ij), "cell": m.cell.tolist(),
            "gb_t_effective_frac_yz": gb_t_eff.tolist(), "gb_t_dydz_seed_frac_yz": None,
            "fixed_indices": m.roles.fixed_idx.tolist(), "master_indices": m.roles.master_idx.tolist(),
            "slave_indices": m.roles.slave_idx.tolist(),
            "master_sub_lo_indices": m.roles.master_sub_lo_idx.tolist(),
            "master_sub_hi_indices": m.roles.master_sub_hi_idx.tolist(),
            "slave_sub_lo_indices": m.roles.slave_sub_lo_idx.tolist(),
            "slave_sub_hi_indices": m.roles.slave_sub_hi_idx.tolist(),
            "recon": {
                "slave_atom_indices": m.recon.slave_atom_indices.tolist(),
                "slave_master_indices": m.recon.slave_master_indices.tolist(),
                "rotation": m.recon.rotation.tolist(), "b0": m.recon.b0.tolist(),
                "t_coeff_inplane": m.recon.t_coeff_inplane.tolist(), "sub_block": m.recon.sub_block.tolist(),
                "fit_residual_A": m.recon.fit_residual_A,
            },
            "t0_inplane": [0.0, 0.0], "reference_cnid": REFERENCE_CNID, "recon_residual_A": resid,
        }
        assert list(payload) == MANIFEST_KEYS, list(payload)
        E.write_json(out_dir / f"{cnid}.json", payload, a.force)
        summary["templates"][cnid] = {"poscar": str(poscar), "fit_residual_A": m.fit_residual_A,
                                      "recon_residual_A": resid, "gb_t_effective_frac_yz": gb_t_eff.tolist()}
        n_ok += 1
        print(f"[OK] {cnid}: fit_residual={m.fit_residual_A:.3e} recon={resid:.3e} gb_t_eff={gb_t_eff}", flush=True)
    summary["n_written"] = n_ok
    E.write_json(res / "stage_e_manifests_summary.json", summary, a.force)
    print(f"WROTE {n_ok} manifests in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
