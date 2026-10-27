#!/usr/bin/env python3
"""fixed_identity.py -- identity check of the fixed atoms of the generated structures of the proposed model: P2-A (FIXED_A) and P2-B (FIXED_B), and the FAIL validation.

Background: the atom order of proposed is pos = cat[frac_a0, frac_b0, primary, replica] (generate_symmetry_constrained.py:429).
  FIXED_A = indices 0..37 are the Cartesian anchors of the reference template (common to all structures), FIXED_B = indices 38..75 are reconstructed from gb_dx and gb_t of each structure
  (symmetry_ops.reconstruct_fixed_b_cart: R.FIXED_A + b0 + (gb_dx,0,0) + (0, t_y*b, t_z*c). R is applied to Cartesian coordinates).
  Therefore "the first 76 atoms agree with the template index by index" does not hold, and P2 is split into the following two.
  P2-A: Cartesian coordinates of indices 0..37 = fixed_a_cart of the template (minimum image, threshold max <= 1e-4 A)
  P2-B: for d_i = r_B[i] - R.r_A[i] (i=0..37, r_B[i] is index 38+i, R = _R_REPLICA_MATRIX applied to **Cartesian** coordinates)
        e_i = minimum image(d_i - d_0), max_i |e_i| <= 1e-4 A (the translation is common to the 38 atoms)
  The deviation is the Euclidean norm of the 3-component vector of each atom. The minimum image is taken with the cell of the structure itself (rounding of fractional coordinates).

[Execution environment] the venv of mattergen (${GEN_PY}). On the compute nodes of the CPU partition, the import of the whole generate_symmetry_constrained.py
  fails because torch_scatter requires GLIBC_2.32 (measured; the libc of the node is 2.28).
  So, following the rule, build_reference_template and ReferenceTemplate are copied verbatim below (the source file:line is noted at each place).
  The helper functions used are imported from the same module as the generate script (it was confirmed that individual imports do not hit the GLIBC problem):
    mattergen.symmetry_constrained.layout / mattergen.symmetry_constrained.constants / mattergen.symmetry_constrained.variable_length.dataset / mattergen.symmetry_constrained.projection
  The environment variables SCM_HOST_Z=13, SCM_SOLUTE_ZS= (empty) and PYTHONPATH=${GBREL_CODE} are set by the caller (sbatch).
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

P2_THRESHOLD_A = 1e-4
_CODE_DIR = _gbrel.CODE

_loaded = {}


def _mg():
    """Lazy import of the modules of mattergen (the SCM_* environment variables must already be set before the import)."""
    if _loaded:
        return _loaded
    import os
    pth = _CODE_DIR if os.path.isdir(_CODE_DIR) else "/mnt" + _CODE_DIR
    if pth not in sys.path:
        sys.path.insert(0, pth)
    from mattergen.symmetry_constrained.constants import N_BULK_PER_GRAIN
    from mattergen.symmetry_constrained.layout import build_canonical_layout, load_bulk_membership, load_gamma_csv, read_poscar_frac_species
    from mattergen.symmetry_constrained.projection import _R_REPLICA_MATRIX
    from mattergen.symmetry_constrained.variable_length.dataset import resolve_view
    _loaded.update(N_BULK_PER_GRAIN=N_BULK_PER_GRAIN, build_canonical_layout=build_canonical_layout,
                   load_bulk_membership=load_bulk_membership, load_gamma_csv=load_gamma_csv,
                   read_poscar_frac_species=read_poscar_frac_species, R=np.array(_R_REPLICA_MATRIX, dtype=float),
                   resolve_view=resolve_view)
    return _loaded


# verbatim copy of: ${GBREL_CODE}/scripts/generate_symmetry_constrained.py:192-194
REFERENCE_CNID = "cnid_0_0"
REFERENCE_K = 0
_DX0_TAG = "dx_p0.000"


# verbatim copy of: ${GBREL_CODE}/scripts/generate_symmetry_constrained.py:214-232 (class ReferenceTemplate)
class ReferenceTemplate:
    """Everything :func:`build_generation_graph` needs, computed ONCE per process from the single
    reference structure (module docstring "Generation-time contract" above)."""

    def __init__(
        self,
        fixed_a_cart: np.ndarray,
        a_template: float,
        b_template: float,
        c_template: float,
        primary_frac_ref: np.ndarray,
        replica_frac_ref: np.ndarray,
        reference_dir: str,
    ):
        self.fixed_a_cart = fixed_a_cart  # (N_BULK_PER_GRAIN, 3) Cartesian, Angstrom
        self.a_template = a_template
        self.b_template = b_template
        self.c_template = c_template
        self.primary_frac_ref = primary_frac_ref  # (m_ref, 3) fractional, [0,1)
        self.replica_frac_ref = replica_frac_ref  # (m_ref, 3) fractional, [0,1)
        self.m_ref = primary_frac_ref.shape[0]
        self.reference_dir = reference_dir


# verbatim copy of: ${GBREL_CODE}/scripts/generate_symmetry_constrained.py:238-325 (function build_reference_template)
# the only modification: the helper functions of mattergen (load_gamma_csv etc.) are referenced through _mg() (the logic is verbatim).
def build_reference_template(
    gamma_csv_path: str, bulk_membership_csv_path: str, manifest_dir: str
) -> ReferenceTemplate:
    """Locate the unique converged (``REFERENCE_CNID``, ``k=REFERENCE_K``, ``dx_p0.000``) row and
    build the FIXED_A Cartesian template + a/b/c template axes + PRIMARY/REPLICA placeholder-source
    fractional coordinates from it (module docstring above)."""
    mg = _mg()
    load_gamma_csv = mg["load_gamma_csv"]
    load_bulk_membership = mg["load_bulk_membership"]
    read_poscar_frac_species = mg["read_poscar_frac_species"]
    build_canonical_layout = mg["build_canonical_layout"]
    resolve_view = mg["resolve_view"]
    N_BULK_PER_GRAIN = mg["N_BULK_PER_GRAIN"]

    rows = load_gamma_csv(gamma_csv_path)
    dx0_marker = f"/{_DX0_TAG}/k{REFERENCE_K}_s"
    candidates = [
        r
        for r in rows
        if r.cnid == REFERENCE_CNID and r.k == REFERENCE_K and r.converged and dx0_marker in r.dir
    ]
    if len(candidates) == 0:
        # (2026-10-01): the Al CSV (integrated_gamma_dedup.csv) names no dx tag in the dir of
        # its dx=0 original rows (".../structures/cnid_0_0/k0_s0"); only its dx!=0 augmentation rows
        # carry ".../dx_m0.750"-style dirs. A row with a dx-tagged directory name is handled by the rule above. Fallback: the converged (REFERENCE_CNID, REFERENCE_K) rows whose dir has no
        # "/dx_" path component. Still exactly one or an exception .
        candidates = [
            r
            for r in rows
            if r.cnid == REFERENCE_CNID
            and r.k == REFERENCE_K
            and r.converged
            and "/dx_" not in r.dir
        ]
    if len(candidates) != 1:
        raise ValueError(
            f"build_reference_template: expected exactly one converged {_DX0_TAG!r} row for "
            f"cnid={REFERENCE_CNID!r} k={REFERENCE_K}, found {len(candidates)}: "
            f"{[c.dir for c in candidates]} (refusing to guess which one is the "
            "reference)."
        )
    row = candidates[0]

    membership_map = load_bulk_membership(bulk_membership_csv_path)
    key = (row.cnid, row.k, row.seed)
    if key not in membership_map:
        raise ValueError(f"build_reference_template: no bulk_membership row for {key}")
    membership = membership_map[key]

    poscar_path = Path(resolve_view(str(Path(row.dir) / "POSCAR_relaxed")))
    frac, cell, species = read_poscar_frac_species(poscar_path)

    #: The dx=0 template manifest is named "{cnid}_dx_p0.000.json" or "{cnid}.json" (same
    # either-or rule as SymmetricTiltGBDataset._template_axes: exactly one must exist).
    _cand_tagged = Path(manifest_dir) / f"{REFERENCE_CNID}_{_DX0_TAG}.json"
    _cand_al = Path(manifest_dir) / f"{REFERENCE_CNID}.json"
    _exists = [p for p in (_cand_tagged, _cand_al) if p.is_file()]
    if len(_exists) != 1:
        raise FileNotFoundError(
            f"build_reference_template: expected exactly one template manifest in {manifest_dir} "
            f"({_cand_tagged.name} or {_cand_al.name}), found {[p.name for p in _exists]} "
            "(neither -> missing, both -> ambiguous)."
        )
    manifest_path = _exists[0]
    with open(manifest_path) as f:
        manifest_payload = json.load(f)
    manifest_cell = manifest_payload["cell"]
    a_t = float(manifest_cell[0][0])
    b_t = float(manifest_cell[1][1])
    c_t = float(manifest_cell[2][2])

    rng = np.random.default_rng(0)
    layout = build_canonical_layout(
        frac=frac, cell=cell, membership=membership, manifest=None, rng=rng, species_frac=species
    )
    fa_idx = layout.source_fixed_a_local_idx  # (N_BULK_PER_GRAIN,)
    if fa_idx.shape[0] != N_BULK_PER_GRAIN:
        raise ValueError(
            f"build_reference_template: reference FIXED_A count {fa_idx.shape[0]} != "
            f"N_BULK_PER_GRAIN={N_BULK_PER_GRAIN}"
        )
    m_idx = layout.source_primary_local_idx  # (m_ref,) canonical sub_lo->sub_hi order
    s_idx = layout.source_replica_local_idx  # (m_ref,)
    if m_idx.shape[0] != s_idx.shape[0]:
        raise ValueError(
            f"build_reference_template: reference PRIMARY count {m_idx.shape[0]} != REPLICA count "
            f"{s_idx.shape[0]}"
        )

    fixed_a_cart = frac[fa_idx] @ cell  # (N_BULK_PER_GRAIN, 3) Cartesian, Angstrom

    return ReferenceTemplate(
        fixed_a_cart=fixed_a_cart,
        a_template=a_t,
        b_template=b_t,
        c_template=c_t,
        primary_frac_ref=frac[m_idx],
        replica_frac_ref=frac[s_idx],
        reference_dir=row.dir,
    )


# ---------------------------------------------------------------------------
# P2-A / P2-B
# ---------------------------------------------------------------------------
def _mic(diff, cell):
    cell = np.asarray(cell, dtype=float)
    d = np.asarray(diff, dtype=float)
    frac = d @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


def dev_p2a(positions, cell, fixed_a_cart) -> float:
    """Maximum [A] of the per-atom norm of the difference (minimum image) between the Cartesian coordinates of indices 0..37 and fixed_a_cart of the template."""
    n = fixed_a_cart.shape[0]
    d = _mic(np.asarray(positions)[:n] - fixed_a_cart, cell)
    return float(np.linalg.norm(d, axis=1).max())


def dev_p2b(positions, cell, R, n_a=38) -> float:
    """d_i = r_B[i] - R.r_A[i] (R is applied to Cartesian coordinates. r_B[i] is index n_a+i), e_i = minimum image(d_i - d_0), max_i |e_i| [A]."""
    pos = np.asarray(positions, dtype=float)
    r_a = pos[:n_a]
    r_b = pos[n_a:2 * n_a]
    d = r_b - r_a @ np.asarray(R, dtype=float).T      # (R r_A[i]) = R @ r_A[i] -> for row vectors r_A @ R.T
    e = _mic(d - d[0], cell)
    return float(np.linalg.norm(e, axis=1).max())


def check_structure(atoms, template, R):
    """(P2-A deviation, P2-B deviation) of one structure."""
    pos = atoms.positions
    cell = atoms.cell[:]
    return dev_p2a(pos, cell, template.fixed_a_cart), dev_p2b(pos, cell, R, n_a=template.fixed_a_cart.shape[0])


def fail_validationes(atoms, template, R, threshold=P2_THRESHOLD_A):
    """FAIL validation (makes broken inputs from one sound structure):
    (W-A) swap the positions of indices 0 and 76 -> P2-A deviation >> 1e-4 A and FAIL
    (W-B) move one atom of FIXED_B (index 38+5) by +0.01 A in the x direction -> P2-B deviation ~ 1e-2 A and FAIL
    Returns: dict (sound values, deviations of each validation case, whether it FAILed)."""
    a0, b0 = check_structure(atoms, template, R)
    wa = atoms.copy()
    pos = wa.positions.copy()
    pos[[0, 76]] = pos[[76, 0]]
    wa.positions = pos
    wa_a, wa_b = check_structure(wa, template, R)
    wb = atoms.copy()
    pos = wb.positions.copy()
    pos[38 + 5, 0] += 0.01
    wb.positions = pos
    wb_a, wb_b = check_structure(wb, template, R)
    return {
        "healthy": {"p2a_A": a0, "p2b_A": b0, "p2a_pass": a0 <= threshold, "p2b_pass": b0 <= threshold},
        "swap_idx0_idx76": {"p2a_A": wa_a, "p2b_A": wa_b, "p2a_fail_detected": wa_a > threshold},
        "move_fixed_b_atom43_by_0.01A": {"p2a_A": wb_a, "p2b_A": wb_b, "p2b_fail_detected": wb_b > threshold},
        "threshold_A": threshold,
        "all_fail_validationes_detected": bool(wa_a > threshold and wb_b > threshold),
        "healthy_passes": bool(a0 <= threshold and b0 <= threshold),
    }
