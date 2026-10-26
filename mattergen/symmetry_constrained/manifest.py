"""Tilt-STGB Sigma5(210)[001] template manifest: role assignment + rigid projection fit.

Pure ``numpy``/``scipy``/``ase`` geometry helpers (no ``torch``). Geometry facts (screw operation R,
atom-count role convention, MASTER/SLAVE cross-block correspondence) were established empirically for the
Sigma5(210)[001] template bank; the tilt role/projection topology (two cross-mapped MASTER/SLAVE sub-blocks,
atom-count-based FIXED/bulk membership) is specific to this layout.

Role convention:
    FIXED  = bulk_membership.csv frozen_indices (76 atoms: 38/grain, both grain images)
    core   = all remaining atoms (GB1+GB2 core, padding-eligible zone)
    MASTER = core atoms with frac-x closer to 0.5 than to round(x)   (GB2 side, x~0.5)
    SLAVE  = core atoms with frac-x closer to round(x) than to 0.5  (GB1 side, x~0 or 1)

Cross-block projection (established by ``recon_smoke*.py`` probes, residual ~1e-14A on all 10
P1 pilot CNID templates): applying the self-symmetry operation R=diag(+1,-1,-1), t=(0.5,0.5,0.5)
maps MASTER's *low-x sub-block* (x in (0.4,0.5], adjacent to grain-A) onto SLAVE's *high-x
sub-block* (x>0.9, the grain-B image wrapped near the PBC seam), and MASTER's *high-x sub-block*
(x in (0.5,0.6), adjacent to grain-B) onto SLAVE's *low-x sub-block* (x in [0,0.1), grain-A) --
the correspondence is a genuine x+0.5 (mod 1) periodic shift, not a same-grain self-mirror. Both
sub-blocks share T_coeff = I_2 (2x2 identity, in-plane [y,z] fractional axes): the grain-B
in-plane rigid displacement (the CNID probe's ``dydz``, i.e. ``gb_t``) is applied verbatim
(not R-transformed) to both blocks' additive translation term, because ``dydz`` is added to
grain-B atoms only at build time and grain-A atoms are exactly invariant under it (verified,
max shift 0.0A for all 10 CNID templates).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from mattergen.symmetry_constrained.constants import ATOMS_PER_PLANE, N_BULK_PER_GRAIN, N_PLANES

R_SLAVE = np.diag([1.0, -1.0, -1.0])  # fractional, GB-normal axis = a-axis = index 0
# In-plane (axis [1,2]) response of the additive translation term to gb_t; identical for both
# cross-mapped sub-blocks (see module docstring / recon probes).
T_COEFF_INPLANE = np.eye(2)
COARSE_GRID_STEP = 0.05  # fractional grid step for the mandatory coarse t-search (
# SS7.1 note (i): initial-value-only refinement collapses to a spurious residual up to 4.86A on
# 7/10 CNID points; the coarse grid is not optional.
N_REFINE_ITERS = 8

# internal notes A-2: half-width of the
# MASTER/SLAVE sub-block x-bands around x=0.5 (MASTER) / x=0,1 (SLAVE), generalized from the
# n_bulk=38 hardcoded 0.10 so that the core (non-FIXED) region always splits evenly into the two
# cross-mapped sub-blocks regardless of N_BULK_PER_GRAIN. n_bulk=38 -> 0.25 - 18/120 = 0.10
# (unchanged); n_bulk=26 -> 0.25 - 12/120 = 0.15.
BAND_HALF = 0.25 - (N_BULK_PER_GRAIN / ATOMS_PER_PLANE - 1) / (2 * N_PLANES)


class SymmetricTiltGeometryError(ValueError):
    """Raised when a tilt template's role/projection geometry cannot be fit within tolerance."""

    def __init__(self, message: str, fit_err: float | None = None):
        super().__init__(message)
        self.fit_err = fit_err


def _mic(delta: np.ndarray) -> np.ndarray:
    """Fold a fractional coordinate delta into the minimum image [-0.5, 0.5)."""
    return delta - np.round(delta)


def _circular_mean(delta: np.ndarray, axis: int = 0) -> np.ndarray:
    """Boundary-safe circular mean of fractional deltas (handles the +-0.5 wrap ambiguity;
    mirrors ``mattergen.gb.manifest._circular_mean`` / ``sym_common``'s refinement step)."""
    ang = 2 * np.pi * delta
    return np.arctan2(np.sin(ang).mean(axis=axis), np.cos(ang).mean(axis=axis)) / (2 * np.pi)


def _pairwise_mic_dist(cell: np.ndarray, frac_a: np.ndarray, frac_b: np.ndarray) -> np.ndarray:
    diff = frac_a[:, None, :] - frac_b[None, :, :]
    diff = _mic(diff)
    cart = diff @ cell
    return np.linalg.norm(cart, axis=2)


def _hungarian_match(cell: np.ndarray, frac_mapped: np.ndarray, frac_target: np.ndarray):
    """Match ``frac_mapped`` (already R/b-transformed source) onto ``frac_target`` (destination).

    Returns ``(row, col, residuals)`` with ``frac_mapped[row[k]]`` corresponding to
    ``frac_target[col[k]]``.
    """
    if frac_mapped.shape[0] != frac_target.shape[0]:
        return None, None, None
    dist = _pairwise_mic_dist(cell, frac_mapped, frac_target)
    row, col = linear_sum_assignment(dist)
    return row, col, dist[row, col]


def _coarse_search_b0(
    cell: np.ndarray, frac_src: np.ndarray, frac_dst: np.ndarray, rotation: np.ndarray, grid_step: float = COARSE_GRID_STEP
) -> np.ndarray:
    """Grid search over the in-plane translation (axis0 fixed at 0.5) minimizing the total
    Hungarian-matched residual. Mandatory (note (i)): a fixed initial guess
    collapses to a wrong local optimum for most CNID points."""
    best_cost = np.inf
    best_b0 = None
    grid = np.arange(0.0, 1.0, grid_step)
    for by in grid:
        for bz in grid:
            b0 = np.array([0.5, by, bz])
            mapped = _mic(frac_src @ rotation.T + b0)
            _row, _col, residuals = _hungarian_match(cell, mapped, frac_dst)
            if residuals is None:
                continue
            cost = float(residuals.sum())
            if cost < best_cost:
                best_cost = cost
                best_b0 = b0
    if best_b0 is None:
        raise SymmetricTiltGeometryError("coarse b0 search found no valid assignment (atom count mismatch)")
    return best_b0


def _refine_b0(
    cell: np.ndarray,
    frac_src: np.ndarray,
    frac_dst: np.ndarray,
    rotation: np.ndarray,
    b0_init: np.ndarray,
    n_iter: int = N_REFINE_ITERS,
):
    """Iterative Hungarian-match + circular-mean translation refinement (mirrors
    ``sym_common._refine_t`` / ``mattergen.gb.manifest._fit_translation_fixed_R``)."""
    b0 = b0_init.copy()
    row = col = residuals = None
    for _ in range(n_iter):
        mapped = _mic(frac_src @ rotation.T + b0)
        row, col, residuals = _hungarian_match(cell, mapped, frac_dst)
        if residuals is None:
            raise SymmetricTiltGeometryError("refine b0: atom count mismatch")
        delta = _mic(frac_dst[col] - mapped[row])
        cart_delta = delta @ cell
        mean_cart = cart_delta.mean(axis=0)
        b0 = b0 + mean_cart @ np.linalg.inv(cell)
    mapped = _mic(frac_src @ rotation.T + b0)
    row, col, residuals = _hungarian_match(cell, mapped, frac_dst)
    return b0, row, col, residuals


@dataclass
class RoleAssignment:
    """Per-template role/index bookkeeping (indices are into the template's own atom array)."""

    fixed_idx: np.ndarray  # (76,) sorted ascending, from bulk_membership.csv frozen_indices
    master_idx: np.ndarray  # (n_core/2,) sorted ascending; GB2 core (x~0.5)
    slave_idx: np.ndarray  # (n_core/2,) sorted ascending; GB1 core (x~0 or 1)
    master_sub_lo_idx: np.ndarray  # subset of master_idx with x in (0.4, 0.5] (grain-A side)
    master_sub_hi_idx: np.ndarray  # subset of master_idx with x in (0.5, 0.6) (grain-B side)
    slave_sub_lo_idx: np.ndarray  # subset of slave_idx with x in [0.0, 0.1) (grain-A side)
    slave_sub_hi_idx: np.ndarray  # subset of slave_idx with x > 0.9 (grain-B side)


def assign_roles(frac: np.ndarray, cell: np.ndarray, fixed_idx: np.ndarray) -> RoleAssignment:
    """Split the non-fixed ("core") atoms into MASTER/SLAVE (note (ii)):
    ``|x-0.5| < |x-round(x)|`` -> MASTER (GB2 side), else SLAVE (GB1 side). Further splits each
    into the two cross-mapped sub-blocks used by the projection (module docstring).

    **Sub-block assignment (internal notes
    r3 Step G-2, 2026-09-21):** MASTER's lo/hi split uses the same ``x <= 0.5`` seam threshold as
    before (unchanged, bit-identical for every atom that is not exactly on the seam). SLAVE's
    lo/hi split is **no longer computed independently from SLAVE's own raw x** (r1/r2 both did
    this and it broke for boundary atoms: a SLAVE atom whose true position is the periodic image
    of the seam can be stored as x=0.0 exactly, which an independent raw-x threshold always
    classifies "lo" even when its MASTER partner is "hi" -- r2's own investigation,
    ``results/ _crossmap_semantics.md`` SS2-3/2-5, traced the r1 failures and the r2
    regression to exactly this asymmetry). Instead, the master<->slave correspondence (pairing) is
    established FIRST via one block-independent Hungarian match over the *entire* core atom set
    (``R_SLAVE`` + a jointly-fit in-plane translation, using the same ``_coarse_search_b0``/
    ``_refine_b0`` machinery ``fit_projection`` uses per sub-block -- Step F-1's measured margin,
    3.3e-16A correct vs 0.151A wrong, is >>14 orders of magnitude, so this is unambiguous), and
    each SLAVE atom then simply inherits its bucket from its matched MASTER partner's bucket
    (MASTER lo -> SLAVE hi, MASTER hi -> SLAVE lo). This guarantees I-1 (completeness, both sides)
    and I-2 (bucket sizes match) *by construction* rather than by hoping two independent
    measurements agree, and I-3 (correspondence integrity) trivially holds since SLAVE's bucket
    literally *is* its partner's bucket. This correspondence-only fit does not feed into the
    actual reconstruction geometry (``fit_projection`` below still fits its own per-block b0/
    residual from the resulting index sets, unchanged) -- it exists solely to resolve the
    boundary-atom bucket ambiguity.
    """
    n = frac.shape[0]
    fixed_set = set(int(i) for i in fixed_idx)
    core_idx = np.array([i for i in range(n) if i not in fixed_set], dtype=np.int64)
    x = frac[core_idx, 0]
    dist_half = np.abs(x - 0.5)
    dist_zero = np.abs(x - np.round(x))
    is_master = dist_half < dist_zero
    master_idx = np.sort(core_idx[is_master])
    slave_idx = np.sort(core_idx[~is_master])

    xm = frac[master_idx, 0]
    if not np.all((xm > 0.5 - BAND_HALF) & (xm < 0.5 + BAND_HALF)):
        raise SymmetricTiltGeometryError(
            f"master core atom(s) outside expected band ({0.5 - BAND_HALF},{0.5 + BAND_HALF}): "
            f"x range=[{xm.min():.6f},{xm.max():.6f}] n_master={len(master_idx)}"
        )
    xs = frac[slave_idx, 0]
    if not np.all((xs < BAND_HALF) | (xs > 1.0 - BAND_HALF)):
        raise SymmetricTiltGeometryError(
            f"slave core atom(s) outside expected band [0,{BAND_HALF})u({1.0 - BAND_HALF},1]: "
            f"x range=[{xs.min():.6f},{xs.max():.6f}] n_slave={len(slave_idx)}"
        )

    # Establish the master<->slave correspondence (block-independent; see docstring above). Raises
    # SymmetricTiltGeometryError (via _coarse_search_b0) if the two core sets have unequal atom counts.
    f_master = frac[master_idx]
    f_slave = frac[slave_idx]
    b0_corr_coarse = _coarse_search_b0(cell, f_master, f_slave, R_SLAVE)
    _b0_corr, row, col, residuals = _refine_b0(cell, f_master, f_slave, R_SLAVE, b0_corr_coarse)
    if residuals is None:
        raise SymmetricTiltGeometryError(
            "master<->slave correspondence fit failed (atom count mismatch after band checks)"
        )
    partner_local = np.empty(len(master_idx), dtype=np.int64)
    partner_local[row] = col
    slave_partner_of_master = slave_idx[partner_local]  # aligned with master_idx order

    is_master_lo = xm <= 0.5
    master_sub_lo = master_idx[is_master_lo]
    master_sub_hi = master_idx[~is_master_lo]
    # Block A: master_sub_lo -> slave_sub_hi ; Block B: master_sub_hi -> slave_sub_lo (module
    # docstring). SLAVE bucket = partner's opposite-of-MASTER-bucket, not an independent threshold.
    slave_sub_hi = np.sort(slave_partner_of_master[is_master_lo])
    slave_sub_lo = np.sort(slave_partner_of_master[~is_master_lo])

    return RoleAssignment(
        fixed_idx=np.sort(np.asarray(fixed_idx, dtype=np.int64)),
        master_idx=master_idx,
        slave_idx=slave_idx,
        master_sub_lo_idx=master_sub_lo,
        master_sub_hi_idx=master_sub_hi,
        slave_sub_lo_idx=slave_sub_lo,
        slave_sub_hi_idx=slave_sub_hi,
    )


@dataclass
class ReconstructionParams:
    """Slave reconstruction parameters, one row per SLAVE atom.

    ``slave = master[slave_master_indices[k]] @ R.T + b0[k] + T_coeff[k] @ gb_t_inplane``
    (``gb_t_inplane`` = the 2-vector diffused/read latent, in-plane [axis1, axis2] only; axis 0
    of the translation is pinned at 0.5, not diffused).
    """

    slave_atom_indices: np.ndarray  # (n_slave,) ascending, template-local index (into full atom array)
    slave_master_indices: np.ndarray  # (n_slave,) master atom global index each slave maps from
    rotation: np.ndarray  # (n_slave, 3, 3) = R_SLAVE broadcast (uniform)
    b0: np.ndarray  # (n_slave, 3) per-atom translation constant (uniform per sub-block: (0.5,0.5,0.5))
    t_coeff_inplane: np.ndarray  # (n_slave, 2, 2) = identity broadcast (uniform)
    sub_block: np.ndarray  # (n_slave,) 0 = master_sub_lo -> slave_sub_hi, 1 = master_sub_hi -> slave_sub_lo
    fit_residual_A: float  # max Hungarian-matched residual across both sub-blocks (template frame)


def fit_projection(frac: np.ndarray, cell: np.ndarray, roles: RoleAssignment) -> ReconstructionParams:
    """Fit b0 (per sub-block, mandatory coarse grid search then refine) and build the per-slave
    reconstruction table. R and T_coeff are architectural constants (not fit per-template; the
    fixed R=diag(1,-1,-1) has been verified to give ~1e-14A residual with the fit b0 on all 10
    P1 pilot CNID templates -- if a template deviates, this raises so the caller can escalate).
    """
    # Block A: master_sub_lo -> slave_sub_hi
    f_msrc_a = frac[roles.master_sub_lo_idx]
    f_dst_a = frac[roles.slave_sub_hi_idx]
    b0_a_coarse = _coarse_search_b0(cell, f_msrc_a, f_dst_a, R_SLAVE)
    b0_a, row_a, col_a, res_a = _refine_b0(cell, f_msrc_a, f_dst_a, R_SLAVE, b0_a_coarse)

    # Block B: master_sub_hi -> slave_sub_lo
    f_msrc_b = frac[roles.master_sub_hi_idx]
    f_dst_b = frac[roles.slave_sub_lo_idx]
    b0_b_coarse = _coarse_search_b0(cell, f_msrc_b, f_dst_b, R_SLAVE)
    b0_b, row_b, col_b, res_b = _refine_b0(cell, f_msrc_b, f_dst_b, R_SLAVE, b0_b_coarse)

    fit_residual = float(max(res_a.max(), res_b.max()))

    n_slave = len(roles.slave_idx)
    slave_master_indices = np.zeros(n_slave, dtype=np.int64)
    b0_full = np.zeros((n_slave, 3))
    sub_block = np.zeros(n_slave, dtype=np.int64)

    slave_pos = {int(a): i for i, a in enumerate(roles.slave_idx)}

    # Block A fills slave_sub_hi entries; row_a indexes master_sub_lo, col_a indexes slave_sub_hi
    for k in range(len(roles.master_sub_lo_idx)):
        master_atom = roles.master_sub_lo_idx[row_a[k]]
        slave_atom = roles.slave_sub_hi_idx[col_a[k]]
        i = slave_pos[int(slave_atom)]
        slave_master_indices[i] = master_atom
        b0_full[i] = b0_a
        sub_block[i] = 0

    for k in range(len(roles.master_sub_hi_idx)):
        master_atom = roles.master_sub_hi_idx[row_b[k]]
        slave_atom = roles.slave_sub_lo_idx[col_b[k]]
        i = slave_pos[int(slave_atom)]
        slave_master_indices[i] = master_atom
        b0_full[i] = b0_b
        sub_block[i] = 1

    rotation = np.broadcast_to(R_SLAVE, (n_slave, 3, 3)).copy()
    t_coeff = np.broadcast_to(T_COEFF_INPLANE, (n_slave, 2, 2)).copy()

    return ReconstructionParams(
        slave_atom_indices=roles.slave_idx.copy(),
        slave_master_indices=slave_master_indices,
        rotation=rotation,
        b0=b0_full,
        t_coeff_inplane=t_coeff,
        sub_block=sub_block,
        fit_residual_A=fit_residual,
    )


def read_gb_t_inplane(
    frac: np.ndarray, frac_template: np.ndarray, master_sub_hi_idx: np.ndarray
) -> np.ndarray:
    """Read out the effective in-plane gb_t (fractional [y,z]) of a structure relative to its
    own k=0 template, from the master_sub_hi (grain-B) sub-block displacement (the sub-block
    that actually carries the CNID/relaxation in-plane shift; master_sub_lo is invariant, design
    : canonicalization must re-fit from the relaxed structure, not reuse the as-built
    seed). Uses a circular mean to avoid the +-0.5 wrap cancellation failure mode (verified: naive
    ``.mean()`` collapses cnid_2_0's true dydz=(0,0.5) to a spurious near-zero value).
    """
    raw = _mic(frac[master_sub_hi_idx] - frac_template[master_sub_hi_idx])
    return _circular_mean(raw[:, 1:3], axis=0)


def reconstruct_slave(
    frac_master: np.ndarray,
    roles: RoleAssignment,
    recon: ReconstructionParams,
    gb_t_inplane: np.ndarray,
    t0_inplane: np.ndarray = np.zeros(2),
) -> np.ndarray:
    """Deterministic slave reconstruction: for each slave atom,
    ``slave = master[src] @ R.T + b0 + [0, T_coeff @ (gb_t - t0)]`` (axis 0 of the additive term
    is always 0 here since b0 already carries the fixed 0.5 normal-axis offset).

    ``frac_master`` must be indexable by the *global* master atom indices in
    ``recon.slave_master_indices`` (i.e. the full per-template fractional coordinate array, not
    a pre-sliced master-only array).
    """
    n_slave = len(recon.slave_atom_indices)
    src = frac_master[recon.slave_master_indices]  # (n_slave, 3)
    rotated = np.einsum("kij,kj->ki", recon.rotation, src)  # slave = src @ R.T i.e. R @ src (R diag)
    inplane_term = np.einsum("kij,j->ki", recon.t_coeff_inplane, gb_t_inplane - t0_inplane)  # (n_slave,2)
    add = np.zeros((n_slave, 3))
    add[:, 1:3] = inplane_term
    return _mic(rotated + recon.b0 + add)


@dataclass
class TiltTemplateManifest:
    """Full manifest payload for one CNID template."""

    cnid: str
    natoms: int
    cell: np.ndarray
    roles: RoleAssignment
    recon: ReconstructionParams
    gb_t_frac_yz: np.ndarray  # (2,) CNID dydz (fractional, in-plane), design doc SS4-2 point 3
    gb_t_cnid_ij: tuple[int, int]
    fit_residual_A: float


def build_tilt_template_manifest(
    frac: np.ndarray,
    cell: np.ndarray,
    fixed_idx: np.ndarray,
    cnid: str,
    gb_t_frac_yz: np.ndarray,
    cnid_ij: tuple[int, int],
    symmetry_tolerance: float = 1e-6,
) -> TiltTemplateManifest:
    """Build the full manifest entry for one CNID k=0 template (as-built).

    Raises :class:`SymmetricTiltGeometryError` if the projection fit residual exceeds
    ``symmetry_tolerance`` (: "T_coeff closed-form fit residual > 1e-6A ->
    implementation halt, escalate to design").
    """
    roles = assign_roles(frac, cell, fixed_idx)
    recon = fit_projection(frac, cell, roles)
    if recon.fit_residual_A > symmetry_tolerance:
        raise SymmetricTiltGeometryError(
            f"{cnid}: projection fit residual {recon.fit_residual_A:.3e}A exceeds "
            f"tolerance {symmetry_tolerance:.3e}A",
            fit_err=recon.fit_residual_A,
        )
    return TiltTemplateManifest(
        cnid=cnid,
        natoms=frac.shape[0],
        cell=np.asarray(cell, dtype=float),
        roles=roles,
        recon=recon,
        gb_t_frac_yz=np.asarray(gb_t_frac_yz, dtype=float),
        gb_t_cnid_ij=cnid_ij,
        fit_residual_A=recon.fit_residual_A,
    )
