"""Canonical [FIXED_A | FIXED_B | MASTER | SLAVE] atom layout for tilt-STGB Sigma5(210)[001]
P1 (bulk-frozen relaxed) training structures (design doc workspace/docs/design/19_
mt3_diffusion_wiring.md SS3.2, M-t3a).

Pure ``numpy``/``ase`` geometry, mirroring ``mattergen.symmetry_constrained.manifest``'s module-docstring
convention (no ``torch`` here; the torch-side ChemGraph assembly lives in :mod:`dataset`).

Canonical layout (``N_PAD = 128``)::

    index:  [ 0 .. 37 ] [ 38 .. 75 ] [ 76 .. 101 ] [ 102 .. 127 ]
    block:    FIXED_A     FIXED_B      MASTER(26)     SLAVE(26)
    count:      38          38        m real + (26-m) padding

``m = 22 - k`` (k = 0..4 removal count -> m = 22,21,20,19,18); ``N = 76 + 2m = 120 - 2k``.

Role-assignment / MASTER<->SLAVE-correspondence / effective-gb_t algorithm (
[2026-08-05 correction] + SS4 M-t3a pseudocode, SECOND REVISION):

- FIXED_A / FIXED_B: read directly off ``bulk_membership.csv``'s ``bulkA_indices`` /
  ``bulkB_indices`` columns for the row's own (cnid, k, seed) -- these are LOCAL (per-structure)
  atom indices, already ordered consistently with the cnid's own k=0 template's bulk-A/bulk-B
  reference-coordinate order (verified empirically across all 10 cnid x 5 k combinations sampled:
  same-cnid, same-slot coordinates match to machine precision across k values despite the raw
  index VALUES differing due to post-removal renumbering; see the M-t3 implementation report
  SS2.1 for the numeric verification this claim rests on).
- MASTER / SLAVE (core atoms = all atoms minus FIXED): role split via
  ``mattergen.symmetry_constrained.manifest.assign_roles`` (the same ``|x-0.5| < |x-round(x)|`` convention as
  the P0 ``recon_residual_check.py`` precedent), applied directly to
  each structure's own frac coordinates + its own ``frozen_indices`` (from
  ``bulk_membership.csv``).
- MASTER<->SLAVE correspondence: ``mattergen.symmetry_constrained.manifest.fit_projection`` (coarse b0 grid
  search + iterative refinement, per sub-block), applied to the structure's OWN
  ``assign_roles`` output. This coarse-search refit remains necessary purely to establish the
  MASTER<->SLAVE atom-index correspondence (a combinatorial/indexing problem); it is NOT how the
  projection formula itself is parameterised at training/generation time (see below).
- **Effective gb_t (, 2026-08-05 correction, supersedes the first M-t3
  implementation's use of the CNID TEMPLATE's ``gb_t_effective_frac_yz``)**: the graph-level
  ``gb_t`` diffusion label stored for a structure is NOT the manifest's per-cnid
  ``gb_t_effective_frac_yz`` (that value is only correct for the as-built k=0 template; k>0
  removal + core relaxation slides the GB1/GB2 core by a rigid in-plane offset d, which (because
  R's in-plane component is -1) shifts the TRUE effective translation by 2d relative to the
  template value -- original prediction, confirmed by 3 diagnostic runs,
  ). Instead it is refit PER STRUCTURE from ``fit_projection``'s own output:
  with the canonical gauge fixed at ``B_CANONICAL = (0.5, 0.5, 0.5)`` (; the
  alternate gauge (0.5,0,0) differs by the periodic fcc sub-lattice degeneracy (0,0.5,0.5) and is
  NOT used), ``g = wrap01(b0_sub - B_CANONICAL)[1:3]`` per sub-block (sub-block A = master_sub_lo
  -> slave_sub_hi, sub-block B = master_sub_hi -> slave_sub_lo). The two sub-blocks agree in
  402/410 structures; this module uses sub-block A's value as the primary
  effective gb_t (matches the diagnostic run's "A convention primary column",
  ``experiments/earlier_experiment/effective_gbt_per_structure.csv`` columns
  ``g_A_y``/``g_A_z``/``delta_g_y``/``delta_g_z``). Substituting ``T_total = B_CANONICAL +
  (0, g_y, g_z)`` directly (no fit) reproduces the true SLAVE coordinates to ~1e-12A across all
  410 structures x 2 sub-blocks -- this closed form (not the raw
  ``fit_projection`` output) is what ``projection.py`` (M-t3b) implements as a fixed, MASTER-only
  function of the (learned/diffused) ``gb_t`` latent.
- **FIXED_B's gb_t treatment -- RETRACTED claim, current implementation differs** (
  SS2.5-C / SS3.3 step 2 originally claimed **FIXED_B does NOT receive a gb_t-based rigid shift**,
  correcting the first M-t3 implementation which wrongly applied the template gb_t to FIXED_B, a
  stacked-grain ``grain2_bulk = template + gb_t`` convention that does not hold for tilt + bulk-
  freezing; that SS2.5-C claim anchored the bulk (FIXED) registry to a SINGLE shared reference
  template (``cnid_0_0``'s k=0 structure) not moving with the core's effective gb_t -- verified to
  ~1e-15A across all 410 structures at the time, F4 in the diagnostic run).
  **This claim is RETRACTED** by design doc
  ``internal notes`` (2026-08-06) SS1.2/SS2.2: "SS2.5-C
  ('FIXED_B does NOT receive a gb_t-based rigid shift') is RETRACTED --
  SS1.2/SS2.2 -- and FIXED_B is reconstructed accordingly" (``projection.py:5-6``). The actual
  generation-time reconstruction that runs today DOES apply a gb_t-based shift to FIXED_B:
  "FIXED_B is reconstructed as R @ FIXED_A_template + b0 + (0, gb_t_y, gb_t_z)"
  (``projection.py:26``). **This module (layout.py) is unaffected by either the original claim or
  its retraction**: it still stores each structure's own FIXED_A/FIXED_B coordinates verbatim (they
  ARE already the correct, cnid/k/seed-specific bulk positions read from the structure's own
  POSCAR_relaxed + bulk_membership.csv row -- no reconstruction needed at the data layer). The
  retracted-vs-current distinction matters only for projection.py's generation-time FIXED_B
  reconstruction from the template bank, not for this module's per-sample verbatim readout.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from ase.io import read as ase_read

from mattergen.symmetry_constrained.constants import (
    SCM_HOST_Z,
    SCM_SOLUTE_ZS,
    GBRole,
    N_BULK_PER_GRAIN,
    N_MASTER_MAX,
)
from mattergen.symmetry_constrained.manifest import (
    BAND_HALF,
    SymmetricTiltGeometryError,
    ReconstructionParams,
    RoleAssignment,
    _mic,
    assign_roles,
    fit_projection,
)

# ---------------------------------------------------------------------------
# Canonical block layout. N_FIXED_A/N_FIXED_B derive from
# N_BULK_PER_GRAIN (internal notes A-1;
# n_bulk=38 -> 38, unchanged default).
# ---------------------------------------------------------------------------
N_FIXED_A = N_BULK_PER_GRAIN
N_FIXED_B = N_BULK_PER_GRAIN
N_MASTER_SLOTS = N_MASTER_MAX  # 26
N_SLAVE_SLOTS = N_MASTER_MAX  # 26 (SLAVE is 1:1 with MASTER, same slot count)
N_PAD = N_FIXED_A + N_FIXED_B + N_MASTER_SLOTS + N_SLAVE_SLOTS  # 128

FIXED_A_SLICE = slice(0, N_FIXED_A)
FIXED_B_SLICE = slice(N_FIXED_A, N_FIXED_A + N_FIXED_B)
MASTER_SLICE = slice(N_FIXED_A + N_FIXED_B, N_FIXED_A + N_FIXED_B + N_MASTER_SLOTS)
SLAVE_SLICE = slice(N_FIXED_A + N_FIXED_B + N_MASTER_SLOTS, N_PAD)

PAD_Z_TILT: Literal[101] = 101  # padding-token atomic number (: the tilt stack
# uses atomic-number 101 for padded slots). A literal (not imported) to keep layout.py torch-import-free,
# matching manifest.py/cnid_geometry.py's pure-numpy style.

# (r1) Step B: HOST_Z / ALLOWED_REAL_SPECIES are now generalized to any
# host+solute atomic-number set via SCM_HOST_Z / SCM_SOLUTE_ZS (mattergen.symmetry_constrained.constants,
#), instead of a hardcoded host/solute pair. Unset env vars reproduce the earlier
# behavior exactly (HOST_Z=13, ALLOWED_REAL_SPECIES=={13,12}) -- "do not change the existing behaviour of the Al version by even a single bit".
# "do not change it by even a single bit".
HOST_Z = SCM_HOST_Z
ALLOWED_REAL_SPECIES = frozenset({HOST_Z} | SCM_SOLUTE_ZS)  # padding (PAD_Z_TILT=101) is
# NOT a real atom, so it is intentionally excluded from this set.

# through into MASTER/SLAVE real-atom slots.: kept as
# the solute literal for callers/tests that still reference it directly; NOT used to derive
# ALLOWED_REAL_SPECIES any longer (HOST_Z/ALLOWED_REAL_SPECIES above is the single source of truth).

# NOT necessarily aluminium: equals SCM_HOST_Z (the host atomic number of the run).
# Do not use in new code. (labels do not preserve provenance.)

GB_GAMMA_PROPERTY_KEY: Literal["gb_gamma_J_m2"] = "gb_gamma_J_m2"  # single
# constant name note ("key name ... defined as a code Literal").

# Canonical gauge for the additive translation b0. The other periodic
# gauge (0.5, 0.0, 0.0) differs from this one by the fcc sub-lattice degeneracy (0, 0.5, 0.5) and
# represents the SAME physical point; fixing this one canonically is what lets the effective gb_t
# be read off as a single well-defined graph-level scalar pair per structure.
B_CANONICAL = np.array([0.5, 0.5, 0.5])


class LayoutError(ValueError):
    """Raised when a structure's role/correspondence geometry cannot be resolved (
    SS6 edge case 4: FIXED consistency or bijection failures must halt loudly, not be silently
    dropped)."""


@dataclass
class BulkMembershipRow:
    cnid: str
    k: int
    seed: int
    natoms: int
    frozen_indices: np.ndarray  # (76,) local index, ascending (as stored in the CSV)
    bulkA_indices: np.ndarray  # (38,) local index, cnid-k0-template-bulkA-slot order
    bulkB_indices: np.ndarray  # (38,) local index, cnid-k0-template-bulkB-slot order


def load_bulk_membership(csv_path: str | Path) -> dict[tuple[str, int, int], BulkMembershipRow]:
    """Load ``bulk_membership.csv`` into a ``(cnid,k,seed) -> row`` map.

    Only ``status == "ok"`` rows are kept.
    """
    out: dict[tuple[str, int, int], BulkMembershipRow] = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            key = (r["cnid"], int(r["k"]), int(r["seed"]))
            out[key] = BulkMembershipRow(
                cnid=r["cnid"],
                k=int(r["k"]),
                seed=int(r["seed"]),
                natoms=int(r["natoms"]),
                frozen_indices=np.array(
                    [int(x) for x in r["frozen_indices"].split(";") if x], dtype=np.int64
                ),
                bulkA_indices=np.array(
                    [int(x) for x in r["bulkA_indices"].split(";") if x], dtype=np.int64
                ),
                bulkB_indices=np.array(
                    [int(x) for x in r["bulkB_indices"].split(";") if x], dtype=np.int64
                ),
            )
    return out


@dataclass
class GammaRow:
    cnid: str
    k: int
    seed: int
    dir: str
    converged: bool
    natoms: int
    energy_eV: float
    area_A2: float
    gamma_Jm2: float


def load_gamma_csv(csv_path: str | Path) -> list[GammaRow]:
    """Load ``relabel_gamma.csv`` as an ordered list of rows."""
    rows: list[GammaRow] = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                GammaRow(
                    cnid=r["cnid"],
                    k=int(r["k"]),
                    seed=int(r["seed"]),
                    dir=r["dir"],
                    converged=(r["converged"] == "True"),
                    natoms=int(r["natoms"]),
                    energy_eV=float(r["energy_eV"]),
                    area_A2=float(r["area_A2"]),
                    gamma_Jm2=float(r["gamma_Jm2"]),
                )
            )
    return rows


@dataclass
class TiltManifestLite:
    """Minimal manifest fields this module needs (subset of
    ``mattergen.symmetry_constrained.manifest.TiltTemplateManifest``'s JSON payload)."""

    cnid: str
    natoms: int
    cell: np.ndarray  # (3,3)
    gb_t_effective_frac_yz: np.ndarray  # (2,) -- the ONLY gb_t field to use




@dataclass
class CanonicalLayoutResult:
    """Output of :func:`build_canonical_layout` for one structure."""

    pos: np.ndarray  # (128, 3) fractional, padding slots filled with U(0,1)^3 draws
    atomic_numbers: np.ndarray  # (128,) int, PAD_Z_TILT for padded MASTER/SLAVE slots
    gb_role: np.ndarray  # (128,) int (GBRole values): 0=FIXED,1=MASTER,2=SLAVE (padding slots
    # keep their block's role "padding slots excluded from pos loss via role
    # mask, included in atomic_numbers/D3PM loss")
    is_real: np.ndarray  # (128,) bool: True for real atoms, False for padding padding
    n_real_master: int  # m = 22 - k
    fit_residual_A: float  # MASTER<->SLAVE projection fit residual for this structure
    effective_gb_t_yz: np.ndarray  # (2,) this structure's OWN refit in-plane translation (design
    #, sub-block A convention) -- the graph-level gb_t diffusion label. NOT the
    # cnid template's gb_t_effective_frac_yz.
    effective_gb_t_subblock_diff: float  # |g_A - g_B| (F1 diagnostic
    # column), reported for Gate a2's cross-check; PASS threshold is on effective_gb_t_yz's match
    # to the reference CSV, not on this value being zero (402/410 structures agree exactly, 8/410
    # differ by the (0.5,0.5) gauge-degenerate pair).
    source_master_local_idx: np.ndarray  # (m,) local POSCAR index for each real MASTER slot,
    # in canonical slot order
    source_slave_local_idx: np.ndarray  # (m,) local POSCAR index for each real SLAVE slot
    source_fixed_a_local_idx: np.ndarray  # (38,)
    source_fixed_b_local_idx: np.ndarray  # (38,)


def _master_slave_order(
    roles: RoleAssignment, recon: ReconstructionParams
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(master_order, slave_order)``: MASTER local indices in canonical
    sub_lo -> sub_hi order, and the corresponding SLAVE local indices
    (1:1, via ``recon.slave_master_indices``).

    ``roles.master_idx`` is already ``sub_lo`` followed by ``sub_hi`` in ascending order within
    each sub-block (``assign_roles`` builds ``master_sub_lo_idx``/``master_sub_hi_idx`` via
    boolean masks over the already-ascending ``master_idx``, so concatenating them preserves
    ascending order within each half); we use ``master_sub_lo_idx``/``master_sub_hi_idx``
    directly to make the ordering explicit and independent of ``assign_roles``' internal
    concatenation order.
    """
    master_order = np.concatenate([roles.master_sub_lo_idx, roles.master_sub_hi_idx])
    # slave_master_indices[i] gives, for recon.slave_atom_indices[i], which MASTER atom it maps
    # from. Build the inverse lookup: master_local_idx -> slave_local_idx.
    master_to_slave = {
        int(m): int(s)
        for s, m in zip(recon.slave_atom_indices.tolist(), recon.slave_master_indices.tolist())
    }
    missing = [int(m) for m in master_order if int(m) not in master_to_slave]
    if missing:
        raise LayoutError(
            f"master atoms {missing} have no SLAVE correspondence in fit_projection output "
            "(bijection failure)"
        )
    slave_order = np.array([master_to_slave[int(m)] for m in master_order], dtype=np.int64)
    return master_order, slave_order


def _wrap01(v: np.ndarray, decimals: int = 6) -> np.ndarray:
    """Wrap into ``[0, 1)`` with a pre-rounding step (mirrors the diagnostic run's ``wrap01``,
    ``experiments/earlier_experiment/scripts/effective_gbt_refit.py``): rounding to
    ``decimals`` BEFORE the mod avoids ``np.mod`` returning a value at/near 1.0 for inputs that
    are supposed to be exactly 0.0 but carry a tiny negative floating-point residual (e.g.
    -1e-16), which ``np.mod(-1e-16, 1.0)`` otherwise resolves to ``0.9999999999999999`` rather
    than ``0.0``.
    """
    return np.round(np.mod(np.round(np.asarray(v, dtype=float), decimals), 1.0), decimals)


def _effective_gb_t(recon: ReconstructionParams) -> tuple[np.ndarray, float]:
    """Refit in-plane effective gb_t from this structure's own ``fit_projection`` output (design
    ): ``g_sub = wrap01(b0_sub - B_CANONICAL)[1:3]`` per sub-block, sub-block A
    (``sub_block == 0``, master_sub_lo -> slave_sub_hi) as the primary value (matches the
    diagnostic run's ``g_A`` / "A convention primary column"). Returns ``(g_yz, |g_A - g_B|)``;
    the latter is a diagnostic-only cross-check (: 402/410 structures agree
    exactly, 8/410 differ by the gauge-degenerate (0.5,0.5) pair -- Gate a2 does not require this
    to be zero).
    """
    sub_block = np.asarray(recon.sub_block)
    b0 = np.asarray(recon.b0)
    b0_a = b0[sub_block == 0][0]
    b0_b = b0[sub_block == 1][0]
    g_a = _wrap01(b0_a - B_CANONICAL)[1:3]
    g_b = _wrap01(b0_b - B_CANONICAL)[1:3]
    diff = float(np.linalg.norm(_mic(g_a - g_b)))
    return g_a, diff


def _validate_inherited_roles(
    frac: np.ndarray, roles: RoleAssignment, membership: BulkMembershipRow
) -> None:
    """Sanity guard for a dx-shifted structure reusing a dx=0-derived ``RoleAssignment`` (case C
    of the periodic-boundary-seam fix, an internal review, constraint 4; silent fallback is prohibited). This is a
    forward-looking safeguard against a future ``|dx|`` large enough to physically move an atom out of
    its inherited MASTER/SLAVE band -- for the dx range this fix targets, every check here is
    expected to pass trivially (the dx-shift only translates GB-normal-axis cartesian positions,
    it never reorders atoms or moves them across the MIC-safe x~0.5 / x~0,1 bands). Raises
    ``SymmetricTiltGeometryError`` (caught by the caller below and re-raised as ``LayoutError``) on any
    deviation -- this must fail loudly, not silently relabel/drop atoms.

    Checks (SS3-1 constraint 4 (i)-(ii); (iii), the existing 4 count-mismatch ``LayoutError``s in
    :func:`build_canonical_layout`, are unaffected by this function and continue to run as-is):
      (i) the inherited roles' atom count (``fixed_idx`` + ``master_idx`` + ``slave_idx``) equals
          this structure's own ``frac.shape[0]``.
      (ii) at this structure's own (possibly dx-shifted) coordinates, every inherited MASTER atom
           still has ``|x-0.5| < 0.1`` and every inherited SLAVE atom still has
           ``|x-round(x)| < 0.1`` -- the same MIC-safe half-width band ``assign_roles`` itself
           relies on (module docstring / ``manifest.py`` SS role convention), evaluated here
           against the coordinates the caller is about to reconstruct with the inherited roles.
    """
    natoms = frac.shape[0]
    n_from_roles = len(roles.fixed_idx) + len(roles.master_idx) + len(roles.slave_idx)
    if n_from_roles != natoms:
        raise SymmetricTiltGeometryError(
            f"inherited RoleAssignment atom count {n_from_roles} "
            f"(fixed={len(roles.fixed_idx)}+master={len(roles.master_idx)}+"
            f"slave={len(roles.slave_idx)}) != this structure's frac natoms {natoms} "
            f"({membership.cnid} k={membership.k} seed={membership.seed})"
        )
    x_master = frac[roles.master_idx, 0]
    off_band_master = roles.master_idx[np.abs(x_master - 0.5) >= BAND_HALF]
    if len(off_band_master) > 0:
        raise SymmetricTiltGeometryError(
            f"inherited MASTER atoms {off_band_master.tolist()} have |x-0.5| >= {BAND_HALF} at "
            f"this structure's coordinates (outside the MIC-safe role-inheritance band); "
            f"{membership.cnid} k={membership.k} seed={membership.seed}"
        )
    x_slave = frac[roles.slave_idx, 0]
    off_band_slave = roles.slave_idx[np.abs(x_slave - np.round(x_slave)) >= BAND_HALF]
    if len(off_band_slave) > 0:
        raise SymmetricTiltGeometryError(
            f"inherited SLAVE atoms {off_band_slave.tolist()} have |x-round(x)| >= {BAND_HALF} "
            f"at this structure's coordinates (outside the MIC-safe role-inheritance band); "
            f"{membership.cnid} k={membership.k} seed={membership.seed}"
        )


def _validate_species_frac(
    species_frac: np.ndarray,
    frac: np.ndarray,
    membership: BulkMembershipRow,
    master_order: np.ndarray,
    slave_order: np.ndarray,
    *,
    check_master_slave: bool = True,
) -> np.ndarray:
    """Validate a per-POSCAR-row chemical species array before it is used to populate
    :class:`CanonicalLayoutResult`'s ``atomic_numbers`` (workspace design doc
    68_layout_species_passthrough.md SS3 step 3). Raises :class:`LayoutError` for any violation
    (silent fallback is prohibited: an unexpected species is never silently coerced to
    ``HOST_Z``). Returns ``species_frac`` as an ``int64`` array (the validated/normalized form the
    caller should use for the MASTER/SLAVE assignment).

    Checks (steps 3 a-d; implementation of the SLAVE mirror-consistency check at the layout level.
    : host/solute species is generalized, see ``ALLOWED_REAL_SPECIES`` above):
      a. ``len(species_frac) == len(frac)`` -- one entry per POSCAR row.
      b. ``set(np.unique(species_frac)) <= ALLOWED_REAL_SPECIES`` -- only host/solute (real-atom
         species this module currently supports; padding's ``PAD_Z_TILT`` is a padding token,
         never present in ``species_frac``, which is POSCAR-row-indexed).
      c. every FIXED_A/FIXED_B POSCAR row (``membership.bulkA_indices``/``bulkB_indices``) has
         species ``HOST_Z`` -- frozen-bulk regulation ("rules to be kept": solute may
         only decorate MASTER/SLAVE real-atom slots, never the frozen bulk).
      d. MASTER<->SLAVE mirror pairing: for every canonical ``(master_order[i], slave_order[i])``
         POSCAR-row pair, the species must match -- a solute decorates both mirror-image atoms of
         a MASTER/SLAVE pair together, never just one side.

    L-a: ``check_master_slave`` (default ``True`` = unchanged behavior) skips check **d
    only** when ``False``; checks a-c always run. A caller that checks the MASTER<->SLAVE species on
    its own correspondence (the variable_length training loader, which orders its output by the op_A
    correspondence, not by this module's per-sub-block ``slave_order``) passes ``False``.
    """
    if len(species_frac) != len(frac):
        raise LayoutError(
            f"species_frac length {len(species_frac)} != frac length {len(frac)}"
        )
    species_frac = np.asarray(species_frac, dtype=np.int64)

    unique_z = set(np.unique(species_frac).tolist())
    disallowed = unique_z - ALLOWED_REAL_SPECIES
    if disallowed:
        raise LayoutError(
            f"species_frac contains disallowed atomic number(s) {sorted(disallowed)}; allowed "
            f"real species are {sorted(ALLOWED_REAL_SPECIES)}"
        )

    fixed_idx = np.concatenate([membership.bulkA_indices, membership.bulkB_indices])
    fixed_species = species_frac[fixed_idx]
    bad_fixed_mask = fixed_species != HOST_Z
    if np.any(bad_fixed_mask):
        bad_rows = fixed_idx[bad_fixed_mask]
        raise LayoutError(
            f"species_frac has non-HOST_Z species at FIXED_A/FIXED_B POSCAR row(s) "
            f"{bad_rows.tolist()} (species {fixed_species[bad_fixed_mask].tolist()}) -- "
            f"the frozen bulk must remain pure host species (Z={HOST_Z})"
        )

    if check_master_slave:
        master_species = species_frac[master_order]
        slave_species = species_frac[slave_order]
        mismatch_mask = master_species != slave_species
        if np.any(mismatch_mask):
            bad_master_rows = master_order[mismatch_mask]
            bad_slave_rows = slave_order[mismatch_mask]
            raise LayoutError(
                "species_frac MASTER<->SLAVE mirror mismatch at POSCAR row pair(s) "
                f"{list(zip(bad_master_rows.tolist(), bad_slave_rows.tolist()))}: MASTER species "
                f"{master_species[mismatch_mask].tolist()} != SLAVE species "
                f"{slave_species[mismatch_mask].tolist} "
            )

    return species_frac


def build_canonical_layout(
    *,
    frac: np.ndarray,
    cell: np.ndarray,
    membership: BulkMembershipRow,
    manifest: TiltManifestLite,
    rng: np.random.Generator,
    roles: RoleAssignment | None = None,
    species_frac: np.ndarray | None = None,
    check_master_slave_species: bool = True,
) -> CanonicalLayoutResult:
    """Build the canonical [FIXED_A|FIXED_B|MASTER|SLAVE] layout for one structure (
    SS3.2/SS4 M-t3a).

    Args:
        frac: ``(natoms, 3)`` fractional coordinates (this structure's own POSCAR_relaxed, wrapped
            to ``[0,1)``).
        cell: ``(3,3)`` cell matrix (this structure's own, or the manifest's -- they are identical
            across all 410 P1 structures; caller decides which to pass).
        membership: this structure's ``bulk_membership.csv`` row.
        manifest: this structure's cnid's tilt template manifest (for ``gb_t_effective_frac_yz``;
            not used for MASTER<->SLAVE correspondence, see module docstring).
        rng: numpy random Generator for the padding-slot ``U(0,1)^3`` draws (padding-slot
            convention of the removed fixed-length layout).
        roles: optional pre-computed :class:`RoleAssignment` to reuse instead of deriving one from
            ``frac``/``membership.frozen_indices`` via ``assign_roles`` (periodic-boundary-seam
            fix, case C: an internal review). ``None`` (the default) reproduces the exact prior behavior -- calls
            ``assign_roles`` on this structure's own coordinates, unchanged -- so the training data
            path (``mattergen.symmetry_constrained.dataset.TiltGBDataset.__getitem__``, which never passes this
            argument) is bit-for-bit unaffected by this parameter's existence. When given, the
            caller (the dx-sweep generation template bank, ``mattergen.scripts.generate_symmetry_constrained.
            build_template_bank_graph``) is expected to have derived ``roles`` once from the SAME
            reference (dx=0) template's coordinates and to pass that SAME object for every dx
            level, including dx=0.0 -- this sidesteps the ``np.mod``-vs-ase-``%%=``-twice periodic
            seam-representation mismatch (dx!=0 shifted coordinates can, purely as a floating-point
            representation artifact, place a continuity-seam atom at the *opposite* raw-x boundary
            representation from the reference, tripping ``assign_roles``' fixed x-bands even though
              the atom has not moved). ``fit_projection`` is still re-run every call on THIS
            structure's own (possibly dx-shifted) coordinates (constraint 5: the MASTER<->SLAVE
            correspondence table and additive translation ``b0`` are geometric quantities computed
            from where the atoms actually are, not inherited).
        species_frac: optional ``(natoms,)`` per-POSCAR-row atomic number array, same row order as
            ``frac``.
            ``None`` (the default) reproduces the exact prior behavior -- every real (non-padding)
            slot gets ``HOST_Z`` -- so existing callers that never pass this argument (all pre-
            existing tests, and any call site not yet migrated) are bit-for-bit unaffected. When
            given, it is validated (see :func:`_validate_species_frac`: length match, allowed
            species only, frozen bulk must stay pure host species, MASTER<->SLAVE mirror pairing)
            BEFORE it is used to populate MASTER/SLAVE ``atomic_numbers``; FIXED_A/FIXED_B and
            padding slots are unaffected by this argument (, always
            ``HOST_Z``/``PAD_Z_TILT``).
        check_master_slave_species: ``True`` (default) = the MASTER<->SLAVE
            species check d of :func:`_validate_species_frac` runs, i.e. unchanged behavior for
            every existing caller. ``False`` skips check d **only** (a-c still run). Only
            meaningful when ``species_frac`` is given; with ``species_frac is None`` no species
            validation happens at all, as before. Used by the variable_length training loader, which
            checks the species on its own op_A correspondence.

    Raises:
        LayoutError: FIXED_A/FIXED_B slot count mismatch, MASTER/SLAVE count mismatch or
            asymmetry (removal must remove a MASTER/SLAVE pair edge case 7),
            MASTER count exceeding ``N_MASTER_MAX`` (edge case handled by Gate a4), a
            role/projection geometry error from ``manifest.py``, an inherited-role sanity-guard
            violation (:func:`_validate_inherited_roles`, when ``roles`` is given), or a
            ``species_frac`` validation failure (:func:`_validate_species_frac`, when
            ``species_frac`` is given).
    """
    natoms = frac.shape[0]
    if membership.natoms != natoms:
        raise LayoutError(
            f"{membership.cnid} k={membership.k} seed={membership.seed}: bulk_membership natoms "
            f"{membership.natoms} != POSCAR natoms {natoms}"
        )
    if len(membership.bulkA_indices) != N_FIXED_A or len(membership.bulkB_indices) != N_FIXED_B:
        raise LayoutError(
            f"{membership.cnid} k={membership.k} seed={membership.seed}: bulkA/bulkB slot counts "
            f"{len(membership.bulkA_indices)}/{len(membership.bulkB_indices)} != "
            f"{N_FIXED_A}/{N_FIXED_B}"
        )

    try:
        if roles is None:
            roles = assign_roles(frac, cell, membership.frozen_indices)
        else:
            _validate_inherited_roles(frac, roles, membership)
        recon = fit_projection(frac, cell, roles)
    except SymmetricTiltGeometryError as e:
        raise LayoutError(
            f"{membership.cnid} k={membership.k} seed={membership.seed}: role/projection fit "
            f"failed: {e}"
        ) from e

    n_master = len(roles.master_idx)
    n_slave = len(roles.slave_idx)
    if n_master != n_slave:
        raise LayoutError(
            f"{membership.cnid} k={membership.k} seed={membership.seed}: MASTER/SLAVE count "
            f"mismatch ({n_master} vs {n_slave}) -- removal must remove a symmetric pair "
            " "
        )
    if n_master > N_MASTER_SLOTS:
        raise LayoutError(
            f"{membership.cnid} k={membership.k} seed={membership.seed}: MASTER count {n_master} "
            f"exceeds N_MASTER_MAX={N_MASTER_SLOTS}"
        )

    master_order, slave_order = _master_slave_order(roles, recon)
    effective_gb_t_yz, effective_gb_t_subblock_diff = _effective_gb_t(recon)
    n_pad = N_MASTER_SLOTS - n_master

    # species_frac validation: run BEFORE any atomic_numbers
    # assignment below, so a violation halts loudly (LayoutError) instead of the caller
    # observing a partially-built CanonicalLayoutResult.
    if species_frac is not None:
        species_frac = _validate_species_frac(
            species_frac, frac, membership, master_order, slave_order,
            check_master_slave=check_master_slave_species,
        )

    pos = np.zeros((N_PAD, 3), dtype=np.float64)
    atomic_numbers = np.full((N_PAD,), HOST_Z, dtype=np.int64)
    gb_role = np.zeros((N_PAD,), dtype=np.int64)
    is_real = np.ones((N_PAD,), dtype=bool)

    # FIXED_A / FIXED_B: slot order = bulk_membership's own
    # bulkA_indices / bulkB_indices order (already cnid/k-consistent, module docstring).
    pos[FIXED_A_SLICE] = frac[membership.bulkA_indices]
    pos[FIXED_B_SLICE] = frac[membership.bulkB_indices]
    atomic_numbers[FIXED_A_SLICE] = HOST_Z
    atomic_numbers[FIXED_B_SLICE] = HOST_Z
    gb_role[FIXED_A_SLICE] = int(GBRole.FIXED)
    gb_role[FIXED_B_SLICE] = int(GBRole.FIXED)

    # MASTER: real atoms first (sub_lo -> sub_hi order), padding padding at
    # the tail.
    master_start = MASTER_SLICE.start
    pos[master_start : master_start + n_master] = frac[master_order]
    if species_frac is not None:
        atomic_numbers[master_start : master_start + n_master] = species_frac[master_order]
    else:
        atomic_numbers[master_start : master_start + n_master] = HOST_Z
    gb_role[MASTER_SLICE] = int(GBRole.MASTER)
    pad_slice = slice(master_start + n_master, MASTER_SLICE.stop)
    if n_pad > 0:
        pos[pad_slice] = rng.random((n_pad, 3))
        atomic_numbers[pad_slice] = PAD_Z_TILT
        is_real[pad_slice] = False

    # SLAVE: position mirrors MASTER's real/padding
    # state ("padding slots copied identically"); *coordinates* here are the structure's own
    # relaxed SLAVE atoms for the real slots (used as the training TARGET the projection loss is
    # scored against) -- the deterministic reconstruction formula itself
    # lives in projection.py (M-t3b), not here.
    slave_start = SLAVE_SLICE.start
    pos[slave_start : slave_start + n_master] = frac[slave_order]
    if species_frac is not None:
        atomic_numbers[slave_start : slave_start + n_master] = species_frac[slave_order]
    else:
        atomic_numbers[slave_start : slave_start + n_master] = HOST_Z
    gb_role[SLAVE_SLICE] = int(GBRole.SLAVE)
    if n_pad > 0:
        slave_pad_slice = slice(slave_start + n_master, SLAVE_SLICE.stop)
        # Padding slot COPY: reuse
        # the exact same draws so the (as-yet-unwired) projection.py invariant "SLAVE padding
        # state == MASTER padding state" already holds at the data layer, not just at generation
        # time.
        pos[slave_pad_slice] = pos[pad_slice]
        atomic_numbers[slave_pad_slice] = PAD_Z_TILT
        is_real[slave_pad_slice] = False

    return CanonicalLayoutResult(
        pos=pos,
        atomic_numbers=atomic_numbers,
        gb_role=gb_role,
        is_real=is_real,
        n_real_master=n_master,
        fit_residual_A=recon.fit_residual_A,
        effective_gb_t_yz=effective_gb_t_yz,
        effective_gb_t_subblock_diff=effective_gb_t_subblock_diff,
        source_master_local_idx=master_order,
        source_slave_local_idx=slave_order,
        source_fixed_a_local_idx=membership.bulkA_indices.copy(),
        source_fixed_b_local_idx=membership.bulkB_indices.copy(),
    )


def read_poscar_frac_species(poscar_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a VASP POSCAR/CONTCAR-format file, returning ``(frac, cell, species)`` -- the
    chemistry-aware superset of :func:`read_poscar_frac` (workspace design doc
    68_layout_species_passthrough.md SS3 step 2). ``species`` is the per-row atomic number
    (``ase``'s ``get_atomic_numbers()``, ``int64``, same row order/length as ``frac``), consumed
    by :func:`build_canonical_layout`'s ``species_frac`` argument to pass solute chemistry
    through the canonical layout instead of hardcoding ``HOST_Z`` for every real
    slot.
    """
    atoms = ase_read(str(poscar_path), format="vasp")
    frac = np.array(atoms.get_scaled_positions(wrap=True))
    cell = np.array(atoms.get_cell())
    species = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    return frac, cell, species




