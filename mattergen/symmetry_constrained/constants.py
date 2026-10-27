"""Shared constants for the tilt-STGB Sigma5(210)[001] manifest/role scheme.

internal notes

PRIMARY-slot expansion (internal notes Stage
A): ``N_BULK_PER_GRAIN`` is the single source of truth for the FIXED/PRIMARY/REPLICA slot counts
(``PRIMARY_ATOMS_K0``/``N_PRIMARY_MAX`` here, ``N_FIXED_A``/``N_FIXED_B`` in ``layout.py``, the
``BAND_HALF`` role-split band in ``manifest.py``). It is overridable via the
``SCM_N_BULK_PER_GRAIN`` environment variable; the default (38, this repo's other users'
existing behavior) is unchanged unless that variable is set (no silent fallback --
the resolved value is always printed to stderr at import time, and an out-of-family value raises
immediately rather than silently falling back).
"""

from __future__ import annotations

import os
import sys
from enum import IntEnum



class GBRole(IntEnum):
    """Mirrors ``mattergen.gb.constraints.GBRole`` (same enum values); kept as an independent
    definition in this package rather than imported, per the symmetry_constrained/gb package-separation
    rationale in ``mattergen.symmetry_constrained.manifest`` module docstring."""

    FIXED = 0
    PRIMARY = 1
    REPLICA = 2


# (4): A_TEMPLATE = 6*C_TEMPLATE spans the bicrystal's full stacking axis and
# is n_bulk-independent; at the FIXED-selection layer spacing (C_TEMPLATE/10, 2 atoms/layer) this
# is exactly 60 atomic layers (both grains combined) -- 60*2 = 120 = the k=0 total atom count.
N_PLANES = 60
ATOMS_PER_PLANE = 2

_N_BULK_PER_GRAIN_ENV_VAR = "SCM_N_BULK_PER_GRAIN"
_N_BULK_PER_GRAIN_DEFAULT = 38
# (2): FIXED is selected in "8 atoms per step" increments (a 2-atom center
# layer + a symmetric 4-atom shell, per grain-pair), so the only geometrically realizable values
# are n_bulk = 2 + 4k for k = 0..9 (i.e. {2,6,10,14,18,22,26,30,34,38}); FIXED=50 (k not integral)
# is not realizable (the setting FIXED=50 is not realizable).
_N_BULK_PER_GRAIN_ALLOWED = frozenset(2 + 4 * k for k in range(10))


def _resolve_n_bulk_per_grain() -> int:
    raw = os.environ.get(_N_BULK_PER_GRAIN_ENV_VAR)
    if raw is None:
        value = _N_BULK_PER_GRAIN_DEFAULT
        source = "default"
    else:
        try:
            value = int(raw)
        except ValueError as e:
            raise ValueError(
                f"{_N_BULK_PER_GRAIN_ENV_VAR}={raw!r} is not an integer"
            ) from e
        source = "env"
    if value not in _N_BULK_PER_GRAIN_ALLOWED:
        raise ValueError(
            f"{_N_BULK_PER_GRAIN_ENV_VAR} resolved to {value} ({source}), which is not one of "
            f"the FIXED-selection-realizable values {sorted(_N_BULK_PER_GRAIN_ALLOWED)} "
            "(n_bulk must be 2+4k for k=0..9 (2)). Refusing to silently fall "
            "back ."
        )
    print(f"[symmetry_constrained] N_BULK_PER_GRAIN={value} ({source})", file=sys.stderr)
    return value


# (2026-08-05 user-confirmed): bulk (FIXED) membership is defined by atom
# count, not layer/distance banding. Default 38 atoms/grain (76 total), frozen at template-build
# time and stored in bulk_membership.csv (workspace, read-only reference; not recomputed at
# runtime). See module docstring for the override mechanism.
N_BULK_PER_GRAIN = _resolve_n_bulk_per_grain()

# (2026-08-05 user-confirmed, reviewer R2-3): N_primary_max = core atom
# count (i.e. the GB2/PRIMARY-side atom count, design doc SS3.2's "max(template GB2 core atom
# count)") + 4 margin atoms (leaves room for future insertion, not just removal).
# Measured PRIMARY-side atom count at k=0 (no removal) for n_bulk=38 across all 10 P1 pilot CNID
# templates: 22 atoms (bulk_membership.csv core=44 total per template, split 22/22 primary/replica).
# A-1: generalized to N_PLANES - N_BULK_PER_GRAIN (n_bulk=38 -> 22 ✓ unchanged;
# n_bulk=26 -> 34).
PRIMARY_ATOMS_K0 = N_PLANES - N_BULK_PER_GRAIN
N_PRIMARY_MARGIN = 4
N_PRIMARY_MAX = PRIMARY_ATOMS_K0 + N_PRIMARY_MARGIN  # n_bulk=38 -> 26 (unchanged); n_bulk=26 -> 38








# internal notes (2026-09-20): the
# host (majority) species and the allowed solute species are read from environment variables
# rather than a per-system enum lookup table, so that adding a new host/solute element
# requires zero code changes (deliberately not a ``SCM_SYSTEM`` enum pattern).
# Same resolution shape as ``_resolve_n_bulk_per_grain`` above (env
# read at import time, default preserved unless the var is set, out-of-family values raise
# immediately -- no silent fallback).
_HOST_Z_ENV_VAR = "SCM_HOST_Z"
_HOST_Z_DEFAULT = 13  # Al (mattergen.symmetry_constrained.layout.AL_Z's earlier hardcoded value)
_SOLUTE_ZS_ENV_VAR = "SCM_SOLUTE_ZS"
_SOLUTE_ZS_DEFAULT = "12"  # atomic number 12 -- reproduces the earlier ALLOWED_REAL_SPECIES = {13, 12} exactly
_RESERVED_Z = frozenset({101, 102})  # PAD_Z_TILT (layout.py), and the MASK token (Z=102)
# -- table, both realized as literals here (not imported) to avoid a reverse import
# edge (this module is imported by layout.py, which owns PAD_Z_TILT; importing torch-side
# modules here would also pull torch into this otherwise torch-free constants module).
_Z_RANGE = range(1, 101)  # 1 <= Z <= 100


def _parse_z_env(raw: str, env_var: str) -> int:
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{env_var}={raw!r} is not an integer") from e


def _resolve_host_z() -> tuple[int, str]:
    raw = os.environ.get(_HOST_Z_ENV_VAR)
    if raw is None:
        return _HOST_Z_DEFAULT, "default"
    return _parse_z_env(raw, _HOST_Z_ENV_VAR), "env"


def _resolve_solute_zs() -> tuple[frozenset[int], str]:
    raw = os.environ.get(_SOLUTE_ZS_ENV_VAR)
    if raw is None:
        source = "default"
        raw = _SOLUTE_ZS_DEFAULT
    else:
        source = "env"
    raw = raw.strip()
    if raw == "":
        #: an explicitly empty string means "no solute" (an empty set) -- this is
        # how a run without a solute (SCM_SOLUTE_ZS=) is expressed, and is a normal, valid value, not an
        # error (does not require erroring on an intentionally-empty selection).
        return frozenset(), source
    zs = []
    for token in raw.split(","):
        token = token.strip()
        zs.append(_parse_z_env(token, _SOLUTE_ZS_ENV_VAR))
    return frozenset(zs), source


def _resolve_host_and_solute() -> tuple[int, frozenset[int]]:
    host_z, host_source = _resolve_host_z()
    solute_zs, solute_source = _resolve_solute_zs()

    #: validation (all ValueError -- no silent fallback).
    for z in [host_z, *sorted(solute_zs)]:
        if z not in _Z_RANGE:
            raise ValueError(
                f"SCM_HOST_Z/SCM_SOLUTE_ZS: atomic number {z} is out of range "
                f"(1 <= Z <= 100 required); host={host_z} ({host_source}), "
                f"solutes={sorted(solute_zs)} ({solute_source})."
            )
    if host_z in solute_zs:
        raise ValueError(
            f"SCM_HOST_Z={host_z} ({host_source}) also appears in "
            f"SCM_SOLUTE_ZS={sorted(solute_zs)} ({solute_source}) -- host and solute species "
            "must be disjoint."
        )
    collision = ({host_z} | solute_zs) & _RESERVED_Z
    if collision:
        raise ValueError(
            f"SCM_HOST_Z/SCM_SOLUTE_ZS collide with reserved atomic number(s) "
            f"{sorted(collision)} ({sorted(_RESERVED_Z)} are reserved for PAD_Z_TILT/"
            f"the MASK token); host={host_z} ({host_source}), solutes={sorted(solute_zs)} "
            f"({solute_source})."
        )

    print(f"[symmetry_constrained] SCM_HOST_Z={host_z} ({host_source})", file=sys.stderr)
    print(f"[symmetry_constrained] SCM_SOLUTE_ZS={sorted(solute_zs)} ({solute_source})", file=sys.stderr)
    return host_z, solute_zs


SCM_HOST_Z: int
SCM_SOLUTE_ZS: "frozenset[int]"
SCM_HOST_Z, SCM_SOLUTE_ZS = _resolve_host_and_solute()
