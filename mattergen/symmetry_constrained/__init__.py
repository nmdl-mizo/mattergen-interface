"""Tilt-STGB (Sigma5(210)[001]) template manifest utilities.

Tilt grain-boundary template manifest utilities: the tilt role layout (two cross-mapped MASTER/SLAVE
sub-blocks, atom-count-based FIXED/bulk membership) is kept in its own package.
"""

from mattergen.symmetry_constrained.constants import N_BULK_PER_GRAIN, N_MASTER_MAX, GBRole
from mattergen.symmetry_constrained.manifest import (
    SymmetricTiltGeometryError,
    build_tilt_template_manifest,
)

__all__ = [
    "GBRole",
    "N_BULK_PER_GRAIN",
    "N_MASTER_MAX",
    "SymmetricTiltGeometryError",
    "build_tilt_template_manifest",
]
