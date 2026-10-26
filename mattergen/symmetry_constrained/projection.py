"""Projection constants shared by the symmetry-constrained model's generation script and its
variable-length projector.

The fixed-length (128-slot) projector class that used to live here was removed from the public
release: it is not reachable from the training entry, the generation script, the proposed/baseline
configurations, the evaluation code or the structure builder (see NOTICE). Only the two constants
below are still referenced.

Design constants: R = diag(+1,-1,-1) (the SLAVE grain is the mirror image of the MASTER grain);
b0 = (0.5, 0.5, 0.5) (fractional offset of the mirror operation).
"""

from __future__ import annotations

_R_SLAVE_MATRIX = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
_B0_CONST = [0.5, 0.5, 0.5]
