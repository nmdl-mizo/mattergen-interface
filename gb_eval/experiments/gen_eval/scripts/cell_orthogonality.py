#!/usr/bin/env python3
"""cell_orthogonality.py -- decision function of non-orthogonality (defined in one place and shared).

Decision: non-orthogonal if any of alpha, beta, gamma of Cell.cellpar() of ase deviates from 90 deg by **more than** ANGLE_TOL_DEG (0.01 deg).
  (The component decision |cell[0,1]|,|cell[0,2]| > 1e-6 of r1 is always False for cells read from a CIF, because the a axis always lies on the x axis.
   0.01 deg is sufficiently larger than the rounding by the printed digits of the angles in the CIF and sufficiently smaller than a breakdown of the structure.)
Users: check_chunk0.py (check 0) and compute_gamma_pre_proposed.py (error=nonorthogonal).
Dependency: ase (evaluation environment). Not executed on the login node.
"""
from __future__ import annotations

import numpy as np

ANGLE_TOL_DEG = 0.01


def cell_angles_deg(cell_or_atoms) -> tuple[float, float, float]:
    """(alpha, beta, gamma) [deg]. The argument is any of ase Atoms, ase Cell and a 3x3 array. Uses Cell.cellpar() of ase."""
    from ase.cell import Cell

    cell = getattr(cell_or_atoms, "cell", cell_or_atoms)
    cp = Cell(np.asarray(cell, dtype=float)).cellpar()
    return float(cp[3]), float(cp[4]), float(cp[5])


def max_angle_dev_deg(cell_or_atoms) -> float:
    return max(abs(a - 90.0) for a in cell_angles_deg(cell_or_atoms))


def is_nonorthogonal(cell_or_atoms) -> bool:
    """True if any of alpha, beta, gamma deviates from 90 deg by more than ANGLE_TOL_DEG (exactly 0.01 deg is orthogonal)."""
    return max_angle_dev_deg(cell_or_atoms) > ANGLE_TOL_DEG
