#!/usr/bin/env python3
"""
config.py -- common settings of the data production (Al STGB segregation dataset).

Plan: see the experiment description.

The initial phase (experiments/bulk_reference/) only freezes and references its products.
This module has constants for the data production only and does not import the config.py of the initial phase
(to avoid confusing it with the 6-boundary set of the initial phase). However, E_bulk/atom, the MACE settings and the
resolve() logic agree numerically with the initial phase.

Path policy:
  - Release version: there are no path constants (the root of inputs and outputs is SB_WORK of al_common). The MACE model is read with mace_model_path() from the environment variable MACE_MODEL_PATH.

Each data-production script imports this module and shares the constants.
"""
from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# paths (differences of the file system are absorbed by resolve(); no local absolute paths are hard-coded)
# ---------------------------------------------------------------------------
# (Release version: the absolute-path constants of internal run directories, Al.cif and interface_master were removed. Al.cif is in structure_builder/inputs/Al.cif,
#  and interface_master is in structure_builder/vendor/interfacemaster)
# mirror constructor ([110] systems): scripts/gb_build/mirror_stgb.py (resolved by the same self-location as this module)

# ---------------------------------------------------------------------------
# MACE settings (GPU) -- the same as Phase 0
# ---------------------------------------------------------------------------
def mace_model_path() -> str:
    """Path of the MACE-MH-1 model file (environment variable MACE_MODEL_PATH, required, no default).
    Read when called (the relaxation stage), not at import time, so the CPU-only stages do not need it."""
    return os.environ["MACE_MODEL_PATH"]
MACE_HEAD  = "omat_pbe"
FMAX       = 0.02
MAX_STEPS  = 500

EV_PER_A2_TO_J_PER_M2 = 16.0218   # eV/Å² → J/m²
E_BULK_PER_ATOM = -3.7569042674314765   # bulk/energy.json of the initial phase (to be reused)

# ---------------------------------------------------------------------------
# the 16 target STGB (sigma <= 33)
#
# 2 kinds of constructors:
#   builder="im"     : interface_master (8 boundaries of the [001] system). Method verified in the initial phase.
#                      pure relaxation = atomic positions + strain along the a axis (GB normal) (mode=pure_im).
#   builder="mirror" : mirror constructor mirror_stgb.py (8 boundaries of the [110] system). A truly symmetric STGB.
#                      pure relaxation = positions only (mode=pure_mirror). GB normal = c axis.
#
# fields of the im system: (name, axis, angle_deg, plane, sigma, a_mult)
#   a_mult: extension factor along the GB normal (a axis). The smallest value that gives an a-axis length >= 48 A (probed).
#           For the 3 boundaries common to the initial phase, a_mult of its config is followed strictly.
# fields of the mirror system: (name, hkl, sigma, misori_deg, merge_tol)
#   merge_tol: threshold for merging close atoms on the mirror plane (sigma3(111)=0.6, exact duplicates only; others=2.0 = about 0.7 d_NN).
#              Same criterion as gen_mirror_grid.py of the diagnostic.
# ---------------------------------------------------------------------------
BOUNDARIES_IM = [
    # name          axis       angle    plane      sigma a_mult
    ("Sigma5_210",  [0, 0, 1], 53.130,  [2, 1, 0],  5,  3),   # follows the initial phase (120 atoms)
    ("Sigma5_310",  [0, 0, 1], 36.870,  [3, 1, 0],  5,  4),   # follows the initial phase (80 atoms)
    ("Sigma13_510", [0, 0, 1], 22.620,  [5, 1, 0], 13,  3),   # follows the initial phase (120 atoms)
    ("Sigma13_320", [0, 0, 1], 67.380,  [3, 2, 0], 13,  2),   # probe: N=208 a=58.2Å
    ("Sigma17_410", [0, 0, 1], 28.072,  [4, 1, 0], 17,  2),   # probe: N=272 a=66.6Å
    ("Sigma17_530", [0, 0, 1], 61.928,  [5, 3, 0], 17,  3),   # probe: N=204 a=70.6Å
    ("Sigma25_430", [0, 0, 1], 73.740,  [4, 3, 0], 25,  2),   # probe: N=400 a=80.7Å
    ("Sigma29_520", [0, 0, 1], 43.603,  [5, 2, 0], 29,  2),   # probe: N=464 a=86.9Å
]

BOUNDARIES_MIRROR = [
    # name          hkl        sigma  misori   merge_tol
    ("Sigma3_111",  (1, 1, 1),  3,  70.529, 0.6),   # coherent twin (low gamma)
    ("Sigma3_112",  (1, 1, 2),  3, 109.471, 2.0),
    ("Sigma9_221",  (2, 2, 1),  9,  38.942, 2.0),
    ("Sigma11_113", (1, 1, 3), 11,  50.479, 2.0),
    ("Sigma11_332", (3, 3, 2), 11, 129.521, 2.0),
    ("Sigma19_331", (3, 3, 1), 19,  26.525, 2.0),
    ("Sigma27_115", (1, 1, 5), 27, 148.412, 2.0),   # the in_plane_basis guard is required (see the note below)
    ("Sigma33_441", (4, 4, 1), 33,  20.050, 2.0),
]

# all boundary names (before exclusion)
ALL_BOUNDARY_NAMES = [b[0] for b in BOUNDARIES_IM] + [b[0] for b in BOUNDARIES_MIRROR]

# upper limit of the number of atoms (above 1200 is excluded and recorded in the exclusion list).
MAX_NATOMS = 1200

# thickness of one grain (mirror system) [A]. The whole cell has c about 2*thickness. Requirement "grain thickness >= 12 A".
# Author's choice: thickness is quantized to the number of atomic layers. With 13.0, sigma3(111) gives 11.65 A per grain (< 12 A, a deviation),
#   so 14.0 is set (sigma3(111) -> 13.98 A per grain, and the other boundaries also keep >= 12 A). The diagnostic uses 13.0, but
#   W1 is production data and satisfies the specification strictly. The structures are still light with 12-90 atoms.
MIRROR_THICKNESS = 14.0
# lower limit of the target GB-normal (a axis) length (im system). Satisfies grain thickness >= 12 A x 2 periodic GBs (same level as the initial phase, >= 48 A).
IM_ALEN_MIN = 48.0

# ---------------------------------------------------------------------------
# sampling of the pure GB microscopic states
# ---------------------------------------------------------------------------
CNID_GRID = [4, 4]              # im system: CNID in-plane 4x4
MIRROR_RBT_N = 4               # mirror system: RBT in-plane 4x4
MIRROR_GAPS = [-0.2, 0.0, 0.4]  # mirror system: 3 gap points (same as the diagnostic)

# threshold of the number of atoms above which the geometric dedup (StructureMatcher) is skipped to avoid O(n^2)
# (dedup may be skipped above 400 atoms).
# Author's choice: for the large [001] cells with 200-400 atoms (sigma13_320/sigma17/sigma25/sigma29), group_structures of StructureMatcher
#   is impractically slow (confirmed by a probe), so the threshold is lowered to 150 and only the 3 small-cell boundaries
#   (sigma5_310=80, sigma5_210/sigma13_510=120) are deduplicated. The large cells keep all CNID points (an increase of data is
#   rather an advantage; dedup is optional). gamma subtracts the real N, so redundant points do not bias it.
DEDUP_MAX_NATOMS = 150

# number K of low-gamma microscopic states selected per boundary (3 points to secure diversity).
TOP_K_MICRO = 3


def resolve(p: str) -> str:
    """Public release: no NFS-mount translation; the path is returned unchanged."""
    return p


def to_remote(p: str) -> str:
    """Public release: no NFS-mount translation; the path is returned unchanged."""
    return str(p)
