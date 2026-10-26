#!/usr/bin/env python3
"""Common definitions of the data rebuild (shared by stages A, B and the checks). Runs on compute nodes. Paths are written with the <CLUSTER_MOUNT> notation.

Note: dx_lib.py adds BULKFREEZE_SCRIPTS to sys.path in the <CLUSTER_MOUNT> notation, so on a compute node (no /mnt)
build_bulk_membership cannot be imported. Therefore the path in the <CLUSTER_MOUNT> notation is
added to sys.path before dx_lib is imported (dx_lib itself is not changed).
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

GC = Path('<ORIGINAL_ROOT>/workspace')
RUN = GC / "runs" / "data_rebuild"
GS = Path('<ORIGINAL_ROOT>/tmp/cnid_grid_sweep')
GROUPS_CSV = GS / "unified_10x10" / "groups_10x10_unified.csv"
STRUCT_G10 = GS / "structures" / "g10"

RESULTS = RUN / "results"
STRUCTURES = RUN / "structures"
VERIFICATION = RUN / "verification"

for _p in (
    GC / "runs" / "bulk_membership" / "scripts",   # build_bulk_membership, sym_common
    GC / "runs" / "dx_sweep" / "scripts",      # dx_lib
    GC / "runs" / "atom_removal" / "scripts",         # removal_builder, sym_common
):
    sys.path.insert(0, str(_p))

# section 2-2(2) of the plan
DX10 = [-0.75, -0.60, -0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45, 0.60]
A_BASE = 54.15601152562611          # criterion a of the acceptance condition on the data rebuild
N_ATOMS = 120
N_BULK = 38
N_FROZEN = 76
K_SUCCESS = {0: 1, 1: 5, 2: 5, 3: 5, 4: 5}   # number of structures per template and dx (21 in total)
MAX_A = 20                                    # trial number a = 0..19
N_TEMPLATES = 31
STRUCT_PER_TEMPLATE_DX = sum(K_SUCCESS.values())   # 21
CELL_TOL = 1e-9


def seed_of(dx_idx: int, k: int, a: int) -> int:
    """§2-5: seed = 10000 + 1000*dx_idx + 100*k + a"""
    return 10000 + 1000 * dx_idx + 100 * k + a


def dx_tag(dx: float) -> str:
    sign = "p" if dx >= 0 else "m"
    return f"dx_{sign}{abs(dx):.3f}"


def struct_dir(cnid: str, dx: float, k: int, seed: int, root: Path = STRUCTURES) -> Path:
    return Path(root) / cnid / dx_tag(dx) / f"k{k}_s{seed}"


def read_template_map(results_dir: Path = RESULTS) -> list[dict]:
    with open(Path(results_dir) / "template_map.csv", newline="") as f:
        return list(csv.DictReader(f))


def select_cnids(all_cnids: list[str], spec: str | None) -> list[str]:
    """If spec is empty, all. If given as a comma-separated list, all of them are required to be contained in all_cnids."""
    if not spec:
        return list(all_cnids)
    want = [s for s in spec.split(",") if s]
    unknown = [s for s in want if s not in all_cnids]
    if unknown:
        raise ValueError(f"unknown cnids: {unknown}")
    return [c for c in all_cnids if c in want]
