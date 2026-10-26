#!/usr/bin/env python3
"""Common definitions (shared by stages A and B and the checks). Release version: the root of inputs and outputs is the environment variable SB_WORK (required. No default value. KeyError if unset).

Under SB_WORK are placed: stage0/structures/g10 (output of stage 0), stage1/groups_10x10_unified.csv (output of stage 1),
results/ (results of stages A to E), structures/ (structures), verification/.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

SB_WORK = Path(os.environ["SB_WORK"])   # required, no default (KeyError if unset)
GS = SB_WORK / "stage0"
GROUPS_CSV = SB_WORK / "stage1" / "groups_10x10_unified.csv"
STRUCT_G10 = GS / "structures" / "g10"

RESULTS = SB_WORK / "results"
STRUCTURES = SB_WORK / "structures"
VERIFICATION = SB_WORK / "verification"

# the sibling modules (dx_lib, removal_builder, sym_common, build_bulk_membership, config, runner_guards) live in this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
