#!/usr/bin/env python3
"""Common parts of stages D1 and D2 of the data rebuild (generation of matching candidates, grouping, reading and writing of the CSV). Runs on compute nodes.

Group = same dx and same natoms (if k is the same, natoms is also the same). Only rows with converged=True are matched.
r2: the matching candidates are by default all pairs within a group (i<j; no energy window). Only when --window-ev W (or the environment variable DATA_WINDOW_EV)
is given explicitly are they narrowed to pairs with |Delta E| <= W (Delta E is the difference of energy_eV) (behavior of r1; reproduced with W=0.05).
The number of pairs per task of D2 is --pairs-per-task (or the environment variable DATA_D2_PAIRS_PER_TASK. Default PAIRS_PER_TASK).
The results directory is switched with --results-dir or the environment variable DATA_RESULTS_DIR (the default is the production RUN/results).
"""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "lib"))  # structure_builder/lib
import builder_common as C

W_EV = None                       # r2 default: no energy window (all pairs). A window is used only when --window-ev is given explicitly
PAIRS_PER_TASK = 1000             # default number of pairs per task of D2 (0.3 s/pair x 1000 = about 5 min. Production about 483 tasks <= MaxArraySize 30001)
LIMIT_C4_HOURS = 100.0         # check 4: estimate <= 100 CPU hours (without a window, estimated with P_all)
LIMIT_C5_EV = 0.01             # check 5: Delta E_max <= 0.01 eV (in r2 without a window it is not decided but "reference". Decided only with a window)
CONV_MIN = 0.98                   # C3
PILOT_K = 2
PILOT_DX = 0.0
GAMMA_ALL = "gamma_all.csv"
GAMMA_DEDUP = "gamma_dedup.csv"
COLS = ["cnid", "k", "seed", "dir", "converged", "natoms", "energy_eV", "area_A2", "gamma_Jm2", "dx_A"]
UNI = dict(stol=0.025, primitive_cell=False, scale=False, attempt_supercell=False)   # ltol and angle_tol are the defaults


def results_dir(arg=None) -> Path:
    return Path(arg or os.environ.get("DATA_RESULTS_DIR") or C.RESULTS)


def add_window_arg(ap):
    ap.add_argument("--window-ev", type=float, default=None,
                    help="narrow the candidates with the window |Delta E|<=W only when given (default: no window = all pairs). Also possible with the environment variable DATA_WINDOW_EV")


def window_ev(arg=None):
    """Window width (eV). None = no window (all pairs). Argument > environment variable DATA_WINDOW_EV (no window if empty) > default W_EV(None)."""
    if arg is not None:
        return float(arg)
    env = os.environ.get("DATA_WINDOW_EV", "").strip()
    return float(env) if env else W_EV


def pairs_per_task(arg=None) -> int:
    n = int(arg) if arg is not None else int(os.environ.get("DATA_D2_PAIRS_PER_TASK") or PAIRS_PER_TASK)
    if n < 1:
        raise ValueError(f"pairs-per-task must be >= 1: {n}")
    return n


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_gamma(path: Path) -> list[dict]:
    """Read with types. row = 0-based row position in the CSV (rows with converged=False are also counted)."""
    out = []
    with open(path, newline="") as f:
        for n, r in enumerate(csv.DictReader(f)):
            out.append({
                "row": n, "cnid": r["cnid"], "k": int(r["k"]), "seed": int(r["seed"]), "dir": r["dir"],
                "converged": r["converged"] == "True", "natoms": int(r["natoms"]),
                "energy_eV": float(r["energy_eV"]), "area_A2": float(r["area_A2"]),
                "gamma_Jm2": float(r["gamma_Jm2"]), "dx_A": float(r["dx_A"]), "raw": r,
            })
    return out


def group_key(r: dict) -> str:
    return f"{C.dx_tag(r['dx_A'])}_n{r['natoms']}"


def pilot_group_key() -> str:
    return f"{C.dx_tag(PILOT_DX)}_n{C.N_ATOMS - 2 * PILOT_K}"


def groups_of(rows: list[dict]) -> dict[str, list[dict]]:
    """Rows with converged=True by group (ascending key; within a group ascending row position)."""
    g: dict[str, list[dict]] = {}
    for r in rows:
        if r["converged"]:
            g.setdefault(group_key(r), []).append(r)
    return {k: sorted(g[k], key=lambda r: r["row"]) for k in sorted(g)}


def all_pairs(grows: list[dict]) -> list[tuple[int, int]]:
    rs = [r["row"] for r in grows]
    return [(rs[a], rs[b]) for a in range(len(rs)) for b in range(a + 1, len(rs))]


def windowed_pairs(grows: list[dict], w=None) -> list[tuple[int, int]]:
    """Candidate pairs (row_i<row_j) (ascending row position). w=None (default) = all pairs within a group.
    When w is a number, sorts energy_eV in ascending order and returns only the pairs with |Delta E| <= w."""
    if w is None:
        return all_pairs(grows)
    s = sorted(grows, key=lambda r: (r["energy_eV"], r["row"]))
    out = []
    for a in range(len(s)):
        for b in range(a + 1, len(s)):
            if s[b]["energy_eV"] - s[a]["energy_eV"] > w:
                break
            out.append((min(s[a]["row"], s[b]["row"]), max(s[a]["row"], s[b]["row"])))
    return sorted(out)


def d2_plan(rows: list[dict], reuse_pilot: bool, w=None):
    """Targets of the matching tasks of D2 (w=None is no window = all pairs). With reuse_pilot=True the check group is excluded (replaced by the result of the all-pair matching of D1).
    Returns: (pairs=[(group_key,row_i,row_j)], group_sizes, pilot_pairs_all(= all pairs of the check group))"""
    g = groups_of(rows)
    pairs, sizes = [], {k: len(v) for k, v in g.items()}
    pk = pilot_group_key()
    for k, grows in g.items():
        if reuse_pilot and k == pk:
            continue
        pairs.extend((k, i, j) for (i, j) in windowed_pairs(grows, w))
    pilot_all = [(pk, i, j) for (i, j) in all_pairs(g[pk])] if (reuse_pilot and pk in g) else []
    return pairs, sizes, pilot_all


def load_relaxed(r: dict):
    from pymatgen.core.structure import Structure
    return Structure.from_file(str(Path(r["dir"]) / "POSCAR_relaxed"))


def read_chunks(d: Path) -> list[dict]:
    rows = []
    for p in sorted(d.glob("chunk_*.csv")):
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows
