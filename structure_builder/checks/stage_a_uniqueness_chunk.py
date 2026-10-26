#!/usr/bin/env python3
"""Data rebuild, stage A-5: uniqueness check after dx (chunk for an array job). The split follows GS/unified_10x10/unified_chunk.py.

  stage_a_uniqueness_chunk.py plan            # print the number of tasks
  stage_a_uniqueness_chunk.py chunk <task_id> # 1 task matches 20 pairs and writes results/stage_a_uniqueness/chunk_<id>.csv
  stage_a_uniqueness_chunk.py selfcheck       # match cnid_0_0 with dx=0.0 against itself (validation with fit=True)

Matching: within each dx, all pairs of the 31 structures (C(31,2)=465 pairs x 10 dx = 4,650 pairs).
Unified settings StructureMatcher(stol=0.025, primitive_cell=False, scale=False, attempt_supercell=False)
(ltol and angle_tol are the pymatgen defaults). Exceptions are recorded in the error column and fit is left empty (not treated as "non-duplicate").
The result is a separate file per task. If a result file already exists, nothing is done.
"""
from __future__ import annotations

import csv
import itertools
import os
import sys
import time
from importlib.metadata import version

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "lib"))  # structure_builder/lib
import builder_common as C

UNI = dict(stol=0.025, primitive_cell=False, scale=False, attempt_supercell=False)
CS = 20
OUT_DIR = C.RESULTS / "stage_a_uniqueness"
COLS = ["dx_A", "dx_idx", "cnid_a", "cnid_b", "fit", "sec", "error"]


def cnid_names(results_dir=C.RESULTS):
    names = [r["cnid"] for r in C.read_template_map(results_dir)]
    if len(names) != C.N_TEMPLATES:
        raise ValueError(f"template_map rows = {len(names)}")
    return names


def all_pairs(names):
    out = []
    for dx_idx in range(len(C.DX10)):
        for a, b in itertools.combinations(range(len(names)), 2):
            out.append((dx_idx, a, b))
    return out


def n_pairs_total():
    n = C.N_TEMPLATES
    return len(C.DX10) * n * (n - 1) // 2


def n_tasks():
    return (n_pairs_total() + CS - 1) // CS


def load_struct(cnid, dx_idx):
    from pymatgen.core.structure import Structure
    dx = C.DX10[dx_idx]
    p = C.struct_dir(cnid, dx, 0, C.seed_of(dx_idx, 0, 0)) / "POSCAR"
    return Structure.from_file(str(p))


def chunk(tid):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    names = cnid_names()
    pairs = all_pairs(names)
    assert len(pairs) == n_pairs_total(), (len(pairs), n_pairs_total())
    assert 0 <= tid < n_tasks(), (tid, n_tasks())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fn = OUT_DIR / f"chunk_{tid:04d}.csv"
    if fn.exists():
        print("exists, skip", fn)
        return
    sm0 = StructureMatcher(**UNI)
    print(f"pymatgen {version('pymatgen')}  matcher: stol={sm0.stol} ltol={sm0.ltol} angle_tol={sm0.angle_tol} "
          f"prim={sm0._primitive_cell} scale={sm0._scale} supercell={sm0._supercell}", flush=True)
    cache = {}
    rows = []
    t0 = time.time()
    for (dx_idx, a, b) in pairs[tid * CS:(tid + 1) * CS]:
        t1 = time.time()
        fit, err = "", ""
        try:
            for nm in (names[a], names[b]):
                if (nm, dx_idx) not in cache:
                    cache[(nm, dx_idx)] = load_struct(nm, dx_idx)
            fit = int(bool(StructureMatcher(**UNI).fit(cache[(names[a], dx_idx)], cache[(names[b], dx_idx)])))
        except Exception as e:  # noqa: BLE001 exceptions go to the error column (not regarded as non-duplicate)
            err = repr(e)
        rows.append([repr(C.DX10[dx_idx]), dx_idx, names[a], names[b], fit, f"{time.time() - t1:.3f}", err])
    tmp = fn.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    os.replace(tmp, fn)
    n_fit = sum(1 for r in rows if r[4] == 1)
    n_err = sum(1 for r in rows if r[6])
    print(f"done {fn.name} rows={len(rows)} fits={n_fit} errors={n_err} {time.time() - t0:.0f}s", flush=True)


def selfcheck():
    from pymatgen.analysis.structure_matcher import StructureMatcher
    s = load_struct("cnid_0_0", C.DX10.index(0.0))
    fit = bool(StructureMatcher(**UNI).fit(s, s))
    print(f"selfcheck cnid_0_0 dx=0.0 vs itself: fit={fit}")
    if not fit:
        sys.exit(3)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        print(n_tasks())
    elif cmd == "chunk":
        chunk(int(sys.argv[2]))
    elif cmd == "selfcheck":
        selfcheck()
    else:
        raise SystemExit(f"unknown command {cmd}")
