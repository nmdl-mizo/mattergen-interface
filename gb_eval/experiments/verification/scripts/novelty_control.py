#!/usr/bin/env python3
"""novelty_control.py -- positive and negative controls of the novelty instrument (pymatgen StructureMatcher).

Plan: see the experiment description.
Usage (mace-mh python on a compute node. Not run on the login node):
  --select               : deterministically selects 20 base structures from train_ref_manifest.csv and writes bases.csv (standard library only)
  --base-id <k>          : performs 15 matchings (10 on route H + 5 on route V) for base structure k (0..19) and writes rows_base<kk>.csv and DONE_base<kk>
Aggregation, gate and --self_check are in aggregate_novelty_control.py (standard library only).

The production matching implementation imports and uses the RUN version of common.py (RUN_COMMON below) (not copied or modified).
This directory also has a common.py, but it is a copy with different content. Never do a plain `import common`
(sys.path[0] is this directory, so that one would be read). Load it by specifying the path with importlib.
The matcher is called as K.make_matcher() without override.
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import csv
import importlib.util
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

THIS = Path(__file__).resolve()
VRUN_SCRIPTS = THIS.parent
DEFAULT_OUT_DIR = VRUN_SCRIPTS.parent / "results" / "novelty_control"

_RUN_COMMON_CANDIDATES = [
    _gbrel.CODE + '/gb_eval/experiments/relax_novelty/scripts/common.py',
    _gbrel.CODE + '/gb_eval/experiments/relax_novelty/scripts/common.py',
]
RUN_COMMON = next((p for p in _RUN_COMMON_CANDIDATES if os.path.isfile(p)), _RUN_COMMON_CANDIDATES[0])
_TRAIN_MANIFEST_NAME = "/workspace/experiments/relax_novelty/results/train_ref_manifest.csv"

NATOMS_KINDS = 5
N_BASES = 20
PER_NATOMS = 4
ROWS_PER_BASE = 15
RNG_SEED0 = 20261008
RMS_DEGEN_TOL = 1e-6          # upper limit of rms_to_source for exactly identical transformations (a degeneracy check of the construction)

# order of the matching rows (15 rows per base structure). Random numbers are drawn in the order P1 -> P1u -> P2 -> P3 -> P6 -> P7 -> N1
PATH_H = "H"
PATH_V = "V"
TRANSFORMS_H = ["P0", "P1", "P1u", "P2", "P3", "P4", "P5", "P6", "P7", "N1"]
TRANSFORMS_V = ["P0", "P1", "P2", "P6", "P7"]
DEGEN_H = ["P0", "P1", "P1u", "P2", "P6"]
DEGEN_V = ["P0", "P1", "P2", "P6"]          # route V has no P1u
ROW_COLS = ["base_id", "row_index", "natoms", "transform", "path", "applied", "skip_reason", "novel", "match_row_index",
            "match_is_source", "n_compared", "rms_to_source", "max_dist_to_source", "max_disp_applied_A", "elapsed_s"]
BASES_COLS = ["base_id", "row_index", "cnid", "natoms", "dx_A", "k", "poscar_relaxed_path"]


class ControlError(RuntimeError):
    """Failure of the construction, the inputs or the recomputation of the controls (non-zero exit)."""


def _pfx_path(p: str) -> str:
    """Of the two notations of the NFS mount, the one that exists on this node (same convention as K.P of the RUN version. Held independently for select)."""
    pfx = '/CLUSTER_ROOT_B' if os.path.isdir(_gbrel.WORK) else '/mnt/CLUSTER_ROOT_A'
    for pre in ('/mnt/CLUSTER_ROOT_A', '/CLUSTER_ROOT_B'):
        if p == pre or p.startswith(pre + "/"):
            return pfx + p[len(pre):]
    return p


def train_manifest_path() -> str:
    return _pfx_path(_gbrel.WORK + _TRAIN_MANIFEST_NAME)


def load_K():
    """Loads the RUN version of common by specifying the path (so that the copy with different content is not read by mistake)."""
    if not os.path.isfile(RUN_COMMON):
        raise ControlError(f"RUN common.py not found: {RUN_COMMON}")
    spec = importlib.util.spec_from_file_location("common_run", RUN_COMMON)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["common_run"] = mod
    spec.loader.exec_module(mod)
    if os.path.realpath(mod.__file__) != os.path.realpath(RUN_COMMON):
        raise ControlError(f"loaded common is not the RUN version: {mod.__file__}")
    return mod


# ---------------------------------------------------------------------------
# --select
# ---------------------------------------------------------------------------
def select_bases(manifest_path: str):
    """Takes the rows with converged == "True", sorts them by row_index ascending for each natoms, and takes the 4 rows at positions floor((j+0.5)*n_g/4) (j=0..3)."""
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_n = {}
    for r in rows:
        if r["converged"] == "True":
            by_n.setdefault(int(r["natoms"]), []).append(r)
    if len(by_n) != NATOMS_KINDS:
        raise ControlError(f"natoms kinds among converged rows = {sorted(by_n)} (expected {NATOMS_KINDS} kinds)")
    out = []
    for n in sorted(by_n):
        g = sorted(by_n[n], key=lambda r: int(r["row_index"]))
        ng = len(g)
        pos = [int(math.floor((j + 0.5) * ng / PER_NATOMS)) for j in range(PER_NATOMS)]
        if len(set(pos)) != PER_NATOMS:
            raise ControlError(f"natoms={n}: positions not distinct: {pos} (n_g={ng})")
        for p in pos:
            out.append(g[p])
    if len(out) != N_BASES:
        raise ControlError(f"selected {len(out)} bases (expected {N_BASES})")
    res = []
    for i, r in enumerate(out):
        res.append({"base_id": i, "row_index": int(r["row_index"]), "cnid": r["cnid"], "natoms": int(r["natoms"]),
                    "dx_A": r["dx_A"], "k": r["k"], "poscar_relaxed_path": r["poscar_relaxed_path"]})
    return res


def _bases_text(bases) -> str:
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=BASES_COLS, lineterminator="\n")
    w.writeheader()
    for b in bases:
        w.writerow(b)
    return buf.getvalue()


def cmd_select(out_dir: Path) -> int:
    mpath = train_manifest_path()
    bases = select_bases(mpath)
    text = _bases_text(bases)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "bases.csv"
    if dest.exists():
        if dest.read_text() == text:
            print(f"SKIP select: {dest} already exists with identical content", flush=True)
            return 0
        print(f"FATAL: {dest} exists with different content; not overwriting", file=sys.stderr)
        return 2
    tmp = out_dir / "bases.csv.tmp"
    tmp.write_text(text)
    os.replace(tmp, dest)
    with open(dest, newline="") as f:
        n = len(list(csv.DictReader(f)))
    if n != N_BASES:
        print(f"FATAL: bases.csv has {n} rows (expected {N_BASES})", file=sys.stderr)
        return 2
    counts = {}
    for b in bases:
        counts[b["natoms"]] = counts.get(b["natoms"], 0) + 1
    print(f"select: wrote {dest} rows={n} natoms_counts={dict(sorted(counts.items()))} manifest={mpath}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# transformation
# ---------------------------------------------------------------------------
def build_transforms(at0, base_id: int):
    """Builds the transformed Atoms from at0 (ase.Atoms without constraints). Returns {ID: (Atoms, max_disp_applied_A or None)}.
    One default_rng(20261008 + base_id) is used, and random numbers are drawn in the order P1 -> P1u -> P2 -> P3 -> P6 -> P7 -> N1."""
    import numpy as np

    rng = np.random.default_rng(RNG_SEED0 + base_id)
    n = len(at0)
    cell0 = np.array(at0.cell.array, dtype=float)
    s0 = at0.get_scaled_positions(wrap=False)
    a, b, c = cell0[0].copy(), cell0[1].copy(), cell0[2].copy()
    out = {}

    def with_scaled(s, cell=None):
        at = at0.copy()
        if cell is not None:
            at.set_cell(cell, scale_atoms=False)
        at.set_scaled_positions(s)
        return at

    def rewrap(at):
        s = at.get_scaled_positions(wrap=False) % 1.0
        at.set_scaled_positions(s)
        return at

    # P0 identity
    out["P0"] = (at0.copy(), None)
    # P1 rigid translation (with wrap)
    u = rng.random(3)
    out["P1"] = (with_scaled((s0 + u) % 1.0), None)
    # P1u rigid translation (without wrap)
    u2 = rng.random(3)
    out["P1u"] = (with_scaled(s0 + u2), None)
    # P2 atom order
    perm = rng.permutation(n)
    out["P2"] = (at0[perm], None)
    # P3 rigid rotation (proper rotation from uniform random numbers)
    A = rng.normal(size=(3, 3))
    Q, Rr = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(Rr))[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    if not (abs(np.linalg.det(Q) - 1.0) < 1e-9):
        raise ControlError(f"P3: rotation det != +1: {np.linalg.det(Q)}")
    at3 = at0.copy()
    at3.set_cell(cell0 @ Q.T, scale_atoms=True)
    out["P3"] = (at3, None)
    # P4 re-choice of the basis [a, b, c+b] (Cartesian coordinates unchanged) + wrap
    at4 = at0.copy()
    at4.set_cell(np.array([a, b, c + b]), scale_atoms=False)
    out["P4"] = (rewrap(at4), None)
    # P5 cyclic permutation of the axes [b, c, a] (Cartesian coordinates unchanged) + wrap
    at5 = at0.copy()
    at5.set_cell(np.array([b, c, a]), scale_atoms=False)
    out["P5"] = (rewrap(at5), None)
    # P6 two-fold rotation about the a axis + translation: (x, y, z) -> (x, -y, -z) + u'', wrap. The cell is unchanged
    u3 = rng.random(3)
    s6 = s0.copy()
    s6[:, 1] = -s6[:, 1]
    s6[:, 2] = -s6[:, 2]
    out["P6"] = (with_scaled((s6 + u3) % 1.0), None)
    # P7 displacement within the tolerance N(0, 0.01 A)
    d7 = rng.normal(0.0, 0.01, size=(n, 3))
    at7 = at0.copy()
    at7.set_positions(at0.get_positions() + d7)
    out["P7"] = (at7, float(np.linalg.norm(d7, axis=1).max()))
    # N1 displacement outside the tolerance N(0, 0.15 A)
    d1 = rng.normal(0.0, 0.15, size=(n, 3))
    atn = at0.copy()
    atn.set_positions(at0.get_positions() + d1)
    out["N1"] = (atn, float(np.linalg.norm(d1, axis=1).max()))
    return out


# ---------------------------------------------------------------------------
# matching (worker. _G is inherited by fork)
# ---------------------------------------------------------------------------
_G = {}


def _check_one(job):
    """job = (transform, path). Performs one matching and returns a row dict (the columns are ROW_COLS other than base_id)."""
    transform, path = job
    K = _G["K"]
    at = _G["transforms"][transform][0]
    max_disp = _G["transforms"][transform][1]
    source = _G["source"]
    row = {"row_index": _G["row_index"], "natoms": _G["natoms"], "transform": transform, "path": path,
           "applied": True, "skip_reason": "", "novel": "", "match_row_index": "", "match_is_source": "",
           "n_compared": "", "rms_to_source": "", "max_dist_to_source": "",
           "max_disp_applied_A": "" if max_disp is None else repr(max_disp), "elapsed_s": ""}
    t0 = time.time()
    if path == PATH_V:
        V = _G["V"]
        dev = V.VAN.angle_dev_deg(at)
        if not (dev <= K.ANGLE_TOL_DEG):
            row["applied"] = False
            row["skip_reason"] = "angle_fail"
            row["elapsed_s"] = f"{time.time() - t0:.3f}"
            return row
        try:
            at_c = V.VAN.canonicalize(at)[0]
        except Exception as e:  # do not swallow it. The parent makes the exit non-zero
            print(f"FATAL canonicalize exception: base={_G['base_id']} transform={transform}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            raise
        struct = K.to_pmg(at_c)
    else:
        struct = K.to_pmg(at)
    res = K.novelty_vs_train(struct, _G["train"], _G["matcher"])
    row["novel"] = bool(res["novel"])
    row["match_row_index"] = res["match_row_index"]
    row["match_is_source"] = (str(res["match_row_index"]) == str(_G["row_index"])) if not res["novel"] else ""
    row["n_compared"] = res["n_compared"]
    rd = _G["matcher"].get_rms_dist(struct, source)
    if rd is not None:
        row["rms_to_source"] = repr(float(rd[0]))
        row["max_dist_to_source"] = repr(float(rd[1]))
    row["elapsed_s"] = f"{time.time() - t0:.3f}"
    return row


def _job_list():
    jobs = [(t, PATH_H) for t in TRANSFORMS_H] + [(t, PATH_V) for t in TRANSFORMS_V]
    if len(jobs) != ROWS_PER_BASE:
        raise ControlError(f"job list size {len(jobs)} != {ROWS_PER_BASE}")
    return jobs


def _degeneracy_check(rows):
    """Checks that rms_to_source of P0, P1, P1u, P2 and P6 is not blank and not > 1e-6, else ControlError. Prints the maximum."""
    bad = []
    mx = {PATH_H: 0.0, PATH_V: 0.0}
    for r in rows:
        degen = (r["path"] == PATH_H and r["transform"] in DEGEN_H) or (r["path"] == PATH_V and r["transform"] in DEGEN_V)
        if not degen:
            continue
        v = r["rms_to_source"]
        if v == "":
            bad.append((r["path"], r["transform"], "empty"))
            continue
        x = float(v)
        mx[r["path"]] = max(mx[r["path"]], x)
        if x > RMS_DEGEN_TOL:
            bad.append((r["path"], r["transform"], v))
    print(f"degeneracy: max rms_to_source over P0/P1/P1u/P2/P6 (rms is normalized by (V/N)^(1/3), as returned by get_rms_dist): "
          f"H={mx[PATH_H]:.3e} V={mx[PATH_V]:.3e} (limit {RMS_DEGEN_TOL:g})", flush=True)
    if bad:
        raise ControlError(f"construction degenerate (rms_to_source empty or > {RMS_DEGEN_TOL:g}): {bad}")


def cmd_base(base_id: int, out_dir: Path, procs: int) -> int:
    if not (0 <= base_id < N_BASES):
        print(f"FATAL: base-id must be 0..{N_BASES - 1} (got {base_id})", file=sys.stderr)
        return 2
    tag = f"{base_id:02d}"
    rows_file = out_dir / f"rows_base{tag}.csv"
    done_file = out_dir / f"DONE_base{tag}"
    if rows_file.exists():
        if done_file.is_file():
            with open(rows_file, newline="") as f:
                n = len(list(csv.DictReader(f)))
            if n == ROWS_PER_BASE:
                print(f"SKIP base {base_id}: {rows_file} already complete", flush=True)
                return 0
        print(f"FATAL: {rows_file} exists without a valid DONE/{ROWS_PER_BASE} rows; not overwriting", file=sys.stderr)
        return 2
    bases_file = out_dir / "bases.csv"
    if not bases_file.is_file():
        print(f"FATAL: {bases_file} not found (the select stage is required)", file=sys.stderr)
        return 2
    with open(bases_file, newline="") as f:
        bases = list(csv.DictReader(f))
    if len(bases) != N_BASES:
        print(f"FATAL: {bases_file} has {len(bases)} rows (expected {N_BASES})", file=sys.stderr)
        return 2
    brow = bases[base_id]
    if int(brow["base_id"]) != base_id:
        print(f"FATAL: bases.csv row {base_id} has base_id {brow['base_id']}", file=sys.stderr)
        return 2

    t_start = time.time()
    K = load_K()
    V = K.load_V()
    from ase.io import read as ase_read

    print(f"base={base_id} row_index={brow['row_index']} natoms={brow['natoms']} cnid={brow['cnid']} dx_A={brow['dx_A']} "
          f"RUN_common={K.__file__}", flush=True)
    at0 = ase_read(K.P(brow["poscar_relaxed_path"]))
    at0.set_constraint()
    assert len(at0.constraints) == 0, f"constraints not removed: {at0.constraints}"
    print(f"constraints={len(at0.constraints)} natoms_read={len(at0)}", flush=True)
    if len(at0) != int(brow["natoms"]):
        raise ControlError(f"natoms mismatch: POSCAR {len(at0)} vs bases.csv {brow['natoms']}")

    transforms = build_transforms(at0, base_id)
    source = K.to_pmg(at0)
    mpath = K.P(train_manifest_path())
    train_by_n = K.load_train_structures(mpath)
    matcher = K.make_matcher()
    print(f"train loaded: " + " ".join(f"{n}:{len(v)}" for n, v in sorted(train_by_n.items())) +
          f" matcher_kwargs={K.load_UNI()}", flush=True)

    _G.update({"K": K, "V": V, "transforms": transforms, "source": source, "train": train_by_n, "matcher": matcher,
               "row_index": int(brow["row_index"]), "natoms": int(brow["natoms"]), "base_id": base_id})
    jobs = _job_list()
    t_run = time.time()
    if procs > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(procs, len(jobs))) as pool:
            results = pool.map(_check_one, jobs, chunksize=1)
    else:
        results = [_check_one(j) for j in jobs]
    rows = []
    for r in results:
        r = dict(r)
        r["base_id"] = base_id
        rows.append(r)
    for r in rows:
        print(f"  {r['path']} {r['transform']:>3} applied={r['applied']} novel={r['novel']} match={r['match_row_index']} "
              f"src={r['match_is_source']} n_cmp={r['n_compared']} rms={r['rms_to_source']} maxd={r['max_dist_to_source']} "
              f"disp={r['max_disp_applied_A']} t={r['elapsed_s']}s", flush=True)
    _degeneracy_check(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"rows_base{tag}.csv.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_COLS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ROW_COLS})
    with open(tmp, newline="") as f:
        n = len(list(csv.DictReader(f)))
    if n != ROWS_PER_BASE:
        raise ControlError(f"rows written {n} != {ROWS_PER_BASE}; DONE not written")
    os.replace(tmp, rows_file)
    done_file.write_text(f"base_id={base_id} rows={n}\n")
    print(f"wrote {rows_file} rows={n} and {done_file.name}; total {time.time() - t_start:.1f}s (checks {time.time() - t_run:.1f}s, procs={procs})",
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select", action="store_true")
    g.add_argument("--base-id", type=int)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--procs", type=int, default=1)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    try:
        if args.select:
            return cmd_select(out_dir)
        return cmd_base(args.base_id, out_dir, max(1, args.procs))
    except Exception as e:  # non-zero exit (do not swallow it)
        traceback.print_exc()
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
