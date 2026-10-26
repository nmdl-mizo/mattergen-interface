#!/usr/bin/env python3
"""common.py -- common parts of the relaxation, novelty and uniqueness stages P / R / N / U+A / Z (only for the scripts of this experiment).

Plan: see the experiment description.
External code imported read-only (not copied):
  - gb_eval/experiments/gen_eval/scripts/compute_gamma_pre_proposed.py       (H: CIF reading read_cif, min_dist_A, etc.)
  - gb_eval/experiments/gen_eval/scripts/compute_gamma_pre_baseline.py    (V: V.VAN = axis-permutation canonicalize, angle_dev_deg, load_chunk)
  - gb_eval/experiments/data_rebuild/scripts/stage_d_common.py                (UNI = StructureMatcher settings; not copied)
Path notation: <CLUSTER_MOUNT>/... on compute nodes and on the login node alike. P() converts to the form that exists. Do not start python on the login node.
"""
from __future__ import annotations

import csv
import importlib.util
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
_PFX = '/CLUSTER_ROOT_B' if os.path.isdir('<ORIGINAL_ROOT>') else '/mnt/CLUSTER_ROOT_A'


def P(p) -> str:
    """Convert so that it works whether the path is written with the compute-node or the login-node mount prefix, to the notation that exists on this node."""
    s = str(p)
    for pre in ('/mnt/CLUSTER_ROOT_A', '/CLUSTER_ROOT_B'):
        if s == pre or s.startswith(pre + "/"):
            return _PFX + s[len(pre):]
    return s


def norm_compute(p) -> str:
    """Canonical form when writing into the manifest (compute-node notation <CLUSTER_MOUNT>)."""
    s = str(p)
    if s.startswith('/mnt/CLUSTER_ROOT_A'):
        s = s[len("/mnt"):]
    return s


GC = '<ORIGINAL_ROOT>/workspace'
RUN = f"{GC}/experiments/relax_novelty"
RUN_REF = f"{GC}/experiments/gen_eval"
RUN_MAIN = f"{GC}/experiments/range_targets"
DATA = f"{GC}/experiments/data_rebuild"
GEN_SCRIPTS = f"{RUN_REF}/scripts"
DATA_SCRIPTS = f"{DATA}/scripts"
CONFIG_DIR = f"{GC}/scripts/config"
RUNNER_GUARDS_DIR = '<ORIGINAL_ROOT>/bin'
CARD_PATH = f"{GC}/model_spec.yaml"
MACE_PY = '<ORIGINAL_ROOT>/miniconda3/envs/mace-mh/bin/python'
VENV_PY = '<ORIGINAL_ROOT>/scm-work/.venv/bin/python'

GAMMA_CSV = f"{DATA}/results/gamma_dedup.csv"
BULK_MEMBERSHIP_CSV = f"{DATA}/results/bulk_membership.csv"
MANIFEST_DIR = f"{DATA}/results/manifests"
SPLIT_JSON = f"{DATA}/results/split.json"
TEMPLATE_A = 54.15601152562611   # cell a of the Al template [A] (same value as TEMPLATE_A of sbatch_eval.sh)

# layers: additional = quantile targets, w = 1 to 4 (RUN_REF); main = width-based targets, w = 2.0 (RUN_MAIN)
LAYERS = {
    "additional": {"run_root": RUN_REF, "ws": [1.0, 2.0, 3.0, 4.0]},
    "main": {"run_root": RUN_MAIN, "ws": [2.0]},
}
ARMS = ["proposed", "baseline"]            # order of array_map: proposed -> baseline
N_CHUNKS = 14
PER_CHUNK = 25
N_TARGETS = 7
CHUNKS_PER_TARGET = 2                 # target_id = chunk // 2
PER_TARGET = CHUNKS_PER_TARGET * PER_CHUNK   # 50
N_PER_ARM_W = N_CHUNKS * PER_CHUNK    # 350
INNER_MAX_TARGET_ID = 4
N_FIXED_PROPOSED = 76
ANGLE_TOL_DEG = 1.0                   # default of --angle-tol-deg of compute_gamma_pre_baseline.py:114
NEAR_TOL_J = 0.10                     # width of the neighborhood of the target (defined in the model specification; do not change)
FMAX = 0.02                           # model specification D2 (checked against config.FMAX at run time)
MAX_STEPS = 500                       # model specification D2 (checked against config.MAX_STEPS at run time)
GAMMA_FACTOR = 16.0218                # D4: config.EV_PER_A2_TO_J_PER_M2


def wtag(w) -> str:
    return f"w{float(w):.1f}"


def armw(arm, w) -> str:
    return f"{arm}_{wtag(w)}"


def layer_run_root(layer) -> str:
    return LAYERS[layer]["run_root"]


def gen_dir(layer, arm, w) -> str:
    return P(f"{layer_run_root(layer)}/results/gen/{arm}/{wtag(w)}")


def eval_dir(layer, arm, w) -> str:
    return P(f"{layer_run_root(layer)}/results/eval/{armw(arm, w)}")


def chunk_name(c) -> str:
    return f"chunk{int(c):02d}"


def layer_arm_w_list(layer):
    return [(arm, w) for arm in ARMS for w in LAYERS[layer]["ws"]]


def array_map_rows(layer):
    """Lexicographic order of (arm, w, chunk) (arm: proposed -> baseline, w ascending, chunk ascending)."""
    rows = []
    for arm in ARMS:
        for w in sorted(LAYERS[layer]["ws"]):
            for c in range(N_CHUNKS):
                rows.append({"layer": layer, "arm": arm, "w": f"{float(w):.1f}", "chunk": c})
    for i, r in enumerate(rows):
        r["array_index"] = i
    return rows


# ---------------------------------------------------------------------------
# loading of external modules (read only, not copied)
# ---------------------------------------------------------------------------
def _ensure_path(p):
    p = P(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_H():
    """compute_gamma_pre_proposed (has read_cif, min_dist_A, V2)."""
    _ensure_path(GEN_SCRIPTS)
    import compute_gamma_pre_proposed as H  # noqa: WPS433
    return H


def load_V():
    """compute_gamma_pre_baseline (V.VAN has canonicalize, angle_dev_deg, load_chunk)."""
    _ensure_path(GEN_SCRIPTS)
    import compute_gamma_pre_baseline as V  # noqa: WPS433
    return V


def load_UNI() -> dict:
    """stage_d_common.UNI (kwargs of StructureMatcher). The settings are not copied."""
    _ensure_path(DATA_SCRIPTS)
    import stage_d_common as D  # noqa: WPS433
    return dict(D.UNI)


def load_CFG():
    _ensure_path(CONFIG_DIR)
    import config as CFG  # noqa: WPS433
    return CFG


def make_matcher(uni=None, **override):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    kw = dict(uni if uni is not None else load_UNI())
    kw.update(override)
    return StructureMatcher(**kw)


# ---------------------------------------------------------------------------
# CSV utilities (count rows with the csv module; do not use awk -F,)
# ---------------------------------------------------------------------------
def read_csv_rows(path):
    with open(P(path), newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, fieldnames, rows, *, overwrite=False):
    path = Path(P(path))
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fieldnames})


def to_bool(s):
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ("1", "true", "yes")


def to_float(s, default=None):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


# ---------------------------------------------------------------------------
# gamma_pre.csv (read by column name)
# ---------------------------------------------------------------------------
GAMMA_PRE_REQUIRED = ["arm", "gamma_target", "cif_index", "gamma_unrelaxed", "natoms", "area_A2", "error",
                      "w", "chunk", "target_id", "is_inner", "non_al_count"]


class InputError(RuntimeError):
    pass


def read_gamma_pre(layer, arm, w):
    """Reads gamma_pre.csv of (arm, w) by column name, checks 350 rows, unique keys and chunks 0..13 x cif_index 0..24, and returns it."""
    path = Path(eval_dir(layer, arm, w)) / "gamma_pre.csv"
    if not path.is_file():
        raise InputError(f"gamma_pre.csv not found: {path} (wait if the evaluation stage is not finished)")
    rows = read_csv_rows(path)
    if rows:
        miss = [c for c in GAMMA_PRE_REQUIRED if c not in rows[0]]
        if miss:
            raise InputError(f"{path}: missing columns {miss} (read by column name)")
    if len(rows) != N_PER_ARM_W:
        raise InputError(f"{path}: {len(rows)} rows != {N_PER_ARM_W}")
    keys = set()
    for r in rows:
        k = (int(r["chunk"]), int(r["cif_index"]))
        if k in keys:
            raise InputError(f"{path}: duplicate key {k}")
        keys.add(k)
        if r["arm"] != armw(arm, w):
            raise InputError(f"{path}: arm label {r['arm']!r} != {armw(arm, w)!r}")
        if abs(float(r["w"]) - float(w)) > 0:
            raise InputError(f"{path}: w {r['w']} != {w}")
        if int(r["target_id"]) != int(r["chunk"]) // CHUNKS_PER_TARGET:
            raise InputError(f"{path}: target_id {r['target_id']} != chunk//2 for chunk {r['chunk']}")
    expect = {(c, i) for c in range(N_CHUNKS) for i in range(PER_CHUNK)}
    if keys != expect:
        raise InputError(f"{path}: key set != chunk 0..13 x cif_index 0..24")
    return rows


def excluded_pre_reason(r) -> str:
    """Reason why a structure was excluded in the evaluation before relaxation (empty = valid). Non-empty error, non_al_count>0 (angle_fail of baseline is in the error column).
    The reason is entered as it is; if there are several they are joined with ';'."""
    parts = []
    err = (r.get("error") or "").strip()
    if err:
        parts.append(err.replace(";", ","))
    try:
        nal = int(float(r.get("non_al_count") or 0))
    except ValueError:
        nal = 0
    if nal > 0:
        parts.append(f"non_al_count={nal}")
    if not parts and to_float(r.get("gamma_unrelaxed")) is None:
        parts.append("gamma_pre_missing")
    return ";".join(parts)


EXCL_CATEGORIES = ["angle_fail", "nonorthogonal", "nonfinite", "non_al", "error", "gamma_pre_missing"]


def excluded_category(reason: str) -> str:
    """Classification for the aggregation per reason (one classification per row; the priority order is EXCL_CATEGORIES). '' if empty."""
    if not reason:
        return ""
    toks = reason.split(";")
    def has(prefix):
        return any(t.startswith(prefix) for t in toks)
    if has("angle_fail"):
        return "angle_fail"
    if has("nonorthogonal"):
        return "nonorthogonal"
    if has("nonfinite"):
        return "nonfinite"
    if has("gamma_pre_missing") and len(toks) == 1:
        return "gamma_pre_missing"
    if any(not t.startswith("non_al_count=") for t in toks):
        return "error"
    return "non_al"


# ---------------------------------------------------------------------------
# loading of generated structures (handled with the same function, orientation and area as the evaluation of gamma before relaxation)
# ---------------------------------------------------------------------------
def load_chunk_atoms(layer, arm, w, chunk, angle_tol_deg=ANGLE_TOL_DEG):
    """list[(atoms | None, note)] of 25 structures.
    proposed: reads chunkNN/cif/gen_<i>.cif with H.read_cif (the route of compute_gamma_pre_proposed.py; in numerical order).
    baseline: extxyz of V.VAN.load_chunk(chunkNN, 25). Structures whose angles are outside 90 deg +- angle_tol are (None, 'angle_fail');
             for the others the Atoms returned by V.VAN.canonicalize(atoms) are used as they are (not reconstructed from the perm column)."""
    cdir = Path(gen_dir(layer, arm, w)) / chunk_name(chunk)
    if arm == "proposed":
        H = load_H()
        cif_dir = cdir / "cif"
        if not cif_dir.is_dir():
            raise InputError(f"{cif_dir} not found (cif/ of proposed is made by the extract of the generation stage)")
        named = []
        for p in cif_dir.glob("*.cif"):
            m = H._CIF_RE.match(p.name)
            if m is None:
                raise InputError(f"unexpected cif file name {p.name!r} in {cif_dir}")
            named.append((int(m.group(1)), p))
        named.sort()
        if [i for i, _ in named] != list(range(PER_CHUNK)):
            raise InputError(f"{cif_dir}: cif indices != 0..{PER_CHUNK - 1}")
        return [(H.read_cif(p), "") for _, p in named]
    V = load_V()
    _meta, atoms_list = V.VAN.load_chunk(cdir, PER_CHUNK)
    out = []
    for atoms in atoms_list:
        dev = V.VAN.angle_dev_deg(atoms)
        if not (dev <= angle_tol_deg):
            out.append((None, "angle_fail"))
            continue
        canon_atoms, _info = V.VAN.canonicalize(atoms)
        out.append((canon_atoms, ""))
    return out


def to_pmg(atoms):
    """ase Atoms -> pymatgen Structure (fractional coordinates are not wrapped; StructureMatcher handles the periodicity itself)."""
    from pymatgen.core import Lattice, Structure
    return Structure(Lattice(atoms.cell[:]), [int(z) for z in atoms.numbers], atoms.get_scaled_positions(wrap=False),
                     coords_are_cartesian=False)


# ---------------------------------------------------------------------------
# minimum image and area
# ---------------------------------------------------------------------------
def minimum_image(diff, cell):
    """Maps the Cartesian difference diff(...x3) to the minimum image of the cell(3x3) (by rounding fractional coordinates; exact for orthogonal cells and small differences)."""
    import numpy as np
    cell = np.asarray(cell, dtype=float)
    d = np.asarray(diff, dtype=float)
    frac = d @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


def area_bc(atoms) -> float:
    """Same function and same axis as gamma before relaxation (V2.in_plane_area_bc = |b x c|)."""
    return float(load_H().V2.in_plane_area_bc(atoms))


# ---------------------------------------------------------------------------
# matcher (novelty and uniqueness)
# ---------------------------------------------------------------------------
def novelty_vs_train(struct, train_by_natoms, matcher):
    """Matches struct (pymatgen Structure) with matcher.fit(struct, train) only against training structures with the same number of atoms.
    train_by_natoms: {natoms: [(row_index, cnid, Structure), ...]}。
    Returns: dict(novel, match_row_index, match_cnid, n_compared, novel_vacuous).
    If even one is True, novel=False (the first matching training structure is recorded). If all are False, novel=True.
    If there is no training structure with the same number of atoms, novel=True, n_compared=0, novel_vacuous=True (vacuous novelty)."""
    refs = train_by_natoms.get(len(struct), [])
    n = 0
    for (ri, cnid, tstruct) in refs:
        n += 1
        if matcher.fit(struct, tstruct):
            return {"novel": False, "match_row_index": ri, "match_cnid": cnid, "n_compared": n, "novel_vacuous": False}
    return {"novel": True, "match_row_index": "", "match_cnid": "", "n_compared": n, "novel_vacuous": (len(refs) == 0)}


def load_train_structures(train_manifest_path):
    """Reads POSCAR_relaxed of all rows of train_ref_manifest.csv as pymatgen Structures and returns {natoms: [(row_index, cnid, Structure)]}."""
    from pymatgen.core import Structure
    rows = read_csv_rows(train_manifest_path)
    by_n = {}
    for r in rows:
        s = Structure.from_file(P(r["poscar_relaxed_path"]))
        if len(s) != int(r["natoms"]):
            raise InputError(f"train row {r['row_index']}: natoms {len(s)} != manifest {r['natoms']}")
        by_n.setdefault(len(s), []).append((int(r["row_index"]), r["cnid"], s))
    return by_n


# ---------------------------------------------------------------------------
# connected components
# ---------------------------------------------------------------------------
def connected_components(n, edges):
    """Labels (0..k-1) of the connected components of a graph with n vertices and edges edges(i,j), and the number of components. Union-find."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(n)]
    uniq = sorted(set(roots))
    idx = {r: k for k, r in enumerate(uniq)}
    return [idx[r] for r in roots], len(uniq)


def count_components_by_fit(n, fit_fn):
    """Takes fit_fn(i, j) for all pairs (i<j) of n vertices and returns the number of connected components of the graph whose edges are the pairs that are true.
    Pairs that are already in the same component are not checked (the number of components does not change; it is the connected components of all pairs, not a greedy match with a representative).
    Returns (number of components, list of pairs confirmed as edges, number of pairs for which fit_fn was actually called)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges, n_calls = [], 0
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            n_calls += 1
            if fit_fn(i, j):
                edges.append((i, j))
                ri, rj = find(i), find(j)
                parent[max(ri, rj)] = min(ri, rj)
    return len({find(i) for i in range(n)}), edges, n_calls


def sbatch_range_spec(indices):
    """Converts a list of integers to the 'a-b,c,d-e' form for SLURM --array."""
    idx = sorted(set(indices))
    out, start, prev = [], None, None
    for x in idx:
        if start is None:
            start = prev = x
        elif x == prev + 1:
            prev = x
        else:
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = x
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(out)


def resolve_array_task(layer, task_id, array_map=None):
    """Looks up the row of SLURM_ARRAY_TASK_ID in array_map_<layer>.csv (written by stage P; columns array_index, layer, arm, w, chunk)."""
    path = array_map or P(f"{RUN}/results/array_map_{layer}.csv")
    if not os.path.isfile(path):
        raise InputError(f"{path} not found (stage P is required)")
    for r in read_csv_rows(path):
        if int(r["array_index"]) == int(task_id):
            if r["layer"] != layer:
                raise InputError(f"{path}: row {task_id} has layer {r['layer']!r} != {layer!r}")
            return {"layer": r["layer"], "arm": r["arm"], "w": float(r["w"]), "chunk": int(r["chunk"])}
    raise InputError(f"{path}: array_index {task_id} not found")
