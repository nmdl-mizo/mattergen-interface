#!/usr/bin/env python3
"""gb_stage2_local.py -- stage 2: equivalence decision of the two grain boundaries in a local window around the grain boundaries (Al, post and two_gb structures).

Plan: see the experiment description.
The implementation of stages 0 and 1 (gb_symfit.py of the verification version) is imported and used (not copied, not changed).

Usage (compute node. The python of the evaluation environment. Do not start python on the login node):
  --mode validation      : W1 to W6 of the plan (+ W4b report only). VRUN/results/gb_stage2/gb_stage2_validation.json (the run is refused unless all_pass is true)
  --mode r0              : stage R0 (dry aggregation before the decision). dry_run_counts.md. exit 4 if a stop condition applies
  --mode run [--h H]     : stage R. H=4.0 (main) -> gb_stage2_<layer>.csv, 3.0 -> _h3.csv, 6.0 -> _h6.csv (validation all_pass is a precondition)
  --mode aggregate       : aggregation -> stage2_check_inputs.json and gb_stage2_summary.md (the CSV of h=4.0 and the CSV of h=3.0/6.0 are preconditions)
  --mode self-check-criterion   : verdict validation of the decision function stage2_checks() -> self_check_criterion.json
Exit codes: 0 = success / 2 = failure of the inputs or checks / 3 = validation or self check FAIL / 4 = R0 stop condition
Outputs are never overwritten (if they exist, FileExistsError -> exit 2).
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gb_symfit as S  # noqa: E402  (verification version. Functions of stages 0 and 1)

K = S.K
if os.path.abspath(S.__file__).replace('/mnt/CLUSTER_ROOT_A', '/CLUSTER_ROOT_B') != os.path.abspath(str(HERE / "gb_symfit.py")).replace('/mnt/CLUSTER_ROOT_A', '/CLUSTER_ROOT_B'):
    raise RuntimeError(f"gb_symfit resolved to {S.__file__}, expected {HERE}")

DESIGN = _gbrel.WORK + '/an internal design note that is not included'

# ---------------------------------------------------------------------------
# preregistered parameters (do not change)
# ---------------------------------------------------------------------------
H_MAIN = 4.0                 # half width h of the core [A] (main value)
H_SENS = (3.0, 6.0)          # for the report of the sensitivity
M_WIN = 4.5                  # margin m of the search window [A] (frozen)
THICK_DIFF_A = 4.5           # thickness_differs_A: |t_A - t_B| > 4.5 Å
THETA_DECISION = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "0.032", "0.064", "0.1"]
THETA_REPORT = ["1e-5", "3e-5", "0.15", "0.2", "0.3"]
THETA_ALL = ["1e-5", "3e-5"] + THETA_DECISION + ["0.15", "0.2", "0.3"]
THETA_VAL = {t: float(t) for t in THETA_ALL}
OPS = ["C2n", "m_p", "C2t", "inv"]
LAYERS_PACK = ["main", "additional", "baseline_regen1", "baseline_regen2", "baseline_regen3", "proposed_regen1", "proposed_regen2"]
BASELINE_JUDGE = ["main", "baseline_regen1", "baseline_regen2", "baseline_regen3"]
N_REQ = 350
G1_K_PASS = 17               # PASS: k_r + k_near_r <= 17
WILSON_FAIL_P = 0.05
K_NEAR_LO, K_NEAR_HI = 0.1, 0.3
REFINE_MAX_ITER = S.REFINE_MAX_ITER
REFINE_TOL = S.REFINE_TOL

OUT_DIRNAME = "gb_stage2"
VALIDATION_NAME = "gb_stage2_validation.json"

# constants of the validation (plan)
W_SEEDS = {"W1": 14001, "W2": 14002, "W4": 14003}
W_PERT_CORE_A = 2.0          # half width of the inner core [A]
W_PERT_AMP_W1 = 0.15
W_PERT_AMP_W2 = 0.30
W_EXCL_A = 6.0
W_SCAN_A, W_STEP_A = 0.3, 0.005
W_MATCH_TOL_A = 0.05
W_MIN_POP = 8
W_MIN_RATE = 0.95
W_DEV_LP_A = 0.15
W_MAX_CANDS = 10
W_W3_SHIFT_A = 8.0


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vrun_root() -> Path:
    return Path(K.P(K.VRUN))


_OUT_ROOT_OVERRIDE = None     # for tests only (--out-root). Not used in the production computation. The inputs are read from the default VRUN/results/gb_stage2/inputs (read only)


def out_root() -> Path:
    if _OUT_ROOT_OVERRIDE is not None:
        return Path(_OUT_ROOT_OVERRIDE)
    return vrun_root() / "results" / OUT_DIRNAME


def inputs_dir() -> Path:
    return vrun_root() / "results" / OUT_DIRNAME / "inputs"


def write_text_once(path: Path, text: str) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        f.write(text)


def write_json_once(path: Path, doc) -> None:
    write_text_once(path, json.dumps(doc, indent=2, ensure_ascii=False, default=float) + "\n")


# ---------------------------------------------------------------------------
# grain-boundary positions (step 3 of the plan)
# ---------------------------------------------------------------------------
def prune_runs_idx(pairs):
    """Same rule as S.prune_runs (:292-311) (removal of runs of length 1 until a fixed point), applied to a sequence of (bin number, label) pairs while keeping the indices.
    The rotation of the beginning to a run boundary is also the same as in S.prune_runs (the final sequence can be rotated)."""
    seq = list(pairs)
    while True:
        m = len(seq)
        if m == 0 or len({lab for _i, lab in seq}) < 2:
            return seq
        s = next(k for k in range(m) if seq[k][1] != seq[k - 1][1])
        rot = seq[s:] + seq[:s]
        runs = []
        for it in rot:
            if runs and runs[-1][0][1] == it[1]:
                runs[-1].append(it)
            else:
                runs.append([it])
        new = [it for run in runs if len(run) >= 2 for it in run]
        if len(new) == m:
            return new
        seq = new


def _is_rotation(a, b) -> bool:
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    return any(b[k:] + b[:k] == a for k in range(len(b)))


def gb_positions(bin_seq: str):
    """From the raw bin sequence (characters A/B/.) returns the fractional positions of GB_AB and GB_BA. (z_AB, z_BA, info).
    ValueError('switches=...') unless there are exactly 2 switches (one A->B and one B->A). The self check of the plan is also done (a discrepancy is an AssertionError)."""
    nb = len(bin_seq)
    pairs = [(i, 1 if c == "A" else 2) for i, c in enumerate(bin_seq) if c in "AB"]
    pruned = prune_runs_idx(pairs)
    ref = S.prune_runs([lab for _i, lab in pairs])
    if not _is_rotation([lab for _i, lab in pruned], list(ref)):
        raise AssertionError(f"check-I: prune_runs_idx labels != prune_runs (rotation-insensitive): {bin_seq}")
    sw = []
    m = len(pruned)
    for k in range(m):
        (i, a), (j, b) = pruned[k], pruned[(k + 1) % m]
        if a != b:
            jj = j if j >= i + 1 else j + nb
            z = (((i + 1) + jj) / 2.0 / nb) % 1.0
            sw.append(("AB" if (a, b) == (1, 2) else "BA", z))
    kinds = sorted(k for k, _z in sw)
    if kinds != ["AB", "BA"]:
        raise ValueError(f"switches={[k for k, _z in sw]}")
    zd = dict(sw)
    return zd["AB"], zd["BA"], {"nb": nb}


def thickness_differs_bins(bin_seq: str) -> bool:
    """The same definition as theta_sensitivity.py:74-83 (comparison of the number of bins on the stage-1 side. True even for a difference of 1 bin)."""
    seq = [1 if c == "A" else 2 for c in bin_seq if c in "AB"]
    seq = S.prune_runs(seq)
    nA, nB = sum(1 for x in seq if x == 1), sum(1 for x in seq if x == 2)
    if nA == 0 or nB == 0:
        return False
    return nA != nB


# ---------------------------------------------------------------------------
# fit in the local window (step 5 of the plan)
# ---------------------------------------------------------------------------
def _dist_matrix(cell, Y, T):
    """From the fractional coordinates of Y(nc,3) and T(ns,3), the matrix (nc,ns) of minimum-image Cartesian distances."""
    diff = Y[:, None, :] - T[None, :, :]
    diff = diff - np.round(diff)
    return np.linalg.norm(diff @ cell, axis=2)


def fit_window_oneway(frac, cell, rdiag, core_idx, search_idx, dcore):
    """Oriented residual r(R; i->j) [A]: the atoms of C_i are mapped by R and translated by tau_c (candidates made from each atom of S_j), and assigned rectangularly to S_j (to different atoms).
    The tau with the smallest sum of assignment distances is taken and refined in the same way as fit_op (reassign each time, tau += mean(mic(difference)), at most 20 times, |d tau|<1e-10).
    r = maximum of the assignment distances after the refinement. inf if |C_i| > |S_j|."""
    nc, ns = len(core_idx), len(search_idx)
    if nc == 0 or nc > ns:
        return float("inf")
    X = frac[core_idx] * rdiag
    T = frac[search_idx]
    k0 = int(np.argmin(np.abs(dcore)))
    best_cost, best_tau = np.inf, None
    for c in range(ns):
        tau = S._mic(T[c] - X[k0])
        D = _dist_matrix(cell, X + tau, T)
        row, col = linear_sum_assignment(D)
        cost = float(D[row, col].sum())
        if cost < best_cost:
            best_cost, best_tau = cost, tau.copy()
    tau = best_tau.copy()
    for _ in range(REFINE_MAX_ITER):
        Y = X + tau
        D = _dist_matrix(cell, Y, T)
        row, col = linear_sum_assignment(D)
        delta = S._mic(T[col] - Y[row])
        step = delta.mean(axis=0)
        tau = tau + step
        if float(np.max(np.abs(step))) < REFINE_TOL:
            break
    D = _dist_matrix(cell, X + tau, T)
    row, col = linear_sum_assignment(D)
    return float(D[row, col].max())


def window_sets(frac, cell, axes, zAB, zBA, h, m):
    n = axes[0]
    Ln = float(np.linalg.norm(cell[n]))
    fn = frac[:, n]
    d_AB = Ln * S._mic(fn - zAB)
    d_BA = Ln * S._mic(fn - zBA)
    sets = {}
    for tag, d in (("AB", d_AB), ("BA", d_BA)):
        sets[tag] = {"d": d, "core": np.flatnonzero(np.abs(d) <= h), "search": np.flatnonzero(np.abs(d) <= h + m)}
    return sets, Ln


def local_s2(frac, cell, axes, zAB, zBA, h, m):
    """s2 = min_R max(r(R; AB->BA), r(R; BA->AB)). Returns: dict(s2, s2_op, r_<op>_AB2BA, r_<op>_BA2AB, n_core_*, n_search_*)."""
    sets, _Ln = window_sets(frac, cell, axes, zAB, zBA, h, m)
    out = {"n_core_AB": int(len(sets["AB"]["core"])), "n_core_BA": int(len(sets["BA"]["core"])),
           "n_search_AB": int(len(sets["AB"]["search"])), "n_search_BA": int(len(sets["BA"]["search"]))}
    best, best_op = float("inf"), ""
    for op in OPS:
        rd = S.op_diag(op, axes)
        r_ab = fit_window_oneway(frac, cell, rd, sets["AB"]["core"], sets["BA"]["search"], sets["AB"]["d"][sets["AB"]["core"]])
        r_ba = fit_window_oneway(frac, cell, rd, sets["BA"]["core"], sets["AB"]["search"], sets["BA"]["d"][sets["BA"]["core"]])
        out[f"r_{op}_AB2BA"], out[f"r_{op}_BA2AB"] = r_ab, r_ba
        e = max(r_ab, r_ba)
        if e < best:
            best, best_op = e, op
    out["s2_max_A"], out["s2_op"] = best, best_op
    return out


# ---------------------------------------------------------------------------
# analysis of one structure (recomputation of stage 0 -> positions -> thickness -> thin_grain -> stage 2)
# ---------------------------------------------------------------------------
def analyze2(frac, cell, refs, ref_seq, h, m, compute_s2=True, z_BA_shift_A=0.0, axes=None):
    """If ref_seq=None, the recomputed value of the structure itself is used as the reference bin sequence (validation). z_BA_shift_A is only for W3 (stays 0 in the production run).
    status of the returned dict: ok|thin_grain|seq_mismatch|error."""
    out = {"status": "", "note": ""}
    try:
        if axes is None:
            axes = S.identify_axes(np.linalg.norm(cell, axis=1))
        out["axes"] = S.axes_str(axes)
        c0 = S.count_gb(frac, cell, axes, refs)
        seq = c0["gb_bin_seq"]
        out["gb_bin_seq_recomputed"] = seq
        out["gb0_status"] = c0["gb0_status"]
        out["nb"] = len(seq)
        if ref_seq is not None and seq != ref_seq:
            out["status"] = "seq_mismatch"
            out["note"] = f"recomputed bin seq != reference ({sum(a != b for a, b in zip(seq, ref_seq))} bins differ)"
            return out
        try:
            zAB, zBA, _info = gb_positions(seq)
        except ValueError as e:
            out["status"] = "error"
            out["note"] = str(e)
            return out
        Ln = float(np.linalg.norm(cell[axes[0]]))
        tB = Ln * ((zBA - zAB) % 1.0)
        tA = Ln - tB
        out.update({"z_AB": zAB, "z_BA": zBA, "t_A_A": tA, "t_B_A": tB,
                    "thickness_differs_bins": thickness_differs_bins(seq),
                    "thickness_differs_A": bool(abs(tA - tB) > THICK_DIFF_A)})
        sets, _ = window_sets(frac, cell, axes, zAB, zBA, h, m)
        out.update({"n_core_AB": int(len(sets["AB"]["core"])), "n_core_BA": int(len(sets["BA"]["core"])),
                    "n_search_AB": int(len(sets["AB"]["search"])), "n_search_BA": int(len(sets["BA"]["search"]))})
        if min(tA, tB) < 2.0 * (h + m):
            out["status"] = "thin_grain"
            return out
        if not compute_s2:
            out["status"] = "ok"
            return out
        zBA_used = (zBA + z_BA_shift_A / Ln) % 1.0
        out.update(local_s2(frac, cell, axes, zAB, zBA_used, h, m))
        out["status"] = "ok"
        return out
    except Exception as e:  # noqa: BLE001  (a failure of a structure is counted as a miss. It is left in note and not silently swallowed)
        out["status"] = "error"
        out["note"] = f"{type(e).__name__}: {e}"
        return out


# ---------------------------------------------------------------------------
# loading of the pack
# ---------------------------------------------------------------------------
def read_pack(layer: str):
    from ase.io import read as ase_read
    path = inputs_dir() / f"{layer}.extxyz"
    if not path.is_file():
        raise K.InputError(f"{path} not found (pack first)")
    frames = ase_read(str(path), index=":")
    out = []
    for a in frames:
        out.append({"info": dict(a.info), "frac": np.array(a.get_scaled_positions(wrap=False), dtype=float),
                    "cell": np.array(a.cell[:], dtype=float), "species": list(a.get_chemical_symbols())})
    return out


# ---------------------------------------------------------------------------
# tasks of the production run and R0 (fork Pool. The pack is held globally and only the index is passed)
# ---------------------------------------------------------------------------
_PACK = None
_REFS = None
_PARAMS = None


def _task(idx):
    fr = _PACK[idx]
    info = fr["info"]
    h, compute_s2 = _PARAMS["h"], _PARAMS["compute_s2"]
    res = analyze2(fr["frac"], fr["cell"], _REFS, info["gb_bin_seq"], h, M_WIN, compute_s2=compute_s2)
    # assert that axes agrees with the axes of the row. A mismatch is counted as an error
    if res.get("axes") is not None and res["axes"] != info["axes"] and res["status"] != "error":
        res["status"] = "error"
        res["note"] = f"axes {res['axes']} != row axes {info['axes']}"
    if int(info["natoms"]) != len(fr["frac"]):
        res["status"] = "error"
        res["note"] = f"natoms {len(fr['frac'])} != {info['natoms']}"
    return idx, res


def run_pool(pack, refs, h, compute_s2, procs):
    global _PACK, _REFS, _PARAMS
    _PACK, _REFS, _PARAMS = pack, refs, {"h": h, "compute_s2": compute_s2}
    idxs = list(range(len(pack)))
    out = [None] * len(pack)
    if procs <= 1:
        for i in idxs:
            j, r = _task(i)
            out[j] = r
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(min(procs, max(1, len(idxs)))) as pool:
            for k, (j, r) in enumerate(pool.imap_unordered(_task, idxs, chunksize=4)):
                out[j] = r
                if (k + 1) % 200 == 0:
                    _log(f"  {k + 1}/{len(idxs)}")
    return out


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, float) or isinstance(v, np.floating):
        return repr(float(v))
    return v


OUT_COLS = (["layer", "arm", "w", "chunk", "cif_index", "natoms", "axes", "nb", "z_AB", "z_BA", "t_A_A", "t_B_A",
             "thickness_differs_bins", "thickness_differs_A", "n_core_AB", "n_core_BA", "n_search_AB", "n_search_BA",
             "status", "note", "s2_max_A", "s2_op"]
            + [f"r_{op}_{d}" for op in OPS for d in ("AB2BA", "BA2AB")]
            + ["s1_max_A", "s1_op"]
            + [f"{k}_{t}" for t in THETA_ALL for k in ("eq1", "eq2", "eq12")])


def make_row(info, res):
    row = {"layer": info["layer"], "arm": info["arm"], "w": f"{float(info['w']):.1f}", "chunk": int(info["chunk"]),
           "cif_index": int(info["cif_index"]), "natoms": int(info["natoms"]), "axes": info["axes"],
           "s1_max_A": float(info["s1_max_A"]), "s1_op": info["s1_op"]}
    for k in ("nb", "z_AB", "z_BA", "t_A_A", "t_B_A", "thickness_differs_bins", "thickness_differs_A", "n_core_AB", "n_core_BA",
              "n_search_AB", "n_search_BA", "status", "note", "s2_max_A", "s2_op") + tuple(f"r_{op}_{d}" for op in OPS for d in ("AB2BA", "BA2AB")):
        if k in res:
            row[k] = res[k]
    ok = res["status"] == "ok" and "s2_max_A" in res
    for t in THETA_ALL:
        th = THETA_VAL[t]
        e1 = float(info["s1_max_A"]) <= th
        e2 = bool(ok and res["s2_max_A"] <= th)
        row[f"eq1_{t}"], row[f"eq2_{t}"], row[f"eq12_{t}"] = e1, e2, (e1 or e2)
    return row


def write_rows_csv(path: Path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=OUT_COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: _fmt(r.get(k, "")) for k in OUT_COLS})


def load_refs():
    return S.load_validation_refs(Path(K.P(f"{K.VRUN}/results/gb_equivalence")))


def require_validation_pass():
    p = out_root() / VALIDATION_NAME
    if not p.is_file():
        raise K.InputError(f"{p} not found: run --mode validation first")
    doc = json.loads(p.read_text())
    if doc.get("all_pass") is not True:
        raise K.InputError(f"{p}: all_pass != true (failed={doc.get('failed')}): the production run is refused")


def csv_name(layer, h):
    suf = "" if abs(h - H_MAIN) < 1e-12 else f"_h{int(round(h))}"
    return f"gb_stage2_{layer}{suf}.csv"


def mode_run(args):
    require_validation_pass()
    h = float(args.h)
    if not any(abs(h - x) < 1e-12 for x in (H_MAIN,) + H_SENS):
        raise K.InputError(f"--h must be one of {(H_MAIN,) + H_SENS} (see the plan)")
    refs = load_refs()
    layers = [x for x in args.layers.split(",") if x] if args.layers else list(LAYERS_PACK)
    for layer in layers:
        path = out_root() / csv_name(layer, h)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
    for layer in layers:
        pack = read_pack(layer)
        _log(f"run layer={layer} h={h}: frames={len(pack)} procs={args.procs}")
        t0 = time.time()
        res = run_pool(pack, refs, h, True, args.procs)
        rows = [make_row(fr["info"], r) for fr, r in zip(pack, res)]
        keys = [(r["layer"], r["arm"], r["w"], r["chunk"], r["cif_index"]) for r in rows]
        if len(rows) != len(pack) or len(set(keys)) != len(keys):          # check-D
            raise K.InputError(f"check-D: rows={len(rows)} frames={len(pack)} unique keys={len(set(keys))}")
        write_rows_csv(out_root() / csv_name(layer, h), rows)
        st = {}
        for r in rows:
            st[r["status"]] = st.get(r["status"], 0) + 1
        _log(f"  wrote {csv_name(layer, h)} rows={len(rows)} status={st} ({time.time() - t0:.0f} s)")
    return 0


# ---------------------------------------------------------------------------
# R0(§3-4-1)
# ---------------------------------------------------------------------------
def _q(vals, q):
    return float(np.quantile(np.array(vals, dtype=float), q)) if len(vals) else float("nan")


def mode_r0(args):
    refs = load_refs()
    md_path = out_root() / "dry_run_counts.md"
    if md_path.exists():
        raise FileExistsError(f"refusing to overwrite: {md_path}")
    cells = {}
    detail = {}
    t0 = time.time()
    for layer in LAYERS_PACK:
        pack = read_pack(layer)
        res = run_pool(pack, refs, H_MAIN, False, args.procs)
        for fr, r in zip(pack, res):
            info = fr["info"]
            key = (layer, info["arm"], f"{float(info['w']):.1f}")
            c = cells.setdefault(key, {"G2": 0, "ok": 0, "seq_mismatch": 0, "error": 0, "thin_grain": 0, "bins": 0, "A": 0, "dt": [], "tmin": []})
            c["G2"] += 1
            c[r["status"] if r["status"] in ("ok", "seq_mismatch", "error", "thin_grain") else "error"] += 1
            if "t_A_A" in r:
                c["bins"] += int(r["thickness_differs_bins"])
                c["A"] += int(r["thickness_differs_A"])
                c["dt"].append(abs(r["t_A_A"] - r["t_B_A"]))
                c["tmin"].append(min(r["t_A_A"], r["t_B_A"]))
            if r["status"] in ("seq_mismatch", "error"):
                detail.setdefault(key, []).append(f"{info['chunk']}/{info['cif_index']}: {r['status']} {r['note']}")
        _log(f"r0 layer={layer}: frames={len(pack)} ({time.time() - t0:.0f} s)")
    keys = sorted(cells, key=lambda k: (LAYERS_PACK.index(k[0]), k[1] != "proposed", float(k[2])))
    L = []
    A = L.append
    A("# Stage R0 (dry aggregation, before the decision)")
    A("")
    A(f"- Plan: `{DESIGN}`. h = {H_MAIN} A, m = {M_WIN} A (thin_grain threshold {2 * (H_MAIN + M_WIN)} A). host={socket.gethostname()} {time.strftime('%Y-%m-%d %H:%M:%S')}")
    A("- A row for which the recomputation of stage 0 does not agree with the reference bin sequence (gb_bin_seq of the stage-1 CSV) is seq_mismatch, and a row whose grain-boundary switches are not one each of A->B and B->A is error.")
    A("- thick_bins = comparison of the number of bins of theta_sensitivity.py:74-83, thick_A = |t_A - t_B| > 4.5 A. dt = |t_A - t_B|, tmin = min(t_A, t_B).")
    A("")
    A("| layer | arm | w | G2 | ok | seq_mismatch | error | thin_grain | thick_bins | thick_A | dt max [Å] | dt q90 [Å] | tmin min [Å] | tmin median [Å] |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        c = cells[k]
        A(f"| {k[0]} | {k[1]} | {k[2]} | {c['G2']} | {c['ok']} | {c['seq_mismatch']} | {c['error']} | {c['thin_grain']} | {c['bins']} | {c['A']} | "
          f"{(max(c['dt']) if c['dt'] else float('nan')):.3f} | {_q(c['dt'], 0.9):.3f} | {(min(c['tmin']) if c['tmin'] else float('nan')):.3f} | {_q(c['tmin'], 0.5):.3f} |")
    # stop conditions
    stops = []
    for k in keys:
        c = cells[k]
        if k[1] == "proposed":
            if c["thin_grain"] > 0:
                stops.append(f"proposed {k}: thin_grain={c['thin_grain']} > 0")
            if c["seq_mismatch"] > 0:
                stops.append(f"proposed {k}: seq_mismatch={c['seq_mismatch']} > 0")
            if c["dt"] and max(c["dt"]) > M_WIN:
                stops.append(f"proposed {k}: max|t_A-t_B|={max(c['dt']):.3f} > m={M_WIN}")
        else:
            if c["seq_mismatch"] + c["error"] > 2:
                stops.append(f"baseline {k}: seq_mismatch+error={c['seq_mismatch'] + c['error']} > 2")
            if k[0] in BASELINE_JUDGE and c["thin_grain"] > 5:
                stops.append(f"baseline judge layer {k}: thin_grain={c['thin_grain']} > 5")
    A("")
    A("## Stop conditions")
    A("- thin_grain > 0 or seq_mismatch > 0 in any layer of proposed")
    A("- seq_mismatch + error > 2 in any (layer, w) of baseline")
    A("- thin_grain > 5 in any of the 4 deciding layers of baseline (main, baseline_regen1 to 3)")
    A("- the maximum of |t_A - t_B| of proposed > m = 4.5 A")
    A("")
    A(f"**R0 VERDICT: {'STOP' if stops else 'CLEAR'}**" + ("" if not stops else f" ({len(stops)} cases)"))
    for s in stops:
        A(f"- STOP: {s}")
    if detail:
        A("")
        A("## Breakdown of seq_mismatch and error (first 5 per cell)")
        for k, v in detail.items():
            A(f"- {k}: " + " ; ".join(v[:5]))
    write_text_once(md_path, "\n".join(L) + "\n")
    _log(f"wrote {md_path}: R0 {'STOP' if stops else 'CLEAR'}")
    return 4 if stops else 0


# ---------------------------------------------------------------------------
# validation(§3-3)
# ---------------------------------------------------------------------------
def lattice_rcart(cell, axes, name="C2n"):
    nh, th, ph = S.lattice_frame(cell, axes)
    sn, st, sp = S.OPS_NTP[name]
    return sn * np.outer(nh, nh) + st * np.outer(th, th) + sp * np.outer(ph, ph)


def stage1_all_ops(frac, cell, axes):
    return {op: float(S.fit_op(frac, cell, S.op_diag(op, axes))["max_A"]) for op in OPS}


def perturb_c2n_symmetric(frac, cell, axes, zAB, core_half, amp, seed):
    """W1-2: uses the whole correspondence p(k) of C2n of B0 and gives U(-amp,amp)^3 A to each atom k of the core {|d_AB| <= core_half} and R_cart.u to its partner p(k)."""
    r = S.fit_op(frac, cell, S.op_diag("C2n", axes))
    rd = S.op_diag("C2n", axes)
    mapped = S._mic(frac * rd + r["t"])
    row, col, _res = S._hungarian_match(cell, mapped, frac)
    p = np.empty(len(frac), dtype=int)
    p[row] = col
    invol = bool(np.all(p[p] == np.arange(len(frac))))
    n = axes[0]
    Ln = float(np.linalg.norm(cell[n]))
    d_AB = Ln * S._mic(frac[:, n] - zAB)
    core = np.flatnonzero(np.abs(d_AB) <= core_half)
    rng = np.random.default_rng(seed)
    Rc = lattice_rcart(cell, axes, "C2n")
    disp = np.zeros((len(frac), 3))
    assigned = np.zeros(len(frac), dtype=bool)
    n_fixed = 0
    nh = S.lattice_frame(cell, axes)[0]
    for k in core:                                       # deterministic in ascending order
        u = rng.uniform(-amp, amp, 3)
        if assigned[k] or assigned[p[k]]:
            continue
        if p[k] == k:                                    # atoms on the two-fold axis: only the component (n direction) for which R u = u
            u = (u @ nh) * nh
            n_fixed += 1
        disp[k] = u
        disp[p[k]] = Rc @ u if p[k] != k else u
        assigned[k] = True
        assigned[p[k]] = True
    f2 = frac + disp @ np.linalg.inv(cell)
    return f2, {"n_core": int(len(core)), "n_displaced": int(assigned.sum()), "involution": invol, "n_fixed_points": n_fixed,
                "max_disp_A": float(np.abs(disp).max()), "c2n_fit_max_A_unperturbed": float(r["max_A"])}


def find_period(frac, cell, axes, zAB, zBA, Lp):
    """W1-4: period d of crystal grain A along n. Returns (d or None, info)."""
    n = axes[0]
    Ln = float(np.linalg.norm(cell[n]))
    u = Ln * ((frac[:, n] - zBA) % 1.0)
    tA = Ln * ((zAB - zBA) % 1.0)
    interior = (u >= W_EXCL_A) & (u <= tA - W_EXCL_A)
    d0 = Lp
    info = {"d0": d0, "L_p": Lp, "t_A_A": tA, "n_interior": int(interior.sum())}

    def pop_for(d):
        return interior & (u + d >= W_EXCL_A) & (u + d <= tA - W_EXCL_A)

    n0 = int(pop_for(d0).sum())
    info["n_pop_at_d0"] = n0
    if n0 < W_MIN_POP:
        info["reason"] = f"population_too_small({n0}<{W_MIN_POP})"
        return None, info
    ds = d0 + np.arange(-W_SCAN_A, W_SCAN_A + 1e-9, W_STEP_A)
    scan = []
    for d in ds:
        pop = pop_for(d)
        npop = int(pop.sum())
        if npop < W_MIN_POP:
            scan.append((float(d), npop, None))
            continue
        tgt = frac[pop].copy()
        tgt[:, n] += d / Ln
        D = _dist_matrix(cell, tgt, frac)
        rate = float(np.mean(D.min(axis=1) <= W_MATCH_TOL_A))
        scan.append((float(d), npop, rate))
    cand = [(d, npop, rate) for d, npop, rate in scan if rate is not None and rate >= W_MIN_RATE and abs(d - Lp) <= W_DEV_LP_A]
    info["n_scan"] = len(scan)
    info["scan_best_rate"] = max([r for _d, _n, r in scan if r is not None], default=None)
    info["n_pass"] = len(cand)
    if not cand:
        info["reason"] = "no_period_passes"
        return None, info
    best_rate = max(r for _d, _n, r in cand)
    top = [c for c in cand if c[2] == best_rate]
    d_sel = min(top, key=lambda c: abs(c[0] - d0))[0]
    pop = pop_for(d_sel)
    tgt = frac[pop].copy()
    tgt[:, n] += d_sel / Ln
    D = _dist_matrix(cell, tgt, frac)
    nn = D.argmin(axis=1)
    ok = D.min(axis=1) <= W_MATCH_TOL_A
    dn = Ln * S._mic(frac[nn[ok], n] - tgt[ok, n])
    d_ref = float(d_sel + dn.mean())
    info.update({"d_selected": d_sel, "rate_selected": best_rate, "n_pop_selected": int(pop.sum()), "d_refined": d_ref,
                 "refine_shift_A": float(dn.mean())})
    if abs(d_ref - Lp) > W_DEV_LP_A:
        info["reason"] = f"refined_d_off({d_ref:.4f} vs L_p {Lp:.4f})"
        return None, info
    return d_ref, info


def remove_center_layer(frac, cell, axes, zAB, zBA, D):
    """W1-5: removes [zc - D/2, zc + D/2) along n around the center zc of crystal grain A and closes the upper side by D. L_n -> L_n - D."""
    n = axes[0]
    Ln = float(np.linalg.norm(cell[n]))
    tA = Ln * ((zAB - zBA) % 1.0)
    zc = (zBA + 0.5 * tA / Ln) % 1.0
    s = Ln * ((frac[:, n] - (zc - 0.5 * D / Ln)) % 1.0)
    keep = s >= D
    Lnew = Ln - D
    f2 = frac[keep].copy()
    f2[:, n] = (s[keep] - D) / Lnew
    c2 = cell.copy()
    c2[n] = cell[n] * (Lnew / Ln)
    return f2, c2, {"D": float(D), "n_removed": int((~keep).sum()), "n_kept": int(keep.sum()), "zc": float(zc), "L_n_new": float(Lnew),
                    "removed_interval_A": [float(-0.5 * D), float(0.5 * D)]}


def _short(res):
    keys = ("status", "note", "gb0_status", "nb", "z_AB", "z_BA", "t_A_A", "t_B_A", "n_core_AB", "n_core_BA", "n_search_AB", "n_search_BA",
            "s2_max_A", "s2_op") + tuple(f"r_{op}_{d}" for op in OPS for d in ("AB2BA", "BA2AB"))
    return {k: res[k] for k in keys if k in res}


def w1_candidate(fr, ci, refs, h, m):
    """Performs steps 1 to 6 of W1 for one candidate structure. Returns (rec, B0 or None). When None, the reason is in rec['skipped']."""
    info = fr["info"]
    tag = f"chunk{info['chunk']}/cif{info['cif_index']}"
    rec = {"candidate": tag, "index": ci}
    frac, cell = fr["frac"], fr["cell"]
    axes = S.identify_axes(np.linalg.norm(cell, axis=1))
    assert S.axes_str(axes) == info["axes"], (S.axes_str(axes), info["axes"])
    Lp = float(np.linalg.norm(cell[axes[2]]))
    a0 = analyze2(frac, cell, refs, None, h, m, compute_s2=True, axes=axes)
    rec["B0_analysis"] = _short(a0)
    if a0["status"] != "ok":
        rec["skipped"] = f"B0 status={a0['status']}"
        return rec, None
    zAB, zBA = a0["z_AB"], a0["z_BA"]
    # steps 1 to 3: C2n symmetric perturbation
    f1, pinfo = perturb_c2n_symmetric(frac, cell, axes, zAB, W_PERT_CORE_A, W_PERT_AMP_W1, W_SEEDS["W1"])
    rec["perturb"] = pinfo
    s1 = stage1_all_ops(f1, cell, axes)
    rec["stage1_after_perturb"] = s1
    if not (s1["C2n"] <= 1e-4 and all(s1[o] >= 0.05 for o in ("C2t", "inv", "m_p"))):
        rec["skipped"] = "stage1 after perturb: need C2n<=1e-4 and C2t,inv,m_p>=0.05"
        return rec, None
    # step 4: period d
    d, dinfo = find_period(f1, cell, axes, zAB, zBA, Lp)
    rec["period"] = dinfo
    if d is None:
        rec["skipped"] = f"period: {dinfo.get('reason')}"
        return rec, None
    k = int(math.ceil(4.0 / d))
    D = k * d
    # step 5: removal of a layer
    f2, c2, rinfo = remove_center_layer(f1, cell, axes, zAB, zBA, D)
    rinfo["k"] = k
    rec["layer_removal"] = rinfo
    axes2 = S.identify_axes(np.linalg.norm(c2, axis=1))
    # step 6: check
    a1 = analyze2(f2, c2, refs, None, h, m, compute_s2=True, axes=axes2)
    s1b = stage1_all_ops(f2, c2, axes2)
    rec["W1_final_analysis"] = _short(a1)
    rec["W1_final_stage1"] = s1b
    step6 = {"n_removed_gt_0": rinfo["n_removed"] > 0, "gb0_two_gb": a1.get("gb0_status") == "two_gb",
             "abs_tA_tB_ge_3.5": ("t_A_A" in a1) and abs(a1["t_A_A"] - a1["t_B_A"]) >= 3.5,
             "stage1_all_ops_ge_0.05": all(v >= 0.05 for v in s1b.values()),
             # [Author's addition: a check that is not in the plan, W1-6] If the final structure is thin_grain, s2 is not evaluated (the expectation of W1-7 cannot be measured).
             # In a trial (a scan of the first 10 candidates), candidates appeared whose crystal grain A became 15.7 to 17.0 A after the layer removal, so this was added. Proceed to the next candidate and record the reason for skipping
             "final_not_thin_grain": a1["status"] != "thin_grain"}
    rec["step6"] = step6
    if not all(step6.values()):
        rec["skipped"] = f"step6 failed: {[kk for kk, vv in step6.items() if not vv]}"
        return rec, None
    B0 = {"fr": fr, "tag": tag, "axes": axes, "axes2": axes2, "d": d, "k": k, "D": D, "f1": f1, "f2": f2, "c2": c2, "a0": a0, "a1": a1, "rec": rec, "Lp": Lp}
    return rec, B0


def mode_validation(args):
    out_path = out_root() / VALIDATION_NAME
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite: {out_path}")
    refs = load_refs()
    h, m = H_MAIN, M_WIN
    doc = {"design": DESIGN, "script": os.path.abspath(__file__), "host": socket.gethostname(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "constants": {"h": h, "m": m, "thin_grain_A": 2 * (h + m), "seeds": W_SEEDS, "pert_core_A": W_PERT_CORE_A, "amp_W1": W_PERT_AMP_W1,
                         "amp_W2": W_PERT_AMP_W2, "excl_A": W_EXCL_A, "scan_A": W_SCAN_A, "step_A": W_STEP_A, "match_tol_A": W_MATCH_TOL_A,
                         "min_pop": W_MIN_POP, "min_rate": W_MIN_RATE, "dev_Lp_A": W_DEV_LP_A, "W3_shift_A": W_W3_SHIFT_A,
                         "refs_deg": refs},
           "candidates_skipped": [], "results": {}, "failed": None, "all_pass": False}

    def finish(failed=None):
        doc["failed"] = failed
        doc["all_pass"] = failed is None and all(doc["results"].get(w, {}).get("pass") for w in ("W1", "W2", "W3", "W4", "W5", "W6"))
        doc["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json_once(out_path, doc)
        _log(f"wrote {out_path}: all_pass={doc['all_pass']} failed={failed}")
        return 0 if doc["all_pass"] else 3

    pack = read_pack("proposed_regen1")
    pack.sort(key=lambda fr: (int(fr["info"]["chunk"]), int(fr["info"]["cif_index"])))
    cands = [fr for fr in pack if float(fr["info"]["s1_max_A_C2n"]) <= 1e-4]
    _log(f"proposed_regen1 pack frames={len(pack)} candidates(s1_max_A_C2n<=1e-4)={len(cands)}")
    doc["n_candidates_available"] = len(cands)

    B0 = None
    for ci, fr in enumerate(cands[:W_MAX_CANDS]):
        rec, B0 = w1_candidate(fr, ci, refs, h, m)
        if B0 is None:
            doc["candidates_skipped"].append(rec)
            continue
        break
    doc["B0_selected"] = None if B0 is None else B0["tag"]
    if B0 is None:
        doc["results"]["W1"] = {"pass": False, "reason": f"all {min(len(cands), W_MAX_CANDS)} candidates skipped (all W1 candidates are lost: no retry and report)"}
        return finish("W1")
    doc["B0_record"] = B0["rec"]
    frac0, cell0, axes0 = B0["fr"]["frac"], B0["fr"]["cell"], B0["axes"]
    f2, c2, axes2, a1 = B0["f2"], B0["c2"], B0["axes2"], B0["a1"]

    # W1 expectation: s2 <= 1e-3, op=C2n
    w1_ok = (a1["status"] == "ok" and a1["s2_max_A"] <= 1e-3 and a1["s2_op"] == "C2n")
    doc["results"]["W1"] = {"pass": bool(w1_ok), "expect": "s2<=1e-3 & op=C2n & stage1 4 ops>=0.05 & d=L_p±0.15", "d_refined": B0["d"], "L_p": B0["Lp"],
                            "d_minus_Lp": B0["d"] - B0["Lp"], "k": B0["k"], "D": B0["D"], "n_removed": B0["rec"]["layer_removal"]["n_removed"],
                            "n_atoms_final": int(len(f2)), "analysis": _short(a1), "stage1_ops_final": B0["rec"]["W1_final_stage1"],
                            "candidate": B0["tag"], "n_candidates_skipped": len(doc["candidates_skipped"])}

    # W2: iid U(-0.30, 0.30)^3 A on the inner core of GB_BA. The position of the window uses z of the final W1 structure as it is (s2 is the same local_s2)
    zAB1, zBA1 = a1["z_AB"], a1["z_BA"]
    Ln1 = float(np.linalg.norm(c2[axes2[0]]))
    d_BA1 = Ln1 * S._mic(f2[:, axes2[0]] - zBA1)
    core2 = np.flatnonzero(np.abs(d_BA1) <= W_PERT_CORE_A)
    rng2 = np.random.default_rng(W_SEEDS["W2"])
    disp2 = rng2.uniform(-W_PERT_AMP_W2, W_PERT_AMP_W2, (len(core2), 3))
    f_w2 = f2.copy()
    f_w2[core2] += disp2 @ np.linalg.inv(c2)
    r_w2 = local_s2(f_w2, c2, axes2, zAB1, zBA1, h, m)
    a_w2 = analyze2(f_w2, c2, refs, a1["gb_bin_seq_recomputed"], h, m, compute_s2=True, axes=axes2)
    doc["results"]["W2"] = {"pass": bool(r_w2["s2_max_A"] >= 0.1), "expect": "s2>=0.1", "n_perturbed": int(len(core2)), "s2_max_A": r_w2["s2_max_A"],
                            "s2_op": r_w2["s2_op"], "local_s2": {k: r_w2[k] for k in r_w2},
                            "full_pipeline_status_with_W1_ref_seq": _short(a_w2),
                            "note": "s2 is measured with local_s2 with the positions z_AB and z_BA of the final W1 structure fixed at the window position (so that s2 can be evaluated even if the bin sequence changes by the perturbation)"}

    # W3: move the center of the window to z_BA + 8.0 A (inside crystal grain A)
    r_w3 = local_s2(f2, c2, axes2, zAB1, (zBA1 + W_W3_SHIFT_A / Ln1) % 1.0, h, m)
    doc["results"]["W3"] = {"pass": bool(r_w3["s2_max_A"] >= 0.1), "expect": "s2>=0.1", "shift_A": W_W3_SHIFT_A, "s2_max_A": r_w3["s2_max_A"],
                            "s2_op": r_w3["s2_op"], "local_s2": r_w3}

    # W4: invariance
    rng4 = np.random.default_rng(W_SEEDS["W4"])
    nb1 = a1["nb"]
    k_n = int(rng4.integers(0, nb1))
    shift = np.zeros(3)
    shift[axes2[1]], shift[axes2[2]] = rng4.uniform(0.0, 1.0, 2)
    shift[axes2[0]] = k_n / nb1
    perm = rng4.permutation(len(f2))
    f_w4 = (f2 + shift)[perm]
    ref_roll = a1["gb_bin_seq_recomputed"]
    ref_roll = ref_roll[-k_n:] + ref_roll[:-k_n] if k_n else ref_roll
    a_w4 = analyze2(f_w4, c2, refs, ref_roll, h, m, compute_s2=True, axes=axes2)
    if a_w4["status"] == "ok" and a1["status"] == "ok":
        dd = abs(a_w4["s2_max_A"] - a1["s2_max_A"])
        w4_ok = bool(dd <= 1e-9 and a_w4["s2_op"] == a1["s2_op"])
    else:
        dd, w4_ok = None, False
    doc["results"]["W4"] = {"pass": w4_ok, "expect": "|ds2|<=1e-9 & s2_op unchanged", "k_n": k_n, "nb": nb1, "shift_frac": shift.tolist(), "abs_delta_s2": dd,
                            "s2_before": a1.get("s2_max_A"), "s2_after": a_w4.get("s2_max_A"), "op_before": a1.get("s2_op"), "op_after": a_w4.get("s2_op"),
                            "analysis_after": _short(a_w4)}
    # W4b (report only): the n component is also continuous and uniform
    rng4b = np.random.default_rng(W_SEEDS["W4"] + 1)
    shift_b = rng4b.uniform(0.0, 1.0, 3)
    f_w4b = (f2 + shift_b)[rng4b.permutation(len(f2))]
    a_w4b = analyze2(f_w4b, c2, refs, None, h, m, compute_s2=True, axes=axes2)
    doc["results"]["W4b"] = {"pass": None, "report_only": True, "shift_frac": shift_b.tolist(), "status": a_w4b["status"],
                             "abs_delta_s2": (abs(a_w4b["s2_max_A"] - a1["s2_max_A"]) if (a_w4b["status"] == "ok" and a1["status"] == "ok") else None),
                             "analysis": _short(a_w4b), "note": "the reference bin sequence is taken from the structure itself (because the bin phase changes with a continuous translation). Not used for decisions"}

    # W5: firing of thin_grain (D = 2d from the undeformed structure of B0)
    f5, c5, rinfo5 = remove_center_layer(frac0, cell0, axes0, B0["a0"]["z_AB"], B0["a0"]["z_BA"], 2.0 * B0["d"])
    axes5 = S.identify_axes(np.linalg.norm(c5, axis=1))
    a5 = analyze2(f5, c5, refs, None, h, m, compute_s2=True, axes=axes5)
    w5_ok = bool(a5.get("gb0_status") == "two_gb" and a5["status"] == "thin_grain" and min(a5["t_A_A"], a5["t_B_A"]) < 2 * (h + m))
    doc["results"]["W5"] = {"pass": w5_ok, "expect": "gb0=two_gb & status=thin_grain", "layer_removal": rinfo5, "min_t_A": min(a5["t_A_A"], a5["t_B_A"]) if "t_A_A" in a5 else None,
                            "analysis": _short(a5)}

    # W6: undeformed B0
    a6 = B0["a0"]
    w6_ok = bool(a6["status"] == "ok" and a6["s2_max_A"] <= 1e-3)
    doc["results"]["W6"] = {"pass": w6_ok, "expect": "s2<=1e-3", "candidate": B0["tag"], "analysis": _short(a6)}
    return finish(None if all(doc["results"][w]["pass"] for w in ("W1", "W2", "W3", "W4", "W5", "W6")) else
                  next(w for w in ("W1", "W2", "W3", "W4", "W5", "W6") if not doc["results"][w]["pass"]))


# ---------------------------------------------------------------------------
# decision function
# ---------------------------------------------------------------------------
def wilson_lower(k, n, z=1.96):
    if n <= 0:
        return float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return center - half


def stage2_checks(k, k_near, p1_ok, p2_ok, n=N_REQ):
    """Check C1 of the plan. k and k_near are the counts of the 4 deciding layers (main, baseline_regen1, baseline_regen2, baseline_regen3). Order of decisions: FAIL -> PASS -> FLAG (single master).
    Returns: the verdict, the Wilson lower limit and k+k_near of each layer, and the reasons a/b/c of FLAG."""
    k, k_near = list(k), list(k_near)
    wl = [wilson_lower(x, n) for x in k]
    fail_layers = [i for i, w in enumerate(wl) if w > WILSON_FAIL_P]
    sums = [a + b for a, b in zip(k, k_near)]
    if fail_layers:
        verdict, reasons = "FAIL", []
    elif all(s <= G1_K_PASS for s in sums) and p1_ok and p2_ok:
        verdict, reasons = "PASS", []
    else:
        verdict = "FLAG"
        reasons = []
        if any(18 <= x for x in k):
            reasons.append("a")        # k_r is 18 or more (not FAIL)
        if not (p1_ok and p2_ok):
            reasons.append("b")        # P1 and P2 do not hold
        if any(x <= G1_K_PASS and s > G1_K_PASS for x, s in zip(k, sums)):
            reasons.append("c")        # k_r + k_near_r > 17
    return {"verdict": verdict, "flag_reasons": reasons, "k": k, "k_near": k_near, "k_plus_near": sums, "wilson_lower": wl,
            "fail_layers": fail_layers, "check_p1": bool(p1_ok), "check_p2": bool(p2_ok), "n": n}


def mode_self_check_gate(args):
    out_path = out_root() / "self_check_gate.json"
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite: {out_path}")
    cases = [
        ("PASS", dict(k=(0, 3, 1, 1), k_near=(0, 0, 0, 0), p1_ok=True, p2_ok=True), "PASS", []),
        ("FAIL", dict(k=(0, 30, 1, 1), k_near=(0, 0, 0, 0), p1_ok=True, p2_ok=True), "FAIL", []),
        ("FLAG(a)", dict(k=(0, 20, 1, 1), k_near=(0, 0, 0, 0), p1_ok=True, p2_ok=True), "FLAG", ["a"]),
        ("FLAG(b)", dict(k=(0, 3, 1, 1), k_near=(0, 0, 0, 0), p1_ok=False, p2_ok=True), "FLAG", ["b"]),
        ("FLAG(c)", dict(k=(0, 3, 1, 1), k_near=(0, 15, 0, 0), p1_ok=True, p2_ok=True), "FLAG", ["c"]),
        # boundary (cross-check): k=17 is PASS, k=17 with k_near=1 is FLAG (c), k=25 is FLAG (a), k=26 is FAIL
        ("boundary k=17", dict(k=(0, 17, 0, 0), k_near=(0, 0, 0, 0), p1_ok=True, p2_ok=True), "PASS", []),
        ("boundary k=17,k_near=1", dict(k=(0, 17, 0, 0), k_near=(0, 1, 0, 0), p1_ok=True, p2_ok=True), "FLAG", ["c"]),
        ("boundary k=25", dict(k=(0, 25, 0, 0), k_near=(0, 0, 0, 0), p1_ok=True, p2_ok=True), "FLAG", ["a"]),
        ("boundary k=26", dict(k=(0, 26, 0, 0), k_near=(0, 0, 0, 0), p1_ok=True, p2_ok=True), "FAIL", []),
        ("P2 does not hold", dict(k=(0, 3, 1, 1), k_near=(0, 0, 0, 0), p1_ok=True, p2_ok=False), "FLAG", ["b"]),
    ]
    results, allok = [], True
    for name, kw, exp_v, exp_r in cases:
        g = stage2_checks(**kw)
        ok = g["verdict"] == exp_v and g["flag_reasons"] == exp_r
        allok = allok and ok
        results.append({"case": name, "input": {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in kw.items()}, "expected_verdict": exp_v,
                        "expected_flag_reasons": exp_r, "got": g, "pass": bool(ok)})
    wl = {kk: wilson_lower(kk, N_REQ) for kk in (17, 20, 25, 26, 30)}
    # expected Wilson lower limits: k=17 -> 0.0305, k=20 -> 0.0373, k=26 -> 0.0512, k=30 -> 0.0607
    wl_exp = {17: 0.0305, 20: 0.0373, 26: 0.0512, 30: 0.0607}
    wl_ok = all(abs(wl[kk] - v) < 5e-4 for kk, v in wl_exp.items())
    allok = allok and wl_ok
    # check with synthetic inputs of P1, P2 and the aggregation function
    agg_ok, agg_info = _self_check_aggregate()
    allok = allok and agg_ok
    doc = {"design": DESIGN, "host": socket.gethostname(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": results,
           "wilson_lower": {str(kk): v for kk, v in wl.items()}, "wilson_expected_REV_3a": {str(kk): v for kk, v in wl_exp.items()}, "wilson_ok": bool(wl_ok),
           "aggregate_self_check": agg_info, "all_pass": bool(allok),
           "verdicts_seen": sorted({r["got"]["verdict"] + ("(" + "".join(r["got"]["flag_reasons"]) + ")" if r["got"]["flag_reasons"] else "") for r in results})}
    write_json_once(out_path, doc)
    _log(f"wrote {out_path}: all_pass={allok} verdicts={doc['verdicts_seen']}")
    return 0 if allok else 3


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
LOG_EDGES = [10 ** (-6 + 0.5 * i) for i in range(13)]      # 1e-6 .. 1 in half-decade bins (12 bins)


def _b(x):
    return str(x).strip() in ("1", "True", "true")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def aggregate_cell(rows):
    """Makes the count table from the rows (a list of dicts read from the CSV) of one (layer, arm, w)."""
    g2 = len(rows)
    st, ops = {}, {}
    cell = {"G2": g2, "n_req": N_REQ}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
        if r["status"] == "ok":
            ops[r["s2_op"]] = ops.get(r["s2_op"], 0) + 1
    cell["status"], cell["s2_op"] = st, ops
    for kind in ("eq1", "eq2", "eq12"):
        cell[f"E{kind[2:]}"] = {t: sum(1 for r in rows if _b(r[f"{kind}_{t}"])) for t in THETA_ALL}
    s2 = [(_f(r["s2_max_A"]), r) for r in rows if r["status"] == "ok"]
    cell["k_near"] = sum(1 for v, r in s2 if v is not None and K_NEAR_LO < v <= K_NEAR_HI and not _b(r["eq1_0.1"]))
    cell["p2_violations"] = {t: sum(1 for v, r in s2 if v is not None and _b(r[f"eq1_{t}"]) and v > 2 * THETA_VAL[t]) for t in ("0.064", "0.1")}
    cell["eq1_not_eq2"] = {t: sum(1 for v, r in s2 if _b(r[f"eq1_{t}"]) and not _b(r[f"eq2_{t}"])) for t in THETA_DECISION}
    vals = [v for v, _r in s2 if v is not None]
    cell["hist_log"] = {"edges": LOG_EDGES, "underflow(<1e-6)": sum(1 for v in vals if v < LOG_EDGES[0]),
                        "counts": [sum(1 for v in vals if LOG_EDGES[i] <= v < LOG_EDGES[i + 1]) for i in range(len(LOG_EDGES) - 1)],
                        "overflow(>=1)": sum(1 for v in vals if v >= LOG_EDGES[-1])}
    cell["thickness_differs_bins"] = sum(1 for r in rows if _b(r["thickness_differs_bins"]))
    cell["thickness_differs_A"] = sum(1 for r in rows if _b(r["thickness_differs_A"]))
    cell["k_over_G2"] = (cell["E12"]["0.1"] / g2) if g2 else None
    return cell


def check_p1(cells_proposed):
    """cells_proposed: only proposed of {(layer,arm,w): cell}. In each cell #(status=ok)=G2 and E2(theta)=G2 (theta: 6 values of 1e-3 or more of the decision grid)."""
    thetas = ["1e-3", "3e-3", "1e-2", "0.032", "0.064", "0.1"]
    per = {}
    ok = True
    for key, c in cells_proposed.items():
        good = c["status"].get("ok", 0) == c["G2"] and all(c["E2"][t] == c["G2"] for t in thetas)
        per["/".join(key)] = {"G2": c["G2"], "n_ok": c["status"].get("ok", 0), "E2": {t: c["E2"][t] for t in thetas}, "pass": bool(good)}
        ok = ok and good
    return bool(ok), per


def check_p2(cells_all):
    tot = {t: sum(c["p2_violations"][t] for c in cells_all.values()) for t in ("0.064", "0.1")}
    return all(v == 0 for v in tot.values()), tot


def aggregate_all(rows_by_layer):
    """rows_by_layer: {layer: [row dict]}. Returns (cells, gate_inputs, preview)."""
    cells = {}
    for layer, rows in rows_by_layer.items():
        groups = {}
        for r in rows:
            groups.setdefault((r["arm"], f"{float(r['w']):.1f}"), []).append(r)
        for (arm, w), rr in groups.items():
            cells[(layer, arm, w)] = aggregate_cell(rr)
    proposed_cells = {k: c for k, c in cells.items() if k[1] == "proposed"}
    p1, p1_detail = check_p1(proposed_cells)
    p2, p2_tot = check_p2(cells)
    k, kn, g2s = [], [], []
    for layer in BASELINE_JUDGE:
        c = cells[(layer, "baseline", "2.0")]
        k.append(c["E12"]["0.1"])
        kn.append(c["k_near"])
        g2s.append(c["G2"])
    gate_in = {"layers": BASELINE_JUDGE, "k": k, "k_near": kn, "G2": g2s, "k_over_G2": [a / b if b else None for a, b in zip(k, g2s)],
               "e_r": [a / N_REQ for a in k], "C1": {"pass": p1, "per_cell": p1_detail}, "C2": {"pass": p2, "violations": p2_tot},
               "excluded_counts": {layer: {s: v for s, v in cells[(layer, "baseline", "2.0")]["status"].items() if s != "ok"} for layer in BASELINE_JUDGE}}
    preview = stage2_checks(k, kn, p1, p2)
    return cells, gate_in, preview


def _self_check_aggregate():
    """Runs aggregate_all really with synthetic inputs and confirms that the counts are as constructed (finds errors of the aggregation before the production run)."""
    def mk(layer, arm, w, i, s1, status, s2):
        row = {"layer": layer, "arm": arm, "w": w, "chunk": 0, "cif_index": i, "status": status, "s2_max_A": "" if s2 is None else repr(s2),
               "s2_op": "C2n" if s2 is not None else "", "thickness_differs_bins": "1", "thickness_differs_A": "0"}
        for t in THETA_ALL:
            th = THETA_VAL[t]
            e1 = s1 <= th
            e2 = bool(status == "ok" and s2 is not None and s2 <= th)
            row[f"eq1_{t}"], row[f"eq2_{t}"], row[f"eq12_{t}"] = int(e1), int(e2), int(e1 or e2)
        return row
    rows_by_layer = {}
    # proposed layers: all ok, s2=1e-5, s1=1e-5
    for layer in ("main", "proposed_regen1", "proposed_regen2"):
        rows_by_layer.setdefault(layer, [])
        rows_by_layer[layer] += [mk(layer, "proposed", "2.0", i, 1e-5, "ok", 1e-5) for i in range(5)]
    # baseline: 6 rows each = [1 row with s1<=theta only (eq1 only), 1 with s2<=theta only, 1 in the near band s2=0.2, 1 far s2=1.0, 1 thin_grain, 1 seq_mismatch]
    for layer in ("main", "baseline_regen1", "baseline_regen2", "baseline_regen3"):
        rows_by_layer.setdefault(layer, [])
        rows_by_layer[layer] += [mk(layer, "baseline", "2.0", 0, 0.05, "ok", 0.06), mk(layer, "baseline", "2.0", 1, 0.5, "ok", 0.001),
                                 mk(layer, "baseline", "2.0", 2, 0.5, "ok", 0.2), mk(layer, "baseline", "2.0", 3, 0.5, "ok", 1.0),
                                 mk(layer, "baseline", "2.0", 4, 0.5, "thin_grain", None), mk(layer, "baseline", "2.0", 5, 0.5, "seq_mismatch", None)]
    cells, gate_in, preview = aggregate_all(rows_by_layer)
    c = cells[("main", "baseline", "2.0")]
    checks = {
        "G2=6": c["G2"] == 6,
        "E1(0.1)=1": c["E1"]["0.1"] == 1,
        "E2(0.1)=2": c["E2"]["0.1"] == 2,
        "E12(0.1)=2 (the row with eq1 only and the row with eq2 only)": c["E12"]["0.1"] == 2,
        "k_near=1 (the row with s2=0.2)": c["k_near"] == 1,
        "status thin_grain=1,seq_mismatch=1": c["status"].get("thin_grain") == 1 and c["status"].get("seq_mismatch") == 1,
        "total of hist=4 (the 4 rows with ok)": sum(c["hist_log"]["counts"]) + c["hist_log"]["underflow(<1e-6)"] + c["hist_log"]["overflow(>=1)"] == 4,
        "P2: s2=0.06<=0.128 with s1=0.05<=0.064 -> 0 violations": gate_in["C2"]["pass"] and c["p2_violations"]["0.064"] == 0,
        "P1 holds": gate_in["C1"]["pass"],
        "preview verdict=PASS(k=2,k_near=1)": preview["verdict"] == "PASS",
    }
    # validation of a violation: s1=0.05 and s2=0.5 -> 1 case of eq1(0.064) and s2 > 2*theta=0.128
    bad = {"main": [mk("main", "baseline", "2.0", 0, 0.05, "ok", 0.5)]}
    cb = aggregate_cell(bad["main"])
    checks["P2 violation validation: detects 1 case of eq1(0.064) and s2>0.128"] = cb["p2_violations"]["0.064"] == 1 and cb["eq1_not_eq2"]["0.064"] == 1
    return all(checks.values()), {"checks": checks, "main_baseline_cell": {k_: c[k_] for k_ in ("G2", "k_near", "status", "s2_op", "thickness_differs_bins")}}


def read_layer_csv(layer, h):
    path = out_root() / csv_name(layer, h)
    if not path.is_file():
        raise K.InputError(f"{path} not found (run --mode run{'' if abs(h - H_MAIN) < 1e-12 else f' --h {h}'} first)")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mode_aggregate(args):
    out_json = out_root() / "stage2_check_inputs.json"
    out_md = out_root() / "gb_stage2_summary.md"
    for p in (out_json, out_md):
        if p.exists():
            raise FileExistsError(f"refusing to overwrite: {p}")
    rows4 = {layer: read_layer_csv(layer, H_MAIN) for layer in LAYERS_PACK}
    rows_sens = {h: {layer: read_layer_csv(layer, h) for layer in LAYERS_PACK} for h in H_SENS}
    cells, gate_in, preview = aggregate_all(rows4)
    sens_cells = {h: aggregate_all(rows_sens[h])[0] for h in H_SENS}
    proposed_pos = {}
    for layer in LAYERS_PACK:
        for (lyr, arm, w), c in cells.items():
            if lyr == layer and arm == "proposed":
                proposed_pos[f"{lyr}/w{w}"] = {"G2": c["G2"], "thickness_differs_bins": c["thickness_differs_bins"], "thickness_differs_A": c["thickness_differs_A"],
                                          "rate_bins": c["thickness_differs_bins"] / c["G2"], "rate_A": c["thickness_differs_A"] / c["G2"]}
    doc = {"design": DESIGN, "h": H_MAIN, "m": M_WIN, "n_req": N_REQ, "gate_inputs": gate_in, "preview": preview,
           "cells": {"/".join(k): v for k, v in sorted(cells.items())},
           "sensitivity_E12": {f"h{int(h)}": {"/".join(k): {"G2": c["G2"], "E12": c["E12"]} for k, c in sorted(sc.items())} for h, sc in sens_cells.items()},
           "proposed_thickness_false_positive_rates": proposed_pos}
    write_json_once(out_json, doc)
    L = []
    A = L.append
    A("# Stage 2 aggregation (h=4.0 main, m=4.5)")
    A("")
    A(f"- Plan: `{DESIGN}`. Raw output: `{out_json}`. n_req={N_REQ} fixed. seq_mismatch, error and thin_grain are counted as not equivalent (misses).")
    A("- **The decision is made by a separate check.** The preview below is the output of `stage2_checks()` (interpretation by the author).")
    A("")
    A("## Input of C1 (baseline, w=2.0)")
    A("| layer | G2 | k=E12(0.1) | k/350 | k/G2 | k_near | k+k_near | Wilson95 lower limit | status (other than ok) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for i, layer in enumerate(BASELINE_JUDGE):
        A(f"| {layer} | {gate_in['G2'][i]} | {gate_in['k'][i]} | {gate_in['e_r'][i]:.4f} | {gate_in['k_over_G2'][i]:.3f} | {gate_in['k_near'][i]} | "
          f"{preview['k_plus_near'][i]} | {preview['wilson_lower'][i]:.4f} | {gate_in['excluded_counts'][layer]} |")
    A("")
    A(f"- P1 (positive control of proposed, stage 2 alone): **{gate_in['C1']['pass']}**, P2 (consistency): **{gate_in['C2']['pass']}** (number of violations {gate_in['C2']['violations']})")
    A(f"- preview: **{preview['verdict']}** (flag_reasons={preview['flag_reasons']})")
    A("")
    A("## All cells (layer / arm / w)")
    th_show = ["1e-3", "0.032", "0.064", "0.1", "0.3"]
    A("| cell | G2 | status | " + " | ".join(f"E1/E2/E12({t})" for t in th_show) + " | k_near | #(eq1∧¬eq2)(0.064/0.1) | #(eq1∧s2>2θ)(0.064/0.1) | thick_bins | thick_A | s2_op |")
    A("|---|---|---|" + "---|" * (len(th_show) + 6))
    for k_, c in sorted(cells.items(), key=lambda kv: (LAYERS_PACK.index(kv[0][0]), kv[0][1] != "proposed", float(kv[0][2]))):
        A(f"| {'/'.join(k_)} | {c['G2']} | {c['status']} | " + " | ".join(f"{c['E1'][t]}/{c['E2'][t]}/{c['E12'][t]}" for t in th_show)
          + f" | {c['k_near']} | {c['eq1_not_eq2']['0.064']}/{c['eq1_not_eq2']['0.1']} | {c['p2_violations']['0.064']}/{c['p2_violations']['0.1']} | "
          f"{c['thickness_differs_bins']} | {c['thickness_differs_A']} | {c['s2_op']} |")
    A("")
    A("## Report only: E12 at theta=0.064, E12(0.1) at h=3.0 and 6.0 (counts within G2)")
    A("| cell | G2 | E12(0.064) h=4 | E12(0.1) h=3 | E12(0.1) h=4 | E12(0.1) h=6 |")
    A("|---|---|---|---|---|---|")
    for k_, c in sorted(cells.items(), key=lambda kv: (LAYERS_PACK.index(kv[0][0]), kv[0][1] != "proposed", float(kv[0][2]))):
        A(f"| {'/'.join(k_)} | {c['G2']} | {c['E12']['0.064']} | {sens_cells[3.0][k_]['E12']['0.1']} | {c['E12']['0.1']} | {sens_cells[6.0][k_]['E12']['0.1']} |")
    A("")
    A("## Positive rate of the two definitions of thickness for proposed (false-positive rate. The two grain boundaries of proposed are truly equivalent, so 0 is ideal)")
    A("| cell | G2 | thick_bins | rate | thick_A | rate |")
    A("|---|---|---|---|---|---|")
    for k_, v in proposed_pos.items():
        A(f"| {k_} | {v['G2']} | {v['thickness_differs_bins']} | {v['rate_bins']:.3f} | {v['thickness_differs_A']} | {v['rate_A']:.3f} |")
    A("")
    A("## Histogram of s2 (status=ok, logarithmic half-decade bins 1e-6 to 1 A)")
    A("| cell | <1e-6 | " + " | ".join(f"[{LOG_EDGES[i]:.0e},{LOG_EDGES[i + 1]:.0e})" for i in range(len(LOG_EDGES) - 1)) + " | >=1 |")
    A("|---|---|" + "---|" * (len(LOG_EDGES) - 1) + "---|")
    for k_, c in sorted(cells.items(), key=lambda kv: (LAYERS_PACK.index(kv[0][0]), kv[0][1] != "proposed", float(kv[0][2]))):
        hh = c["hist_log"]
        A(f"| {'/'.join(k_)} | {hh['underflow(<1e-6)']} | " + " | ".join(str(x) for x in hh["counts"]) + f" | {hh['overflow(>=1)']} |")
    write_text_once(out_md, "\n".join(L) + "\n")
    _log(f"wrote {out_json} and {out_md}; preview verdict={preview['verdict']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["validation", "r0", "run", "aggregate", "self_check-gate"])
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--h", type=float, default=H_MAIN)
    ap.add_argument("--layers", default="", help="narrow the layers only for run (default: all 7 layers)")
    ap.add_argument("--out-root", default="", help="for tests only: replaces the output destination (not given in the production run). The inputs are read from the default location")
    args = ap.parse_args()
    global _OUT_ROOT_OVERRIDE
    if args.out_root:
        _OUT_ROOT_OVERRIDE = args.out_root
    fn = {"validation": mode_validation, "r0": mode_r0, "run": mode_run, "aggregate": mode_aggregate, "self_check-gate": mode_self_check_gate}[args.mode]
    try:
        return fn(args)
    except FileExistsError as e:
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        return 2
    except K.InputError as e:
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        return 2
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
