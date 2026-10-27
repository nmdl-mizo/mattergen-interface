#!/usr/bin/env python3
"""gb_symfit.py -- stages 0 and 1: measures whether the two grain boundaries in the cell of a generated Al structure are equivalent (residual of the 2_1 screw fit).

Plan: see the experiment description.
  Stage 0: count_gb  = divides the atoms into A/B by the orientation of in-plane bonds and counts the number of grain boundaries n_gb from the sequence of layers along the normal
  Stage 1: gb_symfit = maps the structure onto itself with four operations ops (C2n, m_p, C2t, inv) and gives the residual of the fit with free translation (set matching = Hungarian)
  The decision (EQUIV/INEQUIV) is not given until the threshold theta is frozen. The outputs are only continuous quantities and the classification (status).

Usage (compute node; conda python of the evaluation environment with numpy, scipy, ase; no MLIP is used = CPU):
  python gb_symfit.py --mode validation --out-dir DIR   : runs the self-tests W1 to W11 in order and writes DIR/gb_symfit_validation.json.
                                                          If any of them fails it writes and exits non-zero at that point (thresholds, formulas and parameters are not changed)
  python gb_symfit.py --mode main    --out-dir DIR [--procs N] : production computation (700 rows x {pre,post}). DIR/gb_symfit_main.csv and gb_symfit_summary.md.
                                                          Refuses to start unless gb_symfit_validation.json is all PASS
  python gb_symfit.py --mode additional    --out-dir DIR [--procs N] : additional layer (w=1 to 4, 2,800 structures). DIR/gb_symfit_additional.csv and gb_symfit_summary_additional.md.
                                                          The checks of validation.json, theta and the W series are the same as for main (the instrument is not changed)
Exit codes: 0 = success / 2 = failure of the inputs or checks (number of rows, keys, validation not satisfied, existing output, etc.) / 3 = validation FAIL
Outputs are never overwritten (if they exist, FileExistsError -> exit 2).

The functions of manifest.py (mattergen/symmetry_constrained/manifest.py) are copied verbatim instead of imported (whether the import works on compute nodes
was not measured). Inputs are resolved with the same common (K) as build_manifests.py (settings are not copied).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

DESIGN = '<ORIGINAL_ROOT>/an internal design note that is not included'

# ---------------------------------------------------------------------------
# preregistered parameters (do not change)
# ---------------------------------------------------------------------------
BOND_CUT_A = 3.4            # minimum-image distance within 3.4 A counts as a neighbor bond
INPLANE_T_A = 0.5           # bonds whose t-component has absolute value below 0.5 A are in-plane bonds
MIN_INPLANE_BONDS = 3       # only atoms with 3 or more in-plane bonds get an orientation
LABEL_WIN_DEG = 8.0         # within 8 deg of a reference orientation: A or B
BIN_WIDTH_A = 2.0           # bin width along the normal direction
BIN_DOMINANCE = 0.6         # A if nA >= 0.6 (nA+nB) (and likewise B)
BIN_MIN_LABELED = 2         # §3-0-4: nA+nB >= 2
LABELED_MIN_FRAC = 0.5      # disordered if fewer than 50% of the atoms are labeled
T_LEN_A = 4.04              # t is the edge of 4.04+-0.15 A
T_LEN_TOL_A = 0.15
A0_AL = 4.0366              # W9: a0 of the single crystal [A]
REFINE_MAX_ITER = 20        # §3-2-2c
REFINE_TOL = 1e-10          # (fractional)
SEED = 132                  # seed for the extraction of 20 training structures, the displacement of W3 and the random numbers of W4

OP_NAMES = ["C2n", "m_p", "C2t", "inv"]
# diagonal of R in the table (in the order n,t,p; fractional)
OPS_NTP = {"C2n": (1.0, -1.0, -1.0), "m_p": (1.0, 1.0, -1.0), "C2t": (-1.0, 1.0, -1.0), "inv": (-1.0, -1.0, -1.0)}

VALIDATION_ORDER = ["W6", "W8", "W9", "W11", "W10", "W1", "W5", "W4", "W3", "W2", "W7"]   # §6-1
VALIDATION_ALL = [f"W{i}" for i in range(1, 12)]
VALIDATION_NAME = "gb_symfit_validation.json"
MAIN_CSV_NAME = "gb_symfit_main.csv"
SUMMARY_NAME = "gb_symfit_summary.md"
ADDITIONAL_CSV_NAME = "gb_symfit_additional.csv"             # output name of layer=additional (the name of main is not changed)
ADDITIONAL_SUMMARY_NAME = "gb_symfit_summary_additional.md"
LAYERS = ("main", "additional")

# reference orientations on paper: the angle arccos(3/sqrt(10))=18.43 deg between n=[210] and [110], and its mirror image 71.57 deg (mod 90 deg)
REF_DESK_DEG = (math.degrees(math.acos(3.0 / math.sqrt(10.0))), 90.0 - math.degrees(math.acos(3.0 / math.sqrt(10.0))))
SIGMA5_DIFF_DEG = 36.87     # W8: difference of the two reference orientations (distance on the circle)

STATUS_ORDER = ["fitted", "no_gb", "multi_gb", "disordered", "axis_mismatch", "non_al", "excluded_pre", "error"]


class ValidationFail(RuntimeError):
    pass


class AxisMismatch(ValueError):
    pass


# ---------------------------------------------------------------------------
# verbatim copy from manifest.py
# source: mattergen/symmetry_constrained/manifest.py
# ---------------------------------------------------------------------------
# R_REPLICA = np.diag([1.0, -1.0, -1.0])  # manifest.py:46 (fractional; grain-boundary normal = a axis = index 0). It corresponds to the operation C2n of this module. Use OPS_NTP["C2n"].
def _mic(delta: np.ndarray) -> np.ndarray:  # manifest.py:71-73
    """Fold a fractional coordinate delta into the minimum image [-0.5, 0.5)."""
    return delta - np.round(delta)


def _pairwise_mic_dist(cell: np.ndarray, frac_a: np.ndarray, frac_b: np.ndarray) -> np.ndarray:  # manifest.py:83-87
    diff = frac_a[:, None, :] - frac_b[None, :, :]
    diff = _mic(diff)
    cart = diff @ cell
    return np.linalg.norm(cart, axis=2)


def _hungarian_match(cell: np.ndarray, frac_mapped: np.ndarray, frac_target: np.ndarray):  # manifest.py:90-100
    """Match ``frac_mapped`` (already R/b-transformed source) onto ``frac_target`` (destination).

    Returns ``(row, col, residuals)`` with ``frac_mapped[row[k]]`` corresponding to
    ``frac_target[col[k]]``.
    """
    if frac_mapped.shape[0] != frac_target.shape[0]:
        return None, None, None
    dist = _pairwise_mic_dist(cell, frac_mapped, frac_target)
    row, col = linear_sum_assignment(dist)
    return row, col, dist[row, col]


# ---------------------------------------------------------------------------
# identification of the axes
# ---------------------------------------------------------------------------
def identify_axes(lengths):
    """(idx_n, idx_t, idx_p) = indices of the lattice vectors. n is the longest edge, t is the edge of 4.04+-0.15 A (exactly one), p is the rest.
    If they cannot be identified, AxisMismatch (do not silently use a default order)."""
    L = [float(x) for x in lengths]
    if len(L) != 3 or not all(math.isfinite(x) and x > 0 for x in L):
        raise AxisMismatch(f"invalid cell edge lengths {L}")
    order = sorted(range(3), key=lambda i: -L[i])
    n = order[0]
    if L[order[0]] - L[order[1]] < 1e-6:
        raise AxisMismatch(f"longest edge is not unique: {L}")
    rest = [i for i in range(3) if i != n]
    cand = [i for i in rest if abs(L[i] - T_LEN_A) <= T_LEN_TOL_A]
    if len(cand) != 1:
        raise AxisMismatch(f"t-axis candidates (|L-{T_LEN_A}|<={T_LEN_TOL_A}) among non-n edges = {len(cand)} != 1: edges {L}")
    t = cand[0]
    p = [i for i in rest if i != t][0]
    return n, t, p


def axes_str(axes) -> str:
    n, t, p = axes
    return f"n={'abc'[n]},t={'abc'[t]},p={'abc'[p]}"


def op_diag(name, axes) -> np.ndarray:
    """R is written in the order (n,t,p). Returns the diagonal components rearranged in the order of the lattice vectors."""
    n, t, p = axes
    d = [0.0, 0.0, 0.0]
    d[n], d[t], d[p] = OPS_NTP[name]
    return np.array(d, dtype=float)


# ---------------------------------------------------------------------------
# stage 1: gb_symfit
# ---------------------------------------------------------------------------
def fit_op(frac: np.ndarray, cell: np.ndarray, rdiag: np.ndarray) -> dict:
    """For one operation R=diag(rdiag), the fit that maps mapped = mic(R frac + t) onto frac by the Hungarian method.
    a. candidate translations t_j = mic(frac[j] - R frac[0]) (j=0..N-1). All three components are free (axis0 is not fixed at 0.5).
    b. for each candidate, match and give the total residual.
    c. starting from t_j with the smallest total, iterate t <- t + mean(mic(d_k)) until |dt|<1e-10 or 20 times (re-matching each time; the same procedure as _refine_b0 of manifest.py:128).
    d. give the max/rms Cartesian residuals of the final matching."""
    base = frac * rdiag                      # R is diagonal, so frac @ R.T = frac * rdiag
    t_cands = _mic(frac - base[0])           # (N,3)
    best_cost = np.inf
    best_t = None
    for tj in t_cands:
        mapped = _mic(base + tj)
        row, col, res = _hungarian_match(cell, mapped, frac)
        if res is None:
            raise RuntimeError("atom count mismatch in _hungarian_match")
        cost = float(res.sum())
        if cost < best_cost:
            best_cost, best_t = cost, tj.copy()
    t = best_t.copy()
    n_iter = 0
    for _ in range(REFINE_MAX_ITER):
        n_iter += 1
        mapped = _mic(base + t)
        row, col, _res = _hungarian_match(cell, mapped, frac)
        delta = _mic(frac[col] - mapped[row])
        step = delta.mean(axis=0)
        t = t + step
        if float(np.max(np.abs(step))) < REFINE_TOL:
            break
    mapped = _mic(base + t)
    row, col, _res = _hungarian_match(cell, mapped, frac)
    dcart = _mic(frac[col] - mapped[row]) @ cell
    norms = np.linalg.norm(dcart, axis=1)
    return {"max_A": float(norms.max()), "rms_A": float(np.sqrt(np.mean(norms ** 2))), "t": t, "n_iter": n_iter}


def gb_symfit(frac: np.ndarray, cell: np.ndarray, axes, op_names=None) -> dict:
    """Performs fit_op for each of ops and adopts the operation with the smallest max residual.
    Returns: {"ops": {name: {max_A, rms_A, tau_n, tau_t, tau_p, n_iter}}, "best_op": name}
    tau is the (n,t,p) components of t taken mod 1."""
    op_names = list(op_names) if op_names is not None else list(OP_NAMES)
    n, t, p = axes
    out = {}
    for name in op_names:
        r = fit_op(frac, cell, op_diag(name, axes))
        tm = np.mod(r["t"], 1.0)
        out[name] = {"max_A": r["max_A"], "rms_A": r["rms_A"], "tau_n": float(tm[n]), "tau_t": float(tm[t]), "tau_p": float(tm[p]),
                     "n_iter": r["n_iter"]}
    best = min(op_names, key=lambda k: out[k]["max_A"])   # for equal values, the first
    return {"ops": out, "best_op": best}


# ---------------------------------------------------------------------------
# stage 0: decision of the number of grain boundaries
# ---------------------------------------------------------------------------
def circ_dist_deg(a, b):
    """Distance on the circle mod 90 deg (0..45)."""
    return np.abs(np.mod(np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 45.0, 90.0) - 45.0)


def lattice_frame(cell: np.ndarray, axes):
    """(n, t, p): orthonormal basis (Cartesian) obtained by orthogonalizing the directions of the lattice vectors with the priority n > t > p. cell has row vectors."""
    n, t, p = axes
    vn, vt, vp = cell[n], cell[t], cell[p]
    nh = vn / np.linalg.norm(vn)
    th = vt - (vt @ nh) * nh
    th = th / np.linalg.norm(th)
    ph = vp - (vp @ nh) * nh - (vp @ th) * th
    ph = ph / np.linalg.norm(ph)
    return nh, th, ph


def atom_orientations(frac: np.ndarray, cell: np.ndarray, axes):
    """Returns the orientation phi [deg, mod 90] of each atom (nan if fewer than 3 in-plane bonds) and the number of in-plane bonds.
    Bond = minimum-image distance < 3.4 A. In-plane = absolute value of the t component < 0.5 A. alpha = atan2(p component, n component). phi = (1/4) arg(sum exp(4i alpha))."""
    N = frac.shape[0]
    nh, th, ph = lattice_frame(cell, axes)
    d = _mic(frac[None, :, :] - frac[:, None, :])          # [i,j] = frac[j] - frac[i]
    cart = d @ cell
    dist = np.linalg.norm(cart, axis=2)
    bond = (dist < BOND_CUT_A) & (~np.eye(N, dtype=bool))
    inpl = bond & (np.abs(cart @ th) < INPLANE_T_A)
    alpha = np.arctan2(cart @ ph, cart @ nh)
    z = np.where(inpl, np.exp(4j * alpha), 0.0).sum(axis=1)
    cnt = inpl.sum(axis=1)
    phi = np.mod(np.degrees(np.angle(z)) / 4.0, 90.0)
    phi = np.where(cnt >= MIN_INPLANE_BONDS, phi, np.nan)
    return phi, cnt


def measure_reference_peaks(phis) -> dict:
    """Replaces the reference orientations by the measured values of the training set (W8). [Author's choice] procedure of the peak detection:
    count on the circle mod 90 deg in 0.5 deg bins (180), smooth with a box of +-2 bins, take the maximum as the first peak and the maximum excluding the range within 15 deg of the first peak on the circle as the second peak.
    Each peak is refined by the circular mean (period 90 deg) of the atoms within 5 deg of its center. refs_deg returned in ascending order."""
    ph = np.asarray(phis, dtype=float)
    ph = ph[np.isfinite(ph)]
    if ph.size < 10:
        raise ValidationFail(f"too few oriented atoms to measure reference peaks: {ph.size}")
    nb = 180
    hist = np.bincount(np.minimum((ph / 0.5).astype(int), nb - 1), minlength=nb).astype(float)
    sm = sum(np.roll(hist, s) for s in range(-2, 3))
    c = (np.arange(nb) + 0.5) * 0.5
    i1 = int(np.argmax(sm))
    c1 = c[i1]
    sm2 = np.where(circ_dist_deg(c, c1) > 15.0, sm, -1.0)
    i2 = int(np.argmax(sm2))
    c2 = c[i2]
    refs = []
    for cc in (c1, c2):
        sel = ph[circ_dist_deg(ph, cc) <= 5.0]
        ang = np.radians(4.0 * sel)
        refs.append(float(np.mod(np.degrees(np.angle(np.exp(1j * ang).mean())) / 4.0, 90.0)))
    refs = sorted(refs)
    diff = float(circ_dist_deg(refs[0], refs[1]))
    return {"refs_deg": refs, "diff_deg": diff, "coarse_peak_deg": [float(c1), float(c2)], "n_oriented_atoms": int(ph.size),
            "hist_0p5deg_smoothed_top": {f"{float(c[i]):.2f}": float(sm[i]) for i in np.argsort(-sm)[:6]}}


def label_atoms(phi: np.ndarray, refs) -> np.ndarray:
    """Within 8 deg of a reference orientation: A (=1, refs[0]) or B (=2, refs[1]); otherwise 0. If within both, the nearer one."""
    valid = np.isfinite(phi)
    ph = np.where(valid, phi, 0.0)
    dA = np.where(valid, circ_dist_deg(ph, refs[0]), np.inf)
    dB = np.where(valid, circ_dist_deg(ph, refs[1]), np.inf)
    inA, inB = dA <= LABEL_WIN_DEG, dB <= LABEL_WIN_DEG
    lab = np.zeros(phi.shape[0], dtype=int)
    lab[inA & ~inB] = 1
    lab[inB & ~inA] = 2
    both = inA & inB
    lab[both] = np.where(dA[both] <= dB[both], 1, 2)
    return lab


def prune_runs(seq):
    """From a periodic sequence, removes runs of length 1 (isolated single bins) and closes the gap, repeated until nothing changes.
    A sequence with only one kind of label has a single run covering the whole sequence with no boundary, so nothing is done."""
    seq = list(seq)
    while True:
        m = len(seq)
        if m == 0 or len(set(seq)) < 2:
            return seq
        s = next(k for k in range(m) if seq[k] != seq[k - 1])     # start of the run (periodic)
        rot = seq[s:] + seq[:s]
        runs = []
        for lab in rot:
            if runs and runs[-1][0] == lab:
                runs[-1][1] += 1
            else:
                runs.append([lab, 1])
        new = [lab for lab, ln in runs if ln >= 2 for _ in range(ln)]
        if len(new) == m:
            return new
        seq = new


def cyclic_changes(seq) -> int:
    m = len(seq)
    if m == 0:
        return 0
    return sum(1 for k in range(m) if seq[k] != seq[(k + 1) % m])


def count_gb(frac: np.ndarray, cell: np.ndarray, axes, refs, phi=None) -> dict:
    """Number of grain boundaries. Keys of the return value: n_gb (may be None), frac_A, frac_B, frac_unlabeled, gb0_status, gb_bin_seq, note, phi_hist10.
    gb0_status: disordered / no_gb / multi_gb / two_gb (n_gb=2; go on to stage 1) / error. The order of decisions is as specified."""
    N = frac.shape[0]
    if phi is None:
        phi, _cnt = atom_orientations(frac, cell, axes)
    lab = label_atoms(phi, refs)
    fA, fB = float((lab == 1).mean()), float((lab == 2).mean())
    fU = 1.0 - fA - fB
    n_ax = axes[0]
    Ln = float(np.linalg.norm(cell[n_ax]))
    nb = max(1, int(math.floor(Ln / BIN_WIDTH_A + 0.5)))
    bidx = np.minimum((np.mod(frac[:, n_ax], 1.0) * nb).astype(int), nb - 1)
    nA = np.bincount(bidx[lab == 1], minlength=nb)
    nB = np.bincount(bidx[lab == 2], minlength=nb)
    tot = nA + nB
    bl = np.zeros(nb, dtype=int)
    bl[(tot >= BIN_MIN_LABELED) & (nA >= BIN_DOMINANCE * tot) & (nA >= nB)] = 1
    bl[(tot >= BIN_MIN_LABELED) & (nB >= BIN_DOMINANCE * tot) & (nB > nA)] = 2
    seq0 = [int(x) for x in bl if x != 0]
    seq = prune_runs(seq0)
    bin_seq = "".join(".AB"[int(x)] for x in bl)
    ph_ok = phi[np.isfinite(phi)]
    hist10 = [int(x) for x in np.histogram(ph_ok, bins=10, range=(0.0, 90.0))[0]] if ph_ok.size else [0] * 10
    out = {"n_gb": None, "frac_A": fA, "frac_B": fB, "frac_unlabeled": fU, "gb_bin_seq": bin_seq, "note": "", "phi_hist10": hist10,
           "n_bins": nb}
    if (fA + fB) < LABELED_MIN_FRAC:
        out["gb0_status"] = "disordered"
        out["note"] = f"labeled_frac={fA + fB:.3f}<{LABELED_MIN_FRAC}"
        return out
    if len(seq) == 0:
        # [Author's choice] an edge case that the plan does not specify: the sequence became empty after removing the isolated runs (e.g. alternating ABAB...). The number of grain boundaries is not claimed and the structure is classified as disordered
        out["gb0_status"] = "disordered"
        out["note"] = "empty_after_run_removal"
        return out
    ng = cyclic_changes(seq)
    out["n_gb"] = ng
    if ng % 2 == 1:
        out["gb0_status"] = "error"
        out["note"] = f"odd n_gb={ng}"
        return out
    if ng == 0:
        out["gb0_status"] = "no_gb"
    elif ng >= 4:
        out["gb0_status"] = "multi_gb"
    else:
        out["gb0_status"] = "two_gb"
    return out


# ---------------------------------------------------------------------------
# analysis of one structure (stage 0 -> stage 1)
# ---------------------------------------------------------------------------
def _fmt_cell(cell: np.ndarray) -> dict:
    a, b, c = (np.linalg.norm(cell[i]) for i in range(3))
    def ang(u, v):
        return float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0))))
    al, be, ga = ang(cell[1], cell[2]), ang(cell[0], cell[2]), ang(cell[0], cell[1])
    return {"cell_a_A": float(a), "cell_b_A": float(b), "cell_c_A": float(c), "cell_alpha_deg": al, "cell_beta_deg": be,
            "cell_gamma_deg": ga, "angle_dev_max_deg": max(abs(al - 90.0), abs(be - 90.0), abs(ga - 90.0))}


def analyze_structure(frac: np.ndarray, cell: np.ndarray, species, refs) -> dict:
    """Order of decisions (this order is written only in this function): non_al -> axis_mismatch -> stage 0 (disordered -> no_gb -> multi_gb) -> stage 1.
    excluded_pre is decided beforehand by the caller (manifest)."""
    out = {"status": "", "note": ""}
    out.update(_fmt_cell(cell))
    if any(s != "Al" for s in species):
        out["status"] = "non_al"
        out["note"] = "species=" + ",".join(sorted(set(species)))
        return out
    try:
        axes = identify_axes(np.linalg.norm(cell, axis=1))
    except AxisMismatch as e:
        out["status"] = "axis_mismatch"
        out["note"] = str(e)
        return out
    out["axes"] = axes_str(axes)
    c0 = count_gb(frac, cell, axes, refs)
    for k in ("n_gb", "frac_A", "frac_B", "frac_unlabeled", "gb0_status", "gb_bin_seq"):
        out[k] = c0[k]
    out["note"] = c0["note"]
    gs = c0["gb0_status"]
    if gs in ("disordered", "multi_gb", "error"):
        out["status"] = gs
        return out
    fit = gb_symfit(frac, cell, axes)
    best = fit["best_op"]
    if gs == "no_gb":
        # monitoring columns for W10 (not used for decisions)
        out["status"] = "no_gb"
        out["symfit_op_name_diag"] = best
        out["symfit_max_A_diag"] = fit["ops"][best]["max_A"]
        out["symfit_rms_A_diag"] = fit["ops"][best]["rms_A"]
        for name in OP_NAMES:
            out[f"symfit_max_A_{name}_diag"] = fit["ops"][name]["max_A"]
        return out
    out["status"] = "fitted"
    out["op_name"] = best
    out["gb_symfit_max_A"] = fit["ops"][best]["max_A"]
    out["gb_symfit_rms_A"] = fit["ops"][best]["rms_A"]
    out["tau_n"], out["tau_t"], out["tau_p"] = (fit["ops"][best][k] for k in ("tau_n", "tau_t", "tau_p"))
    for name in OP_NAMES:
        o = fit["ops"][name]
        out[f"max_A_{name}"], out[f"rms_A_{name}"] = o["max_A"], o["rms_A"]
        out[f"tau_n_{name}"], out[f"tau_t_{name}"], out[f"tau_p_{name}"] = o["tau_n"], o["tau_t"], o["tau_p"]
    return out


# ---------------------------------------------------------------------------
# output columns
# ---------------------------------------------------------------------------
ID_FIELDS = ["layer", "arm", "w", "chunk", "cif_index", "stage"]
OUT_FIELDS = (ID_FIELDS + ["status", "note", "natoms", "excluded_pre", "axes",
                           "cell_a_A", "cell_b_A", "cell_c_A", "cell_alpha_deg", "cell_beta_deg", "cell_gamma_deg", "angle_dev_max_deg",
                           "gamma_relaxed",
                           "n_gb", "frac_A", "frac_B", "frac_unlabeled", "gb0_status", "gb_bin_seq",
                           "op_name", "gb_symfit_max_A", "gb_symfit_rms_A", "tau_n", "tau_t", "tau_p"]
              + [f"{m}_{o}" for o in OP_NAMES for m in ("max_A", "rms_A", "tau_n", "tau_t", "tau_p")]
              + ["symfit_op_name_diag", "symfit_max_A_diag", "symfit_rms_A_diag"] + [f"symfit_max_A_{o}_diag" for o in OP_NAMES])


# ---------------------------------------------------------------------------
# loading of inputs
# ---------------------------------------------------------------------------
def manifest_path(layer="main") -> str:
    """Without arguments, main as before (existing call on the validation side)."""
    if layer not in LAYERS:
        raise K.InputError(f"layer must be one of {LAYERS} (got {layer!r})")
    return K.P(f"{K.RUN}/results/inputs_manifest_{layer}.csv")


def train_manifest_path() -> str:
    return K.P(f"{K.RUN}/results/train_ref_manifest.csv")


def relax_chunk_dir(layer, arm, w, chunk) -> Path:
    return Path(K.P(f"{K.RUN}/results/relax/{layer}/{K.armw(arm, w)}/{K.chunk_name(chunk)}"))


def atoms_to_arrays(atoms):
    cell = np.array(atoms.cell[:], dtype=float)
    frac = np.array(atoms.get_scaled_positions(wrap=False), dtype=float)
    return frac, cell, list(atoms.get_chemical_symbols())


def load_train_structures(n=20) -> list:
    """n structures with seed 132 from the relaxed training structures (the same extraction rule as build_r3_inputs of build_manifests: random.Random(seed).sample from sorted(row_index))."""
    from ase.io import read as ase_read
    rows = K.read_csv_rows(train_manifest_path())
    by_idx = {int(r["row_index"]): r for r in rows}
    picks = random.Random(SEED).sample(sorted(by_idx), n)
    out = []
    for ri in picks:
        r = by_idx[ri]
        path = K.P(r["poscar_relaxed_path"])
        at = ase_read(path, format="vasp")
        frac, cell, sp = atoms_to_arrays(at)
        out.append({"row_index": ri, "path": r["poscar_relaxed_path"], "frac": frac, "cell": cell, "species": sp, "natoms": len(at)})
    return out


def build_single_crystal(flip_p: bool):
    """W9: ideal fcc single crystal. n=[210], t=[001], p=[-120], a0=4.0366 A, 6 periods along n (6 a0 sqrt(5) = 54.2 A), 1 period along t, 1 period along p (a0 sqrt(5)).
    flip_p=True is the orientation of grain B (p reversed). The cell is orthogonal with the same ordering as the proposed model (a=n, b=t, c=p). An exception if it does not have 120 atoms."""
    a0 = A0_AL
    A = 6.0 * a0 * np.array([2.0, 1.0, 0.0])
    B = a0 * np.array([0.0, 0.0, 1.0])
    C = a0 * np.array([-1.0, 2.0, 0.0])
    if flip_p:
        C = -C
    M = np.array([A, B, C])
    basis = a0 * np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    pts = []
    for i in range(-3, 15):
        for j in range(-2, 11):
            for k in range(-1, 2):
                for b in basis:
                    pts.append(np.array([i, j, k], dtype=float) * a0 + b)
    pts = np.array(pts)
    fr = pts @ np.linalg.inv(M)
    fr_r = np.round(fr, 9)
    sel = np.all((fr_r >= 0.0) & (fr_r < 1.0), axis=1)
    frac = np.mod(fr[sel], 1.0)
    if frac.shape[0] != 120:
        raise ValidationFail(f"single crystal has {frac.shape[0]} atoms != 120")
    cell = np.diag([np.linalg.norm(A), np.linalg.norm(B), np.linalg.norm(C)])
    return frac, cell


# ---------------------------------------------------------------------------
# validation (self-tests)
# ---------------------------------------------------------------------------
def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json_once(path: Path, doc) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        f.write(json.dumps(doc, indent=2, ensure_ascii=False, default=float) + "\n")


class Validationes:
    def __init__(self):
        self.res = {}
        self.train = None
        self.fit_train = None      # row_index -> gb_symfit (all ops)
        self.refs = None

    # ---- common ----
    def _train(self):
        if self.train is None:
            self.train = load_train_structures(20)
            _log(f"loaded {len(self.train)} train structures (seed {SEED}): row_index={[t['row_index'] for t in self.train]}")
        return self.train

    def _axes_of(self, cell):
        return identify_axes(np.linalg.norm(cell, axis=1))

    # ---- W6 ----
    def w6(self):
        rows = K.read_csv_rows(manifest_path())
        pats = {}
        n_excl = {}
        mism = []
        for r in rows:
            arm = r["arm"]
            if r["excluded_pre"]:
                n_excl[arm] = n_excl.get(arm, 0) + 1
                continue
            L = [float(r["cell_a_A"]), float(r["cell_b_A"]), float(r["cell_c_A"])]
            try:
                ax = identify_axes(L)
                pats.setdefault(arm, {}).setdefault(axes_str(ax), 0)
                pats[arm][axes_str(ax)] += 1
            except AxisMismatch as e:
                mism.append({"arm": arm, "chunk": r["chunk"], "cif_index": r["cif_index"], "msg": str(e)})
        tr = self._train()
        tr_pat = {}
        for t in tr:
            try:
                ax = self._axes_of(t["cell"])
                tr_pat[axes_str(ax)] = tr_pat.get(axes_str(ax), 0) + 1
            except AxisMismatch as e:
                mism.append({"arm": "train", "row_index": t["row_index"], "msg": str(e)})
        exp_h, exp_v, exp_t = "n=a,t=b,p=c", "n=a,t=c,p=b", "n=a,t=b,p=c"
        n_h = sum(pats.get("proposed", {}).values())
        ok = (not mism and pats.get("proposed", {}) == {exp_h: 350} and set(pats.get("baseline", {})) == {exp_v}
              and tr_pat == {exp_t: 20})
        return {"pass": bool(ok), "expected": {"proposed": {exp_h: 350}, "baseline(excl. excluded_pre)": exp_v, "train20": {exp_t: 20}},
                "patterns": pats, "n_excluded_pre": n_excl, "train20_patterns": tr_pat, "axis_mismatch": mism, "n_proposed_non_excluded": n_h}

    # ---- W8 ----
    def w8(self):
        tr = self._train()
        phis, per = [], []
        for t in tr:
            ax = self._axes_of(t["cell"])
            phi, _ = atom_orientations(t["frac"], t["cell"], ax)
            phis.append(phi)
        meas = measure_reference_peaks(np.concatenate(phis))
        refs = meas["refs_deg"]
        self.refs = refs
        for t, phi in zip(tr, phis):
            ax = self._axes_of(t["cell"])
            c = count_gb(t["frac"], t["cell"], ax, refs, phi=phi)
            per.append({"row_index": t["row_index"], "n_gb": c["n_gb"], "gb0_status": c["gb0_status"], "labeled_frac": c["frac_A"] + c["frac_B"],
                        "frac_A": c["frac_A"], "frac_B": c["frac_B"], "bin_seq": c["gb_bin_seq"], "phi_hist10": c["phi_hist10"], "note": c["note"]})
        n2 = sum(1 for p in per if p["n_gb"] == 2 and p["gb0_status"] == "two_gb")
        min_lab = min(p["labeled_frac"] for p in per)
        diff = meas["diff_deg"]
        ok = (n2 == len(per) and abs(diff - SIGMA5_DIFF_DEG) <= 1.0 and min_lab >= 0.7)
        return {"pass": bool(ok), "refs_deg": refs, "ref_desk_deg": list(REF_DESK_DEG), "ref_diff_deg": diff, "ref_diff_expected_deg": SIGMA5_DIFF_DEG,
                "n_two_gb": n2, "n_structures": len(per), "min_labeled_frac": min_lab, "peak_measure": meas, "per_structure": per}

    # ---- W9 ----
    def w9(self):
        out = {}
        ok = True
        for name, flip in (("A", False), ("B", True)):
            frac, cell = build_single_crystal(flip)
            ax = self._axes_of(cell)
            c = count_gb(frac, cell, ax, self.refs)
            lab = c["frac_A"] + c["frac_B"]
            good = (c["gb0_status"] == "no_gb" and c["n_gb"] == 0 and lab >= 1.0 - 1e-12)
            ok = ok and good
            out[name] = {"axes": axes_str(ax), "natoms": int(frac.shape[0]), "n_gb": c["n_gb"], "gb0_status": c["gb0_status"], "labeled_frac": lab,
                         "frac_A": c["frac_A"], "frac_B": c["frac_B"], "bin_seq": c["gb_bin_seq"], "cell_diag_A": [float(cell[i, i]) for i in range(3)], "pass": bool(good)}
        return {"pass": bool(ok), "crystals": out}

    # ---- W11 ----
    def w11(self):
        t = self._train()[0]
        ax = self._axes_of(t["cell"])
        n = ax[0]
        f2 = np.concatenate([t["frac"].copy(), t["frac"].copy()])
        f2[: t["natoms"], n] = t["frac"][:, n] / 2.0
        f2[t["natoms"]:, n] = (t["frac"][:, n] + 1.0) / 2.0
        cell2 = t["cell"].copy()
        cell2[n] = cell2[n] * 2.0
        ax2 = self._axes_of(cell2)
        c = count_gb(f2, cell2, ax2, self.refs)
        ok = (c["gb0_status"] == "multi_gb" and c["n_gb"] == 4)
        return {"pass": bool(ok), "source_row_index": t["row_index"], "natoms": int(f2.shape[0]), "n_gb": c["n_gb"], "gb0_status": c["gb0_status"],
                "bin_seq": c["gb_bin_seq"], "labeled_frac": c["frac_A"] + c["frac_B"]}

    # ---- W10 ----
    def w10(self):
        out = {}
        ok = True
        for name, flip in (("A", False), ("B", True)):
            frac, cell = build_single_crystal(flip)
            ax = self._axes_of(cell)
            fit = gb_symfit(frac, cell, ax)       # apply the fit directly without going through stage 0
            ops = {k: v["max_A"] for k, v in fit["ops"].items()}
            good = ops["inv"] <= 1e-6 and ops["C2t"] <= 1e-6
            ok = ok and good
            out[name] = {"max_A_by_op": ops, "best_op": fit["best_op"], "pass_inv_C2t_le_1e-6": bool(good)}
        return {"pass": bool(ok), "crystals": out}

    # ---- W1 (all ops on 20 structures; reused by W5 and W7) ----
    def _fit_train(self):
        if self.fit_train is None:
            self.fit_train = {}
            for t in self._train():
                ax = self._axes_of(t["cell"])
                self.fit_train[t["row_index"]] = gb_symfit(t["frac"], t["cell"], ax)
            _log("fitted 4 ops on 20 train structures")
        return self.fit_train

    def w1(self):
        ft = self._fit_train()
        vals = {ri: f["ops"]["C2n"]["max_A"] for ri, f in ft.items()}
        mx = max(vals.values())
        return {"pass": bool(mx <= 1e-6), "threshold_A": 1e-6, "max_C2n_A": mx, "C2n_max_A_by_row_index": vals}

    # ---- W5 ----
    def w5(self):
        ft = self._fit_train()
        dev = {ri: float(abs(f["ops"]["C2n"]["tau_n"] - 0.5)) for ri, f in ft.items()}
        mx = max(dev.values())
        return {"pass": bool(mx <= 1e-6), "tolerance": 1e-6, "max_abs_tau_n_minus_0p5": mx,
                "tau_n_by_row_index": {ri: f["ops"]["C2n"]["tau_n"] for ri, f in ft.items()}}

    # ---- W4 ----
    def w4(self):
        ft = self._fit_train()
        rng = np.random.default_rng(SEED)
        d_all = {}
        for t in self._train():
            ax = self._axes_of(t["cell"])
            shift = rng.uniform(0.0, 1.0, 3)
            perm = rng.permutation(t["natoms"])
            f2 = np.mod(t["frac"] + shift, 1.0)[perm]
            r = fit_op(f2, t["cell"], op_diag("C2n", ax))
            d_all[t["row_index"]] = float(abs(r["max_A"] - ft[t["row_index"]]["ops"]["C2n"]["max_A"]))
        mx = max(d_all.values())
        return {"pass": bool(mx <= 1e-9), "threshold_A": 1e-9, "max_abs_change_A": mx, "change_by_row_index": d_all, "seed": SEED}

    # ---- W3(FAIL validation) ----
    def w3(self):
        t = self._train()[0]
        ax = self._axes_of(t["cell"])
        n = ax[0]
        rng = np.random.default_rng(SEED)
        x = np.mod(t["frac"][:, n], 1.0)
        sel = x < 0.5
        disp = rng.uniform(-0.30, 0.30, (int(sel.sum()), 3))      # independent uniform displacement of +-0.30 A in each Cartesian component
        f2 = t["frac"].copy()
        f2[sel] = f2[sel] + disp @ np.linalg.inv(t["cell"])
        fit = gb_symfit(f2, t["cell"], ax)
        best = fit["best_op"]
        mx = fit["ops"][best]["max_A"]
        return {"pass": bool(mx >= 0.1), "threshold_A": 0.1, "gb_symfit_max_A": mx, "best_op": best, "source_row_index": t["row_index"],
                "n_displaced": int(sel.sum()), "max_A_by_op": {k: v["max_A"] for k, v in fit["ops"].items()}, "seed": SEED}

    # ---- W2 ----
    def w2(self):
        rows = [r for r in K.read_csv_rows(manifest_path()) if r["arm"] == "proposed" and not r["excluded_pre"]]
        keyed = {(int(r["chunk"]), int(r["cif_index"])): r for r in rows}
        picks = random.Random(SEED).sample(sorted(keyed), 20)
        cache = {}
        vals = {}
        for (c, i) in picks:
            r = keyed[(c, i)]
            if c not in cache:
                cache[c] = K.load_chunk_atoms(r["layer"], "proposed", float(r["w"]), c)
            atoms = cache[c][i][0]
            if atoms is None:
                raise ValidationFail(f"proposed pre structure unavailable: chunk{c} cif{i}")
            frac, cell, sp = atoms_to_arrays(atoms)
            ax = self._axes_of(cell)
            res = fit_op(frac, cell, op_diag("C2n", ax))
            vals[f"chunk{c:02d}/cif{i}"] = res["max_A"]
        mx = max(vals.values())
        return {"pass": bool(mx <= 1e-4), "threshold_A": 1e-4, "max_C2n_A": mx, "C2n_max_A_by_structure": vals, "seed": SEED}

    # ---- W7 ----
    def w7(self):
        ft = self._fit_train()
        by_op = {o: {ri: f["ops"][o]["max_A"] for ri, f in ft.items()} for o in OP_NAMES}
        mx_c2n = max(by_op["C2n"].values())
        summ = {o: {"min": min(v.values()), "median": float(np.median(list(v.values()))), "max": max(v.values())} for o, v in by_op.items()}
        return {"pass": bool(mx_c2n <= 1e-6), "note": "only C2n is judged (<=1e-6). The other three are only recorded (measured classification of the table on paper)", "summary_by_op": summ,
                "max_A_by_op_and_row_index": by_op}


def run_validation(out_dir: Path) -> int:
    path = out_dir / VALIDATION_NAME
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    W = Validationes()
    doc = {"design": DESIGN, "script": os.path.abspath(__file__), "host": socket.gethostname(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "order": VALIDATION_ORDER,
           "constants": {"BOND_CUT_A": BOND_CUT_A, "INPLANE_T_A": INPLANE_T_A, "MIN_INPLANE_BONDS": MIN_INPLANE_BONDS, "LABEL_WIN_DEG": LABEL_WIN_DEG,
                         "BIN_WIDTH_A": BIN_WIDTH_A, "BIN_DOMINANCE": BIN_DOMINANCE, "BIN_MIN_LABELED": BIN_MIN_LABELED, "LABELED_MIN_FRAC": LABELED_MIN_FRAC,
                         "T_LEN_A": T_LEN_A, "T_LEN_TOL_A": T_LEN_TOL_A, "A0_AL": A0_AL, "REFINE_MAX_ITER": REFINE_MAX_ITER, "REFINE_TOL": REFINE_TOL,
                         "SEED": SEED, "OPS_NTP": OPS_NTP},
           "results": {}, "failed": None, "all_pass": False, "refs_deg": None}
    for wid in VALIDATION_ORDER:
        t0 = time.time()
        try:
            r = getattr(W, wid.lower())()
        except Exception as e:  # an exception is also recorded as a FAIL of the validation (not silently swallowed)
            r = {"pass": False, "exception": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        r["elapsed_s"] = round(time.time() - t0, 2)
        doc["results"][wid] = r
        _log(f"{wid}: {'PASS' if r['pass'] else 'FAIL'} ({r['elapsed_s']} s)")
        if wid == "W8" and r.get("refs_deg"):
            doc["refs_deg"] = r["refs_deg"]
        if not r["pass"]:
            doc["failed"] = wid
            doc["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            write_json_once(path, doc)
            _log(f"STOP: {wid} FAILED. wrote {path}. Stopping without changing thresholds, formulas or parameters")
            return 3
    doc["all_pass"] = all(doc["results"][w]["pass"] for w in VALIDATION_ALL)
    doc["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json_once(path, doc)
    _log(f"wrote {path}: all_pass={doc['all_pass']}")
    return 0 if doc["all_pass"] else 3


def load_validation_refs(out_dir: Path):
    """Check at the start of main: refuse unless validation.json is all PASS. Returns the reference orientations (measured values of W8)."""
    path = out_dir / VALIDATION_NAME
    if not path.is_file():
        raise K.InputError(f"{path} not found: run MODE=validation before main")
    doc = json.loads(path.read_text())
    res = doc.get("results", {})
    bad = [w for w in VALIDATION_ALL if not (w in res and res[w].get("pass") is True)]
    if doc.get("all_pass") is not True or bad:
        raise K.InputError(f"{path}: validation not all PASS (all_pass={doc.get('all_pass')}, not-PASS/missing={bad}, failed={doc.get('failed')})")
    refs = res["W8"].get("refs_deg")
    if not (isinstance(refs, list) and len(refs) == 2):
        raise K.InputError(f"{path}: results.W8.refs_deg missing")
    return [float(x) for x in refs]


# ---------------------------------------------------------------------------
# production computation
# ---------------------------------------------------------------------------
def chunk_task(task):
    """One (arm, chunk, stage) = 25 structures. The manifest rows are received from the caller."""
    arm, w, chunk, stage, mrows, refs = task
    layer = mrows[0]["layer"]
    rdir = relax_chunk_dir(layer, arm, w, chunk)
    if not (rdir / "RELAX_DONE").is_file():
        raise K.InputError(f"{rdir}/RELAX_DONE not found")
    relax = {int(r["cif_index"]): r for r in K.read_csv_rows(rdir / "relax_rows.csv")}
    need = [r for r in mrows if not r["excluded_pre"]]
    atoms_pre = None
    frames = None
    if need and stage == "pre":
        atoms_pre = K.load_chunk_atoms(layer, arm, float(w), chunk)
    if need and stage == "post":
        from ase.io import read as ase_read
        xyz = rdir / "relaxed.extxyz"
        if not xyz.is_file():
            raise K.InputError(f"{xyz} not found")
        frames = ase_read(str(xyz), index=":")
    out = []
    for r in sorted(mrows, key=lambda x: int(x["cif_index"])):
        i = int(r["cif_index"])
        rr = relax.get(i)
        row = {"layer": layer, "arm": arm, "w": f"{float(w):.1f}", "chunk": chunk, "cif_index": i, "stage": stage,
               "natoms": r["natoms"], "excluded_pre": r["excluded_pre"], "gamma_relaxed": (rr or {}).get("gamma_relaxed", "")}
        if r["excluded_pre"]:
            row.update({"status": "excluded_pre", "note": r["excluded_pre"]})
            out.append(row)
            continue
        try:
            if stage == "pre":
                atoms = atoms_pre[i][0]
                if atoms is None:
                    raise K.InputError("pre structure unavailable (manifest says valid)")
            else:
                if rr is None or rr.get("status") != "ok_converged" or rr.get("frame_index", "") == "":
                    raise K.InputError(f"no converged relaxed structure (relax status={None if rr is None else rr.get('status')})")
                at = frames[int(rr["frame_index"])]
                if int(at.info.get("cif_index", -1)) != i:
                    raise K.InputError(f"relaxed.extxyz frame {rr['frame_index']} cif_index {at.info.get('cif_index')} != {i}")
                atoms = at
            if len(atoms) != int(r["natoms"]):
                raise K.InputError(f"natoms {len(atoms)} != manifest {r['natoms']}")
            frac, cell, sp = atoms_to_arrays(atoms)
            row.update(analyze_structure(frac, cell, sp, refs))
        except Exception as e:
            row.update({"status": "error", "note": f"{type(e).__name__}: {e}"})
        out.append(row)
    return out


def run_main(out_dir: Path, procs: int, layer: str = "main") -> int:
    if layer not in LAYERS:
        raise K.InputError(f"layer must be one of {LAYERS} (got {layer!r})")
    csv_name, sum_name = (MAIN_CSV_NAME, SUMMARY_NAME) if layer == "main" else (ADDITIONAL_CSV_NAME, ADDITIONAL_SUMMARY_NAME)
    csv_path, sum_path = out_dir / csv_name, out_dir / sum_name
    for p in (csv_path, sum_path):
        if p.exists():
            raise FileExistsError(f"refusing to overwrite: {p}")
    refs = load_validation_refs(out_dir)
    _log(f"validation all PASS; reference orientations (deg, mod 90) = {refs}")
    mrows = K.read_csv_rows(manifest_path(layer))
    n_man = len(mrows)
    bad_layer = sorted({r.get("layer", "") for r in mrows if r.get("layer", "") != layer})
    if bad_layer:
        raise K.InputError(f"manifest layer column mismatch: expected all rows layer={layer!r}, found {bad_layer}")
    n_req = {}
    for r in mrows:
        n_req[(r["arm"], f"{float(r['w']):.1f}")] = n_req.get((r["arm"], f"{float(r['w']):.1f}"), 0) + 1
    groups = {}
    for r in mrows:
        groups.setdefault((r["arm"], r["w"], int(r["chunk"])), []).append(r)
    tasks = [(arm, w, chunk, stage, rows, refs) for (arm, w, chunk), rows in sorted(groups.items(), key=lambda kv: (kv[0][0] != "proposed", kv[0][1], kv[0][2]))
             for stage in ("pre", "post")]
    _log(f"manifest rows={n_man}; tasks={len(tasks)} (arm x chunk x stage); procs={procs}")
    results = []
    t0 = time.time()
    if procs <= 1:
        for tk in tasks:
            results.extend(chunk_task(tk))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(min(procs, len(tasks))) as pool:
            for k, rs in enumerate(pool.imap_unordered(chunk_task, tasks, chunksize=1)):
                results.extend(rs)
                if (k + 1) % 8 == 0 or k + 1 == len(tasks):
                    _log(f"{k + 1}/{len(tasks)} tasks done ({time.time() - t0:.0f} s)")
    # row count 2 x manifest, no duplicate keys (layer,arm,w,chunk,cif_index,stage) (layer is included)
    keys = [(r["layer"], r["arm"], r["w"], int(r["chunk"]), int(r["cif_index"]), r["stage"]) for r in results]
    if len(results) != 2 * n_man or len(set(keys)) != len(keys):
        raise K.InputError(f"check-C: rows={len(results)} (expected {2 * n_man}), unique keys={len(set(keys))}")
    results.sort(key=lambda r: (r["arm"] != "proposed", r["w"], int(r["chunk"]), int(r["cif_index"]), r["stage"] != "pre"))   # w is constant for main, so the order is as before
    K.write_csv_rows(csv_path, OUT_FIELDS, results)
    _log(f"wrote {csv_path} ({len(results)} rows)")
    train_gmin = min(float(r["gamma_Jm2"]) for r in K.read_csv_rows(train_manifest_path()))
    text = build_summary(results, refs, train_gmin, n_man, layer=layer, n_req=n_req)
    with open(sum_path, "x") as f:
        f.write(text)
    _log(f"wrote {sum_path}")
    return 0


# ---------------------------------------------------------------------------
# summary.md (no figure is made)
# ---------------------------------------------------------------------------
def _fl(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _q(vals, qs=(0, 25, 50, 75, 90, 99, 100)):
    if not vals:
        return ["-"] * len(qs)
    return [f"{float(np.percentile(vals, q)):.3e}" for q in qs]


def build_summary(rows, refs, train_gmin, n_man, layer="main", n_req=None) -> str:
    """layer=main is the output as it was (not changed by a single byte). layer=additional makes each table per (arm,w,stage)."""
    L = []
    A = L.append
    arms = ["proposed", "baseline"]
    stages = ["pre", "post"]
    additional = layer == "additional"
    if additional:
        arm_cells = [(a, w) for a in arms for w in sorted({r["w"] for r in rows if r["arm"] == a})]
    else:
        arm_cells = [(a, None) for a in arms]

    def sel(r, arm, w, st):
        return r["arm"] == arm and (w is None or r["w"] == w) and r["stage"] == st

    def armlab(arm, w):
        return arm if w is None else f"{arm} w={w}"

    A("# gb_symfit stage 0 and stage 1 aggregation")
    A("")
    A(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} / plan: {DESIGN}")
    A(f"- Rows: {len(rows)} (= {n_man} x 2. Checked at start. Duplicate keys: 0)")
    A(f"- Reference orientations (measured value of W8, deg mod 90): {refs[0]:.3f} / {refs[1]:.3f}")
    if not additional:
        A("- theta is not frozen. **The counts of EQUIV/INEQUIV are not given**. Only the quantile table is looked at. The equivalence rate before relaxation is not used for claims.")
    else:
        A("- additional layer. Additional application after theta=0.064 A was frozen. The counts of G2 and E are in gb_symfit_additional_g2_and_e.md")
    A("")
    if additional:
        A("## 1. Counts of the classes (status): arm x w x stage (the denominator is the requested number = number of manifest rows per (arm,w): "
          + ", ".join(f"{a} w={w}: {n_req[(a, w)]}" for a, w in arm_cells) + ")")
    else:
        A("## 1. Counts of the classes (status): arm x stage (the denominator is the requested number = 350)")
    A("")
    A(("| arm / w | stage | " if additional else "| arm | stage | ") + " | ".join(STATUS_ORDER) + (" | total | n_req |" if additional else " | total |"))
    A("|---|---|" + "---|" * (len(STATUS_ORDER) + (2 if additional else 1)))
    for arm, w in arm_cells:
        for st in stages:
            sub = [r for r in rows if sel(r, arm, w, st)]
            cnt = {s: sum(1 for r in sub if r["status"] == s) for s in STATUS_ORDER}
            other = sum(1 for r in sub if r["status"] not in STATUS_ORDER)
            A(f"| {armlab(arm, w)} | {st} | " + " | ".join(str(cnt[s]) for s in STATUS_ORDER) + f" | {len(sub)}" + (f" (unclassified {other})" if other else "")
              + (f" | {n_req[(arm, w)]}" if additional else "") + " |")
    errs = [r for r in rows if r["status"] == "error"]
    if additional:
        A("")
        A(f"Number of error rows (reported only): {len(errs)} (of all {len(rows)} rows)")
    if errs:
        A("")
        A(f"Reasons of the error rows (first 5 / all {len(errs)}): " + "; ".join(
            (f"{r['arm']}/{r['chunk']}/{r['cif_index']}/{r['stage']}" if not additional else f"{r['arm']}/w{r['w']}/{r['chunk']}/{r['cif_index']}/{r['stage']}") + f": {r['note']}" for r in errs[:5]))
    A("")
    A("## 2. Quantiles of gb_symfit_max_A [A] (status=fitted only; structures judged to have n_gb=2)")
    A("")
    A(("| arm / w | stage |" if additional else "| arm | stage |") + " n | min | 25% | 50% | 75% | 90% | 99% | max |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for arm, w in arm_cells:
        for st in stages:
            v = [_fl(r.get("gb_symfit_max_A")) for r in rows if sel(r, arm, w, st) and r["status"] == "fitted"]
            v = [x for x in v if x is not None]
            A(f"| {armlab(arm, w)} | {st} | {len(v)} | " + " | ".join(_q(v)) + " |")
    A("")
    A("## 3. Breakdown of the adopted operations (op_name) (fitted only)")
    A("")
    A(("| arm / w | stage |" if additional else "| arm | stage |") + " " + " | ".join(OP_NAMES) + " |")
    A("|---|---|" + "---|" * len(OP_NAMES))
    for arm, w in arm_cells:
        for st in stages:
            sub = [r for r in rows if sel(r, arm, w, st) and r["status"] == "fitted"]
            A(f"| {armlab(arm, w)} | {st} | " + " | ".join(str(sum(1 for r in sub if r.get("op_name") == o)) for o in OP_NAMES) + " |")
    A("")
    A("Median of the max residual per operation [A] (fitted only; measured classification of the table on paper):")
    A("")
    A(("| arm / w | stage |" if additional else "| arm | stage |") + " " + " | ".join(OP_NAMES) + " |")
    A("|---|---|" + "---|" * len(OP_NAMES))
    for arm, w in arm_cells:
        for st in stages:
            sub = [r for r in rows if sel(r, arm, w, st) and r["status"] == "fitted"]
            cells = []
            for o in OP_NAMES:
                v = [_fl(r.get(f"max_A_{o}")) for r in sub]
                v = [x for x in v if x is not None]
                cells.append(f"{float(np.median(v)):.3e}" if v else "-")
            A(f"| {armlab(arm, w)} | {st} | " + " | ".join(cells) + " |")
    A("")
    A("## 4. Distribution of tau_n (tau_n mod 1 of the adopted operation; fitted only; counts in steps of 0.1)")
    A("")
    A("Note: tau_n of the operations that reverse the direction of n (C2t, inv) depends on the choice of the origin (because the n component of R is -1). tau_n of C2n and m_p does not depend on the origin.")
    A("")
    A(("| arm / w | stage |" if additional else "| arm | stage |") + " " + " | ".join(f"[{k / 10:.1f},{(k + 1) / 10:.1f})" for k in range(10)) + " | 0.5±1e-3 |")
    A("|---|---|" + "---|" * 11)
    for arm, w in arm_cells:
        for st in stages:
            v = [_fl(r.get("tau_n")) for r in rows if sel(r, arm, w, st) and r["status"] == "fitted"]
            v = [x for x in v if x is not None]
            h = np.histogram(v, bins=10, range=(0.0, 1.0))[0] if v else [0] * 10
            n_half = sum(1 for x in v if abs(x - 0.5) <= 1e-3)
            A(f"| {armlab(arm, w)} | {st} | " + " | ".join(str(int(x)) for x in h) + f" | {n_half} |")
    A("")
    A(f"## 5. Cross table of the suspected grain-boundary loss (gamma_relaxed < training minimum of gamma {train_gmin:.4f} J/m2) and gb0_status")
    A("")
    A("gamma_relaxed is the value of relax_rows.csv (the value of the same structure was attached also to the pre rows). For rows with empty gb0_status, status is used.")
    A("")
    cats = ["two_gb", "no_gb", "multi_gb", "disordered", "error", "(stage 0 not reached)"]
    A(("| arm / w | stage |" if additional else "| arm | stage |") + " suspected | " + " | ".join(cats) + " |")
    A("|---|---|---|" + "---|" * len(cats))
    for arm, w in arm_cells:
        for st in stages:
            for lab, cond in (("suspected", lambda g: g is not None and g < train_gmin), ("not suspected", lambda g: g is not None and g >= train_gmin)):
                sub = [r for r in rows if sel(r, arm, w, st) and cond(_fl(r.get("gamma_relaxed")))]
                def cat(r):
                    g = r.get("gb0_status") or ""
                    return g if g in cats[:-1] else "(stage 0 not reached)"
                A(f"| {armlab(arm, w)} | {st} | {lab} | " + " | ".join(str(sum(1 for r in sub if cat(r) == c)) for c in cats) + " |")
    A("")
    A("## 6. Monitoring columns of the no_gb rows (production version of the measurement of W10; not used for decisions)")
    A("")
    A(("| arm / w | stage |" if additional else "| arm | stage |") + " n(no_gb) | median of the max residual of inv [A] | median of the max residual of C2t [A] |")
    A("|---|---|---|---|---|")
    for arm, w in arm_cells:
        for st in stages:
            sub = [r for r in rows if sel(r, arm, w, st) and r["status"] == "no_gb"]
            def med(col):
                v = [_fl(r.get(col)) for r in sub]
                v = [x for x in v if x is not None]
                return f"{float(np.median(v)):.3e}" if v else "-"
            A(f"| {armlab(arm, w)} | {st} | {len(sub)} | {med('symfit_max_A_inv_diag')} | {med('symfit_max_A_C2t_diag')} |")
    A("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["validation", "main", "additional"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--procs", type=int, default=1)
    args = ap.parse_args()
    out_dir = Path(K.P(args.out_dir))
    t0 = time.time()
    try:
        if args.mode == "validation":
            rc = run_validation(out_dir)
        else:
            rc = run_main(out_dir, max(1, args.procs), layer=args.mode)   # mode=main -> layer main / mode=additional -> layer additional
    except (K.InputError, FileExistsError, AxisMismatch) as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)
    _log(f"done rc={rc} ({time.time() - t0:.0f} s)")
    sys.exit(rc)


if __name__ == "__main__":
    main()
