#!/usr/bin/env python3
"""cna.py -- number of grain-boundary slabs by adaptive CNA (a-CNA) (an instrument of a different principle). Only numpy / scipy / ase (ovito is not used).

Plan: see the experiment description.
Definition of the instrument:
  1. 12 neighbors of each atom (with periodic boundaries; cKDTree with 3x3x3 images). Local cutoff r_c = (1+sqrt(2))/2 x (mean of the 12 neighbor distances) (the a-CNA FCC rule of Stukowski 2012).
  2. An atom whose number of neighbors within r_c is not exactly 12 is other. For 12, for each neighbor j compute (number of common neighbors, number of bonds among the common neighbors, longest chain length);
     if all 12 are (4,2,1) it is fcc, if (4,2,1)x6 + (4,2,2)x6 it is hcp, otherwise other.
  3. The grain-boundary normal axis n follows the same rule as identify_axes of gb_symfit (longest edge). Cut 2.0 A bins along n (number = round(L_n / 2.0), the same as count_gb of gb_symfit);
     a bin with a (fcc+hcp) atom fraction >= 0.5 is G, otherwise (including bins without atoms) D. Isolated runs of length 1 are removed from the periodic sequence (the same rule as prune_runs of gb_symfit;
     the information of the removed bins drops out of the sequence but the remaining bins keep their original bin numbers). Among the G runs, those with the number of hcp atoms >= the number of fcc atoms are counted as hcp_runs (report only).
  4. Bin phase: n_slab is obtained for 4 shifts of the bin origin along n by 0, 0.5, 1.0 and 1.5 A, and the mode is taken. A structure for which the mode is not unique
     (ties of the most votes such as 2-2 or 1-1-1-1) gets phase_ambiguous = True. It is counted in neither two_gb_cna nor loss_cna (a miss).
  5. n_slab = number of D runs (periodic sequence). If the overall (fcc+hcp) fraction < 0.5, disordered_cna.
     two_gb_cna = (n_slab == 2) ∧ ¬disordered_cna ∧ ¬phase_ambiguous、loss_cna = (n_slab ≤ 1) ∧ ¬disordered_cna ∧ ¬phase_ambiguous。
[Author's choices]
  - The longest chain length is evaluated only for neighbors with (ncn, nb) = (4, 2) (this is the only case needed for the decision of fcc and hcp. 2 if the two bonds share a vertex, 1 if they are apart).
  - Common neighbors and bonds are decided with r_c(i) only from the vectors of the 12 neighbors of atom i (same as the implementation of Stukowski 2012 / OVITO. Neighbors of neighbors are not searched).
  - If the sequence became empty after removing the isolated runs (example: G and D alternate one bin each), n_slab is not decided (treated as None. That phase is counted as a vote for the mode).
    A structure whose mode is None is treated as ambiguous.
  - n_slab of a periodic sequence of D only (no G) is 1, that of G only is 0 (perfect crystal = 0).
Usage:
  python cna.py --mode run --layer main --out-dir DIR [--symfit-csv CSV] [--procs N]
      -> DIR/cna_<layer>.csv (all manifest rows. Unconverged and excluded structures are counted as misses by the status column)
        With --symfit-csv, DIR/agreement.md and DIR/criterion_inputs_3.json are also written (input values of C3 and a preview of the function output).
  If an existing output exists, FileExistsError -> exit 2.
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import csv
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

DESIGN = _gbrel.WORK + '/an internal design note that is not included'
RC_FACTOR = (1.0 + math.sqrt(2.0)) / 2.0
BIN_WIDTH_A = 2.0
GROUP_FRAC = 0.5                    # G if the (fcc+hcp) fraction in a bin >= 0.5
DISORDERED_FRAC = 0.5               # disordered_cna if the overall (fcc+hcp) fraction < 0.5
PHASES_A = (0.0, 0.5, 1.0, 1.5)
MIN_WIDTH_A = 3.6                   # minimum perpendicular width at which the neighbor search of one image holds with 3x3x3 images (larger than r_c of about 3.46 A)
T_LEN_A = 4.04
T_LEN_TOL_A = 0.15
OTHER, FCC, HCP = 0, 1, 2


class AxisMismatch(ValueError):
    pass


def identify_axes(lengths):
    """Same rule as identify_axes of gb_symfit (copied without importing the function). (idx_n, idx_t, idx_p). AxisMismatch if it cannot be identified."""
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


# ---------------------------------------------------------------------------
# a-CNA of each atom
# ---------------------------------------------------------------------------
def perpendicular_widths(cell):
    cell = np.asarray(cell, dtype=float)
    vol = abs(np.linalg.det(cell))
    return [vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)]


def acna_labels(frac, cell):
    """Returns the a-CNA label of each atom (0=other, 1=fcc, 2=hcp) and r_c. frac is the fractional coordinates (N,3), cell has row vectors (3,3)."""
    frac = np.asarray(frac, dtype=float)
    cell = np.asarray(cell, dtype=float)
    N = frac.shape[0]
    if N < 14:
        raise ValueError(f"too few atoms for a-CNA: {N}")
    w = perpendicular_widths(cell)
    if min(w) < MIN_WIDTH_A:
        raise ValueError(f"cell perpendicular widths {w} < {MIN_WIDTH_A} Å: 3x3x3 images are not enough")
    pos = (frac - np.floor(frac)) @ cell
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float) @ cell      # (27,3)
    pts = (pos[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    tree = cKDTree(pts)
    d, ii = tree.query(pos, k=14)                       # itself + 13 neighbors
    if not np.all(d[:, 0] < 1e-6):
        raise ValueError("nearest hit is not the atom itself (duplicate atoms?)")
    dist = d[:, 1:]                                     # (N,13)
    vec = pts[ii[:, 1:13]] - pos[:, None, :]            # (N,12,3)
    rc = RC_FACTOR * dist[:, :12].mean(axis=1)          # (N,)
    ok12 = (dist <= rc[:, None]).sum(axis=1) == 12      # exactly 12 within r_c (the 13th is outside r_c)
    diff = vec[:, :, None, :] - vec[:, None, :, :]
    pd = np.linalg.norm(diff, axis=3)                   # (N,12,12)
    eye = np.eye(12, dtype=bool)[None, :, :]
    B = (pd <= rc[:, None, None]) & ~eye                # j-k is a bond (decided with r_c(i))
    Bf = B.astype(np.float64)
    ncn = B.sum(axis=2)                                 # (N,12): number of common neighbors of neighbor j
    nb = np.einsum("njk,njl,nkl->nj", Bf, Bf, Bf) / 2.0   # number of bonds among the common neighbors
    # degree of each vertex in the common-neighbor subgraph (used only when ncn==4 and nb==2): deg[n,j,k] = B[n,j,k] * #{l: B[n,j,l] and B[n,k,l]}
    deg = Bf * np.einsum("njl,nkl->njk", Bf, Bf)
    maxdeg = deg.max(axis=2)
    cand = (ncn == 4) & (np.abs(nb - 2.0) < 1e-9)
    s421 = cand & (maxdeg < 1.5)
    s422 = cand & (maxdeg > 1.5)
    is_fcc = ok12 & (s421.sum(axis=1) == 12)
    is_hcp = ok12 & (s421.sum(axis=1) == 6) & (s422.sum(axis=1) == 6)
    lab = np.zeros(N, dtype=int)
    lab[is_fcc] = FCC
    lab[is_hcp] = HCP
    return lab, rc


# ---------------------------------------------------------------------------
# bins, runs, n_slab
# ---------------------------------------------------------------------------
def smooth_cyclic(items):
    """items: periodic sequence of [(label, bin_index)]. Repeats the operation of removing runs of length 1 and closing the gap until nothing changes (same rule as prune_runs of gb_symfit).
    A sequence with only one kind of label is left alone. The remaining bins keep their original bin numbers."""
    items = list(items)
    while True:
        m = len(items)
        labs = [l for l, _ in items]
        if m == 0 or len(set(labs)) < 2:
            return items
        s = next(k for k in range(m) if labs[k] != labs[k - 1])
        rot = items[s:] + items[:s]
        runs = []
        for it in rot:
            if runs and runs[-1][0][0] == it[0]:
                runs[-1].append(it)
            else:
                runs.append([it])
        new = [it for run in runs if len(run) >= 2 for it in run]
        if len(new) == m:
            return new
        items = new


def cyclic_runs(items):
    """Consecutive runs [[(label, bin)...], ...] of the periodic sequence items. One run if there is only one kind of label. [] if empty. If the first and last runs have the same label they are joined."""
    if not items:
        return []
    runs = []
    for it in items:
        if runs and runs[-1][0][0] == it[0]:
            runs[-1].append(it)
        else:
            runs.append([it])
    if len(runs) > 1 and runs[0][0][0] == runs[-1][0][0]:
        runs[0] = runs[-1] + runs[0]
        runs.pop()
    return runs


def n_slab_for_phase(frac_n, Ln, nb, lab, shift_A):
    """Returns n_slab at one bin phase (number of D runs. None if not decided), the raw G/D sequence, the smoothed runs, hcp_runs, etc."""
    u = np.mod(np.asarray(frac_n, dtype=float) * Ln - shift_A, Ln)
    bidx = np.minimum((u / (Ln / nb)).astype(int), nb - 1)
    tot = np.bincount(bidx, minlength=nb)
    nf = np.bincount(bidx[lab == FCC], minlength=nb)
    nh = np.bincount(bidx[lab == HCP], minlength=nb)
    G = (tot > 0) & ((nf + nh) >= GROUP_FRAC * tot)         # a bin without atoms is D (no crystal atoms)
    raw = "".join("G" if g else "D" for g in G)
    items = [(1 if g else 0, i) for i, g in enumerate(G)]    # 1=G, 0=D
    sm = smooth_cyclic(items)
    runs = cyclic_runs(sm)
    if not runs:
        return {"n_slab": None, "raw": raw, "n_runs_G": 0, "hcp_runs": 0, "runs": []}
    labs = [r[0][0] for r in runs]
    if len(runs) == 1:
        n_slab = 1 if labs[0] == 0 else 0
    else:
        n_slab = sum(1 for l in labs if l == 0)
    hcp_runs = 0
    for r in runs:
        if r[0][0] == 1:
            bins = {b for _l, b in r}
            h = int(sum(nh[b] for b in bins))
            f = int(sum(nf[b] for b in bins))
            if h >= f:
                hcp_runs += 1
    return {"n_slab": n_slab, "raw": raw, "n_runs_G": sum(1 for l in labs if l == 1), "hcp_runs": hcp_runs,
            "runs": [("G" if r[0][0] == 1 else "D", [b for _l, b in r]) for r in runs]}


def cna_core(frac, cell, n_axis):
    """Main body of the CNA instrument. Returns a result dict from frac(N,3), cell(3,3) and the index n_axis of the normal axis."""
    frac = np.asarray(frac, dtype=float)
    cell = np.asarray(cell, dtype=float)
    N = frac.shape[0]
    lab, rc = acna_labels(frac, cell)
    f_fcc, f_hcp = float((lab == FCC).mean()), float((lab == HCP).mean())
    f_other = 1.0 - f_fcc - f_hcp
    Ln = float(np.linalg.norm(cell[n_axis]))
    nb = max(1, int(math.floor(Ln / BIN_WIDTH_A + 0.5)))
    fn = np.mod(frac[:, n_axis], 1.0)
    ph = [n_slab_for_phase(fn, Ln, nb, lab, s) for s in PHASES_A]
    votes = Counter(p["n_slab"] for p in ph)
    top = max(votes.values())
    winners = [v for v, c in votes.items() if c == top]
    ambiguous = (len(winners) != 1) or (winners[0] is None)
    n_slab = None if ambiguous else winners[0]
    disordered = (f_fcc + f_hcp) < DISORDERED_FRAC
    two_gb = bool((n_slab == 2) and not disordered and not ambiguous)
    loss = bool((n_slab is not None) and (n_slab <= 1) and not disordered and not ambiguous)
    # hcp_runs and bin_seq are those of the first phase that agrees with the mode (phase 0 if ambiguous)
    rep = next((p for p in ph if p["n_slab"] == n_slab), ph[0]) if not ambiguous else ph[0]
    return {"n_atoms": N, "frac_fcc": f_fcc, "frac_hcp": f_hcp, "frac_other": f_other, "n_slab": n_slab,
            "n_slab_by_phase": [p["n_slab"] for p in ph], "phase_ambiguous": bool(ambiguous), "disordered_cna": bool(disordered),
            "two_gb_cna": two_gb, "loss_cna": loss, "hcp_runs": rep["hcp_runs"], "bin_seq": ph[0]["raw"],
            "bin_seq_by_phase": [p["raw"] for p in ph], "runs_by_phase": [p["runs"] for p in ph], "n_bins": nb, "Ln": Ln,
            "lab": lab}


def cna_structure(frac, cell):
    """Production entrance: axis identification (longest edge) -> cna_core. AxisMismatch if it cannot be identified (a default axis is never used silently)."""
    n_axis, _t, _p = identify_axes(np.linalg.norm(np.asarray(cell), axis=1))
    out = cna_core(frac, cell, n_axis)
    out["n_axis"] = n_axis
    return out


def classify_slab_count(n_slab, disordered=False, ambiguous=False):
    """(two_gb_cna, loss_cna) from the value of n_slab. The same rule as cna_core (the validation with synthetic input goes through this function)."""
    if ambiguous or disordered or n_slab is None:
        return False, False
    return n_slab == 2, n_slab <= 1


# ---------------------------------------------------------------------------
# production: all structures from the outputs of the manifest and relax
# ---------------------------------------------------------------------------
CNA_FIELDS = ["layer", "arm", "chunk", "cif_index", "stage", "status", "note", "n_atoms", "frac_fcc", "frac_hcp", "frac_other", "n_slab",
              "n_slab_by_phase", "phase_ambiguous", "disordered_cna", "two_gb_cna", "loss_cna", "hcp_runs", "bin_seq"]


def _fmt(x):
    return repr(float(x)) if isinstance(x, (float, np.floating)) else x


def chunk_task(task):
    layer, arm, w, chunk, mrows = task
    rdir = Path(f"{K.relax_root_for(layer)}/{K.armw(arm, w)}/{K.chunk_name(chunk)}")
    out = []
    base = lambda i: {"layer": layer, "arm": arm, "chunk": chunk, "cif_index": i, "stage": "post"}
    if not (rdir / "RELAX_DONE").is_file():
        raise K.InputError(f"{rdir}/RELAX_DONE not found")
    relax = {int(r["cif_index"]): r for r in K.read_csv_rows(rdir / "relax_rows.csv")}
    from ase.io import read as ase_read
    frames = None
    for r in sorted(mrows, key=lambda x: int(x["cif_index"])):
        i = int(r["cif_index"])
        row = base(i)
        if r["excluded_pre"]:
            row.update({"status": "excluded_pre", "note": r["excluded_pre"]})
            out.append(row)
            continue
        rr = relax.get(i)
        if rr is None or rr.get("status") != "ok_converged" or rr.get("frame_index", "") == "":
            row.update({"status": "not_converged", "note": f"relax status={None if rr is None else rr.get('status')}"})
            out.append(row)
            continue
        try:
            if frames is None:
                frames = ase_read(str(rdir / "relaxed.extxyz"), index=":")
            at = frames[int(rr["frame_index"])]
            if int(at.info.get("cif_index", -1)) != i:
                raise K.InputError(f"relaxed.extxyz frame {rr['frame_index']} cif_index {at.info.get('cif_index')} != {i}")
            if any(s != "Al" for s in at.get_chemical_symbols()):
                row.update({"status": "non_al", "note": "species=" + ",".join(sorted(set(at.get_chemical_symbols())))})
                out.append(row)
                continue
            res = cna_structure(at.get_scaled_positions(wrap=False), np.array(at.cell[:], dtype=float))
            row.update({"status": "ok", "note": "", "n_atoms": res["n_atoms"], "frac_fcc": _fmt(res["frac_fcc"]), "frac_hcp": _fmt(res["frac_hcp"]),
                        "frac_other": _fmt(res["frac_other"]), "n_slab": "" if res["n_slab"] is None else res["n_slab"],
                        "n_slab_by_phase": "|".join("NA" if v is None else str(v) for v in res["n_slab_by_phase"]),
                        "phase_ambiguous": res["phase_ambiguous"], "disordered_cna": res["disordered_cna"], "two_gb_cna": res["two_gb_cna"],
                        "loss_cna": res["loss_cna"], "hcp_runs": res["hcp_runs"], "bin_seq": res["bin_seq"]})
        except AxisMismatch as e:
            row.update({"status": "axis_mismatch", "note": str(e)})
        except Exception as e:  # noqa: BLE001
            row.update({"status": "error", "note": f"{type(e).__name__}: {e}"})
        out.append(row)
    return out


def run_layer(layer, out_dir: Path, procs: int, log):
    K.layer_cfg(layer)
    csv_path = out_dir / f"cna_{layer}.csv"
    if csv_path.exists():
        raise FileExistsError(f"refusing to overwrite: {csv_path}")
    mrows = K.read_csv_rows(K.manifest_path_for(layer))
    bad = sorted({r["layer"] for r in mrows if r["layer"] != layer})
    if bad:
        raise K.InputError(f"manifest layer column mismatch: expected {layer!r}, found {bad}")
    groups = {}
    for r in mrows:
        K.check_layer_arm(layer, r["arm"])
        groups.setdefault((r["arm"], float(r["w"]), int(r["chunk"])), []).append(r)
    tasks = [(layer, arm, w, c, rows) for (arm, w, c), rows in sorted(groups.items(), key=lambda kv: (kv[0][0] != "proposed", kv[0][1], kv[0][2]))]
    log(f"layer={layer} manifest rows={len(mrows)} tasks={len(tasks)} procs={procs}")
    res = []
    t0 = time.time()
    if procs <= 1:
        for tk in tasks:
            res.extend(chunk_task(tk))
    else:
        with mp.get_context("fork").Pool(min(procs, len(tasks))) as pool:
            for k, rs in enumerate(pool.imap_unordered(chunk_task, tasks, chunksize=1)):
                res.extend(rs)
                if (k + 1) % 7 == 0 or k + 1 == len(tasks):
                    log(f"{k + 1}/{len(tasks)} tasks ({time.time() - t0:.0f} s)")
    keys = [(r["arm"], r["chunk"], r["cif_index"]) for r in res]
    if len(res) != len(mrows) or len(set(keys)) != len(keys):
        raise K.InputError(f"rows {len(res)} != manifest {len(mrows)} or duplicated keys")
    res.sort(key=lambda r: (r["arm"] != "proposed", r["chunk"], r["cif_index"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "x", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CNA_FIELDS)
        wr.writeheader()
        for r in res:
            wr.writerow({k: r.get(k, "") for k in CNA_FIELDS})
    log(f"wrote {csv_path} ({len(res)} rows)")
    return res


# ---------------------------------------------------------------------------
# agreement table and the input values of C3
# ---------------------------------------------------------------------------
def b(x):
    return str(x).strip().lower() in ("true", "1")


def cohen_kappa(n11, n10, n01, n00):
    n = n11 + n10 + n01 + n00
    if n == 0:
        return None
    po = (n11 + n00) / n
    pe = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / (n * n)
    if abs(1 - pe) < 1e-12:
        return None
    return (po - pe) / (1 - pe)


def agreement(cna_rows, symfit_rows, layer="main"):
    """cna_rows: rows (dict) of cna_<layer>.csv. symfit_rows: rows (dict) of the CSV of gb_symfit. Compared per structure (key (chunk, cif_index); not by set matching).
    The denominator is the number of cna_rows per arm (350). A row whose CNA is not ok has two_gb_cna=False (miss)."""
    sym = {}
    for r in symfit_rows:
        if r["layer"] == layer and r["stage"] == "post":
            sym[(r["arm"], int(r["chunk"]), int(r["cif_index"]))] = r
    out = {"arms": {}}
    recs = []
    for r in cna_rows:
        k = (r["arm"], int(r["chunk"]), int(r["cif_index"]))
        s = sym.get(k)
        if s is None:
            raise K.InputError(f"symfit post row missing for {k}")
        recs.append({"arm": r["arm"], "chunk": k[1], "cif_index": k[2], "cna_status": r["status"],
                     "two_gb_cna": b(r["two_gb_cna"]), "loss_cna": b(r["loss_cna"]), "ambiguous": b(r["phase_ambiguous"]) if r["status"] == "ok" else False,
                     "n_slab": (int(r["n_slab"]) if str(r["n_slab"]) != "" else None), "gb0_status": s["gb0_status"],
                     "sym_two_gb": s["gb0_status"] == "two_gb", "frac_A": s.get("frac_A", ""), "frac_B": s.get("frac_B", ""),
                     "sym_bin_seq": s.get("gb_bin_seq", ""), "cna_bin_seq": r.get("bin_seq", "")})
    for arm in ("proposed", "baseline"):
        rs = [x for x in recs if x["arm"] == arm]
        if not rs:
            continue
        n11 = sum(1 for x in rs if x["two_gb_cna"] and x["sym_two_gb"])
        n10 = sum(1 for x in rs if x["two_gb_cna"] and not x["sym_two_gb"])
        n01 = sum(1 for x in rs if not x["two_gb_cna"] and x["sym_two_gb"])
        n00 = sum(1 for x in rs if not x["two_gb_cna"] and not x["sym_two_gb"])
        out["arms"][arm] = {"n": len(rs), "cna_two_gb&sym_two_gb": n11, "cna_two_gb&sym_not": n10, "cna_not&sym_two_gb": n01, "cna_not&neither": n00,
                            "kappa": cohen_kappa(n11, n10, n01, n00), "q_k": sum(1 for x in rs if x["two_gb_cna"]),
                            "qloss_k": sum(1 for x in rs if x["loss_cna"]), "a": sum(1 for x in rs if x["ambiguous"]),
                            "n_status": dict(Counter(x["cna_status"] for x in rs))}
    van = [x for x in recs if x["arm"] == "baseline"]
    v74 = [x for x in van if x["sym_two_gb"]]
    dist = Counter(("ambiguous" if x["ambiguous"] else ("NA" if x["n_slab"] is None else x["n_slab"])) for x in v74)
    s3 = sum(1 for x in v74 if (not x["ambiguous"]) and x["n_slab"] is not None and x["n_slab"] >= 3)
    out["baseline_symfit_two_gb"] = {"n": len(v74), "n_slab_distribution": {str(k): v for k, v in sorted(dist.items(), key=lambda kv: str(kv[0]))},
                                    "ambiguous_in_74": sum(1 for x in v74 if x["ambiguous"]), "s3": s3}
    # input values of C3 and U
    import criteria as GT
    u_rows = [{"arm": x["arm"], "two_gb_cna": x["two_gb_cna"], "gb0_status": x["gb0_status"], "frac_A": x["frac_A"] or 0, "frac_B": x["frac_B"] or 0} for x in recs]
    u_total, u_single = GT.split_u([u for u in u_rows if u["frac_A"] != "" ])
    proposed = out["arms"].get("proposed")
    gi = {"q_k": out["arms"]["baseline"]["q_k"], "qloss_k": out["arms"]["baseline"]["qloss_k"], "a": out["arms"]["baseline"]["a"], "s3": s3,
          "proposed_k": proposed["q_k"] if proposed else None, "u_total": u_total, "u_single": u_single, "n": 350,
          "a_in_symfit_two_gb_74": out["baseline_symfit_two_gb"]["ambiguous_in_74"]}
    out["criterion_inputs_3"] = gi
    if proposed is not None:
        out["criterion_3_preview"] = GT.criterion_3(gi["q_k"], gi["qloss_k"], gi["a"], gi["s3"], gi["proposed_k"], gi["u_total"], gi["u_single"])
    return out, recs


def render_agreement_md(res, recs, desc):
    L = []
    A = L.append
    A("# Agreement of CNA with stages 0/1 (gb_symfit) (report only)")
    A("")
    A(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} / plan: {DESIGN}")
    A(f"- Input: {desc}")
    A("- The comparison is per structure (key (arm, chunk, cif_index)). A row whose CNA is not ok has two_gb_cna=False (miss). The denominator is the requested number per arm (350). kappa and the agreement rate are for reporting only.")
    A("")
    A("## 2x2 agreement table (rows = CNA two_gb_cna, columns = stage 0 gb0_status==two_gb)")
    A("")
    A("| arm | n | CNA yes, stage 0 yes | CNA yes, stage 0 no | CNA no, stage 0 yes | CNA no, stage 0 no | Cohen kappa | q_k | q_loss_k | ambiguous(a) |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for arm, v in res["arms"].items():
        kp = "n/a (one is constant)" if v["kappa"] is None else f"{v['kappa']:.3f}"
        A(f"| {arm} | {v['n']} | {v['cna_two_gb&sym_two_gb']} | {v['cna_two_gb&sym_not']} | {v['cna_not&sym_two_gb']} | {v['cna_not&neither']} | {kp} | {v['q_k']} | {v['qloss_k']} | {v['a']} |")
    A("")
    A("Breakdown of the status of CNA: " + "; ".join(f"{a}: {v['n_status']}" for a, v in res["arms"].items()))
    A("")
    vs = res["baseline_symfit_two_gb"]
    A(f"## Distribution of n_slab (CNA) for the {vs['n']} baseline structures with two_gb at stage 0")
    A("")
    A("| n_slab | count |")
    A("|---|---|")
    for k, v in vs["n_slab_distribution"].items():
        A(f"| {k} | {v} |")
    A("")
    A(f"- ambiguous (among the {vs['n']} with two_gb at stage 0): {vs['ambiguous_in_74']} (not counted in s3). s3 = number with n_slab >= 3 = {vs['s3']}.")
    A("")
    A("## Input values of C3 (the decision is made by a separate check. Below is a preview of criteria.criterion_3)")
    A("")
    A("```json")
    A(json.dumps({"inputs": res["criterion_inputs_3"], "preview": res.get("criterion_3_preview")}, ensure_ascii=False, indent=1, default=str))
    A("```")
    A("")
    A("## Sequence of bin_seq of stage 0 and CNA (baseline; stage 0 two_gb but CNA not two_gb / n_slab != 2; first 10; for diagnosis)")
    A("")
    A("| chunk | cif | CNA n_slab | CNA two_gb | stage 0 bin_seq (. A B) | CNA bin_seq (G D) |")
    A("|---|---|---|---|---|---|")
    n = 0
    for x in sorted((x for x in recs if x["arm"] == "baseline" and x["sym_two_gb"] and not x["two_gb_cna"]), key=lambda x: (x["chunk"], x["cif_index"])):
        A(f"| {x['chunk']} | {x['cif_index']} | {x['n_slab'] if x['n_slab'] is not None else 'NA'} | {x['two_gb_cna']} | `{x['sym_bin_seq']}` | `{x['cna_bin_seq']}` |")
        n += 1
        if n >= 10:
            break
    A("")
    return "\n".join(L) + "\n"


def write_once(path: Path, text):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["run"])
    ap.add_argument("--layer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--symfit-csv", default=None, help="if given, agreement.md and criterion_inputs_3.json are also written (the gb_symfit CSV of the layer)")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    out_dir = Path(K.P(args.out_dir))
    t0 = time.time()

    def log(m):
        print(f"[CNA {time.time() - t0:7.1f}s] {m}", flush=True)

    try:
        for n in ("agreement.md", "criterion_inputs_3.json"):
            if args.symfit_csv and (out_dir / n).exists():
                raise FileExistsError(f"refusing to overwrite: {out_dir / n}")
        run_layer(args.layer, out_dir, max(1, args.procs), log)
        if args.symfit_csv:
            cna_rows = K.read_csv_rows(out_dir / f"cna_{args.layer}.csv")
            sym_rows = K.read_csv_rows(K.P(args.symfit_csv))
            res, recs = agreement(cna_rows, sym_rows, args.layer)
            write_once(out_dir / "agreement.md", render_agreement_md(res, recs, f"cna={out_dir}/cna_{args.layer}.csv symfit={args.symfit_csv}"))
            write_once(out_dir / "criterion_inputs_3.json", json.dumps({"inputs": res["criterion_inputs_3"], "preview": res.get("criterion_3_preview"),
                                                                       "agreement": {"arms": res["arms"], "baseline_symfit_two_gb": res["baseline_symfit_two_gb"]}},
                                                                      ensure_ascii=False, indent=1, default=str) + "\n")
            log("wrote agreement.md and criterion_inputs_3.json")
    except (K.InputError, FileExistsError, FileNotFoundError) as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
    log("CNA DONE")


if __name__ == "__main__":
    main()
