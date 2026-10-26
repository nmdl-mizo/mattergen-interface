#!/usr/bin/env python3
"""aggregate_relax_novelty.py -- uniqueness decision (connected components), aggregation of the metrics, additional items, figures and report.

Plan: see the experiment description.
Usage (compute node of the CPU partition, python of the evaluation environment. No MLIP is used = CPU):
  python aggregate_relax_novelty.py --layers additional|main|both [--selected proposed:2.0,baseline:2.0] [--out-dir DIR] [--procs N]
  --layers additional : interim aggregation. Checks 2,800 rows and writes to results/interim_additional/. The main comparison section is not made (with --selected, exit 2).
  --layers main : interim aggregation of main only (700 rows, results/interim_main/).
  --layers both : final aggregation. Checks 3,500 rows (additional 2,800 + main 700) at the start and writes directly under results/. In production --selected proposed:2.0,baseline:2.0
                  (decision note. The main comparison is w=2.0 of main, the 5 inner targets, denominator 250).
Inputs (checked at the start. If even one is missing, exit 2 without writing): results/inputs_manifest_<layer>.csv, results/relax/<layer>/.../{RELAX_DONE,relax_rows.csv},
  results/novelty/<layer>/.../{NOVELTY_PRE_DONE,NOVELTY_POST_DONE,novelty_{pre,post}_rows.csv}, results/train_ref_manifest.csv
Output: relax_novelty_summary.json, relax_novelty_per_target.csv (layer x arm x w x target = 70 rows for both), relax_novelty_report.md, figs/relax_novelty_grid.{png,pdf}
Metrics: the denominator is the requested number n_req = 50 x number of targets (exclusions and unconverged are misses)
  (3) survive = #converged / n_req
  (4) novel(post) = #{converged and novel-post} / n_req, (4-pre) = #{valid_pre and novel-pre} / n_req
  (6-post) nun = sum_t number of connected components(S_post,t) / n_req, (6-pre) = sum_t number of connected components(S_pre,t) / n_req
     S_post = {converged and novel-post and abs(gamma_relaxed - gamma_target) <= 0.10} (UNI.fit among relaxed structures)
     S_pre  = {valid_pre and novel-pre and abs(gamma_pre - gamma_target) <= 0.10} (UNI.fit among structures before relaxation)
  Main decision (model specification): (1) medianabs(gamma_pre - gamma_target), (2) #{valid_pre and |Delta| <= 0.10}/n_req, (5) MAE (only structures with valid_pre; counts shown)
  Secondary (gamma after relaxation): (1')(2')(5') with the same formulas for gamma_relaxed (converged only)
  Wilson 95% CI for each rate and Newcombe (hybrid score) 95% CI for the difference proposed - baseline. No acceptance threshold is set.
Additional items (not used as acceptance conditions): stratification by the lattice distance delta of dx of proposed, stratification of the cell deviation of baseline by quartiles, vacuous novelty, breakdown of (3) (suspected grain-boundary loss, nearly converged)
Row label of the figure "(2') near (<=0.10)" = rate of (2') abs(gamma_relaxed - gamma_target) <= 0.10 J/m2 (the unit of the vertical axis is a single "Fraction" for the whole figure)
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

Z95 = 1.959963984540054
NEAR = K.NEAR_TOL_J
NEAR_CONV_FMAX = 0.05            # "nearly converged" = among the unconverged, fmax_final <= 0.05 eV/A
DX_FLIP_A = 0.065                # the lower end of the measured flip point (resolution of the atomic displacement)
INNER = [0, 1, 2, 3, 4]
EXTRAP = [5, 6]
ALL_T = list(range(7))
TARGET_SETS = {"inner5": INNER, "extrap2": EXTRAP, "all7": ALL_T}

RATE_METRICS = [   # (id, key of the cell statistics of the numerator, description)
    ("survive", "k_conv", "(3) converged / n_req"),
    ("novel_post", "k_novel_post", "(4) converged AND novel (after relaxation) / n_req"),
    ("novel_pre", "k_novel_pre", "(4-pre) valid_pre AND novel (before relaxation) / n_req"),
    ("nun_post", "cc_post", "(6-post) number of connected components that are novel, unique and near the target (after relaxation) / n_req"),
    ("nun_pre", "cc_pre", "(6-pre) number of connected components that are novel, unique and near the target (before relaxation) / n_req"),
    ("near_pre", "k_near_pre", "(2) valid_pre ∧ abs(γ_pre−γ_target) <= 0.10 / n_req"),
    ("near_post", "k_near_post", "(2') converged ∧ abs(γ_relaxed−γ_target) <= 0.10 / n_req"),
]
VALUE_METRICS = [  # (id, key of the value list, description of the count)
    ("median_abs_dgamma_pre", "abs_d_pre", "(1) median abs(gamma_pre-gamma_target) (valid_pre only)"),
    ("mae_pre", "abs_d_pre", "(5) MAE(gamma_pre, valid_pre only)"),
    ("median_abs_dgamma_post", "abs_d_post", "(1') median abs(gamma_relaxed-gamma_target) (converged only)"),
    ("mae_post", "abs_d_post", "(5') MAE(gamma_relaxed, converged only)"),
]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def bc_dev_axis_matched(cell_b, cell_c, bt, ct):
    """Cell deviation obtained by assigning the shorter edge to the tilt axis (b = bt of train) and the longer edge to c (ct of train). Assumes bt < ct. Returns (dev, whether the axes had been swapped)."""
    if not bt < ct:
        raise K.InputError(f"bc_dev_axis_matched requires bt < ct, got bt={bt} ct={ct}")
    s, l = sorted((cell_b, cell_c))
    return max(abs(s - bt) / bt, abs(l - ct) / ct), cell_b > cell_c


def wilson(k, n, z=Z95):
    if n <= 0:
        return None, None
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / den
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    # so that the end points do not exceed p because of floating-point rounding (negative yerr for 724157/724492)
    return min(max(0.0, c - h), p), max(min(1.0, c + h), p)


def newcombe(k1, n1, k2, n2, z=Z95):
    """Newcombe (hybrid score) interval of the difference p1 - p2."""
    if n1 <= 0 or n2 <= 0:
        return None, None
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mean(xs):
    return (sum(xs) / len(xs)) if xs else None


# ---------------------------------------------------------------------------
# loading of the inputs and completeness check
# ---------------------------------------------------------------------------
def tri(s):
    if s in ("True", "true", True):
        return True
    if s in ("False", "false", False):
        return False
    return None


def join_inputs(layers, args, mrows_by_layer):
    """Joins relax_rows and the novelty rows to the manifest rows (per layer) and returns records and problems (the count check is held by load_inputs)."""
    results = Path(K.P(f"{K.RUN}/results"))
    relax_root = Path(K.P(args.relax_root)) if args.relax_root else results / "relax"
    nov_root = Path(K.P(args.novelty_root)) if args.novelty_root else results / "novelty"
    records, problems = [], []
    for layer in layers:
        mrows = mrows_by_layer[layer]
        by_chunk = {}
        for r in mrows:
            by_chunk.setdefault((r["arm"], f"{float(r['w']):.1f}", int(r["chunk"])), {})[int(r["cif_index"])] = r
        for (arm, w, c), mr in sorted(by_chunk.items()):
            rd = relax_root / layer / K.armw(arm, float(w)) / K.chunk_name(c)
            nd = nov_root / layer / K.armw(arm, float(w)) / K.chunk_name(c)
            if not (rd / "RELAX_DONE").is_file():
                problems.append(f"{rd}/RELAX_DONE missing")
                continue
            rrows = K.read_csv_rows(rd / "relax_rows.csv")
            if len(rrows) != K.PER_CHUNK:
                problems.append(f"{rd}/relax_rows.csv has {len(rrows)} rows != 25")
                continue
            prow = postrow = None
            for ph in ("pre", "post"):
                if not (nd / f"NOVELTY_{ph.upper()}_DONE").is_file():
                    problems.append(f"{nd}/NOVELTY_{ph.upper()}_DONE missing")
                    continue
                nrows = K.read_csv_rows(nd / f"novelty_{ph}_rows.csv")
                if len(nrows) != K.PER_CHUNK:
                    problems.append(f"{nd}/novelty_{ph}_rows.csv has {len(nrows)} rows != 25")
                    continue
                if ph == "pre":
                    prow = {int(x["cif_index"]): x for x in nrows}
                else:
                    postrow = {int(x["cif_index"]): x for x in nrows}
            if prow is None or postrow is None:
                continue
            rr = {int(x["cif_index"]): x for x in rrows}
            for i in range(K.PER_CHUNK):
                m, x = mr[i], rr[i]
                if (x["excluded_pre"] or "") != (m["excluded_pre"] or ""):
                    problems.append(f"{rd}: excluded_pre differs between manifest and relax_rows at cif_index {i}")
                    continue
                status = x["status"]
                if status not in ("ok_converged", "ok_unconverged", "excluded_pre", "nonfinite", "error"):
                    problems.append(f"{rd}: unknown status {status!r} at {i}")
                    continue
                if (status == "excluded_pre") != bool(m["excluded_pre"]):
                    problems.append(f"{rd}: status/excluded_pre mismatch at {i}")
                    continue
                gpre = K.to_float(m["gamma_unrelaxed"]) if not m["excluded_pre"] else None
                records.append({
                    "layer": layer, "arm": arm, "w": w, "chunk": c, "cif_index": i, "target_id": int(m["target_id"]),
                    "is_inner": bool(int(m["is_inner"])), "gamma_target": float(m["gamma_target"]),
                    "excl_cat": m["excluded_cat"], "valid_pre": not m["excluded_pre"], "gamma_pre": gpre,
                    "novel_pre": tri(prow[i]["novel"]), "vac_pre": tri(prow[i]["novel_vacuous"]),
                    "status": status, "converged": status == "ok_converged",
                    "gamma_relaxed": K.to_float(x.get("gamma_relaxed")), "fmax_final": K.to_float(x.get("fmax_final")),
                    "novel_post": tri(postrow[i]["novel"]), "vac_post": tri(postrow[i]["novel_vacuous"]),
                    "gb_dx": K.to_float(m.get("gb_dx_realized_A")),
                    "cell_b": K.to_float(m.get("cell_b_A")), "cell_c": K.to_float(m.get("cell_c_A")),
                    "cell_alpha": K.to_float(m.get("cell_alpha_deg")), "cell_beta": K.to_float(m.get("cell_beta_deg")),
                    "cell_gamma": K.to_float(m.get("cell_gamma_deg")), "natoms": int(m["natoms"]),
                })
    return records, problems


def load_inputs(layers, args, train_info):
    """Reads the manifest, relax_rows and the novelty rows, checks completeness and returns the records. InputError if violated (nothing is written)."""
    import build_manifests as BM
    results = Path(K.P(f"{K.RUN}/results"))
    mrows_by_layer, n_total = {}, 0
    for layer in layers:
        mpath = Path(K.P(args.manifest_dir)) / f"inputs_manifest_{layer}.csv" if args.manifest_dir else results / f"inputs_manifest_{layer}.csv"
        if not mpath.is_file():
            raise K.InputError(f"{mpath} not found (the preceding stage is required)")
        mrows = K.read_csv_rows(mpath)
        BM.validate_manifest(mrows, layer)
        mrows_by_layer[layer] = mrows
        n_total += len(mrows)
    records, problems = join_inputs(layers, args, mrows_by_layer)
    if problems:
        raise K.InputError(f"CR: {len(problems)} problem(s) in inputs; nothing written. first: {problems[:5]}")
    if len(records) != n_total:
        raise K.InputError(f"CR: {len(records)} joined rows != manifest rows {n_total}")
    expect = {("additional",): 2800, ("main",): 700, ("additional", "main"): 3500}.get(tuple(layers))
    if expect is not None and n_total != expect:
        raise K.InputError(f"row count {n_total} != {expect} for layers {layers}")
    # consistency of novel: valid_pre <=> novel_pre is non-NA, converged <=> novel_post is non-NA
    for r in records:
        if r["valid_pre"] != (r["novel_pre"] is not None):
            raise K.InputError(f"novelty-pre NA inconsistent at {r['layer']}/{r['arm']}/w{r['w']}/chunk{r['chunk']}/{r['cif_index']}")
        if r["converged"] != (r["novel_post"] is not None):
            raise K.InputError(f"novelty-post NA inconsistent at {r['layer']}/{r['arm']}/w{r['w']}/chunk{r['chunk']}/{r['cif_index']}")
    return records


def load_train_info(train_manifest):
    rows = K.read_csv_rows(train_manifest)
    if len(rows) != 4419:
        raise K.InputError(f"train manifest rows {len(rows)} != 4419")
    g = [float(r["gamma_Jm2"]) for r in rows]
    dxs = sorted({round(float(r["dx_A"]), 9) for r in rows})

    def uniq(col, nd=6):
        return sorted({round(float(r[col]), nd) for r in rows})
    return {"gamma_min": min(g), "gamma_max": max(g), "dx_values": dxs, "n_dx_values": len(dxs),
            "cell_b_values": uniq("cell_b_A"), "cell_c_values": uniq("cell_c_A"),
            "cell_alpha_values": uniq("cell_alpha_deg", 4), "cell_beta_values": uniq("cell_beta_deg", 4),
            "cell_gamma_values": uniq("cell_gamma_deg", 4), "n_train": len(rows)}


# ---------------------------------------------------------------------------
# connected components (among the structures of S_pre and S_post)
# ---------------------------------------------------------------------------
_G = {}


def _comp_worker(task):
    key, structs = task
    matcher = _G["matcher"]
    n_comp, edges, n_calls = K.count_components_by_fit(len(structs), lambda i, j: matcher.fit(structs[i], structs[j]))
    return key, n_comp, len(edges), n_calls


def s_members(records):
    """(layer,arm,w,target,phase) -> [record]. Definition of S_pre / S_post."""
    out = {}
    for r in records:
        if r["valid_pre"] and r["novel_pre"] and r["gamma_pre"] is not None and abs(r["gamma_pre"] - r["gamma_target"]) <= NEAR:
            out.setdefault((r["layer"], r["arm"], r["w"], r["target_id"], "pre"), []).append(r)
        if r["converged"] and r["novel_post"] and r["gamma_relaxed"] is not None and abs(r["gamma_relaxed"] - r["gamma_target"]) <= NEAR:
            out.setdefault((r["layer"], r["arm"], r["w"], r["target_id"], "post"), []).append(r)
    return out


def compute_components(records, procs, log):
    """Number of connected components of S for each (layer,arm,w,target,phase). pre: UNI.fit among structures before relaxation, post: among relaxed structures."""
    from ase.io import read as ase_read
    S = s_members(records)
    pre_cache, post_cache = {}, {}
    tasks = []
    results = Path(K.P(f"{K.RUN}/results"))
    for key in sorted(S):
        layer, arm, w, t, phase = key
        structs = []
        for r in S[key]:
            c, i = r["chunk"], r["cif_index"]
            if phase == "pre":
                ck = (layer, arm, w, c)
                if ck not in pre_cache:
                    pre_cache[ck] = K.load_chunk_atoms(layer, arm, float(w), c)
                structs.append(K.to_pmg(pre_cache[ck][i][0]))
            else:
                ck = (layer, arm, w, c)
                if ck not in post_cache:
                    rd = Path(K.P(ARGS_RELAX_ROOT or f"{K.RUN}/results/relax")) / layer / K.armw(arm, float(w)) / K.chunk_name(c)
                    rrows = {int(x["cif_index"]): x for x in K.read_csv_rows(rd / "relax_rows.csv")}
                    frames = ase_read(str(rd / "relaxed.extxyz"), index=":")
                    post_cache[ck] = (rrows, frames)
                rrows, frames = post_cache[ck]
                fi = int(rrows[i]["frame_index"])
                if int(frames[fi].info.get("cif_index", -1)) != i:
                    raise K.InputError(f"relaxed.extxyz frame {fi} cif_index mismatch in {layer}/{arm}/w{w}/chunk{c}")
                structs.append(K.to_pmg(frames[fi]))
        tasks.append((key, structs))
    _G["matcher"] = K.make_matcher() if ARGS_STOL is None else K.make_matcher(stol=ARGS_STOL)
    if ARGS_STOL is not None:
        log(f"matcher: stol={_G['matcher'].stol} ltol={_G['matcher'].ltol} angle_tol={_G['matcher'].angle_tol} (--stol override)")
    comps, stats = {}, {"n_groups": len(tasks), "n_calls": 0, "n_edges": 0}
    t0 = time.time()
    if procs > 1 and len(tasks) > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(min(procs, len(tasks))) as pool:
            for key, n_comp, n_e, n_calls in pool.imap_unordered(_comp_worker, tasks, chunksize=1):
                comps[key] = n_comp
                stats["n_calls"] += n_calls
                stats["n_edges"] += n_e
    else:
        for tk in tasks:
            key, n_comp, n_e, n_calls = _comp_worker(tk)
            comps[key] = n_comp
            stats["n_calls"] += n_calls
            stats["n_edges"] += n_e
    log(f"components: {len(tasks)} groups, {stats['n_calls']} UNI.fit calls, {stats['n_edges']} edges, {time.time() - t0:.1f}s")
    return comps, stats


ARGS_RELAX_ROOT = None
ARGS_STOL = None   # --stol (None = the value of UNI as before)


# ---------------------------------------------------------------------------
# metrics (pure functions: also called from the degeneracy check and the FAIL validation)
# ---------------------------------------------------------------------------
def cell_stats(recs, comps, key3, t, train_gamma_min):
    """Raw statistics of one (layer,arm,w,target). recs are the 50 structures of that cell."""
    layer, arm, w = key3
    c = {"n_req": len(recs)}
    abs_pre = [abs(r["gamma_pre"] - r["gamma_target"]) for r in recs if r["valid_pre"] and r["gamma_pre"] is not None]
    abs_post = [abs(r["gamma_relaxed"] - r["gamma_target"]) for r in recs if r["converged"] and r["gamma_relaxed"] is not None]
    c.update({
        "n_valid_pre": sum(1 for r in recs if r["valid_pre"]),
        "excl_by_cat": {cat: sum(1 for r in recs if r["excl_cat"] == cat) for cat in K.EXCL_CATEGORIES},
        "k_conv": sum(1 for r in recs if r["converged"]),
        "n_unconverged": sum(1 for r in recs if r["status"] == "ok_unconverged"),
        "n_nonfinite": sum(1 for r in recs if r["status"] == "nonfinite"),
        "n_error": sum(1 for r in recs if r["status"] == "error"),
        "k_novel_post": sum(1 for r in recs if r["converged"] and r["novel_post"]),
        "k_novel_pre": sum(1 for r in recs if r["valid_pre"] and r["novel_pre"]),
        "k_near_pre": sum(1 for r in recs if r["valid_pre"] and r["gamma_pre"] is not None and abs(r["gamma_pre"] - r["gamma_target"]) <= NEAR),
        "k_near_post": sum(1 for r in recs if r["converged"] and r["gamma_relaxed"] is not None and abs(r["gamma_relaxed"] - r["gamma_target"]) <= NEAR),
        "cc_post": comps.get((layer, arm, w, t, "post"), 0),
        "cc_pre": comps.get((layer, arm, w, t, "pre"), 0),
        "abs_d_pre": abs_pre, "abs_d_post": abs_post,
        "k_vac_post": sum(1 for r in recs if r["converged"] and r["vac_post"]),
        "k_vac_pre": sum(1 for r in recs if r["valid_pre"] and r["vac_pre"]),
        "k_collapse_suspect": sum(1 for r in recs if r["converged"] and r["gamma_relaxed"] is not None and train_gamma_min is not None
                                  and r["gamma_relaxed"] < train_gamma_min),
        "k_near_converged": sum(1 for r in recs if r["status"] == "ok_unconverged" and r["fmax_final"] is not None and r["fmax_final"] <= NEAR_CONV_FMAX),
    })
    return c


def merge_cells(cells):
    """Adds up the cell statistics (statistics of the target set T)."""
    out = {"n_req": 0, "n_valid_pre": 0, "k_conv": 0, "n_unconverged": 0, "n_nonfinite": 0, "n_error": 0, "k_novel_post": 0, "k_novel_pre": 0,
           "k_near_pre": 0, "k_near_post": 0, "cc_post": 0, "cc_pre": 0, "k_vac_post": 0, "k_vac_pre": 0, "k_collapse_suspect": 0,
           "k_near_converged": 0, "abs_d_pre": [], "abs_d_post": [], "excl_by_cat": {cat: 0 for cat in K.EXCL_CATEGORIES}}
    for c in cells:
        for k in out:
            if k in ("abs_d_pre", "abs_d_post"):
                out[k] = out[k] + c[k]
            elif k == "excl_by_cat":
                for cat in K.EXCL_CATEGORIES:
                    out[k][cat] += c[k][cat]
            else:
                out[k] += c[k]
    return out


def derive_metrics(c):
    """Makes the values of the metrics from the statistics c (rates with k, n, p and the Wilson interval; value metrics with counts)."""
    n = c["n_req"]
    m = {"n_req": n}
    for mid, key, _desc in RATE_METRICS:
        k = c[key]
        lo, hi = wilson(k, n)
        m[mid] = {"k": k, "n": n, "p": (k / n) if n else None, "lo": lo, "hi": hi}
    vals = {"abs_d_pre": c["abs_d_pre"], "abs_d_post": c["abs_d_post"]}
    for mid, key, _desc in VALUE_METRICS:
        v = vals[key]
        m[mid] = {"n_used": len(v), "value": (median(v) if mid.startswith("median") else mean(v))}
    m["counts"] = {k: c[k] for k in ("n_valid_pre", "n_unconverged", "n_nonfinite", "n_error", "k_vac_post", "k_vac_pre",
                                     "k_collapse_suspect", "k_near_converged")}
    m["excl_by_cat"] = dict(c["excl_by_cat"])
    return m


def compute_all(records, comps, train_gamma_min):
    """Returns the metrics of all (layer,arm,w) x target sets and the per-target rows.
    Returns: per_target (list[dict]), sets (dict[(layer,arm,w,setname)] -> metrics), cells (dict)."""
    groups = {}
    for r in records:
        groups.setdefault((r["layer"], r["arm"], r["w"], r["target_id"]), []).append(r)
    cells = {}
    for (layer, arm, w, t), recs in sorted(groups.items()):
        cells[(layer, arm, w, t)] = cell_stats(recs, comps, (layer, arm, w), t, train_gamma_min)
    per_target = []
    for (layer, arm, w, t), c in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], float(kv[0][2]), kv[0][3])):
        m = derive_metrics(c)
        row = {"layer": layer, "arm": arm, "w": w, "target_id": t, "is_inner": int(t <= K.INNER_MAX_TARGET_ID), "n_req": c["n_req"]}
        for mid, _k, _d in RATE_METRICS:
            row[mid] = m[mid]["p"]
            row[mid + "_k"] = m[mid]["k"]
        for mid, _k, _d in VALUE_METRICS:
            row[mid] = m[mid]["value"]
            row[mid + "_n"] = m[mid]["n_used"]
        for cat in K.EXCL_CATEGORIES:
            row[f"excl_pre_{cat}"] = c["excl_by_cat"][cat]
        row.update({"n_unconverged": c["n_unconverged"], "n_nonfinite_R": c["n_nonfinite"], "n_error_R": c["n_error"],
                    "k_vacuous_novel_post": c["k_vac_post"], "k_collapse_suspect": c["k_collapse_suspect"], "k_near_converged": c["k_near_converged"]})
        per_target.append(row)
    sets = {}
    lawset = sorted({(k[0], k[1], k[2]) for k in cells}, key=lambda x: (x[0], x[1], float(x[2])))
    for (layer, arm, w) in lawset:
        for sname, ts in TARGET_SETS.items():
            sel = [cells[(layer, arm, w, t)] for t in ts if (layer, arm, w, t) in cells]
            if len(sel) != len(ts):
                continue
            sets[(layer, arm, w, sname)] = derive_metrics(merge_cells(sel))
    return per_target, sets, cells


def differences(sets):
    """proposed - baseline for the same (layer, w, set). Rates with the Newcombe 95% CI, value metrics as differences only."""
    out = {}
    keys = sorted({(k[0], k[2], k[3]) for k in sets}, key=lambda x: (x[0], float(x[1]), x[2]))
    for (layer, w, sname) in keys:
        h, v = sets.get((layer, "proposed", w, sname)), sets.get((layer, "baseline", w, sname))
        if h is None or v is None:
            continue
        d = {}
        for mid, _k, _d in RATE_METRICS:
            lo, hi = newcombe(h[mid]["k"], h[mid]["n"], v[mid]["k"], v[mid]["n"])
            d[mid] = {"diff": h[mid]["p"] - v[mid]["p"], "lo": lo, "hi": hi}
        for mid, _k, _d in VALUE_METRICS:
            hv, vv = h[mid]["value"], v[mid]["value"]
            d[mid] = {"diff": (hv - vv) if (hv is not None and vv is not None) else None}
        out[(layer, w, sname)] = d
    return out


def flat_diff_values(diffs):
    vals = []
    for key in sorted(diffs):
        for mid in sorted(diffs[key]):
            vals.append(((key, mid), diffs[key][mid]["diff"]))
    return vals


# ---------------------------------------------------------------------------
# additional items
# ---------------------------------------------------------------------------
def extra_tables(records, train_info):
    ex = {"train": {k: train_info[k] for k in ("gamma_min", "gamma_max", "n_dx_values", "cell_b_values", "cell_c_values",
                                               "cell_alpha_values", "cell_beta_values", "cell_gamma_values")}}
    dxv = train_info["dx_values"]

    def delta(gdx):
        return min(abs(gdx - d) for d in dxv)
    # 1. stratification of the novelty rate-post by the lattice distance delta of dx of proposed
    proposed = {}
    for r in records:
        if r["arm"] != "proposed" or r["gb_dx"] is None:
            continue
        d = delta(r["gb_dx"])
        s = "delta<=0.065" if d <= DX_FLIP_A else "delta>0.065"
        for scope in ("inner5", "all7"):
            if scope == "inner5" and not r["is_inner"]:
                continue
            e = proposed.setdefault(f"{r['layer']}|w{r['w']}|{scope}", {}).setdefault(s, {"n_rows": 0, "n_converged": 0, "n_novel_post": 0, "deltas": []})
            e["n_rows"] += 1
            e["deltas"].append(d)
            if r["converged"]:
                e["n_converged"] += 1
                e["n_novel_post"] += int(bool(r["novel_post"]))
    for scope in proposed.values():
        for s, e in scope.items():
            e["novel_rate_given_converged"] = (e["n_novel_post"] / e["n_converged"]) if e["n_converged"] else None
            e["delta_median_A"] = median(e["deltas"])
            e["delta_max_A"] = max(e["deltas"]) if e["deltas"] else None
            del e["deltas"]
    ex["proposed_dx_lattice_distance"] = proposed
    # 2. cell deviation of baseline (b and c are assigned to train with the shorter edge as the tilt axis. In baseline the order of b and c is opposite to that of train)
    bt = train_info["cell_b_values"]
    ct = train_info["cell_c_values"]
    cell_ok = len(bt) == 1 and len(ct) == 1
    van = {}
    if cell_ok:
        bt, ct = bt[0], ct[0]
        groups = {}
        for r in records:
            if r["arm"] != "baseline" or r["cell_b"] is None:
                continue
            dev, _swapped = bc_dev_axis_matched(r["cell_b"], r["cell_c"], bt, ct)
            ang = max(abs(r["cell_alpha"] - 90.0), abs(r["cell_beta"] - 90.0), abs(r["cell_gamma"] - 90.0))
            groups.setdefault((r["layer"], r["w"]), []).append((dev, ang, r))
        for (layer, w), lst in sorted(groups.items()):
            conv = [(d, a, r) for d, a, r in lst if r["converged"]]
            e = {"n_rows_with_cell": len(lst), "n_converged": len(conv), "max_angle_dev_deg": max(a for _, a, _ in lst),
                 "median_bc_dev": median([d for d, _, _ in lst]), "max_bc_dev": max(d for d, _, _ in lst), "quartile_edges": None, "strata": [],
                 "n_axis_swapped": sum(1 for _, _, r in lst if r["cell_b"] > r["cell_c"])}  # n_axis_swapped: number of rows with cell_b > cell_c (the order of the axes is opposite to that of train)
            if len(conv) >= 4:
                devs = sorted(d for d, _, _ in conv)
                edges = [devs[int(round(q * (len(devs) - 1)))] for q in (0.25, 0.5, 0.75)]
                e["quartile_edges"] = edges
                strata = [{"n_converged": 0, "n_novel_post": 0} for _ in range(4)]
                for d, _, r in conv:
                    q = 0 if d <= edges[0] else 1 if d <= edges[1] else 2 if d <= edges[2] else 3
                    strata[q]["n_converged"] += 1
                    strata[q]["n_novel_post"] += int(bool(r["novel_post"]))
                for s in strata:
                    s["novel_rate_given_converged"] = (s["n_novel_post"] / s["n_converged"]) if s["n_converged"] else None
                e["strata"] = strata
            van[f"{layer}|w{w}"] = e
    ex["baseline_cell_deviation"] = {"train_b_A": bt if cell_ok else train_info["cell_b_values"],
                                    "train_c_A": ct if cell_ok else train_info["cell_c_values"],
                                    "train_cell_single_valued": cell_ok, "per_layer_w": van}
    return ex


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def fnum(x, nd=3):
    return "NA" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def fci(m):
    return f"{fnum(m['p'])} [{fnum(m['lo'])}, {fnum(m['hi'])}] ({m['k']}/{m['n']})"


def write_report(path, layers, sets, diffs, per_target, extra, selected, stats, train_info, args_str):
    L = []
    A = L.append
    A("# Relaxed evaluation, novelty and uniqueness report of Al")
    A("")
    A(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} / layers: {', '.join(layers)} / command: `{args_str}`")
    A("- **Definition of \"novel\"**: StructureMatcher (stol=0.025, ltol/angle_tol default, primitive_cell=False, scale=False, attempt_supercell=False) matches none of the 4,419 training structures with the same number of atoms. "
      "This means that it is distinguished from the training set by an atomic displacement resolution of ~0.065 A (the lower end of the measured flip point).")
    A("- The **denominator** is the requested number (n_req = 50 x number of targets; 5 inner targets = 250, all 7 targets = 350). Exclusions (angle_fail, non_al, error, nonorthogonal, nonfinite) and unconverged structures are counted as misses.")
    A("- **The novelty rates before relaxation (4-pre) and (6-pre) are only listed in the table and not used for claims**. Structures as generated still have disordered atomic positions and hardly match the relaxed training structures, so they are expected to stick at "
      "99 to 100%. Read them together with (4) and (6-post) after relaxation.")
    A("- No acceptance threshold is set. Values are point estimates and Wilson 95% CI. The difference proposed - baseline is given with the Newcombe 95% CI.")
    A(f"- train: n={train_info['n_train']}, minimum of gamma {train_info['gamma_min']:.4f}, maximum {train_info['gamma_max']:.4f} J/m2, {train_info['n_dx_values']} distinct values of dx.")
    A(f"- Calls of UNI.fit (connected components): {stats['n_groups']} groups, {stats['n_calls']} calls, {stats['n_edges']} edges.")
    A("")
    if selected is not None:
        sl, sw = selected["layer"], selected["w"]
        A("## Main comparison (main layer, 5 inner targets, denominator 250)")
        A("")
        A(f"proposed w{sw['proposed']} vs baseline w{sw['baseline']}.")
        A("")
        A("| metric | proposed | baseline | proposed - baseline [Newcombe 95% CI] |")
        A("|---|---|---|---|")
        h = sets.get((sl, "proposed", sw["proposed"], "inner5"))
        v = sets.get((sl, "baseline", sw["baseline"], "inner5"))
        dd = None
        if sw["proposed"] == sw["baseline"]:
            dd = diffs.get((sl, sw["proposed"], "inner5"))
        for mid, _k, desc in RATE_METRICS:
            dtxt = ""
            if dd is not None:
                d = dd[mid]
                dtxt = f"{fnum(d['diff'])} [{fnum(d['lo'])}, {fnum(d['hi'])}]"
            A(f"| {desc} | {fci(h[mid])} | {fci(v[mid])} | {dtxt} |")
        for mid, _k, desc in VALUE_METRICS:
            A(f"| {desc} | {fnum(h[mid]['value'])} (n={h[mid]['n_used']}) | {fnum(v[mid]['value'])} (n={v[mid]['n_used']}) | "
              f"{fnum(dd[mid]['diff']) if dd else ''} |")
        A("")
    for sname, title in (("inner5", "5 inner targets (main aggregation, n_req=250)"), ("extrap2", "2 extrapolated targets (separate table, n_req=100)"), ("all7", "all 7 targets (n_req=350)")):
        A(f"## Metrics per layer x arm x w: {title}")
        A("")
        head = "| layer | arm | w | " + " | ".join(m[0] for m in RATE_METRICS) + " | " + " | ".join(m[0] for m in VALUE_METRICS) + " |"
        A(head)
        A("|" + "---|" * (3 + len(RATE_METRICS) + len(VALUE_METRICS)))
        for (layer, arm, w, s), m in sorted(sets.items(), key=lambda kv: (kv[0][0], float(kv[0][2]), kv[0][1])):
            if s != sname:
                continue
            A(f"| {layer} | {arm} | {w} | " + " | ".join(fci(m[mid]) for mid, _k, _d in RATE_METRICS) + " | "
              + " | ".join(f"{fnum(m[mid]['value'])} (n={m[mid]['n_used']})" for mid, _k, _d in VALUE_METRICS) + " |")
        A("")
    A("Correspondence of the metrics: " + " / ".join(f"`{mid}` = {desc}" for mid, _k, desc in RATE_METRICS + [(a, b, c) for a, b, c in VALUE_METRICS]))
    A("")
    A("## proposed - baseline at the same w (Newcombe 95% CI)")
    A("")
    A("| layer | w | set | " + " | ".join(m[0] for m in RATE_METRICS) + " |")
    A("|" + "---|" * (3 + len(RATE_METRICS)))
    for (layer, w, sname), d in sorted(diffs.items(), key=lambda kv: (kv[0][0], float(kv[0][1]), kv[0][2])):
        A(f"| {layer} | {w} | {sname} | " + " | ".join(f"{fnum(d[mid]['diff'])} [{fnum(d[mid]['lo'])}, {fnum(d[mid]['hi'])}]" for mid, _k, _d in RATE_METRICS) + " |")
    A("")
    A("## Breakdown of exclusions and unconverged structures (layer x arm x w x target; the denominator is the requested number 50)")
    A("")
    cols = [f"excl_pre_{c}" for c in K.EXCL_CATEGORIES] + ["n_unconverged", "n_nonfinite_R", "n_error_R"]
    A("| layer | arm | w | target | n_req | " + " | ".join(cols) + " | survive(3) |")
    A("|" + "---|" * (6 + len(cols)))
    for r in per_target:
        A(f"| {r['layer']} | {r['arm']} | {r['w']} | {r['target_id']} | {r['n_req']} | " + " | ".join(str(r[c]) for c in cols) + f" | {fnum(r['survive'])} |")
    A("")
    A("## Additional items (not used as acceptance conditions)")
    A("")
    A("### 1. Novelty rate (post) stratified by the lattice distance delta of dx of proposed (minimum distance to the dx values of the training set) (rate among converged structures)")
    A("")
    A("| layer/w/range | stratum | rows | converged | novel-post | novelty rate (among converged) | median delta [A] |")
    A("|---|---|---|---|---|---|---|")
    for scope, d in sorted(extra["proposed_dx_lattice_distance"].items()):
        for s, e in sorted(d.items()):
            A(f"| {scope} | {s} | {e['n_rows']} | {e['n_converged']} | {e['n_novel_post']} | {fnum(e['novel_rate_given_converged'])} | {fnum(e['delta_median_A'])} |")
    A("")
    vc = extra["baseline_cell_deviation"]
    A(f"### 2. Cell deviation of baseline (deviation after assigning the shorter edge to the tilt axis, after canonicalize. b = {vc['train_b_A']} and c = {vc['train_c_A']} of the training set, single value: {vc['train_cell_single_valued']})")
    A("")
    A("| layer/w | rows with a cell | converged | maximum angle deviation [deg] | median of max(abs(db/b), abs(dc/c)) | maximum | number of rows with swapped axes n_axis_swapped | quartile boundaries | novelty rate-post of Q1 to Q4 (among converged) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for k, e in sorted(vc["per_layer_w"].items()):
        qs = ", ".join(f"Q{i + 1}: {s['n_novel_post']}/{s['n_converged']}" for i, s in enumerate(e["strata"])) if e["strata"] else "NA(converged < 4)"
        A(f"| {k} | {e['n_rows_with_cell']} | {e['n_converged']} | {fnum(e['max_angle_dev_deg'], 4)} | {fnum(e['median_bc_dev'], 5)} | {fnum(e['max_bc_dev'], 5)} | "
          f"{e['n_axis_swapped']} | {e['quartile_edges']} | {qs} |")
    A("")
    A("### 3. Vacuous novelty (structures without a training structure of the same number of atoms; included in the novelty rate, with the count shown) and breakdown of (3)")
    A("")
    A(f"Suspected grain-boundary loss = converged and gamma_relaxed < the training minimum {train_info['gamma_min']:.4f} J/m2. Nearly converged = among the unconverged, fmax_final <= {NEAR_CONV_FMAX} eV/A.")
    A("")
    A("| layer | arm | w | set | converged | vacuous novelty (post) | vacuous novelty (pre) | suspected grain-boundary loss | nearly converged (among unconverged) | unconverged |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for (layer, arm, w, s), m in sorted(sets.items(), key=lambda kv: (kv[0][0], float(kv[0][2]), kv[0][1])):
        if s != "inner5":
            continue
        c = m["counts"]
        A(f"| {layer} | {arm} | {w} | {s} | {m['survive']['k']} | {c['k_vac_post']} | {c['k_vac_pre']} | {c['k_collapse_suspect']} | {c['k_near_converged']} | {c['n_unconverged']} |")
    A("")
    Path(path).write_text("\n".join(L) + "\n")


def make_figure(sets, outdir, layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, K.P(K.GEN_SCRIPTS))
    from paperfig import COL_2, OKABE_ITO, save_fig, setup_style
    setup_style()
    cols = []
    for (layer, arm, w, s) in sorted(sets, key=lambda k: (k[0], float(k[2]))):
        if s == "inner5" and arm == "proposed":
            cols.append((layer, w))
    # the row labels are wrapped short (the unit of the vertical axis is a single supylabel "Fraction" for the whole figure). (2') near = |gamma_relaxed-gamma_target| <= 0.10 J/m2
    rows = [("survive", "(3)\nsurvive"), ("novel_post", "(4) novel\n(post)"), ("nun_post", "(6-post) novel\n& unique & near"), ("near_post", "(2') near\n(≤0.10)")]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(COL_2, COL_2 * 0.14 * len(rows) + 0.6), squeeze=False, sharey="row")
    for ci, (layer, w) in enumerate(cols):
        for ri, (mid, ylab) in enumerate(rows):
            ax = axes[ri][ci]
            for ai, (arm, color) in enumerate((("proposed", OKABE_ITO[5]), ("baseline", OKABE_ITO[6]))):
                m = sets.get((layer, arm, w, "inner5"))
                if m is None:
                    continue
                x = m[mid]
                ax.bar(ai, x["p"], color=color, width=0.7)
                if x["lo"] is not None:
                    ax.errorbar(ai, x["p"], yerr=[[max(0.0, x["p"] - x["lo"])], [max(0.0, x["hi"] - x["p"])]], color="black", capsize=1.5, linewidth=0.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["proposed", "baseline"] if ri == len(rows) - 1 else ["", ""])
            ax.set_ylim(0, 1.05)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=7)
            if ri == 0:
                ax.set_title(f"{layer} w={w}")
    fig.supylabel("Fraction", x=0.005, fontsize=9)
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    figdir = Path(outdir) / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, "relax_novelty_grid", str(figdir), pdf=True)
    plt.close(fig)
    return [str(figdir / "relax_novelty_grid.png"), str(figdir / "relax_novelty_grid.pdf")]


PT_FIELD_ORDER = None


def write_outputs(outdir, layers, records, comps, stats, train_info, selected, args_str, procs=1, log=print):
    """Writes per_target, summary, report and figures to outdir (refused if they exist). Also called from the synthetic input of the degeneracy check."""
    outdir = Path(outdir)
    for f in ("relax_novelty_summary.json", "relax_novelty_per_target.csv", "relax_novelty_report.md"):
        if (outdir / f).exists():
            raise FileExistsError(f"refusing to overwrite {outdir / f}")
    outdir.mkdir(parents=True, exist_ok=True)
    per_target, sets, _cells = compute_all(records, comps, train_info["gamma_min"])
    diffs = differences(sets)
    n_laws = len({(r["layer"], r["arm"], r["w"]) for r in records})
    if len(per_target) != n_laws * K.N_TARGETS:
        raise K.InputError(f"check-A1: per_target rows {len(per_target)} != {n_laws} (layer,arm,w) x 7")
    extra = extra_tables(records, train_info)
    fields = list(per_target[0].keys())
    K.write_csv_rows(outdir / "relax_novelty_per_target.csv", fields, per_target)
    summary = {
        "layers": layers, "args": args_str, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "n_records": len(records),
        "n_per_target_rows": len(per_target), "uni_fit_stats": stats, "selected": selected,
        "definitions": {"n_req": "50 x number of targets (inner5=250, extrap2=100, all7=350)", "near_tol_Jm2": NEAR,
                        "novel": "StructureMatcher(UNI) finds no match among same-natoms train structures",
                        "denominator_rule": "exclusions and unconverged count as misses "},
        "sets": {f"{k[0]}|{k[1]}|w{k[2]}|{k[3]}": v for k, v in sets.items()},
        "proposed_minus_baseline": {f"{k[0]}|w{k[1]}|{k[2]}": v for k, v in diffs.items()},
        "extra_r2_6": extra,
    }
    if ARGS_STOL is not None:   # by default (None) the same set of JSON keys as before
        summary["stol"] = ARGS_STOL
    (outdir / "relax_novelty_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_report(outdir / "relax_novelty_report.md", layers, sets, diffs, per_target, extra, selected, stats, train_info, args_str)
    figs = make_figure(sets, outdir, layers)
    log(f"wrote {outdir}/relax_novelty_summary.json, relax_novelty_per_target.csv ({len(per_target)} rows), relax_novelty_report.md, {figs}")
    return per_target, sets, diffs


def parse_selected(s, layers):
    if s is None:
        return None
    if "main" not in layers:
        raise K.InputError("--selected requires --layers main or both (the interim aggregation --layers additional does not make the main comparison section)")
    out = {}
    for tok in s.split(","):
        arm, w = tok.split(":")
        if arm not in K.ARMS:
            raise K.InputError(f"--selected: unknown arm {arm!r}")
        out[arm] = f"{float(w):.1f}"
    if set(out) != set(K.ARMS):
        raise K.InputError("--selected must give both proposed:<W> and baseline:<W>")
    return {"layer": "main", "w": out}


def main():
    global ARGS_RELAX_ROOT, ARGS_STOL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", required=True, choices=["additional", "main", "both"])
    ap.add_argument("--selected", default=None, help="example: proposed:2.0,baseline:2.0 (the main comparison section is made only for --layers main|both)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--manifest-dir", default=None)
    ap.add_argument("--relax-root", default=None)
    ap.add_argument("--novelty-root", default=None)
    ap.add_argument("--train-manifest", default=None)
    ap.add_argument("--stol", type=float, default=None, help="overrides stol of StructureMatcher (default None = as before)")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    ARGS_RELAX_ROOT = args.relax_root
    ARGS_STOL = args.stol
    t0 = time.time()

    def log(msg):
        print(f"[A {time.time() - t0:7.1f}s] {msg}", flush=True)

    layers = ["additional", "main"] if args.layers == "both" else [args.layers]
    default_out = {"additional": "results/interim_additional", "main": "results/interim_main", "both": "results"}[args.layers]
    outdir = Path(K.P(args.out_dir or f"{K.RUN}/{default_out}"))
    try:
        selected = parse_selected(args.selected, layers)
        train_info = load_train_info(args.train_manifest or K.P(f"{K.RUN}/results/train_ref_manifest.csv"))
        records = load_inputs(layers, args, train_info)
        log(f"inputs complete: {len(records)} records (layers {layers})")
        comps, stats = compute_components(records, args.procs, log)
        write_outputs(outdir, layers, records, comps, stats, train_info, selected, " ".join(sys.argv), args.procs, log)
    except (K.InputError, FileExistsError) as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)
    log("A DONE")


if __name__ == "__main__":
    main()
