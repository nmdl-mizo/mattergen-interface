#!/usr/bin/env python3
"""low_gamma_nn.py -- nearest-neighbour matching against train (L->T) of the 10 low-gamma novel proposed structures of Al, and mutual matching among the 10 (descriptive, no gate, no decision).

Plan: see the experiment description.
Execution environment: ${MACE_PY} on a compute node. Not run on the login node.
Input (read only): the 10 rows with arm=proposed of VRUN/results/low_gamma_novel_view/table.csv (the trailing '#' row is skipped),
  <file_stem>.vasp in the same directory, RUN/results/relax/main/proposed_w2.0/chunkNN/relaxed.extxyz, RUN/results/train_ref_manifest.csv.
Output (new files only; stops before the computation if any exists): VRUN/results/low_gamma_novel_view/nn/{nn_L.csv,pairs_L.csv,summary.md,summary.json,NN_L.DONE}
Reused (not copied, not changed): rms_pair / vn13 / load_train / read_G_structures / write_rows / STOL_NN / CHUNK_REFS of VRUN/scripts/grid_interp.py
  and RUN/scripts/common.py (make_matcher, count_components_by_fit), which is read through it.
The definition of the distance is the same as in the grid-interpolation analysis: d_rms = rms_norm x vn13(x) [A] (primary), d_max = maxdist_norm x vn13(x) (secondary). x is the query side.
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import os
import socket
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------
# load grid_interp (not changed). The only side effect of the import is loading common (the RUN version) (only function definitions and constants; no computation or writing).
# VRUN/scripts also has a common.py, so it is loaded directly by file path, not through sys.path.
# ---------------------------------------------------------------------------
_PFX = '/CLUSTER_ROOT_B' if os.path.isdir(_gbrel.WORK) else '/mnt/CLUSTER_ROOT_A'
_GI_PATH = f"{_gbrel.CODE}/gb_eval/experiments/verification/scripts/grid_interp.py"
_spec = importlib.util.spec_from_file_location("grid_interp", _GI_PATH)
GI = importlib.util.module_from_spec(_spec)
sys.modules["grid_interp"] = GI
_spec.loader.exec_module(GI)

K = GI.K
P = K.P
rms_pair, vn13, STOL_NN, CHUNK_REFS = GI.rms_pair, GI.vn13, GI.STOL_NN, GI.CHUNK_REFS
log = GI.log

VRUN = GI.VRUN
VIEW = f"{VRUN}/results/low_gamma_novel_view"
TABLE_CSV = f"{VIEW}/table.csv"
NN_DIR_DEFAULT = f"{VIEW}/nn"
OUT_FILES = ["nn_L.csv", "pairs_L.csv", "summary.md", "summary.json", "NN_L.DONE"]
N_L = 10
EXPECT_TRAIN_N = {112: 1191, 116: 1003}      # specification of the plan
SELF_TOL_A = 1e-6                            # tolerance for comparing .vasp and extxyz [A]
SELF_MATCH_TOL_A = 1e-3                      # tolerance of the self-matching [A] (a numerical error of StructureMatcher of 1.7e-6 A was observed; meaningful distances are 0.02 A or more)
EXIT_AC = GI.EXIT_AC                         # 3
EXIT_INPUT = GI.EXIT_INPUT                   # 2
REF_SOURCE = f"{VRUN}/results/grid_interp/summary.md:3-5"
REF_VALUES = {"d_rms_median_G_to_T_A": 0.1651, "d_rms_median_E_to_T_A": 0.1842, "d_rms_median_dx_adjacent_pairs_A": 0.4551}


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------
def read_L_rows():
    """Rows of table.csv with arm=proposed and an integer rank (rank 1..10). The trailing '# train reference' row is excluded."""
    rows = []
    for r in K.read_csv_rows(TABLE_CSV):
        rk = (r.get("rank") or "").strip()
        if r.get("arm") != "proposed" or not rk.isdigit():
            continue
        rows.append(r)
    rows.sort(key=lambda r: int(r["rank"]))
    if [int(r["rank"]) for r in rows] != list(range(1, N_L + 1)):
        raise K.InputError(f"{TABLE_CSV}: proposed ranks {[r['rank'] for r in rows]} != 1..{N_L}")
    for r in rows:
        if int(r["natoms"]) not in EXPECT_TRAIN_N:
            raise K.InputError(f"rank {r['rank']}: natoms {r['natoms']} not in {sorted(EXPECT_TRAIN_N)}")
    return rows


def read_L_structures(rows):
    """Reads frame number frame_index of the extxyz (the master) with GI.read_G_structures (including the cross-check of cif_index and natoms)."""
    g_rows = [{"chunk": r["chunk"], "cif_index": r["cif_index"], "frame_index": r["frame_index"], "natoms": r["natoms"],
               "relaxed_extxyz": r["relaxed_path"]} for r in rows]
    return GI.read_G_structures(g_rows)


def check_vasp_vs_extxyz(rows, structs):
    """Reads .vasp and the extxyz frame with pymatgen and checks that lattice, coordinates (A, SELF_TOL_A), elements and number of atoms agree.
    Author's choice: the plan says one structure, but all 10 are checked (cheap). Coordinates are compared by periodic differences that remove lattice translations, and the raw Cartesian difference is also given."""
    import numpy as np
    from pymatgen.core import Structure
    out = []
    for r, sx in zip(rows, structs):
        path = f"{VIEW}/{r['file_stem']}.vasp"
        sv = Structure.from_file(P(path))
        if len(sv) != len(sx):
            raise K.InputError(f"{path}: natoms {len(sv)} != extxyz {len(sx)}")
        if [s.specie.Z for s in sv] != [s.specie.Z for s in sx]:
            raise K.InputError(f"{path}: species order differs from extxyz frame")
        dl = float(np.max(np.abs(sv.lattice.matrix - sx.lattice.matrix)))
        df = sv.frac_coords - sx.frac_coords
        raw = float(np.max(np.abs(sv.cart_coords - sx.cart_coords)))
        df -= np.round(df)
        per = float(np.max(np.abs(df @ sx.lattice.matrix)))
        out.append({"rank": int(r["rank"]), "file": os.path.basename(path), "max_abs_lattice_diff_A": dl,
                    "max_abs_cart_diff_raw_A": raw, "max_abs_coord_diff_periodic_A": per})
        log(f"[check-vasp] rank {r['rank']} {os.path.basename(path)}: lattice {dl:.3e} Å, coords raw {raw:.3e} / periodic {per:.3e} Å (tol {SELF_TOL_A:g})")
        if dl >= SELF_TOL_A or per >= SELF_TOL_A:
            raise K.InputError(f"{path}: .vasp vs extxyz mismatch (lattice {dl:.3e}, periodic coords {per:.3e} >= {SELF_TOL_A:g} Å)")
    return out


# ---------------------------------------------------------------------------
# self-matching
# ---------------------------------------------------------------------------
def self_check(x, m06, muni):
    r = rms_pair(m06, x, x)
    if r is None:
        raise K.InputError("[check-5 VIOLATION] self rms_pair returned None (censored)")
    v = vn13(x)
    d_rms, d_max = r[0] * v, r[1] * v
    fit = bool(muni.fit(x, x))
    ok = (d_rms < SELF_MATCH_TOL_A) and fit
    log(f"[check-5] self-match rank 1: d_rms={d_rms:.3e} Å, d_max={d_max:.3e} Å, fit(stol=0.025)={fit} -> {'OK' if ok else 'VIOLATION'}")
    return {"d_rms_A": d_rms, "d_max_A": d_max, "fit_uni_0025": fit, "ok": ok}


# ---------------------------------------------------------------------------
# matching (L->T)
# ---------------------------------------------------------------------------
_W = {}


def _nn_worker(task):
    qi, lo, hi = task
    q = _W["queries"][qi]
    refs = _W["refs_by_n"][len(q)]
    m06 = _W["m06"]
    t0 = time.time()
    res = [(j, rms_pair(m06, q, refs[j][2])) for j in range(lo, hi)]
    return qi, res, time.time() - t0


def nn_row(rank_row, q, refs, meta, res, muni):
    """Record of one structure. res = [(j, (rms_norm, maxdist_norm) | None)]."""
    v = vn13(q)
    res = sorted(res, key=lambda t: t[0])
    ok = [(j, r) for (j, r) in res if r is not None]
    by_rms = sorted(ok, key=lambda t: (t[1][0], t[1][1], t[0]))
    by_max = sorted(ok, key=lambda t: (t[1][1], t[1][0], t[0]))
    row = {"rank": int(rank_row["rank"]), "chunk": rank_row["chunk"], "cif_index": rank_row["cif_index"], "natoms": len(q),
           "gamma_relaxed": float(rank_row["gamma_relaxed"]), "vn13_x_A": v, "n_compared": len(res),
           "n_censored": len(res) - len(ok)}
    tests = {}
    if by_rms:
        j, r = by_rms[0]
        ri = refs[j][0]
        row.update({"d_rms_min_A": r[0] * v, "nn_row_index": ri, "nn_cnid": meta[ri]["cnid"], "nn_k": meta[ri]["k"], "nn_dx_A": meta[ri]["dx_A"],
                    "nn_gamma_Jm2": meta[ri]["gamma_Jm2"], "nn_d_max_A": r[1] * v})
        tests[ri] = refs[j][2]
        if len(by_rms) > 1:
            j2, r2 = by_rms[1]
            ri2 = refs[j2][0]
            row.update({"d_rms_2nd_A": r2[0] * v, "nn2_row_index": ri2, "nn2_cnid": meta[ri2]["cnid"]})
        j3, r3 = by_max[0]
        ri3 = refs[j3][0]
        row.update({"d_max_min_A": r3[1] * v, "dmax_nn_row_index": ri3, "dmax_nn_cnid": meta[ri3]["cnid"]})
        tests[ri3] = refs[j3][2]
    row["n_fit_uni"] = sum(1 for t in tests.values() if muni.fit(q, t))      # computed with 2 structures (at most) only
    row["n_fit_uni_tested"] = len(tests)
    return row


NN_FIELDS = ["rank", "chunk", "cif_index", "natoms", "gamma_relaxed", "vn13_x_A", "n_compared", "n_censored",
             "d_rms_min_A", "nn_row_index", "nn_cnid", "nn_k", "nn_dx_A", "nn_gamma_Jm2", "nn_d_max_A",
             "d_rms_2nd_A", "nn2_row_index", "nn2_cnid",
             "d_max_min_A", "dmax_nn_row_index", "dmax_nn_cnid", "n_fit_uni", "n_fit_uni_tested"]
PAIR_FIELDS = ["rank_a", "rank_b", "natoms", "d_rms_A", "d_max_A", "censored", "fit_uni_0025"]


def _pair_worker(job):
    ia, ib = job
    a, b = _W["L"][ia], _W["L"][ib]
    r = rms_pair(_W["m06"], a, b)
    f = bool(_W["muni"].fit(a, b))
    return job, r, f


def stat3(vals):
    return {"n": len(vals), "min": min(vals), "median": statistics.median(vals), "max": max(vals)} if vals else {"n": 0}


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def build_summary_md(S):
    L = []
    L.append("# Nearest-neighbour matching against train of the 10 low-gamma novel proposed structures of Al, and mutual matching (descriptive, no gate, no decision)\n")
    L.append(f"- Generated: {S['generated_utc']} / host {S['host']} / job {S['slurm_job_id']}")
    L.append(f"- Distance: d_rms = rms_norm x vn13(x) [A] (primary), d_max = maxdist_norm x vn13(x) (secondary). x is the query side. matcher stol={STOL_NN} (cut-off = rms_norm >= {STOL_NN}).")
    L.append(f"- Self-matching: rank 1 vs itself d_rms={S['self_check']['d_rms_A']:.3e} A, fit={S['self_check']['fit_uni_0025']}\n")
    L.append("## 1. L->T nearest train (nn_L.csv)\n")
    L.append("| rank | chunk | cif | natoms | γ_relaxed | n_cmp | n_cens | d_rms_min [Å] | train row | cnid | k | dx [Å] | γ_train | d_max@that train [Å] | d_rms second [Å] | second row/cnid | d_max_min [Å] | that row/cnid | n_fit_uni |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def f(v, n=4):
        return "" if v in (None, "") else (f"{v:.{n}f}" if isinstance(v, float) else str(v))
    for r in S["nn_rows"]:
        L.append(f"| {r['rank']} | {r['chunk']} | {r['cif_index']} | {r['natoms']} | {f(r['gamma_relaxed'])} | {r['n_compared']} | {r['n_censored']} | "
                 f"{f(r.get('d_rms_min_A'))} | {f(r.get('nn_row_index'))} | {f(r.get('nn_cnid'))} | {f(r.get('nn_k'))} | {f(r.get('nn_dx_A'), 2)} | {f(r.get('nn_gamma_Jm2'))} | "
                 f"{f(r.get('nn_d_max_A'))} | {f(r.get('d_rms_2nd_A'))} | {f(r.get('nn2_row_index'))}/{f(r.get('nn2_cnid'))} | {f(r.get('d_max_min_A'))} | "
                 f"{f(r.get('dmax_nn_row_index'))}/{f(r.get('dmax_nn_cnid'))} | {r['n_fit_uni']} |")
    d = S["d_rms_min_stats"]
    L.append("")
    if d["n"]:
        L.append(f"- d_rms_min [A]: range {d['min']:.4f} - {d['max']:.4f}, median {d['median']:.4f} (valid {d['n']}/{N_L})")
    L.append(f"- cnid of the nearest train (smallest d_rms): {len(S['nn_cnid_counts'])} kinds, frequencies " + ", ".join(f"{k}:{v}" for k, v in S["nn_cnid_counts"].items()))
    L.append(f"- dx [A] of the nearest train (smallest d_rms): " + ", ".join(f"{k}:{v}" for k, v in S["nn_dx_counts"].items()))
    L.append(f"- Total of n_fit_uni (number of train structures for which the fit at stol 0.025 is True, checked with the nearest 2): {S['n_fit_uni_total']}\n")
    L.append("## 2. Mutual matching of L (pairs_L.csv)\n")
    p = S["pairs"]
    L.append(f"- Of all {p['n_pairs_all']} pairs, pairs with the same number of atoms: {p['n_pairs_same_natoms']} (112 vs 112: {p['n_pairs_112']}, 116 vs 116: {p['n_pairs_116']}),"
             f"pairs with different numbers of atoms that cannot be matched: {p['n_pairs_cross_natoms']}")
    L.append(f"- Censored pairs: {p['n_censored']}")
    L.append(f"- Pairs with fit (stol 0.025) True: {p['n_fit_true']} / {p['n_pairs_same_natoms']}")
    L.append(f"- Number of connected components built from the pairs with fit True: group of 112: {p['components_112']}/5, group of 116: {p['components_116']}/5, total {p['components_total']} (number of distinct structures by this instrument)")
    dr = p["d_rms_stats"]
    if dr["n"]:
        L.append(f"- d_rms of pairs [A]: range {dr['min']:.4f} - {dr['max']:.4f}, median {dr['median']:.4f} (valid {dr['n']})")
    dm = p["d_max_stats"]
    if dm["n"]:
        L.append(f"- d_max of pairs [A] (secondary): range {dm['min']:.4f} - {dm['max']:.4f}, median {dm['median']:.4f}")
    L.append("")
    L.append(f"## 3. Reference values (listed only for comparison. Source: {REF_SOURCE})\n")
    L.append(f"- Median d_rms G->T {REF_VALUES['d_rms_median_G_to_T_A']}, E->T {REF_VALUES['d_rms_median_E_to_T_A']}, adjacent dx pairs {REF_VALUES['d_rms_median_dx_adjacent_pairs_A']} A\n")
    L.append("No decision statement is written (no gate, descriptive).")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=NN_DIR_DEFAULT)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args(argv)
    out = Path(P(args.out_dir))
    try:
        return run(args, out)
    except (K.InputError, FileExistsError) as e:
        log(f"[FATAL] {type(e).__name__}: {e}")
        return EXIT_INPUT


def run(args, out):
    # if an output already exists, stop before the computation (existing files are not rewritten)
    exist = [str(out / n) for n in OUT_FILES if (out / n).exists()]
    if exist:
        raise FileExistsError(f"existing outputs, refusing to overwrite: {exist}")
    t_start = time.time()
    rows = read_L_rows()
    structs = read_L_structures(rows)
    vcheck = check_vasp_vs_extxyz(rows, structs)
    natoms_list = [len(s) for s in structs]
    n_by = Counter(natoms_list)
    log(f"[L] proposed ranks 1..{N_L}: natoms={natoms_list} counts={dict(n_by)}")
    _rows_t, meta, by_n = GI.load_train(set(natoms_list))
    for n, cnt in EXPECT_TRAIN_N.items():
        if n in n_by and len(by_n.get(n, [])) != cnt:
            raise K.InputError(f"[check-2 precondition] train natoms {n}: {len(by_n.get(n, []))} != expected {cnt}")
    m06 = K.make_matcher(stol=STOL_NN)
    muni = K.make_matcher()
    log(f"[env] host={socket.gethostname()} job={os.environ.get('SLURM_JOB_ID', '')} procs={args.procs} train="
        f"{{{', '.join(f'{n}:{len(by_n[n])}' for n in sorted(by_n))}}} out_dir={out}")

    # self-matching (non-zero exit on failure)
    sc = self_check(structs[0], m06, muni)
    if not sc["ok"]:
        log("[check-5 VIOLATION] self-match check failed; abort")
        return EXIT_AC

    # time taken by the first 5 matchings
    tp = []
    for q in structs[:5]:
        t0 = time.time()
        rms_pair(m06, q, by_n[len(q)][0][2])
        tp.append(time.time() - t0)
    log(f"[TIMING] rms_pair(stol={STOL_NN}) per call: n={len(tp)} mean={sum(tp) / len(tp):.3f}s max={max(tp):.3f}s"
        + (" [FLAG: mean >5s]" if sum(tp) / len(tp) > 5.0 else ""))

    # --- 3-1: L→T ---
    tasks = []
    for qi, q in enumerate(structs):
        n_ref = len(by_n[len(q)])
        for lo in range(0, n_ref, CHUNK_REFS):
            tasks.append((qi, lo, min(lo + CHUNK_REFS, n_ref)))
    _W.update({"queries": structs, "refs_by_n": by_n, "m06": m06, "muni": muni, "L": structs})
    results = {qi: [] for qi in range(len(structs))}
    t0 = time.time()
    if args.procs > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(min(args.procs, len(tasks))) as pool:
            for qi, res, _dt in pool.imap_unordered(_nn_worker, tasks, chunksize=1):
                results[qi].extend(res)
    else:
        for tk in tasks:
            qi, res, _dt = _nn_worker(tk)
            results[qi].extend(res)
    n_calls = sum(len(v) for v in results.values())
    log(f"[L->T] {n_calls} comparisons ({len(tasks)} tasks) in {time.time() - t0:.1f}s wall")
    nn_rows = [nn_row(rows[qi], structs[qi], by_n[len(structs[qi])], meta, results[qi], muni) for qi in range(len(structs))]
    for r in nn_rows:
        log(f"[L->T] rank {r['rank']} natoms {r['natoms']}: n_compared={r['n_compared']} n_censored={r['n_censored']} "
            f"d_rms_min={r.get('d_rms_min_A', float('nan')):.4f} nn=row {r.get('nn_row_index')} {r.get('nn_cnid')} n_fit_uni={r['n_fit_uni']}")

    # --- 3-2: mutual matching of L ---
    pair_jobs = [(a, b) for a in range(N_L) for b in range(a + 1, N_L) if natoms_list[a] == natoms_list[b]]
    expect_pairs = sum(math.comb(c, 2) for c in n_by.values())
    if len(pair_jobs) != expect_pairs:
        raise K.InputError(f"[check-3 precondition] pair jobs {len(pair_jobs)} != sum C(n,2) {expect_pairs}")
    t0 = time.time()
    if args.procs > 1 and len(pair_jobs) > 1:
        with mp.get_context("fork").Pool(min(args.procs, len(pair_jobs))) as pool:
            pres = list(pool.imap_unordered(_pair_worker, pair_jobs, chunksize=1))
    else:
        pres = [_pair_worker(j) for j in pair_jobs]
    log(f"[pairs] {len(pres)} same-natoms pairs in {time.time() - t0:.1f}s wall")
    pres = {job: (r, f) for job, r, f in pres}
    pair_rows = []
    for (a, b) in pair_jobs:
        r, f = pres[(a, b)]
        v = vn13(structs[a])
        pair_rows.append({"rank_a": a + 1, "rank_b": b + 1, "natoms": natoms_list[a], "d_rms_A": (r[0] * v) if r else None,
                          "d_max_A": (r[1] * v) if r else None, "censored": r is None, "fit_uni_0025": f})
    comp = {}
    for n in sorted(n_by):
        idx = [i for i in range(N_L) if natoms_list[i] == n]
        loc = {g: k for k, g in enumerate(idx)}
        fit_tab = {(loc[a], loc[b]): pres[(a, b)][1] for (a, b) in pair_jobs if natoms_list[a] == n}
        ncomp, edges, ncalls = K.count_components_by_fit(len(idx), lambda i, j: fit_tab[(i, j)])
        comp[n] = {"n_structs": len(idx), "n_components": ncomp, "edges_rank": [[idx[i] + 1, idx[j] + 1] for i, j in edges]}
        log(f"[pairs] natoms {n}: {len(idx)} structures -> {ncomp} connected component(s) by fit(stol=0.025); edges(rank)={comp[n]['edges_rank']}")

    # --- output ---
    out.mkdir(parents=True, exist_ok=True)
    GI.write_rows(out / "nn_L.csv", NN_FIELDS, nn_rows, False)
    GI.write_rows(out / "pairs_L.csv", PAIR_FIELDS, pair_rows, False)
    d_rms_min = [r["d_rms_min_A"] for r in nn_rows if "d_rms_min_A" in r]
    pv = [r["d_rms_A"] for r in pair_rows if r["d_rms_A"] is not None]
    pm = [r["d_max_A"] for r in pair_rows if r["d_max_A"] is not None]
    n_all = math.comb(N_L, 2)
    S = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""), "design": "an internal design note that is not included",
        "stol_nn": STOL_NN, "elapsed_wall_s": time.time() - t_start,
        "self_check": sc, "vasp_vs_extxyz": vcheck, "nn_rows": nn_rows,
        "d_rms_min_stats": stat3(d_rms_min),
        "nn_cnid_counts": dict(Counter(r["nn_cnid"] for r in nn_rows if "nn_cnid" in r).most_common()),
        "nn_dx_counts": {str(k): v for k, v in Counter(r["nn_dx_A"] for r in nn_rows if "nn_dx_A" in r).most_common()},
        "n_fit_uni_total": sum(r["n_fit_uni"] for r in nn_rows),
        "pairs": {"n_pairs_all": n_all, "n_pairs_same_natoms": len(pair_rows), "n_pairs_cross_natoms": n_all - len(pair_rows),
                  "n_pairs_112": sum(1 for r in pair_rows if r["natoms"] == 112), "n_pairs_116": sum(1 for r in pair_rows if r["natoms"] == 116),
                  "n_censored": sum(1 for r in pair_rows if r["censored"]), "n_fit_true": sum(1 for r in pair_rows if r["fit_uni_0025"]),
                  "components_112": comp.get(112, {}).get("n_components"), "components_116": comp.get(116, {}).get("n_components"),
                  "components_total": sum(c["n_components"] for c in comp.values()), "components_detail": {str(k): v for k, v in comp.items()},
                  "d_rms_stats": stat3(pv), "d_max_stats": stat3(pm)},
        "reference_values": {"source": REF_SOURCE, **REF_VALUES},
    }
    with open(out / "summary.json", "x") as fh:
        json.dump(S, fh, indent=2, ensure_ascii=False, default=str)
    with open(out / "summary.md", "x") as fh:
        fh.write(build_summary_md(S))
    n_viol = S["n_fit_uni_total"]
    if n_viol:
        log(f"[nn][check-2 VIOLATION] n_fit_uni total = {n_viol} (expected 0); DONE marker not written")
        return EXIT_AC
    log("[nn][check-2 precondition OK] n_fit_uni = 0 for all 10 (nearest by d_rms and by d_max)")
    sys.path.insert(0, P(K.RUNNER_GUARDS_DIR))
    import runner_guards as RG  # noqa: WPS433
    RG.write_done_marker(str(out), "*.csv", 2, marker="NN_L.DONE")
    log(f"[done] wrote {', '.join(OUT_FILES)} in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
