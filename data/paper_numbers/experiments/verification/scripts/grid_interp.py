#!/usr/bin/env python3
"""grid_interp.py -- relation between the novel Al structures and the training lattice of dx values (gives distributions of distances. Descriptive; no decision).

Plan: see the experiment description.
Subcommands: select (list of G and E, task_map) / nn (nearest neighbors G->T and E->T) / dxpairs (dx neighbor pairs within T) / aggregate / --self-check
Execution environment: the python of the evaluation environment (environment-eval.yml) on a compute node. Do not start it on the login node.
Reuse (not copied, not changed): RUN/scripts/common.py is read with importlib under the alias common_run
  (VRUN/scripts also has common.py, and a plain `import common` would pick the VRUN one).

Return values of get_rms_dist of pymatgen (checked in site-packages/pymatgen/analysis/structure_matcher.py, pymatgen 2025.6.14;
  the same file in the <CLUSTER_MOUNT>/... notation on compute nodes):
  - :643-663 get_rms_dist(struct1, struct2): calls `self._match(..., use_rms=True, break_on_match=False)`, returns None if None,
    otherwise `return match[0], max(match[1])` = (rms, maximum site distance) in this order.
  - :781 `normalization = (len(s1fc) / avg_l.volume) ** (1 / 3)`, the docstring of `_cart_dists` at :520-552 "normalized by (V/atom)^1/3",
    :552 `return new_d2**0.5 * normalization`: distances are per site. Both return values are normalized by (V/N)^(1/3) (V is the volume of the average lattice of the two structures;
    with scale=False the lattice is not rescaled). Distances after removing the mean translation (:543-546 `translation = np.mean(short_vecs)`).
  - :792 `val = np.linalg.norm(dist) / len(dist) ** 0.5 if use_rms else max(dist)`: get_rms_dist chooses the mapping with the **minimum rms**
    (fit uses the minimum of max). The returned maximum distance is the maximum site distance of the "minimum-rms mapping" and the mapping may differ from the one that gives the decision quantity of fit (minimum-max mapping).
  - :801-803 `if best_match and best_match[0] < self.stol: return best_match` / `return None`: the comparison with stol is made for the **rms**
    (with use_rms=True, match[0] is the rms). None = there is no mapping with rms<stol, or the lattice (ltol, angle_tol) or the composition does not match.
    Therefore "censored" means rms_norm >= stol (=0.6); even if the maximum distance exceeds 0.6, a value is returned if rms<0.6 (censored=False).
  - Conversion in this script: d_max = maxdist_norm x (V_x/N)^(1/3), V_x is the volume of the x side (query). The normalization inside pymatgen uses the volume of the average lattice,
    so the conversion is slightly off by the difference between the lattices of x and y (at most 1.5% in the a axis only for dx neighbors and G->T). vn13_x used for the conversion is kept in a column.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import socket
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------
# loading of the RUN version of common (not changed)
# ---------------------------------------------------------------------------
_PFX = '/CLUSTER_ROOT_B' if os.path.isdir('<ORIGINAL_ROOT>') else '/mnt/CLUSTER_ROOT_A'
_RUN_COMMON = f"<ORIGINAL_ROOT>/workspace/experiments/relax_novelty/scripts/common.py"
_spec = importlib.util.spec_from_file_location("common_run", _RUN_COMMON)
K = importlib.util.module_from_spec(_spec)
sys.modules["common_run"] = K
_spec.loader.exec_module(K)

GC = K.GC
RUN = K.RUN
VRUN = f"{GC}/experiments/verification"
DATA = K.DATA
OUT_DEFAULT = f"{VRUN}/results/grid_interp"
TRAIN_MANIFEST = f"{RUN}/results/train_ref_manifest.csv"
INPUTS_MANIFEST_MAIN = f"{RUN}/results/inputs_manifest_main.csv"
PER_TARGET_CSV = f"{RUN}/results/relax_novelty_per_target.csv"
INSTRUMENT_JSON = f"{RUN}/verification/instrument_resolution_dx.json"

DX10 = [-0.75, -0.60, -0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45, 0.60]   # value in the data-rebuild stage
TEMPLATE_A = K.TEMPLATE_A
STOL_NN = 0.6                      # stol for get_rms_dist
STOL_UNI = 0.025                   # stol of UNI (stage_d_common.UNI). Invariant check, check b and the vertical line
E_SEED = 20261008
E_N = 120
N_G_PARTS = 10
N_E_PARTS = 12
N_TASKS = N_G_PARTS + N_E_PARTS + 1    # 23
EXPECT_PAIRS = 3331
EXPECT_SAME = 166
CHUNK_REFS = 25                    # Pool task of nn = (query, 25 consecutive training structures)
EXIT_AC = 3                        # exit code for violations of the checks (select/nn/dxpairs/self-check)
EXIT_INPUT = 2
DELTA_TOL = 0.02
DELTA_UNIFORM = 0.04 / 0.15


def P(p):
    return K.P(p)


def log(msg):
    print(msg, flush=True)


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return repr(float(v))
    return v


def write_rows(path, fields, rows, overwrite):
    K.write_csv_rows(path, fields, [{k: fmt(v) for k, v in r.items()} for r in rows], overwrite=overwrite)


def done_marker(out_dir, glob, marker):
    sys.path.insert(0, P(K.RUNNER_GUARDS_DIR))
    import runner_guards as RG  # noqa: WPS433
    return RG.write_done_marker(str(out_dir), glob, 1, marker=marker)


def task_map_rows():
    """Correspondence task -> (set, split). Must be identical to the bash arithmetic of sbatch_grid_interp.sh."""
    rows = []
    for i in range(N_G_PARTS):
        rows.append({"task_id": i, "set": "G", "part": i, "nparts": N_G_PARTS, "subcommand": "nn", "out_file": f"nn_G_{i:02d}.csv"})
    for j in range(N_E_PARTS):
        rows.append({"task_id": N_G_PARTS + j, "set": "E", "part": j, "nparts": N_E_PARTS, "subcommand": "nn", "out_file": f"nn_E_{j:02d}.csv"})
    rows.append({"task_id": N_G_PARTS + N_E_PARTS, "set": "T", "part": 0, "nparts": 1, "subcommand": "dxpairs", "out_file": "dxpairs.csv"})
    assert len(rows) == N_TASKS
    return rows


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------
def vn13(struct):
    return (struct.volume / len(struct)) ** (1.0 / 3.0)


def rms_pair(matcher, x, y):
    """(rms_norm, maxdist_norm) or None (censored). The order of the return values of get_rms_dist of pymatgen is as in the docstring above (:663)."""
    r = matcher.get_rms_dist(x, y)
    if r is None:
        return None
    return float(r[0]), float(r[1])


def nearest_level(gb_dx):
    """Nearest DX10 level and the difference delta from it, and whether it is outside the range. For outside the range (<-0.75 or >0.60) delta is the difference from the nearest end point (same formula)."""
    lv = min(DX10, key=lambda v: abs(gb_dx - v))
    return lv, gb_dx - lv, (gb_dx < DX10[0] - 1e-9) or (gb_dx > DX10[-1] + 1e-9)


# ---------------------------------------------------------------------------
# loading of train / test
# ---------------------------------------------------------------------------
def load_train(natoms_set=None):
    """All manifest rows and {natoms: [(row_index, cnid, Structure)]} (only those atom numbers if natoms_set is given)."""
    from pymatgen.core import Structure
    rows = K.read_csv_rows(TRAIN_MANIFEST)
    by_n, meta = {}, {}
    for r in rows:
        ri, n = int(r["row_index"]), int(r["natoms"])
        meta[ri] = {"cnid": r["cnid"], "k": int(r["k"]), "seed": int(r["seed"]), "dx_A": float(r["dx_A"]),
                    "gamma_Jm2": float(r["gamma_Jm2"]), "natoms": n}
        if natoms_set is not None and n not in natoms_set:
            continue
        s = Structure.from_file(P(r["poscar_relaxed_path"]))
        if len(s) != n:
            raise K.InputError(f"train row {ri}: natoms {len(s)} != manifest {n}")
        by_n.setdefault(n, []).append((ri, r["cnid"], s))
    return rows, meta, by_n


def read_G_structures(g_rows):
    """Relaxed structure of each row of G_list (frame number frame_index of relaxed.extxyz, cif_index checked). A list of pymatgen Structure."""
    from ase.io import read as ase_read
    cache, out = {}, []
    for r in g_rows:
        c, i, fi = int(r["chunk"]), int(r["cif_index"]), int(r["frame_index"])
        if c not in cache:
            cache[c] = ase_read(P(r["relaxed_extxyz"]), index=":")
        frame = cache[c][fi]
        if int(frame.info.get("cif_index", -1)) != i:
            raise K.InputError(f"chunk {c}: relaxed.extxyz frame {fi} cif_index {frame.info.get('cif_index')} != {i}")
        s = K.to_pmg(frame)
        if len(s) != int(r["natoms"]):
            raise K.InputError(f"chunk {c} cif {i}: natoms {len(s)} != G_list {r['natoms']}")
        out.append(s)
    return out


def read_E_structures(e_rows):
    from pymatgen.core import Structure
    out = []
    for r in e_rows:
        s = Structure.from_file(P(r["poscar_path"]))
        if len(s) != int(r["natoms"]):
            raise K.InputError(f"test pos {r['test_pos']}: natoms {len(s)} != gamma_dedup {r['natoms']}")
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------
def cmd_select(args):
    out = Path(P(args.out_dir))
    # --- G ---
    per_t = {}
    for r in K.read_csv_rows(PER_TARGET_CSV):
        if r["layer"] == "main" and r["arm"] == "proposed" and abs(float(r["w"]) - 2.0) < 1e-9:
            per_t[int(r["target_id"])] = (int(r["nun_post_k"]), int(r["near_post_k"]))
    if sorted(per_t) != list(range(K.N_TARGETS)):
        raise K.InputError(f"{PER_TARGET_CSV}: proposed main w=2.0 rows for targets {sorted(per_t)} != 0..{K.N_TARGETS - 1}")
    mrows = {}
    for r in K.read_csv_rows(INPUTS_MANIFEST_MAIN):
        if r["layer"] == "main" and r["arm"] == "proposed" and abs(float(r["w"]) - 2.0) < 1e-9:
            mrows[(int(r["chunk"]), int(r["cif_index"]))] = r
    if len(mrows) != K.N_PER_ARM_W:
        raise K.InputError(f"{INPUTS_MANIFEST_MAIN}: proposed main w=2.0 rows {len(mrows)} != {K.N_PER_ARM_W}")
    g_rows = []
    for c in range(K.N_CHUNKS):
        rd = Path(P(f"{RUN}/results/relax/main/proposed_w2.0/{K.chunk_name(c)}"))
        nd = Path(P(f"{RUN}/results/novelty/main/proposed_w2.0/{K.chunk_name(c)}"))
        rr = {int(x["cif_index"]): x for x in K.read_csv_rows(rd / "relax_rows.csv")}
        nr = {int(x["cif_index"]): x for x in K.read_csv_rows(nd / "novelty_post_rows.csv")}
        if len(rr) != K.PER_CHUNK or len(nr) != K.PER_CHUNK:
            raise K.InputError(f"chunk {c}: relax_rows {len(rr)} / novelty_post_rows {len(nr)} != {K.PER_CHUNK}")
        for i in range(K.PER_CHUNK):
            x, n = rr[i], nr[i]
            converged = x["status"] == "ok_converged"                     # same as aggregate_relax_novelty.py:180 (converged)
            novel = {"True": True, "False": False}.get(n["novel"])        # same tri()
            gr = K.to_float(x.get("gamma_relaxed"))
            gt = float(mrows[(c, i)]["gamma_target"])
            # master of the definition aggregate_relax_novelty.py:253 (post side of s_members)
            if converged and novel and gr is not None and abs(gr - gt) <= K.NEAR_TOL_J:
                g_rows.append({"chunk": c, "cif_index": i, "target_id": int(mrows[(c, i)]["target_id"]), "gamma_target": gt,
                               "gamma_relaxed": gr, "natoms": int(x["natoms"]),
                               "gb_dx_realized_A": float(mrows[(c, i)]["gb_dx_realized_A"]), "frame_index": int(x["frame_index"]),
                               "relaxed_extxyz": K.norm_compute(rd / "relaxed.extxyz")})
    cnt = {t: sum(1 for r in g_rows if r["target_id"] == t) for t in range(K.N_TARGETS)}
    log("[select] G count per target: " + ", ".join(f"t{t}={cnt[t]} (nun_post_k={per_t[t][0]}, near_post_k={per_t[t][1]})" for t in cnt) + f"; total={len(g_rows)}")
    bad = [t for t in cnt if not (per_t[t][0] <= cnt[t] <= per_t[t][1])]
    tot_lo, tot_hi = sum(v[0] for v in per_t.values()), sum(v[1] for v in per_t.values())
    if bad or not (tot_lo <= len(g_rows) <= tot_hi):
        log(f"[select][check-3 VIOLATION] targets {bad} outside [nun_post_k, near_post_k] or total {len(g_rows)} outside [{tot_lo},{tot_hi}]")
        return EXIT_AC
    log(f"[select][check-3 OK] total {len(g_rows)} in [{tot_lo},{tot_hi}]")
    # --- E ---
    gd = K.read_csv_rows(K.GAMMA_CSV)
    test_idx = json.load(open(P(K.SPLIT_JSON)))["test_indices"]
    if len(test_idx) != 552 or len(set(test_idx)) != 552:
        raise K.InputError(f"split.json test_indices: {len(test_idx)} (unique {len(set(test_idx))}) != 552")
    groups = {}
    for pos, gi in enumerate(test_idx):
        groups.setdefault(int(gd[gi]["natoms"]), []).append((pos, gi))
    quota = {n: E_N * len(v) / len(test_idx) for n, v in groups.items()}
    alloc = {n: int(math.floor(q)) for n, q in quota.items()}
    rest = E_N - sum(alloc.values())
    for n in sorted(quota, key=lambda n: (-(quota[n] - alloc[n]), -len(groups[n]), n))[:rest]:     # largest remainder method (ties: the larger group, then the smaller number of atoms)
        alloc[n] += 1
    rng = random.Random(E_SEED)
    e_rows = []
    for n in sorted(groups):
        for pos, gi in sorted(rng.sample(sorted(groups[n]), alloc[n])):
            r = gd[gi]
            e_rows.append({"test_pos": pos, "gamma_row": gi, "cnid": r["cnid"], "k": r["k"], "seed": r["seed"], "natoms": r["natoms"],
                           "dx_A": r["dx_A"], "gamma_Jm2": r["gamma_Jm2"], "poscar_path": K.norm_compute(r["dir"]) + "/POSCAR_relaxed"})
    e_rows.sort(key=lambda r: (r["test_pos"]))
    if len(e_rows) != E_N:
        raise K.InputError(f"E sample {len(e_rows)} != {E_N}")
    log("[select] E allocation by natoms: " + ", ".join(f"{n}:{alloc[n]}/{len(groups[n])}" for n in sorted(groups)))
    gf = ["chunk", "cif_index", "target_id", "gamma_target", "gamma_relaxed", "natoms", "gb_dx_realized_A", "frame_index", "relaxed_extxyz"]
    ef = ["test_pos", "gamma_row", "cnid", "k", "seed", "natoms", "dx_A", "gamma_Jm2", "poscar_path"]
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "G_list.csv", gf, g_rows, args.overwrite)
    write_rows(out / "E_list.csv", ef, e_rows, args.overwrite)
    write_rows(out / "task_map.csv", ["task_id", "set", "part", "nparts", "subcommand", "out_file"], task_map_rows(), args.overwrite)
    log(f"[select] wrote G_list.csv ({len(g_rows)}), E_list.csv ({len(e_rows)}), task_map.csv ({N_TASKS}) in {out}")
    return 0


# ---------------------------------------------------------------------------
# nn
# ---------------------------------------------------------------------------
_G = {}


def _nn_worker(task):
    qi, lo, hi = task
    q = _G["queries"][qi]
    refs = _G["refs_by_n"][len(q)]
    m06, muni, do_fit = _G["m06"], _G["muni"], _G["do_fit"]
    t0 = time.time()
    res = []
    for j in range(lo, hi):
        t = refs[j][2]
        r = rms_pair(m06, q, t)
        f = bool(muni.fit(q, t)) if do_fit else None
        res.append((j, r, f))
    return qi, res, time.time() - t0


def _nn_track(res, refs, meta, v):
    """Returns (ok, record dict) from the comparison result of one structure res=[(j, (rms_norm, maxdist_norm)|None, fit)] (pymatgen not needed).
    censored (None) enters neither minimum. The columns of the d_max-minimum train series are as before; the rms-minimum train series is the minimum of the rms over all training structures (main metric)."""
    ok = [(r[1], r[0], j) for (j, r, f) in res if r is not None]                     # (maxdist_norm, rms_norm, j)
    ok.sort()
    ok_r = sorted((r[0], r[1], j) for (j, r, f) in res if r is not None)             # (rms_norm, maxdist_norm, j)
    d = {}
    if ok:
        mx, rm, j = ok[0]
        ri = refs[j][0]
        d.update({"d_min_A": mx * v, "d_rms_at_min_A": rm * v, "maxdist_norm_min": mx, "rms_norm_at_min": rm, "nn_row_index": ri,
                  "nn_cnid": meta[ri]["cnid"], "nn_k": meta[ri]["k"], "nn_dx_A": meta[ri]["dx_A"], "nn_gamma_Jm2": meta[ri]["gamma_Jm2"],
                  "d2_A": (ok[1][0] * v) if len(ok) > 1 else None})
        rm0, _mx0, jr = ok_r[0]
        rri = refs[jr][0]
        d.update({"d_rms_min_A": rm0 * v, "rms_nn_row_index": rri, "rms_nn_cnid": meta[rri]["cnid"], "rms_nn_dx_A": meta[rri]["dx_A"],
                  "rms_nn_gamma": meta[rri]["gamma_Jm2"], "d_rms_second_A": (ok_r[1][0] * v) if len(ok_r) > 1 else None})
    return ok, d


def cmd_nn(args):
    S = args.set
    out = Path(P(args.out_dir))
    lst = K.read_csv_rows(out / f"{S}_list.csv")
    nparts = args.nparts or (N_G_PARTS if S == "G" else N_E_PARTS)
    if not (0 <= args.part < nparts):
        raise K.InputError(f"--part {args.part} not in 0..{nparts - 1}")
    sel = [i for i in range(len(lst)) if i % nparts == args.part]
    if not sel:
        raise K.InputError(f"{S} part {args.part}/{nparts}: no queries")
    sub = [lst[i] for i in sel]
    queries = read_G_structures(sub) if S == "G" else read_E_structures(sub)
    natoms_set = {len(q) for q in queries}
    _rows, meta, by_n = load_train(natoms_set)
    do_fit = S == "G"                                    # invariant check (G only)
    m06 = K.make_matcher(stol=STOL_NN)
    muni = K.make_matcher()
    log(f"[nn] host={socket.gethostname()} set={S} part={args.part}/{nparts} queries={len(queries)} natoms={sorted(natoms_set)} "
        f"refs={{{', '.join(f'{n}:{len(by_n.get(n, []))}' for n in sorted(natoms_set))}}} procs={args.procs} do_fit={do_fit}")
    log(f"[nn] matcher_nn kwargs={m06.as_dict() if hasattr(m06, 'as_dict') else ''}")
    # measured cost: seconds per get_rms_dist call for the first 5 structures x 1 ref
    tp = []
    for q in queries[:5]:
        refs = by_n.get(len(q), [])
        if refs:
            t0 = time.time()
            rms_pair(m06, q, refs[0][2])
            tp.append(time.time() - t0)
    if tp:
        log(f"[TIMING] get_rms_dist(stol={STOL_NN}) per call: n={len(tp)} mean={sum(tp) / len(tp):.3f}s max={max(tp):.3f}s "
            f"(estimate: fit 0.45 s/call. If it exceeds 3 times = 1.35 s, report)" + (" [FLAG: >1.35s]" if sum(tp) / len(tp) > 1.35 else ""))
    tasks = []
    for qi, q in enumerate(queries):
        n_ref = len(by_n.get(len(q), []))
        for lo in range(0, n_ref, CHUNK_REFS):
            tasks.append((qi, lo, min(lo + CHUNK_REFS, n_ref)))
    _G.update({"queries": queries, "refs_by_n": by_n, "m06": m06, "muni": muni, "do_fit": do_fit})
    results = {qi: [] for qi in range(len(queries))}
    elapsed = {qi: 0.0 for qi in range(len(queries))}
    t0 = time.time()
    if args.procs > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(min(args.procs, len(tasks))) as pool:
            for qi, res, dt in pool.imap_unordered(_nn_worker, tasks, chunksize=1):
                results[qi].extend(res)
                elapsed[qi] += dt
    else:
        for tk in tasks:
            qi, res, dt = _nn_worker(tk)
            results[qi].extend(res)
            elapsed[qi] += dt
    n_calls = sum(len(v) for v in results.values())
    log(f"[nn] {n_calls} comparisons in {time.time() - t0:.1f}s wall")
    rows, n_viol = [], 0
    for qi, q in enumerate(queries):
        refs = by_n.get(len(q), [])
        res = sorted(results[qi], key=lambda x: x[0])
        v = vn13(q)
        ok, trk = _nn_track(res, refs, meta, v)
        n_cens = sum(1 for (_j, r, _f) in res if r is None)
        fit_true = [refs[j][0] for (j, _r, f) in res if f]
        row = {"query_index": sel[qi], "natoms": len(q), "vn13_x_A": v, "n_compared": len(res), "n_censored": n_cens,
               "all_censored": (len(res) > 0 and not ok), "elapsed_cpu_s": elapsed[qi]}
        if S == "G":
            row.update({"chunk": sub[qi]["chunk"], "cif_index": sub[qi]["cif_index"], "target_id": sub[qi]["target_id"],
                        "gb_dx_realized_A": sub[qi]["gb_dx_realized_A"], "n_fit_true": len(fit_true),
                        "fit_true_row_indices": ";".join(str(x) for x in fit_true)})
            n_viol += len(fit_true)
        else:
            row.update({"test_pos": sub[qi]["test_pos"], "gamma_row": sub[qi]["gamma_row"], "cnid": sub[qi]["cnid"], "k": sub[qi]["k"],
                        "dx_A": sub[qi]["dx_A"]})
        row.update(trk)
        rows.append(row)
    base = ["query_index"] + (["chunk", "cif_index", "target_id", "gb_dx_realized_A"] if S == "G" else ["test_pos", "gamma_row", "cnid", "k", "dx_A"])
    fields = base + ["natoms", "vn13_x_A", "n_compared", "n_censored", "all_censored", "d_min_A", "d_rms_at_min_A", "maxdist_norm_min",
                     "rms_norm_at_min", "nn_row_index", "nn_cnid", "nn_k", "nn_dx_A", "nn_gamma_Jm2", "d2_A",
                     "d_rms_min_A", "rms_nn_row_index", "rms_nn_cnid", "rms_nn_dx_A", "rms_nn_gamma", "d_rms_second_A"] + \
        (["n_fit_true", "fit_true_row_indices"] if S == "G" else []) + ["elapsed_cpu_s"]
    name = f"nn_{S}_{args.part:02d}.csv"
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / name, fields, rows, args.overwrite)
    done_marker(out, name, f"nn_{S}_{args.part:02d}.DONE")
    log(f"[nn] wrote {out / name} ({len(rows)} rows)")
    if S == "G":
        if n_viol:
            log(f"[nn][check-4 VIOLATION] {n_viol} (G, train) pairs with UNI.fit(stol={STOL_UNI}) == True")
            return EXIT_AC
        log(f"[nn][check-4 OK] UNI.fit True count over {n_calls} (G, train) pairs = 0")
    return 0


# ---------------------------------------------------------------------------
# dxpairs
# ---------------------------------------------------------------------------
def build_dx_pairs(rows):
    """Reimplementation of the same rule as instrument_dx in the earlier relaxation check: pairs with the same (cnid, k, seed%1000) whose dx differs by 0.15."""
    groups = {}
    for r in rows:
        key = (r["cnid"], int(r["k"]), int(r["seed"]) % 1000)
        groups.setdefault(key, []).append((float(r["dx_A"]), int(r["row_index"]), int(r["natoms"])))
    pairs = []
    for key, lst in groups.items():
        lst.sort()
        for (d1, r1, n1_), (d2, r2, n2_) in zip(lst, lst[1:]):
            if abs(round(d2 - d1, 6) - 0.15) < 1e-6:
                if n1_ != n2_:
                    raise K.InputError(f"pair {r1},{r2}: natoms differ {n1_} vs {n2_}")
                pairs.append((r1, r2))
    return pairs, len(groups)


def _pair_worker(pair):
    i, j = pair
    x, y = _G["structs"][i], _G["structs"][j]
    t0 = time.time()
    r = rms_pair(_G["m06"], x, y)
    f = bool(_G["muni"].fit(x, y))
    return pair, r, f, time.time() - t0


def cmd_dxpairs(args):
    out = Path(P(args.out_dir))
    rows = K.read_csv_rows(TRAIN_MANIFEST)
    pairs, n_groups = build_dx_pairs(rows)
    log(f"[dxpairs] {len(pairs)} adjacent-dx pairs from {n_groups} groups")
    if len(pairs) != EXPECT_PAIRS:
        log(f"[dxpairs][check-5 VIOLATION] pairs {len(pairs)} != {EXPECT_PAIRS}")
        return EXIT_AC
    _r, meta, by_n = load_train()
    structs = {ri: s for lst in by_n.values() for (ri, _c, s) in lst}
    _G.update({"structs": structs, "m06": K.make_matcher(stol=STOL_NN), "muni": K.make_matcher()})
    res, t0 = {}, time.time()
    if args.procs > 1:
        with mp.get_context("fork").Pool(args.procs) as pool:
            for p, r, f, dt in pool.imap_unordered(_pair_worker, pairs, chunksize=8):
                res[p] = (r, f, dt)
    else:
        for pr in pairs:
            p, r, f, dt = _pair_worker(pr)
            res[p] = (r, f, dt)
    log(f"[dxpairs] {len(pairs)} pairs in {time.time() - t0:.1f}s wall")
    out_rows = []
    for n, (a, b) in enumerate(pairs):
        r, f, dt = res[(a, b)]
        v = vn13(structs[a])
        row = {"pair_id": n, "row_index_i": a, "row_index_j": b, "cnid": meta[a]["cnid"], "k": meta[a]["k"], "seed_mod1000": meta[a]["seed"] % 1000,
               "natoms": meta[a]["natoms"], "dx_i_A": meta[a]["dx_A"], "dx_j_A": meta[b]["dx_A"], "vn13_x_A": v,
               "censored": r is None, "fit_uni_stol0025": f, "elapsed_s": dt}
        if r is not None:
            row.update({"rms_norm": r[0], "maxdist_norm": r[1], "d_max_A": r[1] * v, "d_rms_A": r[0] * v})
        out_rows.append(row)
    fields = ["pair_id", "row_index_i", "row_index_j", "cnid", "k", "seed_mod1000", "natoms", "dx_i_A", "dx_j_A", "vn13_x_A", "censored",
              "rms_norm", "maxdist_norm", "d_max_A", "d_rms_A", "fit_uni_stol0025", "elapsed_s"]
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "dxpairs.csv", fields, out_rows, args.overwrite)
    done_marker(out, "dxpairs.csv", "dxpairs.DONE")
    log(f"[dxpairs] wrote {out / 'dxpairs.csv'} ({len(out_rows)} rows); n_fit_uni_true={sum(1 for r in out_rows if r['fit_uni_stol0025'])}")
    return 0


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
def _f(x):
    return K.to_float(x)


def read_nn(out, S, nparts):
    rows = []
    for p in range(nparts):
        f = out / f"nn_{S}_{p:02d}.csv"
        if not f.is_file() or not (out / f"nn_{S}_{p:02d}.DONE").is_file():
            raise K.InputError(f"{f} or its DONE marker not found")
        part_rows = K.read_csv_rows(f)
        if part_rows and "d_rms_min_A" not in part_rows[0]:
            raise K.InputError(f"{f}: column d_rms_min_A missing (written by old nn code; re-run this part, no fallback to d_rms_at_min_A)")
        rows.extend(part_rows)
    return rows


def dist_stats(vals, n_total, n_censored):
    import numpy as np
    v = np.asarray(vals, dtype=float)
    d = {"n": n_total, "n_valid": int(len(v)), "n_censored_or_blank": n_censored}
    if len(v):
        qs = np.percentile(v, [10, 25, 50, 75, 90])           # numpy default (linear interpolation)
        d.update({"p10": float(qs[0]), "p25": float(qs[1]), "p50": float(qs[2]), "p75": float(qs[3]), "p90": float(qs[4]),
                  "min": float(v.min()), "max": float(v.max())})
    return d


def cmd_aggregate(args):
    import numpy as np
    out = Path(P(args.out_dir))
    for m in [f"nn_G_{p:02d}.DONE" for p in range(N_G_PARTS)] + [f"nn_E_{p:02d}.DONE" for p in range(N_E_PARTS)] + ["dxpairs.DONE"]:
        if not (out / m).is_file():
            raise K.InputError(f"marker not found: {out / m}")
    glist = K.read_csv_rows(out / "G_list.csv")
    elist = K.read_csv_rows(out / "E_list.csv")
    gnn, enn = read_nn(out, "G", N_G_PARTS), read_nn(out, "E", N_E_PARTS)
    dxp = K.read_csv_rows(out / "dxpairs.csv")
    # row count part of the output check, invariant check of dxpairs and nn
    bad = []
    if len(gnn) != len(glist):
        bad.append(f"nn_G rows {len(gnn)} != |G| {len(glist)}")
    if len(enn) != E_N:
        bad.append(f"nn_E rows {len(enn)} != {E_N}")
    if len(dxp) != EXPECT_PAIRS:
        bad.append(f"dxpairs rows {len(dxp)} != {EXPECT_PAIRS}")
    if sum(int(r["n_fit_true"]) for r in gnn) != 0:
        bad.append("G has UNI.fit True against train")
    if len({(r["chunk"], r["cif_index"]) for r in gnn}) != len(gnn) or {(r["chunk"], r["cif_index"]) for r in gnn} != {(r["chunk"], r["cif_index"]) for r in glist}:
        bad.append("nn_G keys != G_list keys")
    if bad:
        log("[aggregate][FAIL] " + "; ".join(bad))
        return EXIT_AC
    S = {}
    # --- 3-4: 3 distributions ---
    g_d = [_f(r["d_min_A"]) for r in gnn if _f(r["d_min_A"]) is not None]
    e_d = [_f(r["d_min_A"]) for r in enn if _f(r["d_min_A"]) is not None]
    p_d = [_f(r["d_max_A"]) for r in dxp if _f(r["d_max_A"]) is not None]
    S["d_min_G"] = dist_stats(g_d, len(gnn), len(gnn) - len(g_d))
    S["d_min_E"] = dist_stats(e_d, len(enn), len(enn) - len(e_d))
    S["d_max_dxpairs"] = dist_stats(p_d, len(dxp), len(dxp) - len(p_d))
    # --- main metric d_rms (columns: nn is d_rms_min_A = minimum of the rms over all training structures, dxpairs is d_rms_A). censored = get_rms_dist is None (column blank) ---
    g_r = [_f(r["d_rms_min_A"]) for r in gnn if _f(r["d_rms_min_A"]) is not None]
    e_r = [_f(r["d_rms_min_A"]) for r in enn if _f(r["d_rms_min_A"]) is not None]
    p_r = [_f(r["d_rms_A"]) for r in dxp if _f(r["d_rms_A"]) is not None]
    S["primary_metric"] = "d_rms (main). d_max is secondary. d_rms(G/E) is d_rms_min_A of nn.csv = the minimum of the rms over all training structures"
    S["d_rms_G"] = dist_stats(g_r, len(gnn), len(gnn) - len(g_r))
    S["d_rms_E"] = dist_stats(e_r, len(enn), len(enn) - len(e_r))
    S["d_rms_dxpairs"] = dist_stats(p_r, len(dxp), len(dxp) - len(p_r))
    S["n_no_same_natoms_train"] = {"G": sum(1 for r in gnn if int(r["n_compared"]) == 0), "E": sum(1 for r in enn if int(r["n_compared"]) == 0)}
    med_pr = float(np.median(p_r)) if p_r else None
    med_er = float(np.median(e_r)) if e_r else None
    S["fractions_d_rms"] = {
        "G_drms_below_median_dxpairs": {"numerator": sum(1 for x in g_r if med_pr is not None and x < med_pr), "denominator_valid": len(g_r), "denominator_all_G": len(gnn), "median_dxpairs_A": med_pr},
        "G_drms_above_median_E": {"numerator": sum(1 for x in g_r if med_er is not None and x > med_er), "denominator_valid": len(g_r), "denominator_all_G": len(gnn), "n_G_all_censored_not_counted": len(gnn) - len(g_r), "median_E_A": med_er},
    }
    med_p = float(np.median(p_d)) if p_d else None
    med_e = float(np.median(e_d)) if e_d else None
    S["fractions"] = {
        "G_dmin_below_median_dxpairs": {"numerator": sum(1 for x in g_d if med_p is not None and x < med_p), "denominator_valid": len(g_d), "denominator_all_G": len(gnn), "median_dxpairs_A": med_p},
        "G_dmin_above_median_E": {"numerator": sum(1 for x in g_d if med_e is not None and x > med_e), "denominator_valid": len(g_d), "denominator_all_G": len(gnn), "n_G_all_censored_not_counted": len(gnn) - len(g_d), "median_E_A": med_e},
    }
    # --- nearest cnid / dx ---
    from collections import Counter
    cn = Counter(r["nn_cnid"] for r in gnn if r["nn_cnid"])
    S["G_nn_cnid"] = {"n_distinct": len(cn), "top5": cn.most_common(5)}
    gb = {(r["chunk"], r["cif_index"]): float(r["gb_dx_realized_A"]) for r in glist}
    same_lv = n_den = 0
    for r in gnn:
        if r["nn_dx_A"] == "":
            continue
        n_den += 1
        lv, _d, _o = nearest_level(gb[(r["chunk"], r["cif_index"])])
        same_lv += int(abs(float(r["nn_dx_A"]) - lv) < 1e-6)
    S["G_nn_dx_equals_nearest_level"] = {"numerator": same_lv, "denominator": n_den}
    # --- δ ---
    all350 = [float(r["gb_dx_realized_A"]) for r in K.read_csv_rows(INPUTS_MANIFEST_MAIN)
              if r["layer"] == "main" and r["arm"] == "proposed" and abs(float(r["w"]) - 2.0) < 1e-9]
    if len(all350) != K.N_PER_ARM_W:
        raise K.InputError(f"proposed main w=2.0 gb_dx rows {len(all350)} != {K.N_PER_ARM_W}")
    g_gb = list(gb.values())
    edges = np.round(np.arange(-0.075, 0.075 + 1e-9, 0.0125), 6)

    def delta_block(vals):
        ds = [nearest_level(v) for v in vals]
        d = np.array([x[1] for x in ds])
        h, _ = np.histogram(d, bins=edges)
        return {"n": len(vals), "n_out_of_grid_range": int(sum(1 for x in ds if x[2])), "n_abs_delta_le_0.02": int(np.sum(np.abs(d) <= DELTA_TOL + 1e-9)),
                "fraction_abs_delta_le_0.02": float(np.mean(np.abs(d) <= DELTA_TOL + 1e-9)) if len(d) else None,
                "uniform_reference": DELTA_UNIFORM, "hist_bin_edges": [float(e) for e in edges], "hist_counts": [int(x) for x in h],
                "n_outside_hist_range": int(np.sum((d < edges[0]) | (d > edges[-1]))), "deltas": [float(x) for x in d]}
    S["gbdx_delta"] = {"all350": delta_block(all350), "G": delta_block(g_gb)}
    # --- per target ---
    bt = {}
    for t in range(K.N_TARGETS):
        rows_t = [r for r in gnn if int(r["target_id"]) == t]
        v_rms = [_f(r["d_rms_min_A"]) for r in rows_t if _f(r["d_rms_min_A"]) is not None]
        v_max = [_f(r["d_min_A"]) for r in rows_t if _f(r["d_min_A"]) is not None]
        bt[f"t{t}"] = {"n_G": len(rows_t),
                       "n_valid_d_rms": len(v_rms), "d_rms_median_A": float(np.median(v_rms)) if v_rms else None,
                       "n_valid": len(v_max), "d_min_median_A": float(np.median(v_max)) if v_max else None}   # keep the existing keys (backward compatibility)
    S["G_dmin_median_by_target"] = bt
    S["G_by_target_note"] = "d_rms_median_A is the main metric. d_min_median_A is of the d_max series (secondary)"
    # --- check b ---
    same = [r for r in dxp if r["fit_uni_stol0025"] == "True"]
    below = [r for r in same if _f(r["maxdist_norm"]) is not None and _f(r["maxdist_norm"]) < STOL_UNI]
    exp_same = json.load(open(P(INSTRUMENT_JSON)))["n_judged_same"]
    S["consistency_b"] = {"n_same_166": len(same), "n_dmax_below_stol": len(below), "n_disagree": len(same) - len(below),
                          "n_same_expected_from_json": exp_same, "definition": "same = pairs with UNI.fit(stol=0.025) True (dxpairs.csv fit_uni_stol0025). below = among them maxdist_norm < 0.025. disagree = same - below"}
    S["consistency_a"] = {"n_fit_true_G_vs_train": 0}
    vn_med = float(np.median([float(r["vn13_x_A"]) for r in gnn]))
    S["stol_line_A"] = {"value": STOL_UNI * vn_med, "stol": STOL_UNI, "vn13_median_G_A": vn_med, "note": "vertical line = stol x (V/N)^(1/3) at the median of G (the about 0.064 A of the plan is this conversion)"}
    S["meta"] = {"design": "146", "host": socket.gethostname(), "nn_matcher_stol": STOL_NN, "uni_kwargs": K.load_UNI(), "n_G": len(gnn), "n_E": len(enn), "n_pairs": len(dxp),
                 "percentile_method": "numpy linear", "E_seed": E_SEED}
    (out / "summary.json").write_text(json.dumps(S, indent=1, ensure_ascii=False) + "\n")
    write_summary_md(out / "summary.md", S)
    try:
        make_figs(out, g_d, e_d, p_d, S, g_r, e_r, p_r)
    except Exception as e:                                 # a failure of the figure does not lose the numbers (the exception is shown and the exit code is non-zero)
        log(f"[aggregate][FIG FAIL] {e!r}")
        return EXIT_INPUT
    done_marker(out, "summary.json", "AGG.DONE")
    log(f"[aggregate] wrote summary.json, summary.md, fig_dmin_hist.png (d_rms), fig_dmin_hist_dmax.png, fig_gbdx_delta.png in {out}")
    return 0


def _n(v, nd=4):
    return "" if v is None else f"{v:.{nd}f}"


def write_summary_md(path, S):
    L = ["| quantity | n | n_valid | censored/blank | p10 | p25 | p50 | p75 | p90 | min | max |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, name in (("d_rms_G", "[main] d_rms(G->T) [A]"), ("d_rms_E", "[main] d_rms(E->T) [A]"), ("d_rms_dxpairs", "[main] d_rms(dx neighbor pairs) [A]"),
                      ("d_min_G", "[secondary] d_max(G->T) [A]"), ("d_min_E", "[secondary] d_max(E->T) [A]"), ("d_max_dxpairs", "[secondary] d_max(dx neighbor pairs) [A]")):
        s = S[key]
        L.append(f"| {name} | {s['n']} | {s['n_valid']} | {s['n_censored_or_blank']} | " + " | ".join(_n(s.get(k)) for k in ("p10", "p25", "p50", "p75", "p90", "min", "max")) + " |")
    L += ["", "Note: d_rms of G/E is d_rms_min_A of nn.csv (the minimum of the rms over all training structures)."]
    fr, fr_r = S["fractions"], S["fractions_d_rms"]
    a, b = fr["G_dmin_below_median_dxpairs"], fr["G_dmin_above_median_E"]
    ar, br = fr_r["G_drms_below_median_dxpairs"], fr_r["G_drms_above_median_E"]
    L += ["", "| fraction | d_rms count/denominator [main] | d_rms reference value [A] | d_max count/denominator [secondary] | d_max reference value [A] |", "|---|---|---|---|---|",
          f"| d_min of G < median of the dx neighbor pairs | {ar['numerator']}/{ar['denominator_valid']} (all G {ar['denominator_all_G']}) | {_n(ar['median_dxpairs_A'])} | {a['numerator']}/{a['denominator_valid']} (all G {a['denominator_all_G']}) | {_n(a['median_dxpairs_A'])} |",
          f"| d_min of G > median of d_min of E | {br['numerator']}/{br['denominator_valid']} (all G {br['denominator_all_G']}; those with all censored {br['n_G_all_censored_not_counted']} not counted) | {_n(br['median_E_A'])} | {b['numerator']}/{b['denominator_valid']} (all G {b['denominator_all_G']}; those with all censored {b['n_G_all_censored_not_counted']} not counted) | {_n(b['median_E_A'])} |"]
    nn = S["G_nn_cnid"]
    L += ["", "| nearest training structure of G | value |", "|---|---|", f"| number of kinds of cnid | {nn['n_distinct']} |",
          "| top 5 (cnid, count) | " + "; ".join(f"{c}:{n}" for c, n in nn["top5"]) + " |",
          f"| dx_A of the nearest neighbor is the same as the DX10 level nearest to the generated gb_dx | {S['G_nn_dx_equals_nearest_level']['numerator']}/{S['G_nn_dx_equals_nearest_level']['denominator']} |"]
    d = S["gbdx_delta"]
    L += ["", "| delta = gb_dx - nearest DX10 level | n | outside the range | \\|delta\\|<=0.02 A | fraction | uniform reference |", "|---|---|---|---|---|---|"]
    for k, nm in (("all350", "proposed main experiment, all 350"), ("G", "G")):
        x = d[k]
        L.append(f"| {nm} | {x['n']} | {x['n_out_of_grid_range']} | {x['n_abs_delta_le_0.02']} | {_n(x['fraction_abs_delta_le_0.02'])} | {_n(x['uniform_reference'], 3)} |")
    L += ["", "| target | n_G | n_valid (d_rms) | median of d_rms [A] (main) | median of d_min of the d_max series [A] (secondary) |", "|---|---|---|---|---|"]
    for t, x in S["G_dmin_median_by_target"].items():
        L.append(f"| {t} | {x['n_G']} | {x['n_valid_d_rms']} | {_n(x['d_rms_median_A'])} | {_n(x['d_min_median_A'])} |")
    c = S["consistency_b"]
    L += ["", "| check | value |", "|---|---|", f"| (a) UNI.fit True between G and train | {S['consistency_a']['n_fit_true_G_vs_train']} |",
          f"| (b) n_same_166 / n_dmax_below_stol / n_disagree | {c['n_same_166']} / {c['n_dmax_below_stol']} / {c['n_disagree']} (n_judged_same={c['n_same_expected_from_json']} in the json) |",
          f"| vertical line stol x (V/N)^(1/3) [A] | {_n(S['stol_line_A']['value'])} |"]
    path.write_text("\n".join(L) + "\n")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for cand in (os.environ.get("PAPERFIG_DIR", "paperfig-dir-not-set"),):
        if os.path.isfile(os.path.join(cand, "paperfig.py")):
            sys.path.insert(0, cand)
            import paperfig  # noqa: WPS433
            paperfig.setup_style()
            return plt, paperfig.OKABE_ITO, paperfig.COL_1, paperfig.COL_2, paperfig.save_fig
    # fallback with the same conventions for nodes that cannot see paperfig.py (axis labels 9 pt, ticks 7 pt, Okabe-Ito, PNG only)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "xtick.direction": "in", "ytick.direction": "in", "pdf.fonttype": 42})
    oi = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

    def save_fig(fig, name, outdir, pdf=False):
        fig.savefig(os.path.join(str(outdir), f"{name}.png"), dpi=300, bbox_inches="tight")
    return plt, oi, 86 / 25.4, 178 / 25.4, save_fig


def make_figs(out, g_d, e_d, p_d, S, g_r, e_r, p_r):
    import numpy as np
    plt, oi, c1, c2, save_fig = _style()
    sv = S["stol_line_A"]["value"]
    for (gv, ev, pv), fname, xlab, line_lab in (
            ((g_r, e_r, p_r), "fig_dmin_hist", "d_rms [A]", f"reference: stol 0.025 converted to a maximum distance / ref: stol 0.025 as max-dist ({sv:.3f} A)"),
            ((g_d, e_d, p_d), "fig_dmin_hist_dmax", "d_max [Å]", f"stol 0.025 ({sv:.3f} Å)")):
        fig, ax = plt.subplots(figsize=(c1, c1 * 0.75))
        for vals, lab, col in ((gv, "G→T nearest", oi[5]), (ev, "E→T nearest", oi[2]), (pv, "T dx-adjacent pairs", oi[4])):
            if len(vals):
                x = np.sort(np.asarray(vals))
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=col, label=f"{lab} (n={len(x)})")
        ax.axvline(sv, color="black", linestyle="--", linewidth=0.8, label=line_lab)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Cumulative fraction")
        ax.legend(frameon=False)
        save_fig(fig, fname, str(out))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(c1, c1 * 0.7))
    d = S["gbdx_delta"]
    edges = np.array(d["all350"]["hist_bin_edges"])
    w = np.diff(edges)
    ax.bar(edges[:-1], np.array(d["all350"]["hist_counts"]) / max(1, d["all350"]["n"]), width=w, align="edge", color=oi[1], alpha=0.7, label=f"all proposed main (n={d['all350']['n']})")
    ax.bar(edges[:-1], np.array(d["G"]["hist_counts"]) / max(1, d["G"]["n"]), width=w, align="edge", edgecolor=oi[5], fill=False, linewidth=0.9, label=f"G (n={d['G']['n']})")
    ax.set_xlabel("δ = gb_dx − nearest grid level [Å]")
    ax.set_ylabel("Fraction")
    ax.legend(frameon=False)
    save_fig(fig, "fig_gbdx_delta", str(out))
    plt.close(fig)


# ---------------------------------------------------------------------------
# self check
# ---------------------------------------------------------------------------
def cmd_self_check(args):
    import numpy as np
    from pymatgen.core import Structure
    rows = K.read_csv_rows(TRAIN_MANIFEST)
    seen, pick = set(), []
    for r in rows:
        if int(r["natoms"]) not in seen:
            seen.add(int(r["natoms"]))
            pick.append(r)
        if len(pick) == 3:
            break
    if len(pick) < 3:
        raise K.InputError("train manifest: fewer than 3 distinct natoms")
    m06, muni = K.make_matcher(stol=STOL_NN), K.make_matcher()
    log(f"[self_check] host={socket.gethostname()} nn matcher stol={STOL_NN}; uni kwargs={K.load_UNI()}")
    fail = []
    xs = []
    for r in pick:
        x = Structure.from_file(P(r["poscar_relaxed_path"]))
        xs.append((r, x))
        t0 = time.time()
        res = rms_pair(m06, x, x)
        dmax = None if res is None else res[1] * vn13(x)
        ok = dmax is not None and dmax <= 1e-6
        log(f"[self_check][check-1] row_index={r['row_index']} natoms={len(x)} d_max(x,x)={dmax} Å ({time.time() - t0:.2f}s) -> {'OK' if ok else 'VIOLATION'}")
        if not ok:
            fail.append("check-1")
    r0, x0 = xs[0]
    sd = x0.site_properties.get("selective_dynamics")
    if sd is None:
        raise K.InputError(f"row {r0['row_index']}: POSCAR_relaxed has no selective_dynamics (the bulk cannot be identified)")
    fixed = np.array([not any(v) for v in sd])
    log(f"[self_check] row {r0['row_index']}: fixed(Bulk) atoms = {int(fixed.sum())} (expected {K.N_FIXED_PROPOSED})")
    if int(fixed.sum()) != K.N_FIXED_PROPOSED:
        raise K.InputError(f"fixed atoms {int(fixed.sum())} != {K.N_FIXED_PROPOSED}")
    rng = np.random.default_rng(E_SEED)
    disp = rng.normal(0.0, 0.15, size=(len(x0), 3))
    disp[fixed] = 0.0
    x1 = Structure(x0.lattice, x0.species, x0.cart_coords + disp, coords_are_cartesian=True)
    t0 = time.time()
    res = rms_pair(m06, x0, x1)
    dmax = None if res is None else res[1] * vn13(x0)
    fit = bool(muni.fit(x0, x1))
    ok = dmax is not None and 0.15 <= dmax <= 0.6 and fit is False
    log(f"[self_check][check-2] sigma=0.15 on {int((~fixed).sum())} non-Bulk atoms, seed={E_SEED}: d_max(x,x')={dmax} Å, d_rms={None if res is None else res[0] * vn13(x0)} Å, "
        f"UNI.fit={fit} ({time.time() - t0:.2f}s) -> {'OK' if ok else 'VIOLATION'} (required 0.15<=d_max<=0.6 and fit False)")
    if not ok:
        fail.append("check-2")
    log("[self_check] RESULT: " + ("PASS" if not fail else f"FAIL {sorted(set(fail))}"))
    return 0 if not fail else EXIT_AC


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-check", action="store_true", help="check of the invariants (run as a small job on a compute node)")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--out-dir", default=OUT_DEFAULT)
        p.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        p.add_argument("--overwrite", action="store_true", help="allow regenerating its own output files (by default existing files are not overwritten)")
    common(sub.add_parser("select"))
    pn = sub.add_parser("nn")
    common(pn)
    pn.add_argument("--set", choices=["G", "E"], required=True)
    pn.add_argument("--part", type=int, required=True)
    pn.add_argument("--nparts", type=int, default=0)
    common(sub.add_parser("dxpairs"))
    common(sub.add_parser("aggregate"))
    args = ap.parse_args(argv)
    try:
        if args.self_check:
            return cmd_self_check(args)
        if args.cmd is None:
            ap.error("subcommand or --self_check required")
        return {"select": cmd_select, "nn": cmd_nn, "dxpairs": cmd_dxpairs, "aggregate": cmd_aggregate}[args.cmd](args)
    except (K.InputError, FileExistsError) as e:
        log(f"[FATAL] {type(e).__name__}: {e}")
        return EXIT_INPUT


if __name__ == "__main__":
    sys.exit(main())
