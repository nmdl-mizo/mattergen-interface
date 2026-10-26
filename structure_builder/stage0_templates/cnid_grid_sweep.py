#!/usr/bin/env python3
"""Coarse CNID grid sweep (10x10, 12x12) of Al Sigma5(210)[001]. Structures before relaxation only.
M-all : matching of the whole structure with StructureMatcher(stol=0.05, ltol=0.05, angle_tol=2)  [same as build_cnid_dense.py]
M-bulk: matching of the 76 fixed bulk atoms only. The matching function match() and the tolerances M1/M2 are transcribed verbatim from cnid_bulkonly_pairwise.py.
        The main metric is M2 (stol=0.025, primitive_cell=False, scale=False, attempt_supercell=False). M1 is for reference.
The construction procedure is a copy of build_cnid_dense.py (only the output destination is changed to the work directory).
The identification of the fixed bulk atoms is a copy of build_template_bulk/locate_frozen_indices of expansion_bulk_membership.py
(template = the structure itself; the as-built k0 has no removal = identical to the template).
Groups are merged transitively by matching all pairs + union-find.
"""
import os, sys, io, csv, json, time, shutil, contextlib, itertools
from pathlib import Path
import multiprocessing as mp
import numpy as np
from numpy import array, pi
from numpy.linalg import det

SB_ROOT = Path(__file__).resolve().parents[1]                  # structure_builder/
WORK = Path(os.environ["SB_WORK"]) / "stage0"                   # required, no default (KeyError if unset)
STRUCT_DIR = WORK / "structures"
SCRATCH = WORK / "scratch"
INTERFACE_MASTER_ROOT = str(SB_ROOT / "vendor" / "interfacemaster")
BULK_AL_CIF = str(SB_ROOT / "inputs" / "Al.cif")
# EXISTING_DIR, PILOT_DIR, RB, RUN0809 (internal earlier runs) were inputs of the comparison part of main(), which is not part of the
# public release (unreachable after the `--build-only` return below); main() only supports `--build-only` here.
EXISTING_DIR = PILOT_DIR = RB = RUN0809 = None
SIGMA5_210 = dict(axis=[0, 0, 1], ang_deg=53.130, plane=[2, 1, 0], exp_sigma=5, a_mult=3)
GARBAGE = ["super_sub.cif", "super_film.cif", "min_sub.cif", "min_film.cif"]
MATCHER_KW = dict(stol=0.05, ltol=0.05, angle_tol=2)
EXISTING_10 = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (3, 2)]
GRIDS = [10]   # the template pipeline uses the 10x10 grid only (the 12x12 grid belonged to the internal comparison)
EXPECT_BULK_PAIRS = {("cnid_0_1", "cnid_0_3"), ("cnid_1_2", "cnid_3_2"), ("cnid_1_3", "cnid_2_1")}
# verbatim from bulkonly_pairwise.py:19-23
MATCH = {
    "M1": dict(stol=0.05, ltol=0.05, angle_tol=2),
    "M2": dict(stol=0.025, primitive_cell=False, scale=False, attempt_supercell=False),
}
# verbatim from expansion_bulk_membership.py:50-54
N_BULK = 38
NORMAL_AXIS = 0
FORCE_R_NAME = "Rnnid_omm"
FIT_RESIDUAL_TOL_A = 1e-6
MEMBERSHIP_TOL_A = 1e-6
NPROC = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

sys.path.insert(0, INTERFACE_MASTER_ROOT)
sys.path.insert(0, str(SB_ROOT / "lib"))
from interfacemaster.cellcalc import get_primitive_hkl, rot
from interfacemaster.interface_generator import core as im_core
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from sym_common import audit_symmetry, _r_candidates_v2, _apply_op
from ase.io import read as ase_read

# ------------------------------------------------------------------ build
def _clean(cwd):
    for f in GARBAGE:
        p = cwd / f
        if p.exists():
            p.unlink()

def build_im_core(cwd):
    s1 = Structure.from_file(BULK_AL_CIF); s2 = Structure.from_file(BULK_AL_CIF)
    old = os.getcwd(); os.chdir(cwd)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            itf = im_core(s1, s2, verbose=False)
            R = rot(SIGMA5_210["axis"], SIGMA5_210["ang_deg"] / 180.0 * pi)
            itf.search_fixed(R, exact=True, tol=1e-1)
            hkl = get_primitive_hkl(array(SIGMA5_210["plane"]), itf.conv_lattice_1, itf.lattice_1)
            itf.compute_bicrystal(hkl, normal_ortho=True, plane_ortho=True,
                                  align_rotation_axis=True, rotation_axis=SIGMA5_210["axis"],
                                  lim=100, tol_ortho=1e-3, tol_integer=1e-3)
        _clean(cwd)
    finally:
        os.chdir(old)
    return itf

def write_poscar(itf, dydz, cwd, path):
    old = os.getcwd(); os.chdir(cwd)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            itf.get_bicrystal(dydz=dydz, dx=0.0, dp1=0, dp2=0,
                              xyz_1=[SIGMA5_210["a_mult"], 1, 1], xyz_2=[SIGMA5_210["a_mult"], 1, 1], vx=0,
                              filename=str(path), filetype="VASP")
        _clean(cwd)
    finally:
        os.chdir(old)

def build_chunk(args):
    n, wid, pts = args
    cwd = SCRATCH / f"g{n}_w{wid}"; cwd.mkdir(parents=True, exist_ok=True)
    itf = build_im_core(cwd)
    sigma = int(round(abs(det(itf.U1))))
    if sigma != SIGMA5_210["exp_sigma"]:
        raise RuntimeError(f"Sigma mismatch {sigma}")
    v1, v2 = itf.CNID.T
    for (i, j) in pts:
        d = STRUCT_DIR / f"g{n}" / f"p_{i:02d}_{j:02d}"; d.mkdir(parents=True, exist_ok=True)
        write_poscar(itf, v1 / n * i + v2 / n * j, cwd, d / "POSCAR")
    shutil.rmtree(cwd, ignore_errors=True)
    return (n, wid, len(pts), sigma)

# ------------------------------------------------------------------ bulk membership (copy of expansion_bulk_membership.py)
def _mic_pairwise_dist(cell, frac_a, frac_b):
    diff = frac_a[:, None, :] - frac_b[None, :, :]
    diff = diff - np.round(diff)
    return np.linalg.norm(diff @ cell, axis=2)

def frozen_from_poscar(path):
    atoms = ase_read(str(path), format="vasp")
    cell = np.array(atoms.get_cell()); frac = np.array(atoms.get_scaled_positions(wrap=True))
    res = audit_symmetry(atoms, normal_axis=NORMAL_AXIS, use_v2_candidates=True, force_r_name=FORCE_R_NAME)
    op_res = res["max_residual_A"]
    out = dict(path=str(path), natoms=len(atoms), op_residual_A=op_res, status="ok", reason="", frozen=None,
               max_pos_diff_A=None)
    if not (op_res < FIT_RESIDUAL_TOL_A):
        out["status"] = "skip"; out["reason"] = f"symmetry fit residual {op_res:.3e}"; return out
    R = _r_candidates_v2(NORMAL_AXIS)[FORCE_R_NAME]; t = np.array(res["best_t"], dtype=float)
    grainA = np.where((frac[:, 0] > 0.0) & (frac[:, 0] < 0.5))[0]
    order = np.argsort(np.abs(frac[grainA, 0] - 0.25), kind="stable")
    bulkA = grainA[order[:N_BULK]]
    moved = _apply_op(frac[bulkA], R, t)
    bulkB = np.argmin(_mic_pairwise_dist(cell, moved, frac), axis=1)
    if len(set(bulkB.tolist())) != N_BULK:
        out["status"] = "skip"; out["reason"] = "bulkB not unique"; return out
    ref = np.vstack([frac[bulkA], frac[bulkB]])
    d = _mic_pairwise_dist(cell, ref, frac)
    matched = np.argmin(d, axis=1); md = float(d[np.arange(len(ref)), matched].max())
    out["max_pos_diff_A"] = md
    if len(set(matched.tolist())) != len(matched):
        out["status"] = "skip"; out["reason"] = "duplicate frozen match"; return out
    if not (md < MEMBERSHIP_TOL_A):
        out["status"] = "skip"; out["reason"] = f"membership residual {md:.3e}"; return out
    out["frozen"] = sorted(int(x) for x in matched)
    return out

# ------------------------------------------------------------------ matching (verbatim from bulkonly_pairwise.py)
def sub(s, idx):
    return Structure.from_sites([s[i] for i in idx])

def match(sa, sb, m):
    sm = StructureMatcher(**MATCH[m])
    fit = bool(sm.fit(sa, sb))
    if not fit:
        return False, None, None
    r = sm.get_rms_dist(sa, sb)
    return True, float(r[0]), float(r[1])

G = {}   # globals inherited by fork

def _pair_all(ab):
    a, b = ab
    return (a, b, bool(StructureMatcher(**MATCHER_KW).fit(G["all"][a], G["all"][b])))

def _pair_bulk(ab):
    a, b = ab
    return (a, b, bool(StructureMatcher(**MATCH["M2"]).fit(G["bulk"][a], G["bulk"][b])),
            bool(StructureMatcher(**MATCH["M1"]).fit(G["bulk"][a], G["bulk"][b])))

def _ex_all(pe):
    p, e = pe
    return (p, e, bool(StructureMatcher(**MATCHER_KW).fit(G["all"][p], G["ex_all"][e])))

def _ex_bulk(pe):
    p, e = pe
    return (p, e, bool(StructureMatcher(**MATCH["M2"]).fit(G["bulk"][p], G["ex_bulk"][e])),
            bool(StructureMatcher(**MATCH["M1"]).fit(G["bulk"][p], G["ex_bulk"][e])))

def run_tasks(fn, tasks, label, chunk=8):
    ctx = mp.get_context("fork")
    out = []; t0 = time.time()
    with ctx.Pool(NPROC) as pool:
        for k, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=chunk)):
            out.append(r)
            if (k + 1) % 1000 == 0 or (k + 1) == len(tasks):
                log(f"  {label}: {k+1}/{len(tasks)} ({time.time()-t0:.0f}s)")
    return out

def union_groups(n, edges):
    par = list(range(n))
    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in edges:
        ra, rb = f(a), f(b)
        if ra != rb:
            par[max(ra, rb)] = min(ra, rb)
    gs = {}
    for i in range(n):
        gs.setdefault(f(i), []).append(i)
    return list(gs.values())

def rep_of(members, pts):
    return min((pts[m] for m in members), key=lambda t: (t[0] + t[1], t[0], t[1]))

def label(p):
    return f"cnid_{p[0]}_{p[1]}"

# ------------------------------------------------------------------ main
def main():
    if "--build-only" not in sys.argv:
        raise SystemExit("cnid_grid_sweep.py (public release) supports --build-only only: the comparison part needs internal earlier runs")
    from importlib.metadata import version
    log("host", os.uname().nodename, "python", sys.executable, "NPROC", NPROC)
    log("pymatgen", version("pymatgen"), "ase", version("ase"), "numpy", np.__version__)
    log("interfacemaster", sys.modules["interfacemaster"].__file__)
    STRUCT_DIR.mkdir(parents=True, exist_ok=True); SCRATCH.mkdir(parents=True, exist_ok=True)
    pts_of = {n: [(i, j) for i in range(n) for j in range(n)] for n in GRIDS}

    # ---- 1. build
    t0 = time.time()
    tasks = []
    for n in GRIDS:
        pts = pts_of[n]
        for w in range(NPROC):
            ch = pts[w::NPROC]
            if ch:
                tasks.append((n, w, ch))
    with mp.get_context("fork").Pool(NPROC) as pool:
        res = pool.map(build_chunk, tasks, chunksize=1)
    log("build done", f"{time.time()-t0:.0f}s", "sigma set", sorted({r[3] for r in res}),
        "points", {n: sum(r[2] for r in res if r[0] == n) for n in GRIDS})
    shutil.rmtree(SCRATCH, ignore_errors=True)
    return  # --build-only: the rest of main() (comparison with internal runs) is not available in the public release

    # ---- 2. load + frozen indices
    S = {n: [Structure.from_file(str(STRUCT_DIR / f"g{n}" / f"p_{i:02d}_{j:02d}" / "POSCAR")) for (i, j) in pts_of[n]] for n in GRIDS}
    EX = [Structure.from_file(f"{EXISTING_DIR}/cnid_{a}_{b}/POSCAR") for (a, b) in EXISTING_10]
    paths = []
    for n in GRIDS:
        paths += [str(STRUCT_DIR / f"g{n}" / f"p_{i:02d}_{j:02d}" / "POSCAR") for (i, j) in pts_of[n]]
    ex_paths = [f"{EXISTING_DIR}/cnid_{a}_{b}/POSCAR" for (a, b) in EXISTING_10]
    t0 = time.time()
    with mp.get_context("fork").Pool(NPROC) as pool:
        fr_res = pool.map(frozen_from_poscar, paths + ex_paths, chunksize=2)
    log("frozen-index done", f"{time.time()-t0:.0f}s")
    FR = {}
    k = 0
    for n in GRIDS:
        FR[n] = fr_res[k:k + len(pts_of[n])]; k += len(pts_of[n])
    FR_EX = fr_res[k:]

    W = []   # validation lines
    gate_ok = True
    # (c1) the existing 10 points: frozen of this method == bulk_membership.csv (k=0, seed=0)
    csvfz = {}
    with open(f"{RB}/bulk_membership.csv") as f:
        for r in csv.DictReader(f):
            if r["k"] == "0" and r["seed"] == "0" and r["status"] == "ok":
                csvfz[r["cnid"]] = sorted(int(x) for x in r["frozen_indices"].split(";"))
    c1 = []
    for (a, b), fr in zip(EXISTING_10, FR_EX):
        c1.append(fr["status"] == "ok" and fr["frozen"] == csvfz.get(f"cnid_{a}_{b}"))
    ok = all(c1)
    gate_ok &= ok
    W.append(f"[{'PASS' if ok else 'FAIL'}] (c1) the fixed indices obtained by applying this method to the existing 10 points (initial POSCAR) agree completely with bulk_membership.csv (k=0,seed=0): {sum(c1)}/10")
    # (c2) all lattice points: 76 atoms, status ok
    for n in GRIDS:
        nf = [len(x["frozen"]) if x["frozen"] else 0 for x in FR[n]]
        nat = sorted({x["natoms"] for x in FR[n]})
        bad = [(pts_of[n][i], FR[n][i]["reason"]) for i in range(len(nf)) if FR[n][i]["status"] != "ok" or nf[i] != 76]
        ok = not bad
        gate_ok &= ok
        mx = max((x["max_pos_diff_A"] or 0.0) for x in FR[n]); mo = max(x["op_residual_A"] for x in FR[n])
        W.append(f"[{'PASS' if ok else 'FAIL'}] (c2) {n}x{n}: n_frozen==76 and status ok for all {len(nf)} points  (natoms={nat}, max membership diff={mx:.2e} A, max sym residual={mo:.2e} A)"
                 + ("" if ok else f" violations={bad[:10]}"))
    # (c3) reference: comparison with the earlier expansion_bulk_membership.csv (cnid_g12, k=0 seed=0). Reference information, not a check
    ebm = {}
    with open(f"{RUN0809}/results/expansion_bulk_membership.csv") as f:
        for r in csv.DictReader(f):
            if r["k"] == "0" and r["seed"] == "0" and r["status"] == "ok" and r["cnid"].startswith("cnid_g12_"):
                ebm[r["cnid"]] = sorted(int(x) for x in r["frozen_indices"].split(";"))
    agree = tot = 0
    for idx, (i, j) in enumerate(pts_of[12]):
        key = f"cnid_g12_{i:02d}_{j:02d}"
        if key in ebm:
            tot += 1; agree += int(FR[12][idx]["frozen"] == ebm[key])
    W.append(f"[INFO] (c3) for the {tot} points of 12x12 that have an ok row in the earlier expansion_bulk_membership.csv the fixed indices agree: {agree}/{tot} (reference, not a check)")
    # identity of the initial POSCAR and the atom_removal k0_s0 POSCAR (reference)
    same = []
    for (a, b), s in zip(EXISTING_10, EX):
        p = Structure.from_file(f"{PILOT_DIR}/cnid_{a}_{b}/k0_s0/POSCAR")
        d = np.abs(p.frac_coords - s.frac_coords).max() if len(p) == len(s) else None
        same.append(d is not None and d < 1e-6 and np.allclose(p.lattice.matrix, s.lattice.matrix, atol=1e-6))
    W.append(f"[INFO] existing 10 points: the initial POSCAR and the atom_removal k0_s0 POSCAR (input of today's bulk matching) agree in coordinates and lattice: {sum(same)}/10")

    # ---- 3. bulk sub-structures
    BULK = {n: [sub(S[n][i], FR[n][i]["frozen"]) for i in range(len(pts_of[n]))] for n in GRIDS}
    EX_BULK = [sub(EX[i], FR_EX[i]["frozen"]) for i in range(10)]

    # ---- 4. among the existing 10 points (b)
    G["all"] = EX; G["bulk"] = EX_BULK
    pairs10 = list(itertools.combinations(range(10), 2))
    ra = run_tasks(_pair_all, pairs10, "existing M-all pairs")
    rb_ = run_tasks(_pair_bulk, pairs10, "existing M-bulk pairs")
    ex_all_hit = sorted((label(EXISTING_10[a]), label(EXISTING_10[b])) for a, b, f in ra if f)
    ex_b2 = sorted((label(EXISTING_10[a]), label(EXISTING_10[b])) for a, b, f2, f1 in rb_ if f2)
    ex_b1 = sorted((label(EXISTING_10[a]), label(EXISTING_10[b])) for a, b, f2, f1 in rb_ if f1)
    ok = set(ex_b2) == EXPECT_BULK_PAIRS and len(ex_b2) == 3
    gate_ok &= ok
    W.append(f"[{'PASS' if ok else 'FAIL'}] (b) pairs matching by M-bulk(M2) among the existing 10 points={ex_b2} (expected: only the 3 pairs 1_3/2_1, 0_1/0_3, 1_2/3_2, of 45 pairs)")
    W.append(f"[INFO] (b') pairs matching by M-bulk(M1)={ex_b1}")
    ok = len(ex_all_hit) == 0
    gate_ok &= ok
    W.append(f"[{'PASS' if ok else 'FAIL'}] (a2) pairs matching by M-all (whole-structure matching) among the existing 10 points={ex_all_hit} (earlier self-check: 10 groups = 0 matching pairs)")

    # ---- 5. each lattice
    R = {}
    for n in GRIDS:
        pts = pts_of[n]; N = len(pts)
        G["all"] = S[n]; G["bulk"] = BULK[n]; G["ex_all"] = EX; G["ex_bulk"] = EX_BULK
        pairs = list(itertools.combinations(range(N), 2))
        log(f"grid {n}x{n}: {N} points, {len(pairs)} pairs")
        t0 = time.time()
        pa = run_tasks(_pair_all, pairs, f"{n}x{n} M-all pairs")
        pb = run_tasks(_pair_bulk, pairs, f"{n}x{n} M-bulk pairs")
        log(f"  pairwise done {time.time()-t0:.0f}s")
        ea = run_tasks(_ex_all, [(p, e) for p in range(N) for e in range(10)], f"{n}x{n} vs existing M-all")
        eb = run_tasks(_ex_bulk, [(p, e) for p in range(N) for e in range(10)], f"{n}x{n} vs existing M-bulk")
        edges = {"M-all": [(a, b) for a, b, f in pa if f],
                 "M-bulk": [(a, b) for a, b, f2, f1 in pb if f2],
                 "M-bulk-M1": [(a, b) for a, b, f2, f1 in pb if f1]}
        mex = {"M-all": {p: [] for p in range(N)}, "M-bulk": {p: [] for p in range(N)}}
        for p, e, f in ea:
            if f: mex["M-all"][p].append(e)
        for p, e, f2, f1 in eb:
            if f2: mex["M-bulk"][p].append(e)
        R[n] = dict(edges=edges, mex=mex)
        R[n]["groups"] = {m: union_groups(N, edges[m]) for m in edges}

    # ---- 6. validation (a), positive control for 3i,3j
    g12 = R[12]["groups"]["M-all"]
    n_all12 = len(g12)
    ok = (n_all12 == 42)
    gate_ok &= ok
    reps12 = sorted(rep_of(g, pts_of[12]) for g in g12)
    sizes12 = sorted(len(g) for g in g12)
    aug = json.load(open(f"{RUN0809}/results/cnid_dense_report.json"))["12x12"]
    aug_reps = sorted(tuple(x) for x in aug["reps"]); aug_sizes = sorted(aug["group_sizes"])
    W.append(f"[{'PASS' if ok else 'FAIL'}] (a) 12x12 M-all (all pairs, union-find) number of representatives={n_all12} (earlier=42)")
    W.append(f"[INFO] (a') agreement with the set of representatives of the earlier 12x12: reps {'agree' if reps12 == aug_reps else 'differ'} / group size distribution {'agree' if sizes12 == aug_sizes else 'differ'} (this time number of reps={len(reps12)}, earlier={len(aug_reps)})")
    # positive control 3i,3j: the (3a,3b) of 12x12 match the existing (a,b) by M-all and M-bulk
    pos_all = []; pos_bulk = []
    for e, (a, b) in enumerate(EXISTING_10):
        p = (3 * a) * 12 + (3 * b)
        pos_all.append(e in R[12]["mex"]["M-all"][p]); pos_bulk.append(e in R[12]["mex"]["M-bulk"][p])
    ok = all(pos_all) and all(pos_bulk)
    gate_ok &= ok
    W.append(f"[{'PASS' if ok else 'FAIL'}] (c4) the (3i,3j) points of 12x12 match the existing cnid_i_j as themselves (positive control of the reproduction of the construction): M-all {sum(pos_all)}/10, M-bulk {sum(pos_bulk)}/10")

    # ---- 7. groups csv
    summ = {}
    for n in GRIDS:
        pts = pts_of[n]; N = len(pts)
        for meth in ("M-all", "M-bulk"):
            gs = R[n]["groups"][meth]
            reps = {tuple(rep_of(g, pts)): g for g in gs}
            order = sorted(reps)
            gid = {}
            for k, r in enumerate(order):
                for m in reps[r]:
                    gid[m] = k
            # clique check
            es = set(map(tuple, R[n]["edges"][meth]))
            nonclique = 0
            for g in gs:
                gg = sorted(g)
                if any((a, b) not in es for a, b in itertools.combinations(gg, 2)):
                    nonclique += 1
            ex_label = {}
            if n == 12:
                for (a, b) in EXISTING_10:
                    ex_label[(3 * a) * 12 + (3 * b)] = f"cnid_{a}_{b}"
            fn = WORK / f"groups_{n}x{n}_{meth}.csv"
            with open(fn, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["grid", "method", "i", "j", "group_id", "group_size", "rep_i", "rep_j", "is_rep",
                            "existing_label_at_3i3j", "matches_existing(M-bulk=M2)", "n_frozen"])
                for k, r in enumerate(order):
                    for m in sorted(reps[r]):
                        me = ";".join(label(EXISTING_10[e]) for e in sorted(R[n]["mex"][meth][m]))
                        w.writerow([f"{n}x{n}", meth, pts[m][0], pts[m][1], k, len(reps[r]), r[0], r[1],
                                    int(pts[m] == r), ex_label.get(m, ""), me, len(FR[n][m]["frozen"] or [])])
            # coverage
            cov = unc = 0; incons = 0; cov_groups = []; unc_groups = []
            for k, r in enumerate(order):
                sets = [frozenset(R[n]["mex"][meth][m]) for m in reps[r]]
                if len(set(sets)) > 1:
                    incons += 1
                if any(sets):
                    cov += 1; cov_groups.append(k)
                else:
                    unc += 1; unc_groups.append(k)
            n_cov_existing = len({e for m in range(N) for e in R[n]["mex"][meth][m]})
            d = dict(n_raw=N, n_groups=len(order), nonclique=nonclique, covered=cov, uncovered=unc,
                     incons=incons, unc_reps=[order[k] for k in unc_groups], cov_reps=[order[k] for k in cov_groups],
                     file=str(fn), n_existing_matched=n_cov_existing,
                     sizes=sorted((len(reps[r]) for r in order), reverse=True))
            if n == 12:
                gid12 = {}
                for e, (a, b) in enumerate(EXISTING_10):
                    gid12[(a, b)] = gid[(3 * a) * 12 + (3 * b)]
                d["existing_group_ids"] = gid12
                d["existing_distinct_groups"] = len(set(gid12.values()))
                d["existing_group_rep"] = {k_: order[v] for k_, v in gid12.items()}
                # number of groups to which the existing points belong
                grp_members = {}
                for k_, v in gid12.items():
                    grp_members.setdefault(v, []).append(label(k_))
                d["existing_same_group"] = [sorted(v) for v in grp_members.values() if len(v) > 1]
            summ[(n, meth)] = d
        d1 = len(R[n]["groups"]["M-bulk-M1"])
        summ[(n, "M-bulk-M1")] = dict(n_groups=d1)
        log(f"{n}x{n}: M-all={summ[(n,'M-all')]['n_groups']} M-bulk(M2)={summ[(n,'M-bulk')]['n_groups']} M-bulk(M1)={d1}")

    # ---- 8. item 6: number of relaxed structures of the earlier arm A
    A = Path(RUN0809) / "structures" / "A"
    dirs = sorted(p for p in A.glob("cnid_g12_*") if p.is_dir())
    n_e = n_conv = n_rel = 0
    for d in dirs:
        ej = list(d.glob("*/energy.json"))
        if ej:
            n_e += 1
            try:
                if json.load(open(ej[0])).get("converged") is True:
                    n_conv += 1
            except Exception:
                pass
        if list(d.glob("*/POSCAR_relaxed")):
            n_rel += 1
    item6 = (len(dirs), n_e, n_conv, n_rel)

    # ---- summary
    L = ["# CNID coarse grid sweep Al Sigma5(210)[001] structures before relaxation", f"# run host={os.uname().nodename} python={sys.executable}",
         f"# all-pair matching + union-find (transitive merging). M-all=StructureMatcher{MATCHER_KW}, M-bulk=only the 76 fixed bulk atoms M2={MATCH['M2']} (M1={MATCH['M1']} is for reference)", ""]
    L.append("## Control (validation)")
    L += W
    L.append("")
    if not gate_ok:
        L.append("STOP: no conclusion is given because the control did not hold. See the FAIL rows above. groups_*.csv were written as raw data but are unverified.")
        L.append("(The following tables are reference values and are not used as conclusions)")
    L.append("## Number of representatives per lattice x method (number of groups after removing duplicates)")
    L.append("grid    n_raw  M-all  M-bulk(M2)  M-bulk(M1 reference)  number of non-clique groups (all/bulk)")
    for n in GRIDS:
        L.append(f"{n}x{n:<4}{n*n:>6}  {summ[(n,'M-all')]['n_groups']:>5}  {summ[(n,'M-bulk')]['n_groups']:>10}  {summ[(n,'M-bulk-M1')]['n_groups']:>13}  {summ[(n,'M-all')]['nonclique']}/{summ[(n,'M-bulk')]['nonclique']}")
    L.append("")
    L.append("## Number of groups covered / not covered by the existing 10 points (from 4x4)")
    for n in GRIDS:
        for meth in ("M-all", "M-bulk"):
            d = summ[(n, meth)]
            L.append(f"{n}x{n} {meth}: of {d['n_groups']} groups in total, with a representative matching an existing point={d['covered']}, without (new structures not in the existing set)={d['unc_reps'].__len__()}  "
                     f"(groups in which the matches with existing points are inconsistent={d['incons']}, number of the existing 10 points that found a partner={d['n_existing_matched']}/10)")
            L.append(f"   representatives (i,j) not matching an existing point: {d['unc_reps']}")
    L.append("")
    L.append("## 12x12: groups containing the existing 10 points (3i,3j)")
    for meth in ("M-all", "M-bulk"):
        d = summ[(12, meth)]
        L.append(f"{meth}: the existing 10 points are distributed over {d['existing_distinct_groups']} groups. Existing points in the same group: {d['existing_same_group']}")
        L.append("   existing point -> group representative (i,j): " + ", ".join(f"cnid_{a}_{b}->{d['existing_group_rep'][(a,b)]}" for (a, b) in EXISTING_10))
    L.append("")
    L.append("## Note on 10x10: the existing points are not on the lattice (except 0 and 1/2), so all 100 points were matched directly against the existing 10 points (covered/uncovered above)")
    L.append("")
    L.append("## Number of relaxed structures of the 32 new points of the earlier 12x12 (structures/A/cnid_g12_*) (read only)")
    L.append(f"Number of cnid_g12_* directories under A={item6[0]} / with energy.json={item6[1]} / of which converged=true {item6[2]} / with POSCAR_relaxed={item6[3]}")
    L.append("")
    L.append("files: " + ", ".join(f"groups_{n}x{n}_{m}.csv" for n in GRIDS for m in ("M-all", "M-bulk")) + ", summary.txt, structures/ (constructed POSCAR)")
    with open(WORK / "summary.txt", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("ALL_DONE gate_ok=", gate_ok)

if __name__ == "__main__":
    main()
