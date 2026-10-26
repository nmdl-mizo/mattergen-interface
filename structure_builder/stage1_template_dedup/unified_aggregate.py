#!/usr/bin/env python3
"""Aggregation of the unified setting: reads results/chunk_*.csv, groups transitively over all pairs with union-find, and writes groups_10x10_unified.csv and summary.txt."""
import sys, csv, glob, itertools
from pathlib import Path
from collections import Counter
from pathlib import Path
from importlib.metadata import version
sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_chunk as K

WORK = K.WORK; OUT = WORK / "stage1"; RES = K.RES
N = 10; NP = N * N
pts = [(i, j) for i in range(N) for j in range(N)]

def load(kind, tot):
    rows = {}
    for fn in sorted(glob.glob(str(RES / f"chunk_{kind}_*.csv"))):
        for r in csv.DictReader(open(fn)):
            k = int(r["k"]); assert k not in rows, (fn, k); rows[k] = r
    miss = [k for k in range(tot) if k not in rows]
    if miss: raise RuntimeError(f"{kind}: not computed {len(miss)}/{tot} e.g. {miss[:5]}")
    return [rows[k] for k in range(tot)]

def union_groups(n, edges):
    par = list(range(n))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in edges:
        ra, rb = f(a), f(b)
        if ra != rb: par[max(ra, rb)] = min(ra, rb)
    gs = {}
    for i in range(n): gs.setdefault(f(i), []).append(i)
    return list(gs.values())

def rep_of(members): return min((pts[m] for m in members), key=lambda t: (t[0] + t[1], t[0], t[1]))

def main():
    D = {k: load(k, t) for k, t in K.KINDS}
    W = []; ok_all = True
    nself = sum(int(r["fit"]) for r in D["s"]); ok = nself == 100; ok_all &= ok
    W.append(f"[{'PASS' if ok else 'FAIL'}] (a) matching of each point with itself: {nself}/100 agree")
    pairs = list(itertools.combinations(range(NP), 2))
    edges = [pairs[int(r["k"])] for r in D["p"] if r["fit"] == "1"]
    es = set(edges)
    groups = union_groups(NP, edges)
    reps = {rep_of(g): g for g in groups}; order = sorted(reps)
    gid = {}
    for k, r in enumerate(order):
        for m in reps[r]: gid[m] = k
    nonclique = [reps[r] for r in order if any((a, b) not in es for a, b in itertools.combinations(sorted(reps[r]), 2))]
    mex = {p: [] for p in range(NP)}   # matches_existing is empty in the public release (no earlier-grid structures)
    with open(OUT / "groups_10x10_unified.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grid", "method", "i", "j", "group_id", "group_size", "rep_i", "rep_j", "is_rep", "matches_existing", "prev_M-all_group_id", "prev_M-bulk_group_id"])  # the last three columns are empty in the public release
        for k, r in enumerate(order):
            for m in sorted(reps[r]):
                w.writerow(["10x10", "unified(stol0.025,prim=F,scale=F)", pts[m][0], pts[m][1], k, len(reps[r]), r[0], r[1], int(pts[m] == r),
                            "", "", ""])
    sizes = sorted((len(reps[r]) for r in order), reverse=True)
    L = ["# All-pair matching 10x10 with the unified setting (Al Sigma5(210)[001] structures before relaxation, 100 points, 120 atoms, no removal, dx=0)",
         f"# pymatgen {version('pymatgen')}. Matching: the whole 120-atom structure with StructureMatcher({K.UNI}, ltol/angle_tol are the pymatgen defaults = 0.2/5.0). All 4,950 pairs + union-find (transitive merging)",
         "", "## Controls (validation)"] + W
    L += ["", "## Number of groups", f"Unified setting: {len(order)} groups (100 points)  number of non-clique groups: {len(nonclique)}  ({len(edges)} matching edges)"]
    L += ["", "## Distribution of group sizes", "size: number of groups = " + ", ".join(f"{s}: {c}" for s, c in sorted(Counter(sizes).items(), reverse=True)),
          f"sizes in descending order: {sizes}"]
    if nonclique:
        L += ["", "Non-clique groups: " + str([[pts[m] for m in g] for g in nonclique])]
    L += ["", "## All groups (group_id: representative (i,j) / size / members)"]
    for k, r in enumerate(order):
        L.append(f"  {k}: rep={r} size={len(reps[r])} members={[pts[m] for m in sorted(reps[r])]}")
    L += ["", "files: groups_10x10_unified.csv, summary.txt, results/chunk_*.csv, unified_chunk.py, unified_aggregate.py"]
    (OUT / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L)); print("ALL_DONE gate_ok=", ok_all, flush=True)
main()
