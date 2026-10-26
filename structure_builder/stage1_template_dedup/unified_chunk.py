#!/usr/bin/env python3
"""All-pair matching with the unified setting (chunk version). A copy of cnid_grid_chunk.py: only the matching function is replaced by the unified setting, and the fixed bulk is not used.
Unified setting = the whole 120-atom structure with StructureMatcher(stol=0.025, ltol/angle_tol=pymatgen defaults, primitive_cell=False, scale=False, attempt_supercell=False)
kind: s=each point with itself (100) / x=among the existing 10 points (45 pairs) / p=among the points of 10x10 (4950 pairs) / e=points of 10x10 x the existing 10 points (1000 pairs)
"""
import sys, os, csv, time, itertools
from pathlib import Path
from importlib.metadata import version
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

WORK = Path(os.environ["SB_WORK"])   # required, no default (KeyError if unset)
RES = WORK / "stage1" / "results"
STRUCT = WORK / "stage0" / "structures" / "g10"
UNI = dict(stol=0.025, primitive_cell=False, scale=False, attempt_supercell=False)   # ltol=0.2 and angle_tol=5 are the defaults
N = 10
CS = 20
KINDS = [("s", 100), ("p", 4950)]   # the kinds x/e (comparison with the 10 earlier-grid structures) are not available in the public release

def plan():
    t = []
    for kind, tot in KINDS:
        for s in range(0, tot, CS):
            t.append((kind, s, min(s + CS, tot)))
    return t

def chunk(tid):
    P = plan()
    assert 0 <= tid < len(P), (tid, len(P))
    kind, s, e = P[tid]
    RES.mkdir(parents=True, exist_ok=True)
    fn = RES / f"chunk_{kind}_{s:06d}.csv"
    if fn.exists():
        print("exists, skip", fn); return
    t0 = time.time()
    sm0 = StructureMatcher(**UNI)
    print(f"pymatgen {version('pymatgen')}  matcher: stol={sm0.stol} ltol={sm0.ltol} angle_tol={sm0.angle_tol} prim={sm0._primitive_cell} scale={sm0._scale} supercell={sm0._supercell}", flush=True)
    S = [Structure.from_file(str(STRUCT / f"p_{i:02d}_{j:02d}" / "POSCAR")) for i in range(N) for j in range(N)]
    if kind == "p": pairs = list(itertools.combinations(range(N * N), 2))
    rows = []
    for k in range(s, e):
        if kind == "s": a = b = k; sa = sb = S[k]
        else: a, b = pairs[k]; sa, sb = S[a], S[b]
        sm = StructureMatcher(**UNI)
        fit = bool(sm.fit(sa, sb))
        if fit:
            r = sm.get_rms_dist(sa, sb); rms, mx = float(r[0]), float(r[1])
        else:
            rms = mx = ""
        rows.append([kind, k, a, b, int(fit), rms, mx])
    tmp = fn.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["kind", "k", "a_idx", "b_idx", "fit", "rms_norm", "max_norm", "sec_chunk"])
        for r in rows: w.writerow(r + [f"{time.time()-t0:.0f}"])
    os.rename(tmp, fn)
    print(f"done {fn.name} rows={len(rows)} {time.time()-t0:.0f}s fits={sum(r[4] for r in rows)}", flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "plan": print(len(plan()))
    else: chunk(int(sys.argv[2]))
