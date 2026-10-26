#!/usr/bin/python3
"""Al main experiment (layer=main, w=2.0): sorts the structures that "converged after relaxation AND are novel after relaxation" in ascending gamma_relaxed and writes the top ones, one structure per file.

Run: /usr/bin/python3 -I export_low_gamma_novel.py   (standard library only. No analysis or interpretation is included)
Master of the matching rules: ${GBREL_CODE}/gb_eval/experiments/relax_novelty/scripts/aggregate_relax_novelty.py
  - converged = relax_rows.status == "ok_converged"           (same :165-166, :178)
  - novel_post = novelty_post_rows.novel == "True"        (same :180 tri())
  - frame = the relax_rows.frame_index-th, check that cif_index of the header agrees (same :283-285)
The output destination is only newly created (stops immediately if it exists).
"""
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)
import csv
import os
import re
import sys

ROOT = _gbrel.WORK + '/workspace/runs'
RUN = f"{ROOT}/relax_novelty"
VER = f"{ROOT}/verification"
OUT = f"{VER}/results/low_gamma_novel_view"
ARMS = ["proposed", "baseline"]
W = 2.0
N_CHUNK = 14
PER_CHUNK = 25
NEAR = 0.10          # common.py:79 NEAR_TOL_J
TOP = {"proposed": 10, "baseline": 5}
N_TRAIN_REF = 3


def fix(p):
    """Translates <CLUSTER_MOUNT>/ in the train manifest to <CLUSTER_MOUNT>/ of the login-node mount."""
    if p.startswith('/CLUSTER_ROOT_B/'):
        return "/mnt" + p
    return p


def rcsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v


def chunk_name(c):
    return f"chunk{c:02d}"


def read_frame_text(path, frame_index, cif_index):
    """Returns the frame_index-th frame from extxyz as text as it is (natoms line, comment line, atom lines)."""
    with open(path) as f:
        lines = f.read().split("\n")
    pos, k = 0, 0
    while pos < len(lines):
        if not lines[pos].strip():
            pos += 1
            continue
        n = int(lines[pos].strip())
        block = lines[pos:pos + n + 2]
        if k == frame_index:
            m = re.search(r"\bcif_index=(\d+)", block[1])
            if m is None or int(m.group(1)) != cif_index:
                raise SystemExit(f"STOP: {path} frame {frame_index} cif_index mismatch ({m and m.group(1)} != {cif_index})")
            return block
        pos += n + 2
        k += 1
    raise SystemExit(f"STOP: {path} frame {frame_index} not found (frames={k})")


def parse_frame(block):
    n = int(block[0].strip())
    m = re.search(r'Lattice="([^"]+)"', block[1])
    lat = [float(x) for x in m.group(1).split()]
    if len(lat) != 9:
        raise SystemExit("STOP: Lattice not 9 numbers")
    atoms = []
    for ln in block[2:2 + n]:
        t = ln.split()
        atoms.append((t[0], float(t[1]), float(t[2]), float(t[3])))
    if len(atoms) != n:
        raise SystemExit("STOP: atom count mismatch in frame")
    return n, lat, atoms


def to_poscar(comment, lat, atoms):
    order = []
    for s, *_ in atoms:
        if s not in order:
            order.append(s)
    lines = [comment, "1.0"]
    for i in range(3):
        lines.append("  " + "  ".join(f"{lat[3 * i + j]:.10f}" for j in range(3)))
    lines.append("  " + "  ".join(order))
    lines.append("  " + "  ".join(str(sum(1 for a in atoms if a[0] == s)) for s in order))
    lines.append("Cartesian")
    for s in order:
        for a in atoms:
            if a[0] == s:
                lines.append(f"  {a[1]:.10f}  {a[2]:.10f}  {a[3]:.10f}")
    return "\n".join(lines) + "\n"


def check_poscar(text, n, lat, tol=1e-9):
    """Re-reads the POSCAR and checks that the number of atoms and the lattice agree with the header (extxyz)."""
    L = text.split("\n")
    lat2 = [float(x) for i in range(2, 5) for x in L[i].split()]
    counts = [int(x) for x in L[6].split()]
    n_atoms_lines = len([x for x in L[8:] if x.strip()])
    ok = (sum(counts) == n) and (n_atoms_lines == n) and all(abs(a - b) <= tol for a, b in zip(lat, lat2))
    if not ok:
        raise SystemExit(f"STOP: POSCAR check failed (counts={counts}, lines={n_atoms_lines}, n={n})")
    return True


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"STOP: {OUT} already exists")
    # ---- reading of the inputs
    manifest = {}
    for r in rcsv(f"{RUN}/results/inputs_manifest_main.csv"):
        if abs(float(r["w"]) - W) < 1e-9:
            manifest[(r["arm"], int(r["chunk"]), int(r["cif_index"]))] = r
    nn = {}
    for i in range(10):
        for r in rcsv(f"{VER}/results/grid_interp/nn_G_{i:02d}.csv"):
            k = (int(r["chunk"]), int(r["cif_index"]))
            if k in nn:
                raise SystemExit(f"STOP: duplicate nn_G key {k}")
            nn[k] = r
    glist = {(int(r["chunk"]), int(r["cif_index"])) for r in rcsv(f"{VER}/results/grid_interp/G_list.csv")}
    if set(nn) != glist:
        raise SystemExit("STOP: nn_G keys != G_list keys")
    sym = {}
    for r in rcsv(f"{VER}/results/gb_equivalence/gb_symfit_main.csv"):
        if r["stage"] == "post" and abs(float(r["w"]) - W) < 1e-9:
            sym[(r["arm"], int(r["chunk"]), int(r["cif_index"]))] = r

    # ---- collecting
    sel = {a: [] for a in ARMS}
    stats = {a: {"n_total": 0, "n_converged": 0, "n_novel_post_of_conv": 0, "n_sel": 0} for a in ARMS}
    for arm in ARMS:
        for c in range(N_CHUNK):
            rd = f"{RUN}/results/relax/main/{arm}_w{W:.1f}/{chunk_name(c)}"
            nd = f"{RUN}/results/novelty/main/{arm}_w{W:.1f}/{chunk_name(c)}"
            rr = {int(x["cif_index"]): x for x in rcsv(f"{rd}/relax_rows.csv")}
            nr = {int(x["cif_index"]): x for x in rcsv(f"{nd}/novelty_post_rows.csv")}
            if len(rr) != PER_CHUNK or len(nr) != PER_CHUNK:
                raise SystemExit(f"STOP: {arm} chunk {c}: rows relax={len(rr)} novelty={len(nr)} != {PER_CHUNK}")
            for i in range(PER_CHUNK):
                x, n = rr[i], nr[i]
                stats[arm]["n_total"] += 1
                conv = x["status"] == "ok_converged"
                nov = {"True": True, "False": False}.get(n["novel"])
                if conv != (nov is not None):
                    raise SystemExit(f"STOP: novelty-post NA inconsistent {arm} c{c} i{i}")
                if conv:
                    stats[arm]["n_converged"] += 1
                if conv and nov:
                    stats[arm]["n_novel_post_of_conv"] += 1
                    gr = fnum(x["gamma_relaxed"])
                    if gr is None:
                        raise SystemExit(f"STOP: gamma_relaxed NA {arm} c{c} i{i}")
                    m = manifest[(arm, c, i)]
                    if int(m["natoms"]) != int(x["natoms"]) or abs(float(m["gamma_target"]) - float(x["gamma_target"])) > 1e-12:
                        raise SystemExit(f"STOP: manifest/relax_rows mismatch {arm} c{c} i{i}")
                    sel[arm].append({"arm": arm, "chunk": c, "cif_index": i, "rr": x, "m": m, "gr": gr,
                                     "relaxed_path": f"{rd}/relaxed.extxyz"})
        stats[arm]["n_sel"] = len(sel[arm])
        sel[arm].sort(key=lambda d: (d["gr"], d["chunk"], d["cif_index"]))

    # ---- building the rows
    rows = []
    for arm in ARMS:
        for rank, d in enumerate(sel[arm][:TOP[arm]], 1):
            x, m = d["rr"], d["m"]
            c, i = d["chunk"], d["cif_index"]
            tid = c // 2
            if int(x["target_id"]) != tid or int(m["target_id"]) != tid:
                raise SystemExit(f"STOP: target_id != chunk//2 at {arm} c{c} i{i}")
            gt = float(x["gamma_target"])
            nnr = nn.get((c, i)) if arm == "proposed" else None
            sy = sym.get((arm, c, i))
            rows.append({
                "rank": rank, "arm": arm, "chunk": c, "cif_index": i, "target_id": tid,
                "gamma_target": gt, "gamma_relaxed": d["gr"], "gamma_unrelaxed": fnum(m["gamma_unrelaxed"]),
                "natoms": int(x["natoms"]), "gb_dx_realized_A": fnum(m["gb_dx_realized_A"]),
                "dev_abs": abs(d["gr"] - gt), "near": abs(d["gr"] - gt) <= NEAR,
                "in_G": nnr is not None,
                "d_rms_min_A": fnum(nnr["d_rms_min_A"]) if nnr else None,
                "nn_cnid": nnr["rms_nn_cnid"] if nnr else "",
                "nn_dx_A": fnum(nnr["rms_nn_dx_A"]) if nnr else None,
                "nn_gamma_Jm2": fnum(nnr["rms_nn_gamma"]) if nnr else None,
                "symfit_status": sy["status"] if sy else "",
                "symfit_gb0_status": sy["gb0_status"] if sy else "",
                "symfit_op_name": sy["op_name"] if sy else "",
                "symfit_max_A": fnum(sy["gb_symfit_max_A"]) if sy else None,
                "frame_index": int(x["frame_index"]), "relaxed_path": d["relaxed_path"],
                "file_stem": "",
            })
    # consistency of the G membership: in proposed, near AND conv AND novel should be in G (and the converse)
    for r in rows:
        if r["arm"] == "proposed" and r["near"] != r["in_G"]:
            raise SystemExit(f"STOP: proposed near={r['near']} but in_G={r['in_G']} at c{r['chunk']} i{r['cif_index']}")

    # ---- writing (only newly created)
    os.mkdir(OUT)
    checks = []
    for r in rows:
        stem = f"{r['rank']:02d}_{r['arm']}_c{r['chunk']:02d}_i{r['cif_index']:02d}_g{r['gamma_relaxed']:.3f}"
        r["file_stem"] = stem
        block = read_frame_text(r["relaxed_path"], r["frame_index"], r["cif_index"])
        n, lat, atoms = parse_frame(block)
        if n != r["natoms"]:
            raise SystemExit(f"STOP: natoms {n} != table {r['natoms']} ({stem})")
        hg = re.search(r"\bgamma_relaxed=(\S+)", block[1])
        if hg is None or abs(float(hg.group(1)) - r["gamma_relaxed"]) > 1e-9:
            raise SystemExit(f"STOP: extxyz header gamma_relaxed != relax_rows ({stem})")
        with open(f"{OUT}/{stem}.extxyz", "x") as f:
            f.write("\n".join(block) + "\n")
        comment = f"{r['arm']} chunk{r['chunk']:02d} cif_index{r['cif_index']:02d} gamma_relaxed={r['gamma_relaxed']:.6f} J/m2 w=2.0 main"
        text = to_poscar(comment, lat, atoms)
        check_poscar(text, n, lat)
        with open(f"{OUT}/{stem}.vasp", "x") as f:
            f.write(text)
        checks.append((stem, n, True))

    # train reference: the 3 structures with the lowest gamma
    train = rcsv(f"{RUN}/results/train_ref_manifest.csv")
    if len(train) != 4419:
        raise SystemExit(f"STOP: train manifest rows {len(train)} != 4419")
    tsort = sorted(train, key=lambda t: (float(t["gamma_Jm2"]), int(t["row_index"])))
    trows = []
    for k, t in enumerate(tsort[:N_TRAIN_REF], 1):
        g = float(t["gamma_Jm2"])
        src = fix(t["poscar_relaxed_path"])
        with open(src) as f:
            text = f.read()
        L = text.split("\n")
        cnt = sum(int(x) for x in L[6].split())
        if cnt != int(t["natoms"]):
            raise SystemExit(f"STOP: train POSCAR natoms {cnt} != manifest {t['natoms']} ({src})")
        name = f"train_{k:02d}_row{t['row_index']}_g{g:.3f}.vasp"
        with open(f"{OUT}/{name}", "x") as f:
            f.write(text)
        trows.append({"rank": k, "file": name, "row_index": int(t["row_index"]), "cnid": t["cnid"], "k": int(t["k"]),
                      "dx_A": float(t["dx_A"]), "gamma_Jm2": g, "natoms": int(t["natoms"]), "src": src})

    # ---- table.csv
    cols = ["rank", "arm", "chunk", "cif_index", "target_id", "gamma_target", "gamma_relaxed", "gamma_unrelaxed", "natoms",
            "gb_dx_realized_A", "dev_abs", "near", "in_G", "d_rms_min_A", "nn_cnid", "nn_dx_A", "nn_gamma_Jm2",
            "symfit_status", "symfit_gb0_status", "symfit_op_name", "symfit_max_A", "file_stem", "frame_index", "relaxed_path"]
    with open(f"{OUT}/table.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r[c] is None else r[c] for c in cols])
        w.writerow([])
        w.writerow(["# train reference (3 lowest gamma)", "rank", "file", "row_index", "cnid", "k", "dx_A", "gamma_Jm2", "natoms", "poscar_relaxed_path"])
        for t in trows:
            w.writerow(["#", t["rank"], t["file"], t["row_index"], t["cnid"], t["k"], t["dx_A"], t["gamma_Jm2"], t["natoms"], t["src"]])

    # ---- README.md
    def fm(v, nd=4):
        return "" if v is None else f"{v:.{nd}f}"
    md = []
    md.append("# Export of low-gamma novel structures for visual inspection (Al main experiment layer=main, w=2.0)\n")
    md.append("Target: structures that converged after relaxation (relax_rows.status==ok_converged) AND are novel after relaxation (novelty_post_rows.novel==True). "
              "Sorted by gamma_relaxed in ascending order per arm. No analysis or interpretation is included.\n")
    md.append("Generating script: `${GBREL_CODE}/gb_eval/experiments/verification/scripts/export_low_gamma_novel.py`\n")
    md.append("## Counts (layer=main, w=2.0, 14 chunks x 25 = 350 requested per arm)\n")
    md.append("| arm | requested | converged (ok_converged) | converged AND novel_post=True (= population of this table) |")
    md.append("|---|---|---|---|")
    for a in ARMS:
        s = stats[a]
        md.append(f"| {a} | {s['n_total']} | {s['n_converged']} | {s['n_sel']} |")
    md.append("")
    md.append("## proposed top 10 / baseline top 5\n")
    md.append("| rank | arm | chunk | cif_index | target_id | gamma_target | gamma_relaxed | gamma_unrelaxed | natoms | gb_dx_realized_A | abs(gamma_relaxed-gamma_target) | near (<=0.10) | G | d_rms_min_A | nearest training cnid | nearest dx_A | nearest gamma | equivalent grain boundaries status / op / max_A | file |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        sy = (f"{r['symfit_status']} / {r['symfit_op_name'] or '-'} / {fm(r['symfit_max_A'], 6) if r['symfit_max_A'] is not None else '-'}"
              if r["symfit_status"] else "")
        md.append(f"| {r['rank']} | {r['arm']} | {r['chunk']} | {r['cif_index']} | {r['target_id']} | {fm(r['gamma_target'])} | "
                  f"{fm(r['gamma_relaxed'])} | {fm(r['gamma_unrelaxed'])} | {r['natoms']} | {fm(r['gb_dx_realized_A'])} | "
                  f"{fm(r['dev_abs'])} | {'Y' if r['near'] else 'N'} | {'Y' if r['in_G'] else 'N'} | {fm(r['d_rms_min_A'])} | "
                  f"{r['nn_cnid']} | {fm(r['nn_dx_A'], 2)} | {fm(r['nn_gamma_Jm2'])} | {sy} | `{r['file_stem']}.{{extxyz,vasp}}` |")
    md.append("")
    md.append("## The 3 training structures with the lowest gamma (reference)\n")
    md.append("| rank | row_index | cnid | k | dx_A | gamma [J/m2] | natoms | file | original file |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for t in trows:
        md.append(f"| {t['rank']} | {t['row_index']} | {t['cnid']} | {t['k']} | {t['dx_A']:.2f} | {t['gamma_Jm2']:.4f} | {t['natoms']} | `{t['file']}` | `{t['src']}` |")
    md.append("")
    md.append("## Definitions of the columns and sources (RUN=${GBREL_WORK}/workspace/experiments/relax_novelty, VER=${GBREL_WORK}/workspace/experiments/verification)\n")
    md.append("- Population (converged AND novel_post): converged is `status==ok_converged` of `RUN/results/relax/main/<model>_w2.0/chunkNN/relax_rows.csv` "
              "(same as aggregate_relax_novelty.py:165-166,178). novel is `novel==True` of `RUN/results/novelty/main/<model>_w2.0/chunkNN/novelty_post_rows.csv` (same: 180). "
              "The consistency converged <=> novel non-NA is checked at run time (same as same: 214-215).")
    md.append("- chunk / cif_index: the `chunk` directory name / `cif_index` column of the CSV above. target_id = chunk//2 (checked to agree with the `target_id` column of relax_rows.csv).")
    md.append("- gamma_target, gamma_relaxed, natoms: the columns of the same name of `relax_rows.csv`. gamma_relaxed is also checked to agree with `gamma_relaxed=` in the extxyz header.")
    md.append("- gamma_unrelaxed, gb_dx_realized_A: the columns `gamma_unrelaxed`, `gb_dx_realized_A` of `RUN/results/inputs_manifest_main.csv` (key arm,w=2.0,chunk,cif_index).")
    md.append("- near: |gamma_relaxed-gamma_target| <= 0.10 J/m2 (`RUN/scripts/common.py:79` NEAR_TOL_J. Post-side definition of aggregate_relax_novelty.py:253).")
    md.append("- G: the set of proposed structures that are near AND converged AND novel_post (`VER/scripts/grid_interp.py:219-224`, the actual file `VER/results/grid_interp/G_list.csv`, 90 rows). "
              "G exists only for proposed. In the proposed rows of this table, near=Y <=> G=Y is checked at run time. baseline is not a target of G, so the d_rms columns are blank.")
    md.append("- d_rms_min_A, cnid / dx_A / gamma of the nearest training structure: the columns `d_rms_min_A`, `rms_nn_cnid`, `rms_nn_dx_A`, `rms_nn_gamma` of `VER/results/grid_interp/nn_G_00..09.csv` (key chunk,cif_index). "
              "These are the values of \"the training structure that gives the minimum d_rms over all training structures\" (not the `nn_*` columns that give d_min).")
    md.append("- Equivalent grain boundaries (status / op / max_A): the columns `status`, `op_name`, `gb_symfit_max_A` of the stage=post rows of `VER/results/gb_equivalence/gb_symfit_main.csv` (key arm,w=2.0,chunk,cif_index). "
              "op_name is one of the 4 classes of the adopted symmetry operation {C2n, m_p, C2t, inv} (gb_symfit_summary.md section 3). C2n of proposed is a structure that satisfies the E decision (theta=0.064 A or less), "
              "status=no_gb is a structure for which two grain boundaries could not be identified, and op is blank. Whether the \"4 classes\" are the same classification as intended in the request is a guess (guess 1).")
    md.append("- Structure files: `<rank 2 digits>_<model>_c<chunk 2 digits>_i<cif_index 2 digits>_g<gamma_relaxed 3 digits>.extxyz` is the frame number "
              "`frame_index` (the `frame_index` of relax_rows.csv, checked to agree with `cif_index` of the header) of `RUN/results/relax/main/<model>_w2.0/chunkNN/relaxed.extxyz`, cut out as text. "
              "`.vasp` is a POSCAR converted from the same frame with the standard library (Cartesian, sorted by element, with arm, chunk, cif_index and gamma in the comment line). "
              "After the conversion the number of atoms and the lattice (9 components, within 1e-9 A) were checked to agree with the header.")
    md.append("- Training reference: the first 3 rows of `RUN/results/train_ref_manifest.csv` sorted by gamma (`gamma_Jm2`) in ascending order. POSCAR is a copy of `poscar_relaxed_path` (<CLUSTER_MOUNT> read as <CLUSTER_MOUNT>). "
              "The minimum of gamma over the whole training set is the value of rank 1.\n")
    md.append("## Result of the conversion check\n")
    md.append("| file stem | natoms | POSCAR number of atoms and lattice check |")
    md.append("|---|---|---|")
    for s, n, ok in checks:
        md.append(f"| {s} | {n} | {'OK' if ok else 'NG'} |")
    md.append("")
    with open(f"{OUT}/README.md", "x") as f:
        f.write("\n".join(md) + "\n")
    print("stats", stats)
    print("wrote", len(rows), "structures x2,", len(trows), "train refs")


if __name__ == "__main__":
    main()
