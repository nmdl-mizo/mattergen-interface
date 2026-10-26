#!/usr/bin/env python3
"""dft_write_inputs.py -- writing of the inputs of the DFT stage: 26 grain boundaries + 13 mu = 39 VASP inputs (POSCAR, INCAR, KPOINTS, POTCAR, run_vasp.sh). numpy / ase.
Transfer to and submission on the cluster that runs VASP are not done here.

Plan: see the experiment description (calculation conditions).
  Source of the inputs: VRUN/results/dft_selection/selection.json (13 selected pairs, 26 structures; the structures after MACE relaxation are used as they are = not relaxed again) and dft_mu_check.mu_cell (lattice of the mu cell).
  INCAR   : byte copy of the INCAR of the reference calculation b00s0 (check: diff 0 lines)
  KPOINTS : Gamma-centered, 1 along the longest vector (normal), ceil(2*pi/|a_i|/0.25) for the others (the same rule and format as make_kpoints_content of make_vasp_inputs.py). Computed by re-reading the written POSCAR.
  POTCAR  : the first 2228 lines of the POTCAR of the reference calculation (Al: PAW_PBE Al 04Jan2001.
            the same file and the same line range as in make_vasp_inputs.py:29-30). The content is not read (only copying the lines and checking the byte agreement with b00s0/POTCAR).
  run_vasp.sh: a transcription of .../inputs/b00s0/run_vasp.sh. Only the job name of the scheduler directive is changed (dft_<name>).
  POSCAR  : comment on line 1, "Al", number of atoms, Direct. For grain-boundary structures the lattice after MACE relaxation and the fractional coordinates (folded into [0,1)) are used as they are. Selective dynamics (fixed atoms) is not added (irrelevant for a single-point calculation with NSW=0).
            The fractional coordinates of the mu cell, a perfect FCC-oriented cell of 120 atoms (the same orientation as W9 of gb_symfit: n=[210]x6 periods, t=[001], p=[-120]), are put on the mu lattice
            with the same order of the lattice vectors as the grain-boundary structure (which of a,b,c is n,t,p) (the in-plane lattice is the t and p vectors of the grain-boundary structure themselves).
  Naming : dft_inputs/gb_<model>_t<target>_<lo|hi>/ (26), dft_inputs/mu_proposed_common/ (1), dft_inputs/mu_baseline_t<target>_<lo|hi>/ (12). The correspondence table is in index.json.
Usage: python dft_write_inputs.py --selection selection.json --out-dir dft_inputs --check-json PATH --check-log PATH
  An existing output directory is refused (exit 2). If even one check (INCAR diff 0, KPOINTS, number of atoms and cell of the POSCAR, POTCAR, run_vasp.sh) disagrees, exit 3 (the output is left).
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import filecmp
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import cna as C  # noqa: E402
import dft_mu_check as MU  # noqa: E402

DESIGN = _gbrel.WORK + '/an internal design note that is not included'
SRC_RUN = Path(K.P(_gbrel.WORK + '/workspace/runs/earlier_internal_run'))
B00S0 = SRC_RUN / "inputs/b00s0"
POTCAR_SRC = Path(K.P(_gbrel.WORK + '/workspace/experiments/earlier_experiment/mu_crosscheck/results/vasp_reference_denser_k/POTCAR'))
POTCAR_AL_END_LINE = 2228
KSPACING = 0.25
A0 = MU.A0_AL


def ideal_fcc_120_frac():
    """Same construction as gb_symfit.build_single_crystal(flip_p=False): n=6a0(2,1,0)/..., t=a0(0,0,1), p=a0(-1,2,0). Returns: fractional coordinates (120,3) with column order = (n, t, p)."""
    a0 = A0
    Av = 6.0 * a0 * np.array([2.0, 1.0, 0.0])
    Bv = a0 * np.array([0.0, 0.0, 1.0])
    Cv = a0 * np.array([-1.0, 2.0, 0.0])
    M = np.array([Av, Bv, Cv])
    basis = a0 * np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    pts = []
    for i in range(-3, 15):
        for j in range(-2, 11):
            for k in range(-1, 2):
                for b in basis:
                    pts.append(np.array([i, j, k], dtype=float) * a0 + b)
    fr = np.array(pts) @ np.linalg.inv(M)
    fr_r = np.round(fr, 9)
    sel = np.all((fr_r >= 0.0) & (fr_r < 1.0), axis=1)
    frac = np.mod(fr[sel], 1.0)
    if frac.shape[0] != 120:
        raise RuntimeError(f"ideal oriented fcc has {frac.shape[0]} atoms != 120")
    return frac


def poscar_text(comment, cell, frac):
    L = [comment, " 1.0000000000000000"]
    for v in cell:
        L.append("  " + "  ".join(f"{x:.16f}" for x in v))
    L += [" Al", f" {len(frac)}", "Direct"]
    for f in frac:
        L.append("  " + "  ".join(f"{x:.16f}" for x in f))
    text = "\n".join(L) + "\n"
    assert all(ord(c) < 128 for c in text), "POSCAR is not ASCII"
    return text


def parse_poscar(path):
    """Independent re-reading (without ase): (cell(3,3), natoms, frac(N,3))."""
    lines = Path(path).read_text(encoding="ascii").splitlines()
    scale = float(lines[1].strip())
    cell = np.array([[float(x) for x in lines[2 + i].split()] for i in range(3)]) * scale
    assert lines[5].split() == ["Al"], lines[5]
    n = int(lines[6].split()[0])
    assert lines[7].strip().lower().startswith("d"), lines[7]
    frac = np.array([[float(x) for x in lines[8 + i].split()[:3]] for i in range(n)])
    assert len(lines) == 8 + n, f"unexpected trailing lines in {path}"
    return cell, n, frac


def kpoints_text_from_poscar(poscar_path):
    """Same rule and format as make_kpoints_content of make_vasp_inputs.py (computed by re-reading the written POSCAR)."""
    lines = Path(poscar_path).read_text(encoding="ascii").splitlines()
    scale = float(lines[1].strip())
    norms = [math.sqrt(sum((float(c) * scale) ** 2 for c in lines[2 + i].split())) for i in range(3)]
    i_normal = max(range(3), key=lambda j: norms[j])
    k = [math.ceil(2.0 * math.pi / norms[j] / KSPACING) for j in range(3)]
    k[i_normal] = 1
    text = "\n".join(["Gamma-centered mesh (in-plane KSPACING 0.25 A-1, GB-normal axis = 1)", "0", "Gamma", f"{k[0]} {k[1]} {k[2]}", "0 0 0"]) + "\n"
    assert all(ord(c) < 128 for c in text)
    return text, k


def make_run_vasp(job_name):
    src = (B00S0 / "run_vasp.sh").read_text(encoding="ascii").splitlines()
    assert src[0] == "#SCHED -N _b00s0", src[0]
    return "\n".join([f"#SCHED -N {job_name}"] + src[1:]) + "\n"


def write_potcar(dst):
    with open(POTCAR_SRC, "r", encoding="ascii", errors="strict") as f:
        lines = f.readlines()
    block = lines[:POTCAR_AL_END_LINE]
    assert len(block) == POTCAR_AL_END_LINE
    with open(dst, "w", encoding="ascii") as f:
        f.writelines(block)


def write_dir(d: Path, name, comment, cell, frac):
    d.mkdir(parents=True, exist_ok=False)
    (d / "POSCAR").write_text(poscar_text(comment, cell, frac), encoding="ascii")
    (d / "INCAR").write_bytes((B00S0 / "INCAR").read_bytes())
    ktext, k = kpoints_text_from_poscar(d / "POSCAR")
    (d / "KPOINTS").write_text(ktext, encoding="ascii")
    write_potcar(d / "POTCAR")
    (d / "run_vasp.sh").write_text(make_run_vasp(f"al_{name}"), encoding="ascii")
    return k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--check-json", required=True)
    ap.add_argument("--check-log", required=True)
    args = ap.parse_args()
    out = Path(K.P(args.out_dir))
    cj, cl = Path(K.P(args.check_json)), Path(K.P(args.check_log))
    for p in (out, cj, cl):
        if p.exists():
            print(f"FATAL: refusing to overwrite: {p}", file=sys.stderr)
            sys.exit(2)
    sel = json.load(open(K.P(args.selection)))
    ideal = ideal_fcc_120_frac()                      # column order (n, t, p)
    index = {"design": DESIGN, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "selection": args.selection, "structures": [], "mu": []}
    rec = []
    proposed_mu_done = False
    try:
        for pr in sel["pairs"]:
            for role in ("lo", "hi"):
                s = pr[role]
                at = MU.load_selected(s)
                cell = np.array(at.cell[:], dtype=float)
                frac = np.mod(at.get_scaled_positions(wrap=False), 1.0)
                axes = C.identify_axes(np.linalg.norm(cell, axis=1))
                name = f"gb_{s['arm']}_t{s['target_id']}_{role}"
                comment = f"GB {s['arm']} target{s['target_id']} {role} chunk{s['chunk']} cif{s['cif_index']} "
                k = write_dir(out / name, name, comment, cell, frac)
                area = float(np.linalg.norm(np.cross(cell[axes[1]], cell[axes[2]])))
                index["structures"].append({"name": name, "arm": s["arm"], "target_id": s["target_id"], "role": role, "chunk": s["chunk"], "cif_index": s["cif_index"],
                                            "natoms": len(at), "gamma_mace_J_m2": s["gamma_relaxed"], "area_A2": area, "kpoints": k,
                                            "axes": f"n={'abc'[axes[0]]},t={'abc'[axes[1]]},p={'abc'[axes[2]]}", "det_cell": float(np.linalg.det(cell)),
                                            "mu_name": "mu_proposed_common" if s["arm"] == "proposed" else f"mu_{s['arm']}_t{s['target_id']}_{role}"})
                rec.append({"name": name, "kind": "gb", "cell": cell, "natoms": len(at), "frac": frac, "selected": s, "axes": axes})
                if s["arm"] == "baseline" or not proposed_mu_done:
                    mu, _Ln = MU.mu_cell(cell, axes)
                    mu_name = "mu_proposed_common" if s["arm"] == "proposed" else f"mu_{s['arm']}_t{s['target_id']}_{role}"
                    n_i, t_i, p_i = axes
                    mfrac = np.zeros((len(ideal), 3))
                    mfrac[:, n_i], mfrac[:, t_i], mfrac[:, p_i] = ideal[:, 0], ideal[:, 1], ideal[:, 2]
                    mcomment = f"mu cell for {name}: 120-atom ideal oriented fcc on the same in-plane cell " if s["arm"] != "proposed" else \
                        "mu cell common for proposed (14 structures share the in-plane cell within 1e-3 A): 120-atom ideal oriented fcc "
                    k2 = write_dir(out / mu_name, mu_name, mcomment, mu, mfrac)
                    index["mu"].append({"name": mu_name, "arm": s["arm"], "natoms": len(ideal), "kpoints": k2, "built_from": name, "det_cell": float(np.linalg.det(mu)),
                                        "inplane_t_p_angle": MU.inplane_signature(mu, axes)})
                    rec.append({"name": mu_name, "kind": "mu", "cell": mu, "natoms": len(ideal), "frac": mfrac, "built_from": name, "axes": axes})
                    if s["arm"] == "proposed":
                        proposed_mu_done = True
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
    with open(out / "index.json", "x") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
        f.write("\n")

    # ---------------- checks ----------------
    checks = []

    def chk(item, expected, actual, ok):
        checks.append({"item": item, "expected": expected, "actual": actual, "pass": bool(ok)})

    names = [r["name"] for r in rec]
    n_gb, n_mu = sum(1 for r in rec if r["kind"] == "gb"), sum(1 for r in rec if r["kind"] == "mu")
    chk("number of directories (26 grain boundaries, 13 mu, 39 in total)", [26, 13, 39], [n_gb, n_mu, len(names)], (n_gb, n_mu, len(names)) == (26, 13, 39))
    bad_files = [n for n in names for f in ("POSCAR", "INCAR", "KPOINTS", "POTCAR", "run_vasp.sh") if not (out / n / f).is_file()]
    chk("POSCAR, INCAR, KPOINTS, POTCAR and run_vasp.sh in each directory", [], bad_files, not bad_files)
    inc_bad = [n for n in names if not filecmp.cmp(out / n / "INCAR", B00S0 / "INCAR", shallow=False)]
    chk("INCAR agrees byte for byte with b00s0/INCAR (diff 0 lines)", 0, len(inc_bad), not inc_bad)
    pot_bad = [n for n in names if not filecmp.cmp(out / n / "POTCAR", B00S0 / "POTCAR", shallow=False)]
    chk("POTCAR agrees byte for byte with b00s0/POTCAR (same file, same line range 1 to 2228)", 0, len(pot_bad), not pot_bad)
    chk("POTCAR has 2228 lines and TITEL = PAW_PBE Al 04Jan2001 (once)", [2228, 1],
        [len((out / names[0] / "POTCAR").read_text(encoding="ascii").splitlines()), sum(1 for l in (out / names[0] / "POTCAR").read_text(encoding="ascii").splitlines() if "TITEL" in l and "PAW_PBE Al 04Jan2001" in l)],
        len((out / names[0] / "POTCAR").read_text(encoding="ascii").splitlines()) == 2228)
    run_bad = []
    ref = (B00S0 / "run_vasp.sh").read_text(encoding="ascii").splitlines()
    for n in names:
        got = (out / n / "run_vasp.sh").read_text(encoding="ascii").splitlines()
        if got[0] != f"#SCHED -N al_{n}" or got[1:] != ref[1:]:
            run_bad.append(n)
    chk("run_vasp.sh is a transcription of b00s0 (only the line of the job-name directive differs)", 0, len(run_bad), not run_bad)
    # POSCAR: the number of atoms, the cell and the coordinates agree with the selected structure (independent re-reading)
    pos_bad = []
    for r in rec:
        cell, n, frac = parse_poscar(out / r["name"] / "POSCAR")
        ok = n == r["natoms"] and np.abs(cell - r["cell"]).max() < 1e-12 and np.abs(np.mod(frac - r["frac"] + 0.5, 1.0) - 0.5).max() < 1e-12
        if r["kind"] == "gb":
            s = r["selected"]
            at = MU.load_selected(s)   # re-read from extxyz (separately from the writing)
            ok = ok and n == len(at) == s["natoms"] and np.abs(cell - np.array(at.cell[:], dtype=float)).max() < 1e-12 \
                and np.abs(np.mod(frac - at.get_scaled_positions(wrap=False) + 0.5, 1.0) - 0.5).max() < 1e-12
        if not ok:
            pos_bad.append(r["name"])
    chk("number of atoms, cell and coordinates of the POSCAR agree with the structure of the selection (re-read from extxyz) / the mu construction (0 mismatches)", 0, len(pos_bad), not pos_bad)
    natoms_gb = sorted({r["natoms"] for r in rec if r["kind"] == "gb"})
    chk("range of the number of atoms of the grain-boundary structures (112 to 120)", "112..120", natoms_gb, all(112 <= x <= 120 for x in natoms_gb))
    chk("number of atoms of the mu cell = 120", 120, sorted({r["natoms"] for r in rec if r["kind"] == "mu"}), all(r["natoms"] == 120 for r in rec if r["kind"] == "mu"))
    # KPOINTS: identical for a grain boundary and its mu, `1 7 3` for all proposed, format
    kp = {r["name"]: (out / r["name"] / "KPOINTS").read_text(encoding="ascii").splitlines()[3] for r in rec}
    kp_bad = []
    for st in index["structures"]:
        if kp[st["name"]] != kp[st["mu_name"]]:
            kp_bad.append((st["name"], kp[st["name"]], kp[st["mu_name"]]))
    chk("KPOINTS: identical for the grain-boundary structure and its mu cell", 0, kp_bad, not kp_bad)
    proposed_k = sorted({kp[st["name"]] for st in index["structures"] if st["arm"] == "proposed"} | {kp["mu_proposed_common"]})
    chk("KPOINTS: the 14 proposed structures + the common mu are all `1 7 3`", ["1 7 3"], proposed_k, proposed_k == ["1 7 3"])
    van_k = sorted({kp[st["name"]] for st in index["structures"] if st["arm"] == "baseline"})
    chk("KPOINTS: the 12 baseline structures (the order of the axes is n=a, t=c, p=b, so `1 3 7` by the rule)", ["1 3 7"], van_k, van_k == ["1 3 7"])
    kfmt_bad = [r["name"] for r in rec if (out / r["name"] / "KPOINTS").read_text(encoding="ascii").splitlines()[:3] != ["Gamma-centered mesh (in-plane KSPACING 0.25 A-1, GB-normal axis = 1)", "0", "Gamma"]]
    chk("format of KPOINTS (same as make_vasp_inputs.py. Gamma-centered)", 0, len(kfmt_bad), not kfmt_bad)
    # right-handedness of the lattice (VASP dislikes left-handed bases)
    left = [r["name"] for r in rec if np.linalg.det(r["cell"]) <= 0]
    chk("lattices of all POSCAR are right-handed (det > 0)", 0, left, not left)
    # in-plane lattice of the mu cell = t and p of the grain-boundary structure (with the written POSCAR, the same quantity as check D2)
    d2 = 0.0
    for st in index["structures"]:
        cg, _n, _f = parse_poscar(out / st["name"] / "POSCAR")
        cm, _n2, _f2 = parse_poscar(out / st["mu_name"] / "POSCAR")
        axes = C.identify_axes(np.linalg.norm(cg, axis=1))
        if st["arm"] == "baseline" or st["name"] == next(x["name"] for x in index["structures"] if x["arm"] == "proposed"):
            d2 = max(d2, float(np.abs(cm[axes[1]] - cg[axes[1]]).max()), float(np.abs(cm[axes[2]] - cg[axes[2]]).max()))
    chk("t and p vectors of the mu POSCAR agree with the grain-boundary POSCAR (maximum difference < 1e-6 A. For proposed the original structure of the common mu)", "<1e-6", d2, d2 < 1e-6)
    # the mu cell is a perfect crystal (all atoms fcc by a-CNA)
    notfcc = []
    for r in rec:
        if r["kind"] == "mu":
            cell, n, frac = parse_poscar(out / r["name"] / "POSCAR")
            lab, _rc = C.acna_labels(frac, cell)
            if not np.all(lab == C.FCC):
                notfcc.append((r["name"], float((lab == C.FCC).mean())))
    chk("the mu cell has all atoms fcc by a-CNA (perfect crystal)", [], notfcc, not notfcc)
    # in-plane lattice of the common mu of proposed = in-plane lattices of all 14 structures (1e-3 A)
    proposed_sig = [st for st in index["structures"] if st["arm"] == "proposed"]
    cm, _n, _f = parse_poscar(out / "mu_proposed_common" / "POSCAR")
    ax_m = C.identify_axes(np.linalg.norm(cm, axis=1))
    spread = 0.0
    for st in proposed_sig:
        cg, _n, _f = parse_poscar(out / st["name"] / "POSCAR")
        ax = C.identify_axes(np.linalg.norm(cg, axis=1))
        spread = max(spread, abs(np.linalg.norm(cg[ax[1]]) - np.linalg.norm(cm[ax_m[1]])), abs(np.linalg.norm(cg[ax[2]]) - np.linalg.norm(cm[ax_m[2]])))
    chk("proposed: lengths of the in-plane lattice of the common mu = lengths of the in-plane lattices of the 14 structures (maximum difference < 1e-3 A)", "<1e-3", spread, spread < 1e-3)
    all_pass = all(c["pass"] for c in checks)
    doc = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(), "out_dir": str(out), "n_dirs": len(names), "all_pass": all_pass, "checks": checks,
           "kpoints_by_dir": kp}
    cj.parent.mkdir(parents=True, exist_ok=True)
    with open(cj, "x") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1, default=str)
        f.write("\n")
    with open(cl, "x") as f:
        f.write(f"# check of the 39 inputs of the DFT stage {doc['date']} host={doc['host']} script={Path(__file__).name}\n")
        for c in checks:
            f.write(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['item']}: expected {c['expected']} / measured {c['actual']}\n")
        f.write(f"all_pass = {all_pass}\n")
    print(f"[dft inputs] dirs={len(names)} all_pass={all_pass}", file=sys.stderr)
    sys.exit(0 if all_pass else 3)


if __name__ == "__main__":
    main()
