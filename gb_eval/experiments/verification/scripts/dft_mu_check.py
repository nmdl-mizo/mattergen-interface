#!/usr/bin/env python3
"""dft_mu_check.py -- computation of the in-plane lattice and KPOINTS of the mu cell and checks D1 and D2. numpy / ase. VASP input files are not written (next step).

Plan: see the experiment description.
  mu (reference energy) cell: a perfect FCC-oriented cell with N=120 with the same in-plane cell as the grain-boundary structure (the lattice vectors of the tilt axis t and the in-plane p themselves = lengths and angle) and the same k points.
    The period along n is L_n = V / |t x p| from the volume of 120 atoms V = 120 a0^3/4 (a0 = 4.0365507720520695 A, the MACE-relaxed value).
    The n vector is L_n along the direction of t x p (the same sign as the n vector of the grain-boundary structure). The order of the lattice vectors (which of a, b, c is n, t, p) is the same as for the grain-boundary structure.
  KPOINTS (rule of the DFT setup): Gamma-centered, 1 along the longest vector (normal), ceil(2*pi/|a_i|/0.25) for the others.
  proposed: confirm that the in-plane cells (|t|, |p|, angle) of the 14 selected structures are of one kind within 1e-3 A and use one common mu. If not of one kind, stop and report (exit 3).
  baseline: one mu cell for each of the 12 selected structures, one to one.
  Check D1: KPOINTS are identical for each grain-boundary structure and its mu cell. The 14 proposed structures + the common mu are all identical (expected: 1 along n, 7 along t, 3 along p). For baseline it is computed from the rule for each structure.
  Check D2: the maximum difference between the t and p vectors of each mu cell and those of the corresponding grain-boundary structure < 1e-6 A.
Usage: python dft_mu_check.py --selection selection.json --out-json PATH --log-d1 PATH --log-d2 PATH
  (the python of the evaluation environment. Reads relaxed.extxyz of the 26 selected structures. A few seconds)
  Exit codes: 0 = D1 and D2 hold / 3 = do not hold, or the in-plane cell of proposed is not of one kind / 2 = failure of the input or output
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
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
import cna as C  # noqa: E402     (identify_axes. The same axis identification rule)

DESIGN = _gbrel.WORK + '/an internal design note that is not included'
A0_AL = 4.0365507720520695      # MACE-relaxed a0
N_MU = 120
KSPACING = 0.25                 # A^-1 (rule of the DFT setup)
PROPOSED_INPLANE_TOL_A = 1e-3       # tolerance for the agreement of the in-plane cells of the 14 proposed structures
D2_TOL_A = 1e-6                 # check-D2


def kpoints_grid(cell):
    """Gamma-centered mesh. 1 along the longest vector, ceil(2*pi/|a_i|/0.25) for the others (the same rule as make_kpoints_content of make_vasp_inputs.py)."""
    norms = [float(np.linalg.norm(cell[i])) for i in range(3)]
    i_normal = max(range(3), key=lambda j: norms[j])
    k = [math.ceil(2.0 * math.pi / norms[j] / KSPACING) for j in range(3)]
    k[i_normal] = 1
    return k, norms, i_normal


def mu_cell(cell, axes, n_atoms=N_MU, a0=A0_AL):
    """From the lattice cell (row vectors 3x3) of the grain-boundary structure and the axes (n, t, p), returns the lattice of a perfect FCC-oriented cell of n_atoms atoms with the same in-plane lattice (the t and p vectors as they are)."""
    cell = np.asarray(cell, dtype=float)
    n, t, p = axes
    tp = np.cross(cell[t], cell[p])
    area = float(np.linalg.norm(tp))
    vol = n_atoms * a0 ** 3 / 4.0
    Ln = vol / area
    nvec = tp / area * Ln
    if float(nvec @ cell[n]) < 0:
        nvec = -nvec
    out = np.zeros((3, 3))
    out[n], out[t], out[p] = nvec, cell[t], cell[p]
    return out, Ln


def inplane_signature(cell, axes):
    """(|t|, |p|, angle between t and p [deg])."""
    _n, t, p = axes
    vt, vp = cell[t], cell[p]
    ang = math.degrees(math.acos(float(vt @ vp) / (np.linalg.norm(vt) * np.linalg.norm(vp))))
    return [float(np.linalg.norm(vt)), float(np.linalg.norm(vp)), ang]


def load_selected(s):
    from ase.io import read as ase_read
    frames = ase_read(K.P(s["relaxed_extxyz"]), index=":")
    at = frames[int(s["frame_index"])]
    if int(at.info.get("cif_index", -1)) != int(s["cif_index"]):
        raise K.InputError(f"cif_index mismatch: frame {s['frame_index']} has {at.info.get('cif_index')} != {s['cif_index']}")
    if len(at) != int(s["natoms"]):
        raise K.InputError(f"natoms {len(at)} != selection {s['natoms']}")
    return at


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--log-d1", required=True)
    ap.add_argument("--log-d2", required=True)
    args = ap.parse_args()
    outs = [Path(args.out_json), Path(args.log_d1), Path(args.log_d2)]
    for p in outs:
        if p.exists():
            print(f"FATAL: refusing to overwrite: {p}", file=sys.stderr)
            sys.exit(2)
    try:
        sel = json.load(open(K.P(args.selection)))
        recs = []
        for pr in sel["pairs"]:
            for role in ("lo", "hi"):
                s = pr[role]
                at = load_selected(s)
                cell = np.array(at.cell[:], dtype=float)
                axes = C.identify_axes(np.linalg.norm(cell, axis=1))
                mu, Ln = mu_cell(cell, axes)
                k_gb, norms_gb, in_gb = kpoints_grid(cell)
                k_mu, norms_mu, in_mu = kpoints_grid(mu)
                n, t, p = axes
                d2 = max(float(np.abs(mu[t] - cell[t]).max()), float(np.abs(mu[p] - cell[p]).max()))
                recs.append({"arm": s["arm"], "target_id": s["target_id"], "role": role, "chunk": s["chunk"], "cif_index": s["cif_index"], "natoms": len(at),
                             "axes": f"n={'abc'[n]},t={'abc'[t]},p={'abc'[p]}", "inplane_t_p_angle": inplane_signature(cell, axes),
                             "kpoints_gb": k_gb, "kpoints_mu": k_mu, "kpoints_same": k_gb == k_mu, "d2_max_abs_diff_A": d2,
                             "mu_Ln_A": Ln, "mu_cell_lengths_A": [float(x) for x in norms_mu], "gb_cell_lengths_A": [float(x) for x in norms_gb],
                             "mu_normal_axis_is_n": in_mu == n, "gb_normal_axis_is_n": in_gb == n,
                             "mu_cell_matrix": mu.tolist()})
    except (K.InputError, FileNotFoundError, KeyError, C.AxisMismatch) as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)

    proposed = [r for r in recs if r["arm"] == "proposed"]
    van = [r for r in recs if r["arm"] == "baseline"]
    sig = np.array([r["inplane_t_p_angle"] for r in proposed])
    spread = (sig.max(axis=0) - sig.min(axis=0)).tolist()
    proposed_one_kind = bool(len(proposed) == 14 and max(spread[0], spread[1]) <= PROPOSED_INPLANE_TOL_A)
    d1_proposed_expected = [1, 7, 3]
    d1 = {"ac": "check-D1",
          "proposed": {"n_structures": len(proposed), "inplane_spread_lengths_A_angle_deg": spread, "one_kind_within_1e-3A": proposed_one_kind,
                   "inplane_t_p_angle_first": proposed[0]["inplane_t_p_angle"] if proposed else None,
                   "kpoints_set": sorted({tuple(r["kpoints_gb"]) for r in proposed}), "kpoints_mu_set": sorted({tuple(r["kpoints_mu"]) for r in proposed}),
                   "all_same_gb_mu": all(r["kpoints_same"] for r in proposed),
                   "expected_kpoints": d1_proposed_expected,
                   "all_equal_expected": all(r["kpoints_gb"] == d1_proposed_expected and r["kpoints_mu"] == d1_proposed_expected for r in proposed)},
          "baseline": {"n_structures": len(van), "n_distinct_inplane_cells_1e-3A": len({(round(r["inplane_t_p_angle"][0] / 1e-3), round(r["inplane_t_p_angle"][1] / 1e-3)) for r in van}),
                      "all_same_gb_mu": all(r["kpoints_same"] for r in van),
                      "rows": [{"target_id": r["target_id"], "role": r["role"], "axes": r["axes"], "kpoints_gb": r["kpoints_gb"], "kpoints_mu": r["kpoints_mu"],
                                "t_p_angle": r["inplane_t_p_angle"]} for r in van]}}
    d1["pass"] = bool(proposed_one_kind and d1["proposed"]["all_same_gb_mu"] and d1["proposed"]["all_equal_expected"] and d1["baseline"]["all_same_gb_mu"])
    d2_max = max(r["d2_max_abs_diff_A"] for r in recs)
    d2 = {"ac": "check-D2", "max_abs_diff_A": d2_max, "tol_A": D2_TOL_A, "pass": bool(d2_max < D2_TOL_A)}
    n_mu_files = (1 if proposed else 0) + len(van)
    doc = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(), "design": DESIGN, "a0_Al_A": A0_AL, "n_mu_atoms": N_MU, "kspacing": KSPACING,
           "n_structures": len(recs), "n_mu_cells": n_mu_files, "n_dft_runs_expected": len(recs) + n_mu_files, "check_D1": d1, "check_D2": d2, "structures": recs}
    outs[0].parent.mkdir(parents=True, exist_ok=True)
    for p in outs[1:]:
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(outs[0], "x") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    for p, d in ((outs[1], d1), (outs[2], d2)):
        with open(p, "x") as f:
            f.write(f"# {d['ac']} raw output {doc['date']} host={doc['host']} script={Path(__file__).name} selection={args.selection}\n")
            f.write(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
    print(f"[mu] structures={len(recs)} mu_cells={n_mu_files} (expected number of DFT runs {len(recs) + n_mu_files}) D1 pass={d1['pass']} D2 pass={d2['pass']} (max diff {d2_max:.2e} A)", file=sys.stderr)
    if not proposed_one_kind:
        print(f"STOP: the in-plane cells of the 14 proposed structures are not of one kind (spread={spread}). Stopping and reporting", file=sys.stderr)
        sys.exit(3)
    sys.exit(0 if (d1["pass"] and d2["pass"]) else 3)


if __name__ == "__main__":
    main()
