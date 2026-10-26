#!/usr/bin/env python3
"""Data rebuild, stages D1-1 to 3: gamma_all.csv and stage_d1_summary.json.

  python stage_d1_build_csv.py [--cnids a,b] [--inputs-dir DIR] [--results-dir DIR]

Input: results/relax_inputs/{cnid}.json (output of stage B) and energy.json and POSCAR_relaxed of each structure directory (output of stage C).
gamma = (energy_eV - natoms*E_BULK_PER_ATOM) / (2*area) * EV_PER_A2_TO_J_PER_M2 (constants imported from config, area from dx_lib.in_plane_area_bc).
Rows with converged=False are kept in the CSV (converged column False), counted and written to the summary. They are not used in the following stages.
Non-zero exit (2) if even one input has no energy.json. If the fraction of converged is below 98%, check 3 FAIL (non-zero exit 3).
P_all is counted from the rows with converged=True (group = same dx and same natoms). The default of r2 is no window (all pairs). P_W is counted only when --window-ev W is given explicitly (P_W=null without a window).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from ase.io import read as ase_read

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "lib"))  # structure_builder/lib
import builder_common as C
import dx_lib
import stage_d_common as D

import config as CFG  # noqa: E402  (structure_builder/lib/config.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnids", default="")
    ap.add_argument("--inputs-dir", default=str(C.RESULTS / "relax_inputs"))
    ap.add_argument("--results-dir", default=None)
    D.add_window_arg(ap)
    args = ap.parse_args()
    out_dir = D.results_dir(args.results_dir)
    W = D.window_ev(args.window_ev)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [r["cnid"] for r in C.read_template_map()]
    sel = C.select_cnids(names, args.cnids)

    print(f"E_BULK_PER_ATOM={CFG.E_BULK_PER_ATOM!r} EV_PER_A2_TO_J_PER_M2={CFG.EV_PER_A2_TO_J_PER_M2!r} "
          f"(from {CFG.__file__})", flush=True)
    rows, missing, unconv = [], [], []
    n_inputs = 0
    for cnid in sel:
        with open(Path(args.inputs_dir) / f"{cnid}.json") as f:
            inputs = json.load(f)
        for key, e in inputs.items():
            n_inputs += 1
            d = Path(e["poscar_path"]).parent
            ej, pr = d / "energy.json", d / "POSCAR_relaxed"
            if not (ej.exists() and pr.exists()):
                missing.append(key)
                continue
            with open(ej) as f:
                info = json.load(f)
            atoms = ase_read(str(pr), format="vasp")
            n = info["natoms"]
            if n != len(atoms) or n != e["natoms"]:
                raise ValueError(f"{key}: natoms mismatch energy.json={n} POSCAR_relaxed={len(atoms)} input={e['natoms']}")
            area = dx_lib.in_plane_area_bc(atoms)
            E = info["energy_eV"]
            gamma = (E - n * CFG.E_BULK_PER_ATOM) / (2.0 * area) * CFG.EV_PER_A2_TO_J_PER_M2
            conv = bool(info["converged"])
            if not conv:
                unconv.append({"key": key, "n_steps": info.get("n_steps"), "fmax_final": info.get("fmax_final")})
            rows.append([cnid, e["k"], e["seed"], str(d), "True" if conv else "False", n, repr(float(E)),
                         repr(float(area)), repr(float(gamma)), repr(float(e["dx_A"]))])

    gp = out_dir / D.GAMMA_ALL
    tmp = gp.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(D.COLS)
        w.writerows(rows)
    os.replace(tmp, gp)

    allrows = D.read_gamma(gp)
    groups = D.groups_of(allrows)
    sizes = {k: len(v) for k, v in groups.items()}
    p_all = sum(n * (n - 1) // 2 for n in sizes.values())
    p_w_by = {k: len(D.windowed_pairs(v, W)) for k, v in groups.items()} if W is not None else None
    p_w = sum(p_w_by.values()) if W is not None else None
    n_conv = len(rows) - len(unconv)
    frac = n_conv / n_inputs if n_inputs else 0.0
    summary = {
        "cnids": sel, "n_inputs": n_inputs, "n_rows": len(rows), "n_missing_energy_json": len(missing),
        "missing_sample": missing[:10],
        "n_converged": n_conv, "n_unconverged": len(unconv), "converged_fraction": frac,
        "C3": {"threshold": D.CONV_MIN, "verdict": "PASS" if (frac >= D.CONV_MIN and not missing) else "FAIL"},
        "unconverged": unconv,
        "constants": {"E_BULK_PER_ATOM": CFG.E_BULK_PER_ATOM, "EV_PER_A2_TO_J_PER_M2": CFG.EV_PER_A2_TO_J_PER_M2,
                      "config_file": CFG.__file__},
        "window_eV": W, "n_groups": len(sizes), "group_sizes": sizes,
        "P_all": p_all, "P_W": p_w, "P_W_by_group": p_w_by, "gamma_csv": str(gp),
        "gamma_csv_sha256": D.sha256_file(gp),
    }
    with open(out_dir / "stage_d1_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"rows={len(rows)}/{n_inputs} missing={len(missing)} converged={n_conv} ({100 * frac:.2f}%) "
          f"groups={len(sizes)} window_eV={W} P_all={p_all} P_W={p_w}  C3={summary['C3']['verdict']}", flush=True)
    if missing:
        print(f"FAIL: {len(missing)} input(s) without energy.json/POSCAR_relaxed, e.g. {missing[:3]}")
        raise SystemExit(2)
    if frac < D.CONV_MIN:
        print("C3 FAIL (report to parent)")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
