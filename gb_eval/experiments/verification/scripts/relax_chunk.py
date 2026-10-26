#!/usr/bin/env python3
"""relax_chunk.py -- relaxation stage: relaxes the generated structures (1 chunk = 25 structures) with MACE and records gamma after relaxation, convergence and displacement.

Plan: see the experiment description.
Source (the most recent working implementation): ${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts/stage_c_relax_chunk.py:38-67 relax_one
  (FixAtoms, FIRE.run(fmax, steps)), constants ${GBREL_CODE}/gb_eval/scripts/config/config.py (MACE_MODEL, MACE_HEAD, FMAX, MAX_STEPS, E_BULK_PER_ATOM).
  MACE is called with the same V2.build_calc() as gamma before relaxation (mace_mp, cuda, float64, head=omat_pbe).

[Normal mode] python relax_chunk.py --layer <layer name of layers.json> --arm proposed|baseline --w W --chunk C --out-dir <dir>
   [--fix-indices proposed|none|<file>]   Default: proposed arm = proposed (indices 0..75), baseline = none. <file> holds, one structure per row (25 rows, in cif_index order), indices separated by ';'.
   Input: the 25 rows of (arm, w, chunk) of results/inputs_manifest_<layer>.csv. Output: <out-dir>/<model>_w<W>/chunk<cc>/{relaxed.extxyz, relax_rows.csv, RELAX_DONE}.
   Idempotent: if the output dir exists and RELAX_DONE and a relax_rows.csv with 25 rows are present, nothing is written and it exits 0 (skip). If the dir exists but RELAX_DONE does not, exit 2 (not overwritten).
   The proposed arm always measures the maximum displacement of indices 0..75 and exits with 4 unless it is 0.0 (the constraint is not effective). A row whose relative difference of the area (re-read vs manifest) exceeds 1e-12 gets status=error.
[Control mode (R3)] --control-csv verification/r3_inputs.csv --out-dir <dir> [--control-json verification/positive_control.json]
   Relaxes the POSCAR_relaxed of train with the atoms fixed by frozen_indices of the csv (separated by ';', 76 bulk atoms) and records the difference from the training label.
--dry-run: does not use MACE (and does not take a GPU); only checks the loading, the fixed indices and the area and writes nothing.
--calc emt: [for tests only] checks only the route of the runner (FIRE, marker, idempotence) on CPU with the EMT of ASE. Refused unless --out-dir contains '/tmp/'. gamma becomes a meaningless value.
Execution environment: the evaluation environment (compute node, GPU). Not executed on the login node. An MLIP always runs on a GPU (MACE on CPU is refused).
Exit codes: 0 = success/skip, 2 = abnormal state of the input or output dir, 4 = the fixed atoms of proposed moved
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

ROW_FIELDS = ["layer", "arm", "w", "chunk", "cif_index", "target_id", "gamma_target", "is_inner", "status", "excluded_pre",
              "converged", "n_steps", "fmax_final", "energy_eV", "natoms", "area_A2", "gamma_pre", "gamma_relaxed",
              "min_dist_relaxed_A", "max_disp_free_A", "max_disp_fixed_A", "n_fixed", "area_rel_diff", "elapsed_s", "calc", "error", "frame_index"]
CONTROL_FIELDS = ["row_index", "poscar_path", "n_fixed", "status", "converged", "n_steps", "fmax_final", "energy_eV", "natoms", "area_A2",
                  "gamma_Jm2_train", "gamma_relaxed", "delta_gamma_Jm2", "delta_E_eV", "max_disp_free_A", "max_disp_fixed_A", "elapsed_s", "calc", "error"]
AREA_REL_TOL = 1e-12   # check R0 (imposed also on proposed)
STATUSES = ["ok_converged", "ok_unconverged", "excluded_pre", "nonfinite", "error"]


class FixedAtomMoved(RuntimeError):
    pass


def parse_fix_indices(spec, arm, rows_n):
    """Interpretation of --fix-indices -> list[list[int]] (per row). 'proposed' = 0..75, 'none' = no fixing, anything else is a file (number of rows = rows_n)."""
    if spec is None:
        spec = "proposed" if arm == "proposed" else "none"
    if spec == "proposed":
        return [list(range(K.N_FIXED_PROPOSED)) for _ in range(rows_n)], "proposed"
    if spec == "none":
        return [[] for _ in range(rows_n)], "none"
    p = Path(K.P(spec))
    if not p.is_file():
        raise K.InputError(f"--fix-indices file not found: {spec}")
    lines = [ln.strip() for ln in p.read_text().splitlines()]
    lines = [ln for ln in lines if ln != ""]
    if len(lines) != rows_n:
        raise K.InputError(f"--fix-indices file has {len(lines)} lines != {rows_n} structures")
    return [[int(x) for x in ln.split(";") if x != ""] for ln in lines], f"file:{spec}"


def make_calc(kind, dry_run, min_free_gib):
    """Makes the calculator. mace: needs a GPU (vram_precheck + torch.cuda.is_available). emt: for tests only. dry-run: None."""
    if dry_run:
        return None
    if kind == "emt":
        from ase.calculators.emt import EMT
        return EMT()
    import torch
    if not torch.cuda.is_available():
        raise K.InputError("GPU not available: an MLIP is not run on CPU")
    sys.path.insert(0, K.P(K.RUNNER_GUARDS_DIR))
    import runner_guards as RG
    free = RG.vram_precheck(min_free_gib=min_free_gib)
    print(f"[GUARD] VRAM free={free:.1f} GiB (>= {min_free_gib} GiB OK)", flush=True)
    calc, model_path = K.load_H().V2.build_calc()
    print(f"MACE model: {model_path}", flush=True)
    return calc


def relax_one(atoms, fixed_idx, calc, cfg, *, proposed_check_idx=None):
    """Relaxes atoms with FIRE (except the fixed indices). converged is not the return value of FIRE but is decided after the end by max(norm(forces[free])) <= fmax.
    Returns: dict (result) and the relaxed Atoms (a copy without constraints and calculator). The displacement of the fixed atoms is measured with proposed_check_idx (if any)."""
    import numpy as np
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.geometry import find_mic
    from ase.optimize import FIRE

    n = len(atoms)
    pos_pre = atoms.positions.copy()
    cell = atoms.cell[:]
    free_mask = np.ones(n, dtype=bool)
    if fixed_idx:
        free_mask[list(fixed_idx)] = False
        atoms.set_constraint(FixAtoms(indices=list(fixed_idx)))
    atoms.calc = calc
    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=cfg["fmax"], steps=cfg["steps"])
    n_steps = int(opt.get_number_of_steps())
    forces = atoms.get_forces(apply_constraint=False)
    energy = float(atoms.get_potential_energy())
    pos_post = atoms.positions.copy()
    fin = bool(np.isfinite(pos_post).all() and np.isfinite(forces).all() and math.isfinite(energy))
    res = {"n_steps": n_steps, "energy_eV": energy, "natoms": n, "finite": fin, "n_fixed": len(fixed_idx)}
    out = Atoms(numbers=atoms.numbers.copy(), positions=pos_post, cell=atoms.cell.copy(), pbc=atoms.pbc.copy())
    if not fin:
        return res, out
    fmax_final = float(np.linalg.norm(forces[free_mask], axis=1).max()) if free_mask.any() else 0.0
    res["fmax_final"] = fmax_final
    res["converged"] = bool(fmax_final <= cfg["fmax"])
    area = K.area_bc(atoms)
    res["area_A2"] = area
    res["gamma_relaxed"] = (energy - n * cfg["e_bulk"]) / (2.0 * area) * cfg["factor"]
    d, _ = find_mic(pos_post - pos_pre, cell, pbc=True)
    dn = np.linalg.norm(d, axis=1)
    res["max_disp_free_A"] = float(dn[free_mask].max()) if free_mask.any() else 0.0
    chk = list(proposed_check_idx) if proposed_check_idx is not None else list(fixed_idx)
    res["max_disp_fixed_A"] = float(dn[chk].max()) if chk else ""
    res["min_dist_relaxed_A"] = K.load_H().min_dist_A(atoms)
    return res, out


def get_cfg():
    CFG = K.load_CFG()
    cfg = {"fmax": float(CFG.FMAX), "steps": int(CFG.MAX_STEPS), "e_bulk": float(CFG.E_BULK_PER_ATOM),
           "factor": float(CFG.EV_PER_A2_TO_J_PER_M2)}
    if cfg["fmax"] != K.FMAX or cfg["steps"] != K.MAX_STEPS or cfg["factor"] != K.GAMMA_FACTOR:
        raise K.InputError(f"config.py ({cfg}) differs from card D2/D4 constants (fmax {K.FMAX}, steps {K.MAX_STEPS}, factor {K.GAMMA_FACTOR})")
    return cfg


def fmt(x):
    return repr(float(x)) if isinstance(x, float) else x


# ---------------------------------------------------------------------------
# normal mode
# ---------------------------------------------------------------------------
def run_regular(args):
    from ase.io import read as ase_read, write as ase_write
    layer, arm, w, chunk = args.layer, args.arm, float(args.w), int(args.chunk)
    K.check_layer_arm(layer, arm)
    K.assert_out_not_in_readonly(layer, args.out_dir)
    manifest = args.manifest or K.manifest_path_for(layer)
    rows = [r for r in K.read_csv_rows(manifest)
            if r["layer"] == layer and r["arm"] == arm and abs(float(r["w"]) - w) < 1e-9 and int(r["chunk"]) == chunk]
    rows.sort(key=lambda r: int(r["cif_index"]))
    if [int(r["cif_index"]) for r in rows] != list(range(K.PER_CHUNK)):
        raise K.InputError(f"manifest {manifest}: ({layer},{arm},w{w},chunk{chunk}) rows are not cif_index 0..24 (got {len(rows)})")
    out_dir = Path(K.P(args.out_dir)) / K.armw(arm, w) / K.chunk_name(chunk)
    marker = out_dir / "RELAX_DONE"
    if out_dir.exists() and not args.dry_run:
        if marker.is_file() and len(K.read_csv_rows(out_dir / "relax_rows.csv")) == K.PER_CHUNK:
            print(f"SKIP: {out_dir} already complete (RELAX_DONE + {K.PER_CHUNK} rows). nothing written.", flush=True)
            return 0
        print(f"FATAL: {out_dir} exists but is incomplete (no RELAX_DONE or row count != {K.PER_CHUNK}); not overwriting. The author decides.", file=sys.stderr)
        return 2
    if args.calc == "emt" and "/tmp/" not in str(out_dir):
        raise K.InputError("--calc emt is TEST ONLY and requires --out-dir under a '/tmp/' directory")
    cfg = get_cfg()
    fixed_lists, fix_label = parse_fix_indices(args.fix_indices, arm, len(rows))
    proposed_idx = list(range(K.N_FIXED_PROPOSED)) if arm == "proposed" else None
    atoms_list = K.load_chunk_atoms(layer, arm, w, chunk)
    n_todo = sum(1 for r in rows if not r["excluded_pre"])
    print(f"layer={layer} arm={arm} w={w} chunk={chunk} fix={fix_label} rows={len(rows)} to_relax={n_todo} calc={args.calc} dry_run={args.dry_run}", flush=True)
    calc = make_calc(args.calc, args.dry_run, args.min_free_gib) if n_todo > 0 else None

    out_rows, frames = [], []
    t_all = time.time()
    for i, r in enumerate(rows):
        base = {"layer": layer, "arm": arm, "w": f"{w:.1f}", "chunk": chunk, "cif_index": i, "target_id": r["target_id"],
                "gamma_target": r["gamma_target"], "is_inner": r["is_inner"], "excluded_pre": r["excluded_pre"],
                "gamma_pre": r["gamma_unrelaxed"], "natoms": r["natoms"], "area_A2": r["area_A2"], "calc": args.calc,
                "n_fixed": len(fixed_lists[i]), "frame_index": ""}
        if r["excluded_pre"]:
            out_rows.append({**base, "status": "excluded_pre"})
            continue
        t0 = time.time()
        try:
            atoms = atoms_list[i][0]
            if atoms is None:
                raise RuntimeError("structure unavailable (angle_fail in the structure but not excluded in the manifest)")
            if len(atoms) != int(r["natoms"]):
                raise RuntimeError(f"natoms {len(atoms)} != manifest {r['natoms']}")
            area0 = K.area_bc(atoms)
            rel = abs(area0 - float(r["area_A2"])) / float(r["area_A2"])
            base["area_rel_diff"] = repr(rel)
            if rel > AREA_REL_TOL:
                raise RuntimeError(f"area_mismatch rel={rel:.3e} > {AREA_REL_TOL} (not handled with the same axes and the same area as in the evaluation before relaxation)")
            if fixed_lists[i] and (max(fixed_lists[i]) >= len(atoms)):
                raise RuntimeError(f"fixed index {max(fixed_lists[i])} >= natoms {len(atoms)}")
            if args.dry_run:
                out_rows.append({**base, "status": "ok_unconverged", "error": "DRY_RUN", "elapsed_s": f"{time.time() - t0:.3f}"})
                continue
            res, relaxed = relax_one(atoms, fixed_lists[i], calc, cfg, proposed_check_idx=proposed_idx)
            dt = time.time() - t0
            if proposed_idx is not None and res.get("max_disp_fixed_A") not in ("", None) and res["max_disp_fixed_A"] != 0.0:
                raise FixedAtomMoved(f"proposed fixed atoms (0..75) moved: max displacement {res['max_disp_fixed_A']!r} Å at cif_index {i}")
            if not res["finite"]:
                out_rows.append({**base, "status": "nonfinite", "n_steps": res["n_steps"], "energy_eV": fmt(res["energy_eV"]), "elapsed_s": f"{dt:.2f}"})
            else:
                relaxed.info.update({"layer": layer, "arm": arm, "w": w, "chunk": chunk, "cif_index": i, "converged": res["converged"],
                                     "gamma_relaxed": res["gamma_relaxed"], "energy_eV": res["energy_eV"]})
                frames.append(relaxed)
                out_rows.append({**base, "status": "ok_converged" if res["converged"] else "ok_unconverged", "converged": res["converged"],
                                 "n_steps": res["n_steps"], "fmax_final": fmt(res["fmax_final"]), "energy_eV": fmt(res["energy_eV"]),
                                 "gamma_relaxed": fmt(res["gamma_relaxed"]), "min_dist_relaxed_A": fmt(res["min_dist_relaxed_A"]),
                                 "max_disp_free_A": fmt(res["max_disp_free_A"]), "max_disp_fixed_A": fmt(res["max_disp_fixed_A"]),
                                 "elapsed_s": f"{dt:.2f}", "frame_index": len(frames) - 1})
            print(f"  [{i + 1}/{len(rows)}] {out_rows[-1]['status']} steps={out_rows[-1].get('n_steps', '')} fmax={out_rows[-1].get('fmax_final', '')} "
                  f"gamma_relaxed={out_rows[-1].get('gamma_relaxed', '')} dt={dt:.2f}s", flush=True)
            if args.calc == "mace":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
        except FixedAtomMoved as e:
            print(f"FATAL: {e}", file=sys.stderr)
            return 4
        except Exception as e:  # noqa: BLE001  catch per row and continue
            traceback.print_exc()
            msg = f"{type(e).__name__}: {e}"[:200]
            out_rows.append({**base, "status": "error", "error": msg, "elapsed_s": f"{time.time() - t0:.2f}"})
    if len(out_rows) != K.PER_CHUNK:
        print(f"FATAL: internal: {len(out_rows)} rows != {K.PER_CHUNK}", file=sys.stderr)
        return 2
    counts = {s: sum(1 for r in out_rows if r["status"] == s) for s in STATUSES}
    print(f"STATUS {counts} elapsed={time.time() - t_all:.1f}s", flush=True)
    if args.dry_run:
        print("DRY-RUN: nothing written", flush=True)
        return 0
    # ---- writing (the output dir is new. If it existed, the check above has already exited) ----
    out_dir.mkdir(parents=True, exist_ok=False)
    if frames:
        ase_write(str(out_dir / "relaxed.extxyz"), frames, format="extxyz")
        if len(ase_read(str(out_dir / "relaxed.extxyz"), index=":")) != len(frames):
            print("FATAL: relaxed.extxyz frame count mismatch after write", file=sys.stderr)
            return 2
    K.write_csv_rows(out_dir / "relax_rows.csv", ROW_FIELDS, out_rows)
    if len(K.read_csv_rows(out_dir / "relax_rows.csv")) != K.PER_CHUNK:
        print("FATAL: relax_rows.csv does not have 25 rows", file=sys.stderr)
        return 2
    n_nonexcl = sum(1 for r in out_rows if r["status"] != "excluded_pre")
    if n_nonexcl == 0 or not frames:
        print(f"FATAL: no relaxed structure to count (non-excluded rows={n_nonexcl}, frames={len(frames)}); RELAX_DONE not written (B3)", file=sys.stderr)
        return 2
    sys.path.insert(0, K.P(K.RUNNER_GUARDS_DIR))
    import runner_guards as RG
    mp = RG.write_done_marker(str(out_dir), "relaxed.extxyz", 1, marker="RELAX_DONE")
    print(f"wrote {out_dir}/relaxed.extxyz ({len(frames)} frames), relax_rows.csv (25 rows), {mp}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# control mode (R3)
# ---------------------------------------------------------------------------
def run_control(args):
    from ase.io import read as ase_read
    out_dir = Path(K.P(args.out_dir))
    cj = Path(K.P(args.control_json or f"{K.RUN}/verification/positive_control.json"))
    if args.calc == "emt" and "/tmp/" not in str(out_dir):
        raise K.InputError("--calc emt is TEST ONLY and requires --out-dir under a '/tmp/' directory")
    if out_dir.exists() or (cj.exists() and not args.dry_run):
        print(f"FATAL: {out_dir} or {cj} already exists (no overwrite)", file=sys.stderr)
        return 2
    cfg = get_cfg()
    rows = K.read_csv_rows(args.control_csv)
    if len(rows) != 5:
        raise K.InputError(f"control csv has {len(rows)} rows != 5")
    calc = make_calc(args.calc, args.dry_run, args.min_free_gib)
    out_rows = []
    for r in rows:
        base = {"row_index": r["row_index"], "poscar_path": r["poscar_path"], "calc": args.calc}
        t0 = time.time()
        try:
            atoms = ase_read(K.P(r["poscar_path"]), format="vasp")
            fixed = [int(x) for x in r["frozen_indices"].split(";") if x != ""]
            if len(atoms) != int(r["natoms"]) or len(fixed) != K.N_FIXED_PROPOSED or len(set(fixed)) != len(fixed) or max(fixed) >= len(atoms):
                raise RuntimeError(f"bad control input: natoms={len(atoms)}/{r['natoms']} n_fixed={len(fixed)}")
            if args.dry_run:
                out_rows.append({**base, "n_fixed": len(fixed), "status": "dry_run", "natoms": len(atoms), "elapsed_s": f"{time.time() - t0:.3f}"})
                continue
            res, _ = relax_one(atoms, fixed, calc, cfg, proposed_check_idx=fixed)
            if res.get("max_disp_fixed_A") not in ("", None) and res["max_disp_fixed_A"] != 0.0:
                raise FixedAtomMoved(f"control row {r['row_index']}: frozen atoms moved {res['max_disp_fixed_A']!r} Å")
            dg = res["gamma_relaxed"] - float(r["gamma_Jm2"])
            out_rows.append({**base, "n_fixed": len(fixed), "status": "ok" if res["finite"] else "nonfinite", "converged": res["converged"],
                             "n_steps": res["n_steps"], "fmax_final": fmt(res["fmax_final"]), "energy_eV": fmt(res["energy_eV"]),
                             "natoms": res["natoms"], "area_A2": fmt(res["area_A2"]), "gamma_Jm2_train": r["gamma_Jm2"],
                             "gamma_relaxed": fmt(res["gamma_relaxed"]), "delta_gamma_Jm2": fmt(dg),
                             "delta_E_eV": fmt(dg * 2.0 * res["area_A2"] / cfg["factor"]),
                             "max_disp_free_A": fmt(res["max_disp_free_A"]), "max_disp_fixed_A": fmt(res["max_disp_fixed_A"]),
                             "elapsed_s": f"{time.time() - t0:.2f}"})
        except FixedAtomMoved as e:
            print(f"FATAL: {e}", file=sys.stderr)
            return 4
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            out_rows.append({**base, "status": "error", "error": f"{type(e).__name__}: {e}"[:200]})
        print(f"  control row {r['row_index']}: {out_rows[-1]}", flush=True)
    if args.dry_run:
        print("DRY-RUN: nothing written", flush=True)
        return 0
    out_dir.mkdir(parents=True, exist_ok=False)
    K.write_csv_rows(out_dir / "relax_rows.csv", CONTROL_FIELDS, out_rows)
    ok = [r for r in out_rows if r["status"] == "ok"]
    n_conv = sum(1 for r in ok if r["converged"] is True)
    steps = [int(r["n_steps"]) for r in ok]
    dgs = [abs(float(r["delta_gamma_Jm2"])) for r in ok]
    des = [abs(float(r["delta_E_eV"])) for r in ok]
    checks = {"all_5_ok": len(ok) == 5, "all_5_converged": n_conv == 5, "n_steps_max": max(steps) if steps else None,
              "n_steps_le_20": bool(steps) and max(steps) <= 20, "max_abs_delta_gamma_Jm2": max(dgs) if dgs else None,
              "max_abs_delta_gamma_le_0.002": bool(dgs) and max(dgs) <= 0.002, "max_abs_delta_E_eV": max(des) if des else None}
    verdict = "PASS" if (checks["all_5_ok"] and checks["all_5_converged"] and checks["n_steps_le_20"] and checks["max_abs_delta_gamma_le_0.002"]) else "FLAG"
    cj.parent.mkdir(parents=True, exist_ok=True)
    cj.write_text(json.dumps({"ac": "check-R3 (r2-5)", "verdict": verdict, "calc": args.calc, "checks": checks, "rows": out_rows,
                              "note": "FLAG is decided by the author (the threshold and the formula are not changed). Expected: 5/5 converged, n_steps 0 to a few, |d gamma|<=0.002 J/m2 (|dE|<=0.0091 eV)",
                              "date": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2, ensure_ascii=False) + "\n")
    print(f"R3 verdict={verdict} {checks}", flush=True)
    print(f"wrote {out_dir}/relax_rows.csv and {cj}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", choices=K.layer_names(), help="layer name of layers.json")
    ap.add_argument("--arm", choices=["proposed", "baseline"])
    ap.add_argument("--w", type=float)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--task-id", type=int, default=None, help="if given together with --layer, (arm, w, chunk) is looked up from results/array_map_<layer>.csv (for sbatch)")
    ap.add_argument("--array-map", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", default=None, help="default: <work_root of the layer>/results/inputs_manifest_<layer>.csv")
    ap.add_argument("--fix-indices", default=None)
    ap.add_argument("--control-csv", default=None)
    ap.add_argument("--control-json", default=None)
    ap.add_argument("--calc", choices=["mace", "emt"], default="mace")
    ap.add_argument("--min-free-gib", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        if args.control_csv:
            sys.exit(run_control(args))
        if args.task_id is not None:
            if args.layer is None or any(getattr(args, k) is not None for k in ("arm", "w", "chunk")):
                ap.error("--task-id needs --layer and excludes --arm/--w/--chunk")
            t = K.resolve_array_task(args.layer, args.task_id, args.array_map)
            args.arm, args.w, args.chunk = t["arm"], t["w"], t["chunk"]
            print(f"task-id {args.task_id} -> layer={args.layer} arm={args.arm} w={args.w} chunk={args.chunk}", flush=True)
        for k in ("layer", "arm", "w", "chunk"):
            if getattr(args, k) is None:
                ap.error(f"--{k} is required (regular mode)")
        sys.exit(run_regular(args))
    except K.InputError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
