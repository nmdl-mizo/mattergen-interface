#!/usr/bin/env python3
"""Data rebuild, stage C: relaxation (1 array task = 1 template). Transcribed from the earlier script stage3_relax_chunk1.py.

  python stage_c_relax_chunk.py [--task-id N] [--dry-run]
    --task-id : SLURM_ARRAY_TASK_ID if omitted. Corresponds to the cnid at that row of template_map.csv (lexicographic order of (i,j)) (0 to 30).
    --dry-run : does not use the GPU or MACE; only checks the loading of the inputs and their consistency (existence of POSCAR, 76 fixed indices, number of atoms).
  Environment variable DATA_RELAX_LIMIT=N : relaxes only the first N structures of that template (for smoke tests).
    The completion marker is then `*.SMOKE_DONE` (`relax_done/{cnid}.SMOKE_DONE`), which is not mixed with the `*.DONE` of the production run.

Protocol: MACE-MH-1, head omat_pbe (config.MACE_MODEL / MACE_HEAD), device=cuda, float64,
  FIRE (fmax=config.FMAX, steps=config.MAX_STEPS), 76 fixed atoms (FixAtoms), fixed cell.
Output: POSCAR_relaxed, energy.json and relax.log in the structure directory. Progress goes to results/relax_progress/{cnid}.jsonl (a separate file per task).
Idempotent: skipped if energy.json exists. converged is the return value of FIRE (converged in energy.json); it is not decided from log strings.
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import builder_common as C

sys.path.insert(0, str(C.GC / "scripts" / "config"))
import config as CFG  # noqa: E402  (read-only import)

sys.path.insert(0, _gbrel.CODE + '/gb_eval/vendor/bin')
import runner_guards as RG  # noqa: E402

MODE = "stage_c"


def relax_one(poscar_path: Path, out_dir: Path, calc, frozen_idx: list[int]) -> dict:
    from ase.constraints import FixAtoms
    from ase.io import read, write
    from ase.optimize import FIRE

    atoms = read(str(poscar_path), format="vasp")
    natoms = len(atoms)
    n_fixed = len(frozen_idx)
    atoms.set_constraint(FixAtoms(indices=frozen_idx))
    atoms.calc = calc

    out_dir.mkdir(parents=True, exist_ok=True)
    opt = FIRE(atoms, logfile=str(out_dir / "relax.log"))
    converged = opt.run(fmax=CFG.FMAX, steps=CFG.MAX_STEPS)
    n_steps = opt.get_number_of_steps()
    forces = atoms.get_forces()
    fmax_final = float(np.sqrt((forces ** 2).sum(axis=1).max())) if len(forces) else 0.0

    write(str(out_dir / "POSCAR_relaxed"), atoms, format="vasp", direct=True)
    energy = float(atoms.get_potential_energy())
    result = {
        "energy_eV": energy, "energy_per_atom_eV": energy / natoms, "natoms": natoms,
        "converged": bool(converged), "mode": MODE,
        "n_fixed": int(n_fixed), "n_relax": int(natoms - n_fixed),
        "fmax_final": fmax_final, "n_steps": int(n_steps),
        "a_len_A": float(np.linalg.norm(atoms.get_cell()[0])),
    }
    with open(out_dir / "energy.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_inputs(cnid: str, limit: int | None) -> dict:
    p = C.RESULTS / "relax_inputs" / f"{cnid}.json"
    with open(p) as f:
        inputs = json.load(f)
    keys = list(inputs.keys())
    if not keys:
        raise ValueError(f"{p}: no inputs")
    if limit is not None:
        keys = keys[:limit]
    return {k: inputs[k] for k in keys}


def dry_run(cnid: str, inputs: dict):
    from ase.io import read
    bad = []
    for key, e in inputs.items():
        p = Path(e["poscar_path"])
        if not p.exists():
            bad.append((key, "poscar missing"))
            continue
        at = read(str(p), format="vasp")
        fi = e["frozen_indices"]
        if len(at) != e["natoms"] or len(fi) != C.N_FROZEN or len(set(fi)) != len(fi) or max(fi) >= len(at):
            bad.append((key, f"natoms={len(at)}/{e['natoms']} n_frozen={len(fi)}"))
    print(f"[dry-run] {cnid}: inputs={len(inputs)} bad={len(bad)} {bad[:3]}")
    print(f"[dry-run] MACE_MODEL={CFG.resolve(CFG.MACE_MODEL)} head={CFG.MACE_HEAD} fmax={CFG.FMAX} steps={CFG.MAX_STEPS}")
    if bad:
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tid = args.task_id if args.task_id is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    rows = C.read_template_map()
    if len(rows) != C.N_TEMPLATES or not (0 <= tid < len(rows)):
        raise ValueError(f"task id {tid} / template_map rows {len(rows)}")
    cnid = rows[tid]["cnid"]
    lim_env = os.environ.get("DATA_RELAX_LIMIT", "")
    limit = int(lim_env) if lim_env else None
    if limit is not None and limit < 1:
        raise ValueError(f"DATA_RELAX_LIMIT={limit}")
    inputs = load_inputs(cnid, limit)
    keys = list(inputs.keys())
    print(f"task={tid} cnid={cnid} inputs={len(keys)}" + (f" (LIMIT {limit})" if limit else ""), flush=True)
    if args.dry_run:
        dry_run(cnid, inputs)
        return

    print("[GUARD] vram_precheck ...", flush=True)
    free_gib = RG.vram_precheck(min_free_gib=10)
    print(f"[GUARD] VRAM free={free_gib:.1f} GiB (>= 10 GiB OK)", flush=True)

    from mace.calculators import mace_mp
    model_path = CFG.resolve(CFG.MACE_MODEL)
    print(f"MACE load: {model_path} head={CFG.MACE_HEAD}", flush=True)
    t0 = time.time()
    calc = mace_mp(model=model_path, device="cuda", default_dtype="float64", head=CFG.MACE_HEAD)   # once per task
    print(f"model load: {time.time() - t0:.1f}s", flush=True)

    (C.RESULTS / "relax_progress").mkdir(parents=True, exist_ok=True)
    log_f = open(C.RESULTS / "relax_progress" / f"{cnid}.jsonl", "a")
    t_relax0 = time.time()
    done = skipped = failed = 0
    import torch
    for i, key in enumerate(keys):
        e = inputs[key]
        out_dir = Path(e["poscar_path"]).parent
        if (out_dir / "energy.json").exists():
            skipped += 1
            continue
        t_s = time.time()
        try:
            result = relax_one(Path(e["poscar_path"]), out_dir, calc, e["frozen_indices"])
            dt = time.time() - t_s
            done += 1
            rec = {"key": key, "status": "ok", "elapsed_s": dt, **result}
            print(f"  [{i + 1}/{len(keys)}] {key}: E={result['energy_eV']:.4f} conv={result['converged']} "
                  f"steps={result['n_steps']} fmax={result['fmax_final']:.4f} dt={dt:.2f}s", flush=True)
        except Exception as ex:  # noqa: BLE001
            dt = time.time() - t_s
            print(f"  [{i + 1}/{len(keys)}] [ERROR] {key}: {ex}", flush=True)
            traceback.print_exc()
            rec = {"key": key, "status": "error", "elapsed_s": dt, "error": str(ex)}
            failed += 1
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        torch.cuda.empty_cache()
    elapsed = time.time() - t_relax0
    log_f.close()
    print(f"{cnid}: new={done} skipped={skipped} failed={failed} / {len(keys)}  elapsed={elapsed:.1f}s "
          f"sec/structure(new only)={(elapsed / done) if done else None}", flush=True)
    if failed:
        raise RuntimeError(f"{failed} structure(s) failed in {cnid}; no DONE marker written")

    # completion marker: written only if the checks pass. write_done_marker writes inside the checked --dir,
    # so it is written to structures/{cnid}/ and the same content is also placed in results/relax_done/.
    smoke = limit is not None
    mname = "RELAX_SMOKE_DONE" if smoke else "RELAX_DONE"
    mpath = RG.write_done_marker(str(C.STRUCTURES / cnid), "POSCAR_relaxed", len(keys), marker=mname, recursive=True)
    (C.RESULTS / "relax_done").mkdir(parents=True, exist_ok=True)
    dst = C.RESULTS / "relax_done" / f"{cnid}.{'SMOKE_DONE' if smoke else 'DONE'}"
    dst.write_text(Path(mpath).read_text())
    print(f"DONE marker: {mpath} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
