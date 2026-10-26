#!/usr/bin/env python3
"""novelty_chunk.py -- novelty stages N-pre and N-post: checks, for each chunk (25 structures), with StructureMatcher (stage_d_common.UNI) whether a generated structure matches any of the 4,419 training structures (novelty).
Matches one chunk (25 structures) at a time with StructureMatcher (stage_d_common.UNI).

Plan: see the experiment description.
  N-pre : matches the generated structures that are valid in the evaluation before relaxation (excluded_pre of the manifest is empty) (as generated; for baseline, the same axis permutation as in the evaluation before relaxation) with train
  N-post: matches the converged structures after relaxation (status=ok_converged of relax_rows.csv, frame_index of relaxed.extxyz) with train
  Only structures of train with the same number of atoms are compared. If even one UNI.fit is True, novel=False (the first matching train is recorded).
  If there is no train structure with the same number of atoms, novel=True, n_compared=0 and novel_vacuous=True (vacuous novelty).
Usage (compute node of the CPU partition, python of the evaluation environment or the mattergen venv. No MLIP is used = CPU):
  python novelty_chunk.py --layer main --arm proposed --w 2.0 --chunk 0 [--phase pre|post|both] [--procs N]
     --out-dir  default results/novelty/<layer>     --relax-dir default results/relax/<layer>
Output: <out-dir>/<model>_w<W>/chunk<cc>/novelty_{pre,post}_rows.csv (25 rows) and the completion marker NOVELTY_{PRE,POST}_DONE.
Idempotent: skipped (exit 0) if the row file and the marker are both present. Exit 2 (not overwritten) if the row file exists but the marker does not.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

NOV_FIELDS = ["layer", "arm", "w", "chunk", "cif_index", "phase", "eligible", "natoms", "novel", "match_cnid", "match_row_index",
              "n_compared", "novel_vacuous", "elapsed_s"]
_G = {}


def _worker(task):
    idx, struct = task
    t0 = time.time()
    res = K.novelty_vs_train(struct, _G["train"], _G["matcher"])
    res["elapsed_s"] = time.time() - t0
    return idx, res


def run_matching(structs, train_by_n, matcher, procs):
    """structs: {cif_index: Structure} -> {cif_index: result}. procs>1 is a fork Pool (train is inherited from the parent)."""
    _G["train"], _G["matcher"] = train_by_n, matcher
    tasks = sorted(structs.items())
    out = {}
    if procs <= 1 or len(tasks) <= 1:
        for t in tasks:
            i, r = _worker(t)
            out[i] = r
        return out
    ctx = mp.get_context("fork")
    with ctx.Pool(min(procs, len(tasks))) as pool:
        for i, r in pool.imap_unordered(_worker, tasks, chunksize=1):
            out[i] = r
    return out


def load_pre_structures(layer, arm, w, chunk, rows):
    atoms_list = K.load_chunk_atoms(layer, arm, w, chunk)
    out = {}
    for r in rows:
        if r["excluded_pre"]:
            continue
        i = int(r["cif_index"])
        atoms = atoms_list[i][0]
        if atoms is None:
            raise K.InputError(f"{layer}/{arm}/w{w}/chunk{chunk}/{i}: structure unavailable but manifest says valid")
        out[i] = K.to_pmg(atoms)
    return out


def load_post_structures(relax_dir: Path, rows_relax):
    from ase.io import read as ase_read
    xyz = relax_dir / "relaxed.extxyz"
    frames = ase_read(str(xyz), index=":") if xyz.is_file() else []
    out = {}
    for r in rows_relax:
        if r["status"] != "ok_converged":
            continue
        i = int(r["cif_index"])
        fi = int(r["frame_index"])
        at = frames[fi]
        if int(at.info.get("cif_index", -1)) != i:
            raise K.InputError(f"{xyz}: frame {fi} cif_index {at.info.get('cif_index')} != {i}")
        out[i] = K.to_pmg(at)
    return out


def do_phase(phase, args, rows, relax_rows, relax_dir, out_dir, train_by_n, matcher, procs):
    rows_file = out_dir / f"novelty_{phase}_rows.csv"
    marker_name = f"NOVELTY_{phase.upper()}_DONE"
    if rows_file.exists():
        if (out_dir / marker_name).is_file() and len(K.read_csv_rows(rows_file)) == K.PER_CHUNK:
            print(f"SKIP {phase}: {rows_file} already complete", flush=True)
            return 0
        print(f"FATAL: {rows_file} exists without {marker_name} (or wrong row count); not overwriting", file=sys.stderr)
        return 2
    layer, arm, w, chunk = args.layer, args.arm, float(args.w), int(args.chunk)
    t0 = time.time()
    if phase == "pre":
        structs = load_pre_structures(layer, arm, w, chunk, rows)
        eligible = {int(r["cif_index"]) for r in rows if not r["excluded_pre"]}
    else:
        if not (relax_dir / "RELAX_DONE").is_file():
            raise K.InputError(f"{relax_dir}/RELAX_DONE not found: the relaxation stage has not finished (N-post is after the relaxation)")
        structs = load_post_structures(relax_dir, relax_rows)
        eligible = {int(r["cif_index"]) for r in relax_rows if r["status"] == "ok_converged"}
    print(f"[{phase}] {len(structs)} structures to match against train (same natoms only), procs={procs}", flush=True)
    res = run_matching(structs, train_by_n, matcher, procs)
    out_rows = []
    nat = {int(r["cif_index"]): r["natoms"] for r in rows}
    for i in range(K.PER_CHUNK):
        base = {"layer": layer, "arm": arm, "w": f"{w:.1f}", "chunk": chunk, "cif_index": i, "phase": phase,
                "eligible": i in eligible, "natoms": nat[i]}
        if i in res:
            r = res[i]
            out_rows.append({**base, "novel": r["novel"], "match_cnid": r["match_cnid"], "match_row_index": r["match_row_index"],
                             "n_compared": r["n_compared"], "novel_vacuous": r["novel_vacuous"], "elapsed_s": f"{r['elapsed_s']:.2f}"})
        else:
            out_rows.append({**base, "novel": "NA", "novel_vacuous": "NA"})   # rows that are not converged / not valid are NA (a miss in the metrics)
    out_dir.mkdir(parents=True, exist_ok=True)
    K.write_csv_rows(rows_file, NOV_FIELDS, out_rows)
    if len(K.read_csv_rows(rows_file)) != K.PER_CHUNK:
        print("FATAL: novelty rows != 25", file=sys.stderr)
        return 2
    if not res:
        print(f"FATAL: [{phase}] no eligible structure (0 matched); marker not written (B3)", file=sys.stderr)
        return 2
    sys.path.insert(0, K.P(K.RUNNER_GUARDS_DIR))
    import runner_guards as RG
    mpth = RG.write_done_marker(str(out_dir), rows_file.name, 1, marker=marker_name)
    n_nov = sum(1 for r in res.values() if r["novel"])
    print(f"[{phase}] done: matched={len(res)} novel={n_nov} vacuous={sum(1 for r in res.values() if r['novel_vacuous'])} "
          f"n_compared_total={sum(r['n_compared'] for r in res.values())} elapsed={time.time() - t0:.1f}s marker={mpth}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", required=True, choices=K.layer_names(), help="layer name of layers.json")
    ap.add_argument("--arm", choices=["proposed", "baseline"])
    ap.add_argument("--w", type=float)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--task-id", type=int, default=None, help="if given, (arm, w, chunk) is looked up from results/array_map_<layer>.csv (for sbatch. Exclusive with --arm/--w/--chunk)")
    ap.add_argument("--array-map", default=None)
    ap.add_argument("--phase", default="both", choices=["pre", "post", "both"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--train-manifest", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--relax-dir", default=None)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    try:
        if args.task_id is not None:
            if any(getattr(args, k) is not None for k in ("arm", "w", "chunk")):
                ap.error("--task-id excludes --arm/--w/--chunk")
            t = K.resolve_array_task(args.layer, args.task_id, args.array_map)
            args.arm, args.w, args.chunk = t["arm"], t["w"], t["chunk"]
            print(f"task-id {args.task_id} -> layer={args.layer} arm={args.arm} w={args.w} chunk={args.chunk}", flush=True)
        for k in ("arm", "w", "chunk"):
            if getattr(args, k) is None:
                ap.error(f"--{k} is required (or --task-id)")
        layer, arm, w, chunk = args.layer, args.arm, float(args.w), int(args.chunk)
        K.check_layer_arm(layer, arm)
        manifest = args.manifest or K.manifest_path_for(layer)
        rows = [r for r in K.read_csv_rows(manifest)
                if r["layer"] == layer and r["arm"] == arm and abs(float(r["w"]) - w) < 1e-9 and int(r["chunk"]) == chunk]
        rows.sort(key=lambda r: int(r["cif_index"]))
        if [int(r["cif_index"]) for r in rows] != list(range(K.PER_CHUNK)):
            raise K.InputError(f"manifest {manifest}: rows for ({layer},{arm},w{w},chunk{chunk}) are not cif_index 0..24")
        base_out = Path(K.P(args.out_dir or K.novelty_root_for(layer)))
        K.assert_out_not_in_readonly(layer, base_out)
        out_dir = base_out / K.armw(arm, w) / K.chunk_name(chunk)
        base_rel = Path(K.P(args.relax_dir or K.relax_root_for(layer)))
        relax_dir = base_rel / K.armw(arm, w) / K.chunk_name(chunk)
        phases = ["pre", "post"] if args.phase == "both" else [args.phase]
        relax_rows = K.read_csv_rows(relax_dir / "relax_rows.csv") if (relax_dir / "relax_rows.csv").is_file() else []
        if "post" in phases and len(relax_rows) != K.PER_CHUNK:
            raise K.InputError(f"{relax_dir}/relax_rows.csv missing or != 25 rows (N-post needs the relaxation output)")
        t0 = time.time()
        train_by_n = K.load_train_structures(args.train_manifest or K.P(f"{K.RUN}/results/train_ref_manifest.csv"))
        print(f"train loaded: {sum(len(v) for v in train_by_n.values())} structures, natoms groups "
              f"{ {n: len(v) for n, v in sorted(train_by_n.items())} } ({time.time() - t0:.1f}s)", flush=True)
        matcher = K.make_matcher()
        print(f"matcher: stol={matcher.stol} ltol={matcher.ltol} angle_tol={matcher.angle_tol} prim={matcher._primitive_cell} "
              f"scale={matcher._scale} supercell={matcher._supercell}", flush=True)
        rc = 0
        for ph in phases:
            rc = do_phase(ph, args, rows, relax_rows, relax_dir, out_dir, train_by_n, matcher, args.procs) or rc
        sys.exit(rc)
    except K.InputError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
