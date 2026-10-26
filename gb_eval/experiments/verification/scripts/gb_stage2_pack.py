#!/usr/bin/env python3
"""gb_stage2_pack.py -- input pack of stage 2 (relaxed structures that are post and two_gb, one extxyz per layer).

Plan: see the experiment description.
  The verification family (main and the regeneration layers) and the relax_novelty family (additional) each have a module of the same name, common.py, so only one of them is imported in one process:
    --source vrun : puts verification/scripts at the head of sys.path (layers main, baseline_regen1 to 3, proposed_regen1, proposed_regen2)
    --source run  : puts relax_novelty/scripts at the head of sys.path (layer additional)
  At start it prints common.__file__ and gb_symfit.__file__ and raises an exception unless they are under the expected directory.
  Read only (the products of relax_novelty, the stage-1 CSV and the relax products are not changed). <out-dir>/<layer>.extxyz is written in one go (an exception if it exists; no append).
Usage (compute node. The python of the evaluation environment. Do not start python on the login node):
  python gb_stage2_pack.py --source vrun --layers main,baseline_regen1,... --out-dir VRUN/results/gb_stage2/inputs
Exit codes: 0 = success / 2 = failure of the inputs or checks
"""
from __future__ import annotations
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)

import argparse
import csv
import inspect
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

GC = _gbrel.WORK + '/workspace'          # on compute nodes the translation equivalent to P() is done by _norm()
VRUN_SCRIPTS = f"{GC}/experiments/verification/scripts"
RUN_SCRIPTS = f"{GC}/experiments/relax_novelty/scripts"


def _norm(p) -> str:
    """Normal form (with /mnt) to compare ignoring the difference between <CLUSTER_MOUNT> and <CLUSTER_MOUNT>. realpath may resolve an alias of NFS, so it is not used."""
    s = os.path.abspath(str(p))
    if s.startswith('/CLUSTER_ROOT_B/'):
        s = "/mnt" + s
    return s


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fail(msg, code=2):
    print(f"FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=["vrun", "run"])
    ap.add_argument("--layers", required=True, help="comma-separated layer names")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dry-run", action="store_true", help="only print the resolution of the inputs and the counts and exit (nothing is written)")
    return ap.parse_args()


def main():
    args = parse_args()
    expect_dir = _norm(VRUN_SCRIPTS if args.source == "vrun" else RUN_SCRIPTS)
    # only one of the two in one process: the chosen side is put at the head of sys.path and the directory of the other is removed from the path (this file is in verification/scripts, so by default it would become path[0])
    sys.path[:] = [p for p in sys.path if _norm(p or ".") not in (_norm(VRUN_SCRIPTS), _norm(RUN_SCRIPTS))]
    sys.path.insert(0, expect_dir if os.path.isdir(expect_dir) else expect_dir.replace('/mnt/CLUSTER_ROOT_A', '/CLUSTER_ROOT_B', 1))
    import common as K  # noqa: E402
    import gb_symfit as S  # noqa: E402
    for mod in (K, S):
        got = _norm(os.path.dirname(os.path.abspath(mod.__file__)))
        print(f"{mod.__name__}.__file__ = {mod.__file__}", flush=True)
        if got != expect_dir:
            raise RuntimeError(f"check-P2: {mod.__name__} resolved to {mod.__file__}; expected under {expect_dir} (source={args.source})")
    print(f"check-P2 OK: source={args.source} common and gb_symfit are under {expect_dir}", flush=True)

    # [author's choice allowed] loading function of the post structures: reproduces in this file the same procedure as S.relax_chunk_dir of the family and its chunk_task (S.chunk_task)
    # (status=ok_converged of relax_rows.csv, frame_index, the cif_index check of relaxed.extxyz). The names of the functions read and the lines are printed.
    for fn in (S.relax_chunk_dir, S.atoms_to_arrays, S.chunk_task):
        src_file = inspect.getsourcefile(fn)
        line = inspect.getsourcelines(fn)[1]
        print(f"loader-ref: {S.__name__}.{fn.__name__} at {src_file}:{line}", flush=True)
    print("loader: post structure = relax_chunk_dir(layer,arm,w,chunk)/{RELAX_DONE,relax_rows.csv,relaxed.extxyz}; the cif_index of info of the frame of frame_index is checked", flush=True)

    from ase.io import read as ase_read, write as ase_write

    layers = [x for x in args.layers.split(",") if x]
    if not layers:
        _fail("--layers is empty")
    out_dir = Path(K.P(args.out_dir))
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    def csv_path_for(layer):
        if args.source == "vrun":
            return Path(K.P(f"{K.VRUN}/results/gb_equivalence/gb_symfit_{layer}.csv"))
        return Path(K.P(f"{K.RUN}/results/gb_equivalence/gb_symfit_{layer}.csv"))

    total = 0
    for layer in layers:
        if args.source == "vrun" and layer not in K.layer_names():
            raise K.InputError(f"layer {layer!r} not in VRUN layers.json {K.layer_names()}")
        if args.source == "run" and layer not in S.LAYERS:
            raise K.InputError(f"layer {layer!r} not in RUN layers {S.LAYERS}")
        cpath = csv_path_for(layer)
        rows = K.read_csv_rows(cpath)
        sel = [r for r in rows if r["stage"] == "post" and r["gb0_status"] == "two_gb"]
        bad = [r for r in sel if r["status"] != "fitted"]
        if bad:
            raise K.InputError(f"{cpath}: {len(bad)} post/two_gb rows with status != fitted (e.g. {bad[0]['arm']}/{bad[0]['chunk']}/{bad[0]['cif_index']}: {bad[0]['status']})")
        layer_col = {r["layer"] for r in sel}
        if layer_col != {layer}:
            raise K.InputError(f"{cpath}: layer column {layer_col} != {{{layer!r}}}")
        by_group = {}
        for r in sel:
            by_group.setdefault((r["arm"], r["w"], int(r["chunk"])), []).append(r)
        expect_cnt = {}
        for r in sel:
            expect_cnt[(r["arm"], r["w"])] = expect_cnt.get((r["arm"], r["w"]), 0) + 1
        _log(f"{layer}: csv={cpath} rows={len(rows)} post&two_gb={len(sel)} by (arm,w)={sorted(expect_cnt.items())}")
        out_path = out_dir / f"{layer}.extxyz"
        if out_path.exists():
            raise FileExistsError(f"refusing to overwrite: {out_path}")
        if args.dry_run:
            print(f"[DRYRUN] {layer}: would write {len(sel)} frames to {out_path}", flush=True)
            continue

        frames = []
        for (arm, w, chunk), grp in sorted(by_group.items(), key=lambda kv: (kv[0][0] != "proposed", float(kv[0][1]), kv[0][2])):
            rdir = S.relax_chunk_dir(layer, arm, float(w), chunk)
            if not (Path(rdir) / "RELAX_DONE").is_file():
                raise K.InputError(f"{rdir}/RELAX_DONE not found")
            relax = {int(r["cif_index"]): r for r in K.read_csv_rows(Path(rdir) / "relax_rows.csv")}
            xyz = Path(rdir) / "relaxed.extxyz"
            if not xyz.is_file():
                raise K.InputError(f"{xyz} not found")
            fr_all = ase_read(str(xyz), index=":")
            for r in sorted(grp, key=lambda x: int(x["cif_index"])):
                i = int(r["cif_index"])
                rr = relax.get(i)
                if rr is None or rr.get("status") != "ok_converged" or rr.get("frame_index", "") == "":
                    raise K.InputError(f"{layer}/{arm}/w{w}/chunk{chunk}/cif{i}: no converged relaxed structure (relax status={None if rr is None else rr.get('status')})")
                at = fr_all[int(rr["frame_index"])]
                if int(at.info.get("cif_index", -1)) != i:
                    raise K.InputError(f"{xyz} frame {rr['frame_index']} cif_index {at.info.get('cif_index')} != {i}")
                if len(at) != int(r["natoms"]):
                    raise K.InputError(f"{layer}/{arm}/w{w}/chunk{chunk}/cif{i}: natoms {len(at)} != csv {r['natoms']}")
                a = at.copy()
                a.info.clear()
                a.info.update({
                    "layer": layer, "arm": arm, "w": float(w), "chunk": int(chunk), "cif_index": i,
                    "gb_bin_seq": r["gb_bin_seq"], "natoms": int(len(at)), "axes": r["axes"],
                    "s1_max_A": float(r["gb_symfit_max_A"]), "s1_op": r["op_name"],
                    "s1_max_A_C2n": float(r["max_A_C2n"]), "s1_max_A_m_p": float(r["max_A_m_p"]),
                    "s1_max_A_C2t": float(r["max_A_C2t"]), "s1_max_A_inv": float(r["max_A_inv"]),
                })
                frames.append(a)
        # count check: for each layer x arm x w, the number of frames written = the number of post and two_gb rows of the CSV
        got_cnt = {}
        for a in frames:
            k = (a.info["arm"], f"{a.info['w']:.1f}")
            got_cnt[k] = got_cnt.get(k, 0) + 1
        exp_cnt = {(arm, f"{float(w):.1f}"): n for (arm, w), n in expect_cnt.items()}
        if got_cnt != exp_cnt or len(frames) != len(sel):
            raise K.InputError(f"check-P1: {layer}: frames by (arm,w) {sorted(got_cnt.items())} != csv {sorted(exp_cnt.items())}")
        ase_write(str(out_path), frames, format="extxyz")        # written in one go (no append)
        # read-back check: the number of frames, the keys, natoms and info (bin sequence, axes) agree with what was written
        back = ase_read(str(out_path), index=":")
        if len(back) != len(frames):
            raise K.InputError(f"readback frames {len(back)} != written {len(frames)}")
        for a, b in zip(frames, back):
            for key in ("layer", "arm", "chunk", "cif_index", "gb_bin_seq", "natoms", "axes", "s1_op"):
                if a.info[key] != b.info[key]:
                    raise K.InputError(f"readback mismatch {key}: {a.info[key]!r} != {b.info[key]!r}")
            if abs(float(a.info["w"]) - float(b.info["w"])) > 0 or len(b) != a.info["natoms"]:
                raise K.InputError("readback mismatch (w or natoms)")
        _log(f"{layer}: wrote {out_path} frames={len(frames)} by (arm,w)={sorted(got_cnt.items())}")
        total += len(frames)
    _log(f"pack done: source={args.source} layers={layers} frames_total={total}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileExistsError as e:
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
