#!/usr/bin/env python3
"""build_manifests.py -- stage P: list of inputs (inputs_manifest_<layer>.csv), array mapping (array_map_<layer>.csv),
list of train references (train_ref_manifest.csv), identity check of the fixed atoms of proposed (P2-A/P2-B and the FAIL validation), inputs for R3.

Plan: see the experiment description.
Usage (compute node of the CPU partition; the python of the mattergen venv; the environment variables are set by sbatch_p.sh):
  python build_manifests.py --layers additional|main|both [--results-dir DIR] [--verification-dir DIR]
  additional: results/inputs_manifest_additional.csv (2,800 rows), array_map_additional.csv (0..111), r1_additional_array_spec.txt
  main: results/inputs_manifest_main.csv (700 rows), array_map_main.csv (0..27). Non-zero exit unless gamma_pre.csv of main has 350 rows x 2.
  Both use / make train_ref_manifest.csv (4,419 rows; made if absent) and verification/r3_inputs.csv.
Exit codes: 0 = success / 2 = failure of the inputs or checks (number of rows, keys, matching) / 3 = P2-A or P2-B FAIL on real data (stop without changing the threshold or the formula)
Existing files are not overwritten (non-zero exit if they exist).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402

AREA_REL_STOP = 1e-9   # detection that re-reading the structure differs from the evaluation before relaxation (a different structure is read). The criterion (1e-12) of R0 is decided by the runner


MANIFEST_FIELDS = [
    "layer", "run_root", "arm", "w", "chunk", "cif_index", "target_id", "gamma_target", "is_inner",
    "gamma_unrelaxed", "natoms", "area_A2", "excluded_pre", "excluded_cat",
    "gb_dx_realized_A", "non_al_count", "min_dist_A",
    "cell_a_A", "cell_b_A", "cell_c_A", "cell_alpha_deg", "cell_beta_deg", "cell_gamma_deg",
]
TRAIN_FIELDS = ["row_index", "cnid", "natoms", "gamma_Jm2", "poscar_relaxed_path",
                "k", "seed", "dx_A", "converged", "cell_a_A", "cell_b_A", "cell_c_A", "cell_alpha_deg", "cell_beta_deg", "cell_gamma_deg"]
R3_FIELDS = ["row_index", "poscar_path", "frozen_indices", "gamma_Jm2", "area_A2", "natoms"]


# ---------------------------------------------------------------------------
# check P1: manifest check per layer
# ---------------------------------------------------------------------------
def validate_manifest(rows, layer):
    """Number of rows per layer, duplicate keys, 350 per (layer,arm,w), 50 per target. InputError if violated."""
    expect_aw = K.layer_arm_w_list(layer)
    n_expect = len(expect_aw) * K.N_PER_ARM_W
    if len(rows) != n_expect:
        raise K.InputError(f"check-P1: {layer} manifest has {len(rows)} rows != {n_expect}")
    keys = set()
    for r in rows:
        k = (r["layer"], r["arm"], r["w"], int(r["chunk"]), int(r["cif_index"]))
        if k in keys:
            raise K.InputError(f"check-P1: duplicate key {k}")
        keys.add(k)
    cnt_aw, cnt_t = {}, {}
    for r in rows:
        a = (r["layer"], r["arm"], f"{float(r['w']):.1f}")
        cnt_aw[a] = cnt_aw.get(a, 0) + 1
        t = a + (int(r["target_id"]),)
        cnt_t[t] = cnt_t.get(t, 0) + 1
    for arm, w in expect_aw:
        a = (layer, arm, f"{float(w):.1f}")
        if cnt_aw.get(a, 0) != K.N_PER_ARM_W:
            raise K.InputError(f"check-P1: {a} has {cnt_aw.get(a, 0)} rows != {K.N_PER_ARM_W}")
        for t in range(K.N_TARGETS):
            if cnt_t.get(a + (t,), 0) != K.PER_TARGET:
                raise K.InputError(f"check-P1: {a + (t,)} has {cnt_t.get(a + (t,), 0)} rows != {K.PER_TARGET}")
    return {"rows": len(rows), "dup_keys": 0, "per_arm_w": K.N_PER_ARM_W, "per_target": K.PER_TARGET}


# ---------------------------------------------------------------------------
# train reference list (P3) and the inputs of R3
# ---------------------------------------------------------------------------
def write_p3_evidence(rows, n_all, ver_dir: Path, log):
    """Leaves the measurement of check P3 in verification/P3_train_ref_manifest.json (not written if it already exists). Reconfirms the existence of all POSCAR_relaxed."""
    out = ver_dir / "P3_train_ref_manifest.json"
    if out.exists():
        return
    missing = [r["poscar_relaxed_path"] for r in rows if not os.path.isfile(K.P(r["poscar_relaxed_path"]))]
    doc = {"ac": "check-P3", "train_ref_manifest_rows": len(rows), "expected": 4419, "gamma_dedup_rows": n_all, "n_poscar_relaxed_missing": len(missing),
           "natoms_counts": {str(n): sum(1 for r in rows if int(r["natoms"]) == n) for n in sorted({int(r["natoms"]) for r in rows})},
           "pass": bool(len(rows) == 4419 and not missing), "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    log(f"wrote {out}: rows={len(rows)} missing={len(missing)}")


def build_train_manifest(results_dir: Path, ver_dir: Path, log):
    from ase.io import read as ase_read
    out = results_dir / "train_ref_manifest.csv"
    gam = K.read_csv_rows(K.GAMMA_CSV)
    n_all = len(gam)
    split = json.loads(Path(K.P(K.SPLIT_JSON)).read_text())
    tr = split["train_indices"]
    if len(tr) != 4419:
        raise K.InputError(f"check-P3: len(train_indices)={len(tr)} != 4419")
    if len(set(tr)) != len(tr) or min(tr) < 0 or max(tr) >= n_all:
        raise K.InputError(f"check-P3: train_indices out of range / duplicated (n_all={n_all})")
    if n_all != 5509:
        raise K.InputError(f"gamma_dedup.csv has {n_all} rows != 5509")
    if out.exists():
        rows = K.read_csv_rows(out)
        if len(rows) != 4419 or [int(r["row_index"]) for r in rows] != list(tr):
            raise K.InputError(f"existing {out} does not match train_indices (refusing to overwrite)")
        log(f"train_ref_manifest.csv exists ({len(rows)} rows); reused")
        write_p3_evidence(rows, n_all, ver_dir, log)
        return rows, False
    rows, missing, n_nonconv = [], [], 0
    for ri in tr:
        g = gam[ri]
        ppath = g["dir"] + "/POSCAR_relaxed"
        pp = K.P(ppath)
        if not os.path.isfile(pp):
            missing.append(ppath)
            continue
        at = ase_read(pp, format="vasp")
        if len(at) != int(g["natoms"]):
            raise K.InputError(f"row {ri}: POSCAR natoms {len(at)} != csv {g['natoms']}")
        cp = at.cell.cellpar()
        if not K.to_bool(g["converged"]):
            n_nonconv += 1
        rows.append({
            "row_index": ri, "cnid": g["cnid"], "natoms": g["natoms"], "gamma_Jm2": g["gamma_Jm2"], "poscar_relaxed_path": ppath,
            "k": g["k"], "seed": g["seed"], "dx_A": g["dx_A"], "converged": g["converged"],
            "cell_a_A": repr(float(cp[0])), "cell_b_A": repr(float(cp[1])), "cell_c_A": repr(float(cp[2])),
            "cell_alpha_deg": repr(float(cp[3])), "cell_beta_deg": repr(float(cp[4])), "cell_gamma_deg": repr(float(cp[5])),
        })
    if missing:
        raise K.InputError(f"check-P3: {len(missing)} POSCAR_relaxed missing, e.g. {missing[:3]}")
    if len(rows) != 4419:
        raise K.InputError(f"check-P3: train manifest {len(rows)} rows != 4419")
    K.write_csv_rows(out, TRAIN_FIELDS, rows)
    log(f"wrote {out} ({len(rows)} rows; non-converged in train={n_nonconv})")
    write_p3_evidence(rows, n_all, ver_dir, log)
    return rows, True


def build_r3_inputs(train_rows, ver_dir: Path, log):
    """Inputs of the positive control R3: 5 structures chosen at random (seed 122) from train. frozen_indices come from bulk_membership.csv (separated by ';')."""
    out = ver_dir / "r3_inputs.csv"
    if out.exists():
        log(f"{out} exists; reused")
        return K.read_csv_rows(out)
    idx_sorted = sorted(int(r["row_index"]) for r in train_rows)
    picks = random.Random(122).sample(idx_sorted, 5)
    by_idx = {int(r["row_index"]): r for r in train_rows}
    gam = K.read_csv_rows(K.GAMMA_CSV)
    mem = {(m["cnid"], m["k"], m["seed"]): m for m in K.read_csv_rows(K.BULK_MEMBERSHIP_CSV)}
    rows = []
    for ri in picks:
        g = gam[ri]
        m = mem.get((g["cnid"], g["k"], g["seed"]))
        if m is None:
            raise K.InputError(f"R3: no bulk_membership row for {(g['cnid'], g['k'], g['seed'])}")
        fi = [int(x) for x in m["frozen_indices"].split(";") if x != ""]
        if len(fi) != K.N_FIXED_PROPOSED or len(set(fi)) != len(fi) or max(fi) >= int(g["natoms"]):
            raise K.InputError(f"R3: row {ri}: frozen_indices count {len(fi)} (expected 76 distinct, < natoms)")
        rows.append({"row_index": ri, "poscar_path": by_idx[ri]["poscar_relaxed_path"], "frozen_indices": m["frozen_indices"],
                     "gamma_Jm2": g["gamma_Jm2"], "area_A2": g["area_A2"], "natoms": g["natoms"]})
    K.write_csv_rows(out, R3_FIELDS, rows)
    log(f"wrote {out} (row_index={picks})")
    return rows


# ---------------------------------------------------------------------------
# manifest of one layer
# ---------------------------------------------------------------------------
def build_layer(layer, log):
    """Returns the manifest rows of the layer and the P2 measurement of the fixed atoms of proposed."""
    import fixed_identity as FI

    template, R = None, None
    has_proposed = any(arm == "proposed" for arm, _ in K.layer_arm_w_list(layer))
    if has_proposed:
        template = FI.build_reference_template(K.P(K.GAMMA_CSV), K.P(K.BULK_MEMBERSHIP_CSV), K.P(K.MANIFEST_DIR))
        R = FI._mg()["R"]
        log(f"reference template: fixed_a_cart {template.fixed_a_cart.shape} a_t={template.a_template} b_t={template.b_template} "
            f"c_t={template.c_template} ref_dir={template.reference_dir}")
    rows = []
    ident = {"n_structures": 0, "max_dev_p2a_A": 0.0, "max_dev_p2b_A": 0.0, "argmax_p2a": None, "argmax_p2b": None,
             "per_arm_w_chunk": {}, "n_fail_p2a": 0, "n_fail_p2b": 0}
    area_rel_max, natoms_mismatch = 0.0, 0
    excl_summary = {}
    run_root = K.norm_compute(K.layer_run_root(layer))
    for arm, w in K.layer_arm_w_list(layer):
        check_eval_complete(layer, arm, w)
        gp = K.read_gamma_pre(layer, arm, w)
        by_key = {(int(r["chunk"]), int(r["cif_index"])): r for r in gp}
        for c in range(K.N_CHUNKS):
            atoms_list = K.load_chunk_atoms(layer, arm, w, c)
            chunk_max_a = chunk_max_b = 0.0
            for i, (atoms, note) in enumerate(atoms_list):
                r = by_key[(c, i)]
                excl = K.excluded_pre_reason(r)
                if (note == "angle_fail") != ("angle_fail" in excl.split(";")):
                    raise K.InputError(f"{layer}/{arm}/w{w}/chunk{c}/{i}: angle_fail judgement differs between the structure ({note!r}) "
                                       f"and gamma_pre.csv error ({r['error']!r}) (cannot be read with the same function)")
                cell_cols = {k: "" for k in ("cell_a_A", "cell_b_A", "cell_c_A", "cell_alpha_deg", "cell_beta_deg", "cell_gamma_deg")}
                if atoms is not None:
                    if len(atoms) != int(r["natoms"]):
                        natoms_mismatch += 1
                        raise K.InputError(f"{layer}/{arm}/w{w}/chunk{c}/{i}: len(atoms)={len(atoms)} != gamma_pre natoms {r['natoms']}")
                    cp = atoms.cell.cellpar()
                    cell_cols = dict(zip(cell_cols.keys(), (repr(float(x)) for x in cp)))
                    ar = K.to_float(r["area_A2"])
                    if not excl and ar is not None:
                        rel = abs(K.area_bc(atoms) - ar) / ar
                        area_rel_max = max(area_rel_max, rel)
                        if rel > AREA_REL_STOP:
                            raise K.InputError(f"{layer}/{arm}/w{w}/chunk{c}/{i}: area from re-read structure differs from gamma_pre "
                                               f"area_A2 by rel {rel:.3e} (> {AREA_REL_STOP}); not the same structure/axes as the pre-relax evaluation")
                    if arm == "proposed":
                        da, db = FI.check_structure(atoms, template, R)
                        ident["n_structures"] += 1
                        if da > ident["max_dev_p2a_A"]:
                            ident["max_dev_p2a_A"], ident["argmax_p2a"] = da, [layer, arm, float(w), c, i]
                        if db > ident["max_dev_p2b_A"]:
                            ident["max_dev_p2b_A"], ident["argmax_p2b"] = db, [layer, arm, float(w), c, i]
                        ident["n_fail_p2a"] += int(da > FI.P2_THRESHOLD_A)
                        ident["n_fail_p2b"] += int(db > FI.P2_THRESHOLD_A)
                        chunk_max_a, chunk_max_b = max(chunk_max_a, da), max(chunk_max_b, db)
                cat = K.excluded_category(excl)
                if excl:
                    key = f"{arm}_w{float(w):.1f}"
                    excl_summary.setdefault(key, {}).setdefault(cat, 0)
                    excl_summary[key][cat] += 1
                rows.append({
                    "layer": layer, "run_root": run_root, "arm": arm, "w": f"{float(w):.1f}", "chunk": c, "cif_index": i,
                    "target_id": r["target_id"], "gamma_target": r["gamma_target"], "is_inner": int(K.to_bool(r["is_inner"])),
                    "gamma_unrelaxed": r["gamma_unrelaxed"], "natoms": r["natoms"], "area_A2": r["area_A2"],
                    "excluded_pre": excl, "excluded_cat": cat,
                    "gb_dx_realized_A": r.get("gb_dx_realized_A", ""), "non_al_count": r["non_al_count"], "min_dist_A": r.get("min_dist_A", ""),
                    **cell_cols,
                })
            if arm == "proposed":
                ident["per_arm_w_chunk"][f"w{float(w):.1f}/chunk{c:02d}"] = {"max_p2a_A": chunk_max_a, "max_p2b_A": chunk_max_b}
        log(f"{layer}/{arm}/w{float(w):.1f}: rows so far {len(rows)}")
    # consistency of is_inner: target_id <= 4
    for r in rows:
        if int(r["is_inner"]) != int(int(r["target_id"]) <= K.INNER_MAX_TARGET_ID):
            raise K.InputError(f"is_inner inconsistent with target_id for {r['arm']} w{r['w']} chunk{r['chunk']}")
    meta = {"area_rel_diff_max": area_rel_max, "excluded_by_category": excl_summary,
            "template": None if template is None else {"a_template": template.a_template, "b_template": template.b_template,
                                                       "c_template": template.c_template, "reference_dir": template.reference_dir,
                                                       "n_fixed_a": int(template.fixed_a_cart.shape[0])}}
    return rows, (ident if has_proposed else None), meta, template, R


def check_eval_complete(layer, arm, w):
    """Completion check of stage E (evaluation of gamma before relaxation): ALL_DONE exists in the eval dir and the verdict of validation.json is PASS. InputError (wait) if not finished."""
    d = Path(K.eval_dir(layer, arm, w))
    if not (d / "ALL_DONE").is_file():
        raise K.InputError(f"{d}/ALL_DONE not found: stage E not finished (wait)")
    gj = d / "g63v.json"
    if not gj.is_file():
        raise K.InputError(f"{gj} not found: stage E not finished (wait)")
    verdict = json.loads(gj.read_text()).get("verdict")
    if verdict != "PASS":
        raise K.InputError(f"{gj}: verdict={verdict!r} != 'PASS'")


def proposed_fail_validation(layer, template, R, log):
    """Makes the FAIL validation of P2 from one proposed structure of real data (proposed with the smallest w, chunk00, cif 0)."""
    import fixed_identity as FI
    arm = "proposed"
    w = sorted(K.LAYERS[layer]["ws"])[0]
    atoms_list = K.load_chunk_atoms(layer, arm, w, 0)
    atoms = atoms_list[0][0]
    wit = FI.fail_validationes(atoms, template, R)
    wit["source_structure"] = [layer, arm, float(w), 0, 0]
    return wit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", required=True, choices=["additional", "main", "both"])
    ap.add_argument("--results-dir", default=K.P(f"{K.RUN}/results"))
    ap.add_argument("--verification-dir", default=K.P(f"{K.RUN}/verification"))
    args = ap.parse_args()
    results_dir, ver_dir = Path(args.results_dir), Path(args.verification_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ver_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[P {time.time() - t0:7.1f}s] {msg}", flush=True)

    layers = ["additional", "main"] if args.layers == "both" else [args.layers]
    rc = 0
    try:
        for layer in layers:
            for f in (f"inputs_manifest_{layer}.csv", f"array_map_{layer}.csv"):
                if (results_dir / f).exists():
                    raise K.InputError(f"refusing to overwrite existing {results_dir / f}")
            ident_name = "proposed_fixed_identity.json" if layer == "additional" else f"proposed_fixed_identity_{layer}.json"
            if (ver_dir / ident_name).exists():
                raise K.InputError(f"refusing to overwrite existing {ver_dir / ident_name}")
            if layer == "additional" and (results_dir / "r1_additional_array_spec.txt").exists():
                raise K.InputError("refusing to overwrite existing r1_additional_array_spec.txt")
        train_rows, _ = build_train_manifest(results_dir, ver_dir, log)
        build_r3_inputs(train_rows, ver_dir, log)
        for layer in layers:
            log(f"=== layer {layer} ===")
            rows, ident, meta, template, R = build_layer(layer, log)
            ident_name = "proposed_fixed_identity.json" if layer == "additional" else f"proposed_fixed_identity_{layer}.json"
            # ---- decision of P2-A / P2-B and FAIL validation (if sound real data FAIL, stop without changing the threshold) ----
            p2_ok = None
            if ident is not None:
                wit = proposed_fail_validation(layer, template, R, log)
                p2_ok = bool(ident["n_fail_p2a"] == 0 and ident["n_fail_p2b"] == 0)
                doc = {
                    "layer": layer, "ac": "check P2-A / check P2-B", "threshold_A": 1e-4,
                    "R_applied_to": "Cartesian (r_B[i] - R @ r_A[i]. Same as reconstruct_fixed_b_cart of the generate script. Not applied to fractional coordinates)",
                    "R_matrix": R.tolist(),
                    "template": meta["template"],
                    "data": ident, "p2_pass": p2_ok,
                    "fail_validation": wit,
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                (ver_dir / ident_name).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
                log(f"wrote {ver_dir / ident_name}: n={ident['n_structures']} max_p2a={ident['max_dev_p2a_A']:.3e} "
                    f"max_p2b={ident['max_dev_p2b_A']:.3e} pass={p2_ok} fail_validation_detected={wit['all_fail_validationes_detected']}")
                if not p2_ok:
                    log("P2 FAILED on healthy data: stopping without changing threshold/formula (r2-2). No manifest written.")
                    rc = 3
                    break
                if not wit["all_fail_validationes_detected"] or not wit["healthy_passes"]:
                    raise K.InputError("P2 FAIL validation did not behave as expected (see " + str(ver_dir / ident_name) + ")")
            # ---- check P: check of the manifest and FAIL validation ----
            chk = validate_manifest(rows, layer)
            try:
                validate_manifest(rows[:-1], layer)
                gp_validation = {"dropped_one_row_rejected": False}
            except K.InputError as e:
                gp_validation = {"dropped_one_row_rejected": True, "message": str(e), "rows_after_drop": len(rows) - 1}
            if not gp_validation["dropped_one_row_rejected"]:
                raise K.InputError("CP FAIL validation failed: manifest with one row dropped was accepted")
            K.write_csv_rows(results_dir / f"inputs_manifest_{layer}.csv", MANIFEST_FIELDS, rows)
            amap = K.array_map_rows(layer)
            K.write_csv_rows(results_dir / f"array_map_{layer}.csv", ["array_index", "layer", "arm", "w", "chunk"], amap)
            info = {"layer": layer, "manifest": chk, "g122p_validation_drop_one_row": gp_validation, "array_map_rows": len(amap)}
            if layer == "additional":
                r0 = [m["array_index"] for m in amap if m["w"] == "3.0" and m["chunk"] == 0]
                if len(r0) != 2:
                    raise K.InputError(f"R0 indices for additional expected 2 (proposed w3.0 chunk0, baseline w3.0 chunk0), got {r0}")
                r1 = [m["array_index"] for m in amap if m["array_index"] not in r0]
                if len(r1) != 110:
                    raise K.InputError(f"R1-additional indices {len(r1)} != 110")
                spec = K.sbatch_range_spec(r1)
                (results_dir / "r1_additional_array_spec.txt").write_text(spec + "\n")
                info.update({"r0_array_indices": r0, "r1_additional_array_spec": spec})
            info.update(meta)
            info["proposed_p2_pass"] = p2_ok
            (ver_dir / f"P_summary_{layer}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
            log(f"wrote inputs_manifest_{layer}.csv ({len(rows)} rows), array_map_{layer}.csv ({len(amap)} rows), P_summary_{layer}.json")
        if rc == 0:
            log("P DONE")
    except (K.InputError, FileExistsError) as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)
    sys.exit(rc)


if __name__ == "__main__":
    main()
