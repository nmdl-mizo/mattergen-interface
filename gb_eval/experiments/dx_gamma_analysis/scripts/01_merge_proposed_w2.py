#!/usr/bin/python3
"""Makes the merged table merged_proposed_w2.csv (standard library only. Run: /usr/bin/python3 01_merge_proposed_w2.py).

Target: the generation of the main part (main experiment) = Al / CFG w=2.0 / constrained model proposed / 7 targets x 50 = 350 structures.
Join key: (chunk, cif_index). It checks that the key is unique in each input CSV and that the number of rows does not increase or decrease from 350,
and saves the result in results/merge_check.json (stops with an exception on failure).

Input (all read only):
  RELAX   relaxation results of 14 chunks: relax/main/proposed_w2.0/chunkNN/relax_rows.csv   (gamma_target, gamma_pre, gamma_relaxed, converged, energy_eV, fmax_final, natoms ...)
  MANIF   inputs_manifest_main.csv (arm=proposed)                               (cell_a_A, gb_dx_realized_A, min_dist_A [before relaxation], gamma_unrelaxed ...)
  SYMFIT  gb_equivalence/gb_symfit_main.csv (arm=proposed, stage=pre/post)      (cell_a_A, n_gb, gb0_status, gb_symfit_max_A ...)
  STAGE2  verification/results/gb_stage2/gb_stage2_main.csv (arm=proposed)      (equivalence of the local window of stage 2: s2_max_A ... after relaxation, two_gb is the target)
  TRAIN   data_rebuild/results/gamma_dedup.csv + split.json       (training dx, gamma, N: separate output train_ref.csv)
"""
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)
import csv
import glob
import json
import os
import sys

RUN = _gbrel.WORK + '/workspace/runs'
RELAX_GLOB = f"{RUN}/relax_novelty/results/relax/main/proposed_w2.0/chunk*/relax_rows.csv"
MANIF = f"{RUN}/relax_novelty/results/inputs_manifest_main.csv"
SYMFIT = f"{RUN}/relax_novelty/results/gb_equivalence/gb_symfit_main.csv"
STAGE2 = f"{RUN}/verification/results/gb_stage2/gb_stage2_main.csv"
TRAIN_CSV = f"{RUN}/data_rebuild/results/gamma_dedup.csv"
TRAIN_SPLIT = f"{RUN}/data_rebuild/results/split.json"
TARGETS_JSON = f"{RUN}/range_targets/results/eval_targets_range.json"

OUT = f"{RUN}/dx_gamma_analysis/results"
A_T = 54.15601152562611            # a_t of the reference template (the same value as dx_range.py and the audit appendix)
DX_LO, DX_HI = -0.75, 0.60         # training range (DSDXLO/DSDXHI of the numbers ledger)
NEAR_TOL = 0.10                    # manuscript P17 (common.py:79 NEAR_TOL_J)
E_BULK = -3.7569042674314765       # scripts/config/config.py:49
FACTOR = 16.0218                   # scripts/config/config.py:48 (eV/Å^2 -> J/m^2)
TARGET_LABELS = ["inner0", "inner1", "inner2", "inner3", "inner4", "extrap_low", "extrap_high"]


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fl(x):
    return float(x) if x not in ("", None) else None


def index_unique(rows, keyf, name, check):
    d = {}
    for r in rows:
        k = keyf(r)
        if k in d:
            raise SystemExit(f"FATAL: duplicate key {k} in {name}")
        d[k] = r
    check[f"{name}_rows"] = len(rows)
    check[f"{name}_unique_keys"] = len(d)
    return d


def main():
    check = {}
    # --- RELAX ---
    files = sorted(glob.glob(RELAX_GLOB))
    check["relax_files"] = len(files)
    relax = []
    for fp in files:
        rr = read(fp)
        check[f"relax_file_rows::{os.path.basename(os.path.dirname(fp))}"] = len(rr)
        relax.extend(rr)
    for r in relax:
        if not (r["layer"] == "main" and r["arm"] == "proposed" and float(r["w"]) == 2.0):
            raise SystemExit(f"FATAL: unexpected row in relax_rows: {r['layer']},{r['arm']},{r['w']}")
    relax_d = index_unique(relax, lambda r: (int(r["chunk"]), int(r["cif_index"])), "relax", check)
    keys = sorted(relax_d)
    if len(keys) != 350:
        raise SystemExit(f"FATAL: relax rows {len(keys)} != 350")

    # --- MANIFEST (proposed only) ---
    man = [r for r in read(MANIF) if r["arm"] == "proposed" and r["layer"] == "main"]
    man_d = index_unique(man, lambda r: (int(r["chunk"]), int(r["cif_index"])), "manifest_proposed", check)
    # --- SYMFIT ---
    sy = [r for r in read(SYMFIT) if r["arm"] == "proposed"]
    sy_pre = index_unique([r for r in sy if r["stage"] == "pre"], lambda r: (int(r["chunk"]), int(r["cif_index"])), "symfit_proposed_pre", check)
    sy_post = index_unique([r for r in sy if r["stage"] == "post"], lambda r: (int(r["chunk"]), int(r["cif_index"])), "symfit_proposed_post", check)
    # --- STAGE2 ---
    s2 = [r for r in read(STAGE2) if r["arm"] == "proposed" and r["layer"] == "main"]
    s2_d = index_unique(s2, lambda r: (int(r["chunk"]), int(r["cif_index"])), "stage2_proposed", check)

    for name, d in (("manifest", man_d), ("symfit_pre", sy_pre), ("symfit_post", sy_post), ("stage2", s2_d)):
        missing = [k for k in keys if k not in d]
        extra = [k for k in d if k not in relax_d]
        check[f"{name}_missing_vs_relax"] = len(missing)
        check[f"{name}_extra_vs_relax"] = len(extra)
        if missing or extra:
            raise SystemExit(f"FATAL: key mismatch {name}: missing {len(missing)} extra {len(extra)}")

    tj = json.load(open(TARGETS_JSON))
    tg = tj["targets"]
    check["targets_json_labels"] = tj["labels"]

    out_rows = []
    maxdiff = {"gamma_target": 0.0, "gamma_pre": 0.0, "natoms": 0, "cell_a_manif_vs_symfit": 0.0, "dx_cell_vs_manifest": 0.0,
               "area_manif_vs_relax": 0.0, "target_vs_json": 0.0}
    for k in keys:
        r, m, a, b, s = relax_d[k], man_d[k], sy_pre[k], sy_post[k], s2_d[k]
        tid = int(r["target_id"])
        if int(m["target_id"]) != tid or tid != k[0] // 2:
            raise SystemExit(f"FATAL: target_id mismatch at {k}")
        gt = float(r["gamma_target"])
        maxdiff["gamma_target"] = max(maxdiff["gamma_target"], abs(gt - float(m["gamma_target"])))
        maxdiff["target_vs_json"] = max(maxdiff["target_vs_json"], abs(gt - tg[tid]))
        maxdiff["gamma_pre"] = max(maxdiff["gamma_pre"], abs(float(r["gamma_pre"]) - float(m["gamma_unrelaxed"])))
        n = int(r["natoms"])
        if not (n == int(m["natoms"]) == int(a["natoms"]) == int(b["natoms"]) == int(s["natoms"])):
            maxdiff["natoms"] += 1
        cell_a = float(m["cell_a_A"])
        maxdiff["cell_a_manif_vs_symfit"] = max(maxdiff["cell_a_manif_vs_symfit"], abs(cell_a - float(a["cell_a_A"])))
        dx = (cell_a - A_T) / 2.0
        dx_m = float(m["gb_dx_realized_A"])
        maxdiff["dx_cell_vs_manifest"] = max(maxdiff["dx_cell_vs_manifest"], abs(dx - dx_m))
        maxdiff["area_manif_vs_relax"] = max(maxdiff["area_manif_vs_relax"], abs(float(m["area_A2"]) - float(r["area_A2"])))
        gam_rel = fl(r["gamma_relaxed"])
        gam_pre = float(r["gamma_pre"])
        conv = r["converged"] == "True"
        area = float(r["area_A2"])
        energy = fl(r["energy_eV"])
        dg = gam_rel - gt if gam_rel is not None else None
        e_pre = gam_pre * 2.0 * area / FACTOR + n * E_BULK          # derived value: back-calculated from the definition of gamma_pre (the output has no column of the energy before relaxation)
        side = "below" if dx < DX_LO else ("above" if dx > DX_HI else "inside")
        out_rows.append({
            "struct_id": f"main_proposed_w2.0_c{k[0]:02d}_i{k[1]:02d}",
            "chunk": k[0], "cif_index": k[1], "target_id": tid, "target_label": TARGET_LABELS[tid],
            "is_inner": r["is_inner"], "gamma_target_Jm2": repr(gt),
            "natoms": n, "area_A2": repr(area),
            "cell_a_A": repr(cell_a), "cell_b_A": m["cell_b_A"], "cell_c_A": m["cell_c_A"], "dx_A": repr(dx), "dx_manifest_A": repr(dx_m),
            "dx_side": side, "dx_in_train_range": int(side == "inside"),
            "gamma_pre_Jm2": repr(gam_pre), "gamma_relaxed_Jm2": repr(gam_rel) if gam_rel is not None else "",
            "dgamma_Jm2": repr(dg) if dg is not None else "",
            "abs_dgamma_Jm2": repr(abs(dg)) if dg is not None else "",
            "converged": int(conv), "status": r["status"],
            "near_P17": int(conv and gam_rel is not None and abs(dg) <= NEAR_TOL),
            "near_0p10_flag_eq_P17": int(conv and gam_rel is not None and abs(dg) <= 0.10),
            "n_steps": r["n_steps"], "fmax_final_eVA": r["fmax_final"],
            "energy_relaxed_eV": r["energy_eV"],
            "energy_pre_eV_derived": repr(e_pre),
            "dE_relax_eV_derived": repr(energy - e_pre) if energy is not None else "",
            "dgamma_relax_Jm2": repr(gam_rel - gam_pre) if gam_rel is not None else "",
            "min_dist_pre_A": m["min_dist_A"], "min_dist_relaxed_A": r["min_dist_relaxed_A"],
            "max_disp_free_A": r["max_disp_free_A"], "max_disp_fixed_A": r["max_disp_fixed_A"], "n_fixed": r["n_fixed"],
            "non_al_count_pre": m["non_al_count"], "excluded_pre": m["excluded_pre"],
            "n_gb_pre": a["n_gb"], "gb0_status_pre": a["gb0_status"], "n_gb_post": b["n_gb"], "gb0_status_post": b["gb0_status"],
            "symfit_op_post": b["op_name"], "symfit_max_A_pre": a["gb_symfit_max_A"], "symfit_max_A_post": b["gb_symfit_max_A"],
            "stage2_status": s["status"], "stage2_s2_max_A": s["s2_max_A"], "stage2_s2_op": s["s2_op"],
            "stage2_eq12_0.1": s["eq12_0.1"], "stage2_eq12_0.032": s["eq12_0.032"],
        })
    check["maxdiff_cross_source"] = maxdiff
    check["n_merged"] = len(out_rows)
    check["join_policy"] = "inner join based on relax_rows (350). Missing 0, surplus 0 and duplicates 0 in all sources have been checked (an exception otherwise)"
    check["A_T"] = A_T
    check["dx_range"] = [DX_LO, DX_HI]
    check["near_tol_J"] = NEAR_TOL
    if len(out_rows) != 350 or maxdiff["natoms"] != 0:
        raise SystemExit("FATAL: merge validation failed")
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/merged_proposed_w2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # --- training reference ---
    train_all = read(TRAIN_CSV)
    split = json.load(open(TRAIN_SPLIT))
    tr_idx = split["train_indices"]
    check["train_csv_rows"] = len(train_all)
    check["train_split_keys"] = sorted(split.keys())
    check["train_n"] = len(tr_idx)
    with open(f"{OUT}/train_ref.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_pos", "split", "cnid", "k", "natoms", "area_A2", "energy_eV", "gamma_Jm2", "dx_A"])
        sets = {}
        for sp in ("train_indices", "val_indices", "test_indices"):
            for i in split.get(sp, []):
                sets[i] = sp.replace("_indices", "")
        for i, r in enumerate(train_all):
            w.writerow([i, sets.get(i, "none"), r["cnid"], r["k"], r["natoms"], r["area_A2"], r["energy_eV"], r["gamma_Jm2"], r["dx_A"]])
    from collections import Counter
    lv = Counter(round(float(train_all[i]["dx_A"]), 2) for i in tr_idx)
    check["train_dx_level_counts_recomputed"] = {str(k): v for k, v in sorted(lv.items())}
    check["train_dx_min_max"] = [min(float(train_all[i]["dx_A"]) for i in tr_idx), max(float(train_all[i]["dx_A"]) for i in tr_idx)]
    gs = [float(train_all[i]["gamma_Jm2"]) for i in tr_idx]
    check["train_gamma_min_max"] = [min(gs), max(gs)]
    with open(f"{OUT}/merge_check.json", "w", encoding="utf-8") as f:
        json.dump(check, f, ensure_ascii=False, indent=1)
    print(json.dumps(check, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
