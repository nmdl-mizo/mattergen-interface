#!/usr/bin/env python3
"""Checks of the acceptance conditions (stages A, B). Prints PASS/FAIL, and exits with non-zero (1) if there is any FAIL.

  python check_data_rebuild.py --stage A [--cnids cnid_0_0] [--results-dir DIR] [--structures-dir DIR]
                                  [--log-dir DIR] [--log-tag TAG]
  python check_ac.py --stage B [same as above] [--no-recompute-audit]
  python check_ac.py --stage C [same as above] [--limit N]
  python check_data_rebuild.py --stage D1 [--results-dir DIR] [--inputs-dir DIR] [--window-ev W]   # CHECK6, CHECK7 (gates 3, 4, 5). The default of r2 is no window (gate 5 is for reference)
  python check_data_rebuild.py --stage D2 [--results-dir DIR] [--no-reference-check] [--window-ev W]  # check-8
  python check_data_rebuild.py --stage E [--results-dir DIR] [--expect-n-templates 31] [--expect-counts a,b,c]
                        (--regression-json dataset_regression.json | --no-regression-check)   # check-9, check-10

  stage A: CHECK1 (template), CHECK2 (dx structure), CHECK3 (refit of the symmetry operation, gate 1)
  stage B: CHECK4 (removal, gate 2)
  stage C: CHECK5 (relaxation). --limit N is for a smoke test of the first N structures (the completion marker is *.SMOKE_DONE, and the agreement of the total is not checked).

--cnids: narrows the range of the check (for a smoke test). CHECK1 always checks all 31 rows of template_map.csv.
The output of each AC is also saved in <log-dir>/AC-<n><log-tag>.log (the default log-dir is RUN/verification).
The last line prints `RESULT AC-1=PASS ... failed=[...]` (for matching against the expectation of the validation).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "lib"))  # structure_builder/lib
import builder_common as C
import removal_builder as RB
from sym_common import audit_symmetry

# CHECK2: the structure with dx=0 is identical to the template (maximum atomic displacement 0). Writing and reading the POSCAR in decimal gives 5e-17 A by rounding
# (measured in a smoke test. Below machine precision). An exact 0 cannot be achieved because of I/O rounding, so 1e-12 A is allowed [author's choice].
DX0_DISP_TOL = 1e-12
OP_RES_TOL = 1e-6           # check-3
ASBUILT_TOL = 1e-3          # check-4
CONV_MIN = 0.98             # check-5 / C3
RELAX_TOL = 1e-9            # CHECK5 cell and fixed atoms


class AC:
    def __init__(self, n: int, log_dir: Path, tag: str, title: str):
        self.n = n
        self.checks: list[tuple[str, bool]] = []
        log_dir.mkdir(parents=True, exist_ok=True)
        self.f = open(log_dir / f"check-{n}{tag}.log", "w")
        self.line(f"=== check-{n} {title} ===")

    def line(self, s: str):
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def check(self, name: str, ok: bool, detail: str):
        self.checks.append((name, bool(ok)))
        self.line(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    def info(self, s: str):
        self.line(f"  [INFO] {s}")

    @property
    def passed(self) -> bool:
        return all(ok for _, ok in self.checks)

    def failed_names(self):
        return [n for n, ok in self.checks if not ok]

    def close(self):
        self.line(f"VERDICT check-{self.n} {'PASS' if self.passed else 'FAIL'}")
        self.f.close()


def check1(tm, ac: AC):
    ac.check("n_rows==31", len(tm) == C.N_TEMPLATES, f"rows={len(tm)}")
    cn = [r["cnid"] for r in tm]
    ac.check("cnid_unique", len(set(cn)) == len(cn), f"unique={len(set(cn))}/{len(cn)}")
    r00 = [r for r in tm if r["cnid"] == "cnid_0_0"]
    ac.check("has_cnid_0_0_at_(0,0)", len(r00) == 1 and (int(r00[0]["i"]), int(r00[0]["j"])) == (0, 0),
             f"{[(r['i'], r['j']) for r in r00]}")
    bad_atoms, max_cell, max_len = [], 0.0, 0.0
    ref = None
    for r in tm:
        at = ase_read(r["src_poscar"], format="vasp")
        if len(at) != C.N_ATOMS or set(at.get_chemical_symbols()) != {"Al"}:
            bad_atoms.append((r["cnid"], len(at), sorted(set(at.get_chemical_symbols()))))
        cell = np.array(at.get_cell())
        if ref is None:
            ref = cell
        max_cell = max(max_cell, float(np.abs(cell - ref).max()))
        L = at.cell.lengths()
        for key, v in zip(("a_A", "b_A", "c_A"), L):
            max_len = max(max_len, abs(float(r[key]) - float(v)))
    ac.check("all_120_atoms_all_Al", not bad_atoms, f"violations={bad_atoms[:3]}")
    ac.check("cell_identical_31", max_cell <= C.CELL_TOL, f"max|cell-cell_ref|={max_cell:.3e} A (tol {C.CELL_TOL:.0e}), "
             f"re-read from src_poscar")
    ac.check("template_map_lengths_match_src", max_len <= C.CELL_TOL, f"max|a,b,c(csv)-src|={max_len:.3e} A")
    ne = [r["cnid"] + ":" + r["matches_existing"] for r in tm if r["matches_existing"]]
    ac.info(f"matches_existing non-empty rows = {len(ne)} (expected 3: cnid_0_0, cnid_0_5, cnid_5_0) {ne}")


def check2(tm, sel, struct_root: Path, ac: AC):
    n_expect = len(sel) * len(C.DX10)
    rows = {r["cnid"]: r for r in tm}
    n_found, max_da, max_disp0, max_other = 0, 0.0, 0.0, 0.0
    missing, extra = [], []
    for cnid in sel:
        tpl = ase_read(rows[cnid]["src_poscar"], format="vasp")
        tcell = np.array(tpl.get_cell())
        for dx_idx, dx in enumerate(C.DX10):
            p = C.struct_dir(cnid, dx, 0, C.seed_of(dx_idx, 0, 0), struct_root) / "POSCAR"
            if not p.exists():
                missing.append(str(p))
                continue
            at = ase_read(str(p), format="vasp")
            n_found += 1
            max_da = max(max_da, abs(float(at.cell[0, 0]) - (C.A_BASE + 2.0 * dx)))
            oc = np.array(at.get_cell()) - tcell
            oc[0, 0] = 0.0
            max_other = max(max_other, float(np.abs(oc).max()))
            if dx == 0.0:
                max_disp0 = max(max_disp0, float(np.abs(at.get_positions() - tpl.get_positions()).max()))
        tagdirs = sorted((Path(struct_root) / cnid).glob("dx_*/k0_s*"))
        if len(tagdirs) != len(C.DX10):
            extra.append((cnid, len(tagdirs)))
    ac.check("n_structures", n_found == n_expect and not missing and not extra,
             f"found={n_found} expected={n_expect} missing={len(missing)} dirs_per_cnid_mismatch={extra[:3]}")
    ac.check("cell_a_formula", max_da <= C.CELL_TOL and n_found == n_expect,
             f"max|a-(54.15601152562611+2dx)|={max_da:.3e} A (tol {C.CELL_TOL:.0e})")
    ac.check("cell_other_components_unchanged", max_other <= C.CELL_TOL, f"max={max_other:.3e} A")
    ac.check("dx0_identical_to_template", max_disp0 <= DX0_DISP_TOL, f"max atomic displacement={max_disp0:.3e} A "
             f"(tol {DX0_DISP_TOL:.0e})")
    a_m075 = C.A_BASE + 2.0 * (-0.75)
    ac.info(f"a(dx=-0.75) expected {a_m075!r} (expected 52.65601152562611)")


def recompute_residual(atoms) -> float:
    res = audit_symmetry(atoms, normal_axis=0, use_v2_candidates=True, force_r_name=RB.FORCE_R_NAME)["max_residual_A"]
    if res >= ASBUILT_TOL:   # the same as removal_builder: if the quick check is not enough, search all 8 candidates
        res = audit_symmetry(atoms, normal_axis=0, use_v2_candidates=True)["max_residual_A"]
    return float(res)


def check3(sel, res_dir: Path, struct_root: Path, ac: AC):
    n_expect = len(sel) * len(C.DX10)
    n_found, max_json, max_re, max_gap = 0, 0.0, 0.0, 0.0
    missing, over = [], []
    for cnid in sel:
        for dx_idx, dx in enumerate(C.DX10):
            jp = Path(res_dir) / "template_bulk" / f"{cnid}_{C.dx_tag(dx)}.json"
            if not jp.exists():
                missing.append(jp.name)
                continue
            n_found += 1
            with open(jp) as f:
                d = json.load(f)
            rj = float(d["op_residual_A"])
            max_json = max(max_json, rj)
            if not (rj < OP_RES_TOL):
                over.append((jp.name, rj))
            at = ase_read(str(C.struct_dir(cnid, dx, 0, C.seed_of(dx_idx, 0, 0), struct_root) / "POSCAR"),
                          format="vasp")
            rr = float(audit_symmetry(at, normal_axis=0, use_v2_candidates=True,
                                      force_r_name=RB.FORCE_R_NAME)["max_residual_A"])
            max_re = max(max_re, rr)
            max_gap = max(max_gap, abs(rr - rj))
            if not (rr < OP_RES_TOL):
                over.append((jp.name + ":recomputed", rr))
    ac.check("n_fits", n_found == n_expect, f"json found={n_found} expected={n_expect} missing={missing[:3]}")
    ac.check("op_residual_json_lt_1e-6", not any(not n.endswith(":recomputed") for n, _ in over) and n_found > 0,
             f"max(json)={max_json:.3e} A  (tol {OP_RES_TOL:.0e})")
    ac.check("op_residual_recomputed_lt_1e-6", not any(n.endswith(":recomputed") for n, _ in over) and n_found > 0,
             f"max(recomputed from POSCAR)={max_re:.3e} A  over={over[:3]}")
    ac.check("json_matches_recomputed", max_gap <= 1e-12, f"max|json-recomputed|={max_gap:.3e} A (tol 1e-12)")


def check4(sel, res_dir: Path, struct_root: Path, recompute: bool, ac: AC):
    res_dir = Path(res_dir)
    with open(res_dir / "stage_b_summary.json") as f:
        S = json.load(f)
    expected = len(sel) * len(C.DX10) * C.STRUCT_PER_TEMPLATE_DX
    cap = math.floor(0.01 * expected)
    ac.check("summary_cnids_match_selection", sorted(S["cnids"]) == sorted(sel), f"summary={len(S['cnids'])} sel={len(sel)}")
    ac.check("summary_total", S["expected_total"] == expected and S["n_structures"] + S["deficit_total"] == expected,
             f"n_structures({S['n_structures']}) + deficit({S['deficit_total']}) = "
             f"{S['n_structures'] + S['deficit_total']} (expected {expected})")
    ac.check("deficit_cap", S["deficit_total"] <= cap, f"deficit={S['deficit_total']} <= {cap} "
             f"(=floor(1% of {expected}); 65 for all 31 templates)")
    with open(res_dir / "bulk_membership.csv", newline="") as f:
        bm = list(csv.DictReader(f))
    ac.check("bm_row_count", len(bm) == S["n_structures"], f"rows={len(bm)} n_structures={S['n_structures']}")
    keys = Counter((r["cnid"], r["k"], r["seed"]) for r in bm)
    dups = [k for k, v in keys.items() if v > 1]
    ac.check("bm_key_unique", not dups, f"(cnid,k,seed) duplicates={len(dups)} {dups[:2]}")
    bad_nf = [(r["cnid"], r["k"], r["seed"], r["n_frozen"], r["status"]) for r in bm
              if not (r["n_frozen"] == str(C.N_FROZEN) and r["status"] == "ok")]
    ac.check("bm_n_frozen_76_status_ok", not bad_nf, f"violations={len(bad_nf)} {bad_nf[:2]}")
    bad_idx, bad_nat = [], []
    for r in bm:
        k, nat = int(r["k"]), int(r["natoms"])
        if nat != C.N_ATOMS - 2 * k:
            bad_nat.append((r["cnid"], r["k"], r["seed"], nat))
        if r["frozen_indices"]:
            fi = [int(x) for x in r["frozen_indices"].split(";")]
            if len(fi) != C.N_FROZEN or len(set(fi)) != len(fi) or min(fi) < 0 or max(fi) >= nat:
                bad_idx.append((r["cnid"], r["k"], r["seed"], len(fi)))
        else:
            bad_idx.append((r["cnid"], r["k"], r["seed"], "empty"))
    ac.check("bm_frozen_indices_valid", not bad_idx, f"violations={len(bad_idx)} {bad_idx[:2]}")
    ac.check("bm_natoms_120-2k", not bad_nat, f"violations={len(bad_nat)} {bad_nat[:2]}")
    # existence of the structure files
    n_poscar = sum(1 for cnid in sel for _ in (Path(struct_root) / cnid).glob("dx_*/k*_s*/POSCAR"))
    ac.check("structure_files", n_poscar == S["n_structures"], f"POSCAR files={n_poscar} n_structures={S['n_structures']}")
    # relax_inputs
    n_ri, ri_bad = 0, []
    bm_keys = {(r["cnid"], int(r["k"]), int(r["seed"])) for r in bm}
    for cnid in sel:
        with open(res_dir / "relax_inputs" / f"{cnid}.json") as f:
            ri = json.load(f)
        n_ri += len(ri)
        for key, v in ri.items():
            if (v["cnid"], v["k"], v["seed"]) not in bm_keys or len(v["frozen_indices"]) != C.N_FROZEN:
                ri_bad.append(key)
    ac.check("relax_inputs", n_ri == S["n_structures"] and not ri_bad, f"keys={n_ri} bad={len(ri_bad)} {ri_bad[:2]}")
    # number of successes per k: cross-check of summary and bulk_membership
    bm_by_k = Counter(r["k"] for r in bm)
    s_by_k = {k: v for k, v in S["success_by_k"].items()}
    ac.check("per_k_counts_match", all(bm_by_k.get(k, 0) == v for k, v in s_by_k.items()),
             f"summary={s_by_k} bm={dict(bm_by_k)}")
    # symmetry audit residual of the removal log
    n_log_ok, max_log, over = 0, 0.0, []
    for cnid in sel:
        for dx in C.DX10:
            lp = res_dir / "removal_log" / f"{cnid}_{C.dx_tag(dx)}.jsonl"
            with open(lp) as f:
                for ln in f:
                    r = json.loads(ln)
                    if r["ok"]:
                        n_log_ok += 1
                        if r["k"] >= 1:
                            v = r["asbuilt_residual_A"]
                            max_log = max(max_log, v)
                            if not (v < ASBUILT_TOL):
                                over.append((cnid, r["k"], r["seed"], v))
    ac.check("removal_log_count", n_log_ok == S["n_structures"], f"ok rows={n_log_ok} n_structures={S['n_structures']}")
    ac.check("asbuilt_residual_log_lt_1e-3", not over, f"max(log)={max_log:.3e} A  over={over[:2]}")
    if recompute:
        max_re, over_re = 0.0, []
        for r in bm:
            if int(r["k"]) == 0:
                continue
            at = ase_read(str(C.struct_dir(r["cnid"], C.DX10[(int(r["seed"]) - 10000) // 1000], int(r["k"]),
                                           int(r["seed"]), struct_root) / "POSCAR"), format="vasp")
            v = recompute_residual(at)
            max_re = max(max_re, v)
            if not (v < ASBUILT_TOL):
                over_re.append((r["cnid"], r["k"], r["seed"], v))
        ac.check("asbuilt_residual_recomputed_lt_1e-3", not over_re,
                 f"max(recomputed from POSCAR)={max_re:.3e} A over={over_re[:2]}")
    ac.info(f"success_by_k={S['success_by_k']} deficit_by_k={S['deficit_by_k']} "
            f"RemovalSkip reasons={S['removal_skip_reason_counts']}")


def check5(sel, res_dir: Path, struct_root: Path, limit: int | None, ac: AC):
    """CHECK5. The input is results/relax_inputs/{cnid}.json (the parent of poscar_path is the structure directory)."""
    from ase.io import read
    suffix = "SMOKE_DONE" if limit else "DONE"
    done_dir = Path(res_dir) / "relax_done"
    done = [c for c in sel if (done_dir / f"{c}.{suffix}").exists()]
    ac.check("done_markers", len(done) == len(sel), f"{len(done)}/{len(sel)} *.{suffix} in {done_dir}")
    n_in = n_relaxed = n_conv = n_json = 0
    max_cell, max_frz = 0.0, 0.0
    miss, nat_bad, unconv, fm_bad, extra = [], [], [], [], []
    for cnid in sel:
        with open(Path(res_dir) / "relax_inputs" / f"{cnid}.json") as f:
            inputs = json.load(f)
        keys = list(inputs.keys())[:limit] if limit else list(inputs.keys())
        n_in += len(keys)
        n_rel_c = 0
        for key in keys:
            e = inputs[key]
            d = Path(e["poscar_path"]).parent
            pr, ej = d / "POSCAR_relaxed", d / "energy.json"
            if not (pr.exists() and ej.exists()):
                miss.append(key)
                continue
            n_rel_c += 1
            n_relaxed += 1
            with open(ej) as f:
                r = json.load(f)
            n_json += 1
            if r["natoms"] != e["natoms"]:
                nat_bad.append(key)
            if r["converged"] is True:
                n_conv += 1
                if not (r["fmax_final"] <= 0.02):
                    fm_bad.append((key, r["fmax_final"]))
            else:
                unconv.append((key, r.get("n_steps"), r.get("fmax_final")))
            a0 = read(e["poscar_path"], format="vasp")
            a1 = read(str(pr), format="vasp")
            max_cell = max(max_cell, float(np.abs(np.array(a1.get_cell()) - np.array(a0.get_cell())).max()))
            fi = e["frozen_indices"]
            df = a1.get_scaled_positions(wrap=False)[fi] - a0.get_scaled_positions(wrap=False)[fi]
            df -= np.round(df)
            max_frz = max(max_frz, float(np.linalg.norm(df @ np.array(a0.get_cell()), axis=1).max()))
        if not limit:
            n_all = len(list((Path(struct_root) / cnid).rglob("POSCAR_relaxed")))
            if n_all != len(keys):
                extra.append((cnid, n_all, len(keys)))
    ac.check("inputs_nonempty", n_in > 0, f"inputs={n_in}")
    ac.check("relaxed_count_eq_inputs", n_relaxed == n_in and not miss and not extra,
             f"POSCAR_relaxed+energy.json={n_relaxed} inputs={n_in} missing={len(miss)} {miss[:2]} "
             f"total_vs_inputs_mismatch={extra[:2]}")
    if not limit:
        S_path = Path(res_dir) / "stage_b_summary.json"
        if S_path.exists():
            S = json.load(open(S_path))
            if sorted(S["cnids"]) == sorted(sel):
                ac.check("inputs_eq_stage_b_structures", n_in == S["n_structures"],
                         f"inputs={n_in} stage_b n_structures={S['n_structures']}")
    frac = (n_conv / n_in) if n_in else 0.0
    ac.check("converged_ge_98pct", frac >= CONV_MIN and n_relaxed == n_in,
             f"converged={n_conv}/{n_in} ({100 * frac:.2f}%) (>= {100 * CONV_MIN:.0f}%) unconverged={unconv[:3]}")
    ac.check("converged_implies_fmax_le_0.02", not fm_bad, f"violations={len(fm_bad)} {fm_bad[:2]}")
    ac.check("energy_json_natoms", not nat_bad, f"violations={len(nat_bad)} {nat_bad[:2]}")
    ac.check("cell_unchanged", max_cell <= RELAX_TOL and n_relaxed > 0,
             f"max|cell_relaxed-cell_input|={max_cell:.3e} A (tol {RELAX_TOL:.0e})")
    ac.check("frozen76_displacement", max_frz <= RELAX_TOL and n_relaxed > 0,
             f"max displacement of the 76 frozen atoms={max_frz:.3e} A (tol {RELAX_TOL:.0e})")


# ---------------------------------------------------------------- stages D1 and D2 (r1)
GAMMA_TOL = 1e-12


def _read_csv(p: Path):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _brute_counts(rows, w=None):
    """Independent implementation (a double loop, not the sweep of stage_d_common). Group = (dx, natoms), only converged=True.
    w=None (no window): p_w=None, per = the number of all pairs C(n,2) of each group. w=number: p_w = the number of pairs within the window, per = the number of pairs within the window of each group."""
    g = {}
    for r in rows:
        if r["converged"] == "True":
            g.setdefault((float(r["dx_A"]), int(r["natoms"])), []).append(float(r["energy_eV"]))
    p_all = sum(len(v) * (len(v) - 1) // 2 for v in g.values())
    p_w = 0 if w is not None else None
    per = {}
    for key, v in g.items():
        c = sum(1 for a in range(len(v)) for b in range(a + 1, len(v)) if w is None or abs(v[a] - v[b]) <= w)
        per[key] = c
        if w is not None:
            p_w += c
    return p_all, p_w, per, {k: len(v) for k, v in g.items()}


def check6(res_dir: Path, inputs_dir: Path, ac: AC, w=None):
    """CHECK6: identity of the formula of gamma (+ number of rows, dir notation, gate 3). The formula is rewritten separately from stage3_chunk1_gates.py:50-51 and cross-checked."""
    import config as CFG
    rows = _read_csv(Path(res_dir) / "gamma_all.csv")
    cn = sorted({r["cnid"] for r in rows})
    n_in, keys_in = 0, set()
    for c in cn:
        with open(Path(inputs_dir) / f"{c}.json") as f:
            inp = json.load(f)
        n_in += len(inp)
        keys_in |= {(v["cnid"], v["k"], v["seed"]) for v in inp.values()}
    keys_rows = {(r["cnid"], int(r["k"]), int(r["seed"])) for r in rows}
    ac.check("rows_eq_inputs", len(rows) == n_in and keys_rows == keys_in,
             f"rows={len(rows)} inputs={n_in} cnids={len(cn)} key_sets_equal={keys_rows == keys_in}")
    ac.info(f"constants imported from {CFG.__file__}: E_BULK_PER_ATOM={CFG.E_BULK_PER_ATOM!r} "
            f"EV_PER_A2_TO_J_PER_M2={CFG.EV_PER_A2_TO_J_PER_M2!r}")
    pick = sorted({0, len(rows) // 2, len(rows) - 1})
    worst = 0.0
    for i in pick:
        r = rows[i]
        d = Path(r["dir"])
        info = json.load(open(d / "energy.json"))
        at = ase_read(str(d / "POSCAR_relaxed"), format="vasp")
        cell = np.array(at.get_cell())
        A = float(np.linalg.norm(np.cross(cell[1], cell[2])))
        N, E = info["natoms"], info["energy_eV"]
        g = (E - N * CFG.E_BULK_PER_ATOM) / (2.0 * A) * CFG.EV_PER_A2_TO_J_PER_M2
        diff = abs(g - float(r["gamma_Jm2"]))
        worst = max(worst, diff)
        ac.info(f"row {i} ({r['cnid']},k={r['k']},seed={r['seed']}): csv gamma={float(r['gamma_Jm2'])!r} "
                f"recomputed={g!r} |diff|={diff:.3e} natoms csv/json={r['natoms']}/{N} E csv/json="
                f"{float(r['energy_eV'])!r}/{E!r}")
        if int(r["natoms"]) != N or float(r["energy_eV"]) != E:
            worst = max(worst, 1.0)
    ac.check("gamma_formula_3rows", worst <= GAMMA_TOL, f"rows {pick}: max|diff|={worst:.3e} J/m2 (tol {GAMMA_TOL:.0e})")
    n_conv = sum(1 for r in rows if r["converged"] == "True")
    frac = n_conv / len(rows) if rows else 0.0
    ac.check("C3_converged_ge_98pct", frac >= 0.98, f"converged={n_conv}/{len(rows)} ({100 * frac:.2f}%)")
    sj = json.load(open(Path(res_dir) / "stage_d1_summary.json"))
    p_all, p_w, per, sizes = _brute_counts(rows, w)
    ac.check("summary_matches_csv", sj["n_rows"] == len(rows) and sj["n_converged"] == n_conv
             and sj["P_all"] == p_all and sj["P_W"] == p_w and sj["window_eV"] == w,
             f"summary n_rows/n_converged/P_all/P_W={sj['n_rows']}/{sj['n_converged']}/{sj['P_all']}/{sj['P_W']} "
             f"independent={len(rows)}/{n_conv}/{p_all}/{p_w}; window_eV summary/arg={sj['window_eV']}/{w}")


def check7(res_dir: Path, ac: AC, w=None):
    """CHECK7: matching cost and misses (gates 4 and 5)."""
    res_dir = Path(res_dir)
    rows = _read_csv(res_dir / "gamma_all.csv")
    pj = json.load(open(res_dir / "stage_d1_match_trial.json"))
    need = ["s_bar_sec", "p95_sec", "P_all", "P_W", "est_cpu_hours", "dE_max_fit_true_eV", "n_fit_true_pairs"]
    ac.check("json_has_fields", all(k in pj for k in need), f"missing={[k for k in need if k not in pj]}")
    p_all, p_w, per, sizes = _brute_counts(rows, w)
    ac.check("window_eV_recorded", pj["window_eV"] == w, f"json window_eV={pj['window_eV']} arg={w} (None = no window)")
    ac.check("P_all_P_W_independent", pj["P_all"] == p_all and pj["P_W"] == p_w,
             f"json P_all={pj['P_all']} P_W={pj['P_W']}; brute-force P_all={p_all} P_W={p_w}")
    pk_dx, pk_n = 0.0, C.N_ATOMS - 4
    n_pilot = sizes.get((pk_dx, pk_n), 0)
    ac.check("pilot_group_size", pj["n_group"] == n_pilot and pj["n_pairs_planned"] == n_pilot * (n_pilot - 1) // 2,
             f"json n_group={pj['n_group']} pairs={pj['n_pairs_planned']}; csv k=2,dx=0.0 converged rows={n_pilot}")
    ac.check("coverage_ok", pj["coverage_ok"] and pj["n_tasks_missing"] == 0 and pj["n_pair_errors"] == 0,
             f"coverage_ok={pj['coverage_ok']} tasks_missing={pj['n_tasks_missing']} pair_errors={pj['n_pair_errors']}")
    ch = []
    for p in sorted((res_dir / "match_trial").glob("chunk_*.csv")):
        ch += _read_csv(p)
    pairs = [r for r in ch if r["kind"] == "pair" and not r["error"]]
    secs = np.array([float(r["sec"]) for r in pairs])
    ok_s = len(secs) > 0 and abs(pj["s_bar_sec"] - secs.mean()) < 1e-9 and abs(pj["p95_sec"] - np.percentile(secs, 95)) < 1e-9
    ac.check("s_bar_p95_recomputed", ok_s, f"json s_bar={pj['s_bar_sec']} p95={pj['p95_sec']}; "
             f"recomputed s_bar={secs.mean() if len(secs) else None} p95={np.percentile(secs, 95) if len(secs) else None}")
    p_est = p_all if w is None else p_w          # no window: expect P_all. with window: P_W
    est = pj["s_bar_sec"] * p_est / 3600.0
    ac.check("est_cpu_hours_formula", abs(est - pj["est_cpu_hours"]) <= 1e-9 * max(1.0, est),
             f"s_bar*{'P_all' if w is None else 'P_W'}({p_est})/3600={est!r} json={pj['est_cpu_hours']!r}")
    rowmap = {i: r for i, r in enumerate(rows)}
    des = [abs(float(rowmap[int(r["row_i"])]["energy_eV"]) - float(rowmap[int(r["row_j"])]["energy_eV"]))
           for r in pairs if r["fit"] == "1"]
    de_max = max(des) if des else None
    ac.check("dE_max_recomputed", (de_max is None and pj["dE_max_fit_true_eV"] is None) or
             (de_max is not None and pj["dE_max_fit_true_eV"] is not None and abs(de_max - pj["dE_max_fit_true_eV"]) < 1e-12)
             and len(des) == pj["n_fit_true_pairs"], f"json={pj['dE_max_fit_true_eV']} recomputed={de_max} "
             f"n_fit_true json/recomputed={pj['n_fit_true_pairs']}/{len(des)}")
    ac.check("validation_self_5of5_fit_true", pj["validation_self_fit_true"] == "5/5", f"{pj['validation_self_fit_true']}")
    ac.check("validation_fit_false_observed", pj["validation_fit_false_observed"] is True, "fit=False pair observed among real pairs")
    ac.check("validation_natoms_mismatch_recorded", pj["validation_natoms_mismatch_outcome"] is not None,
             f"outcome={pj['validation_natoms_mismatch_outcome']}")
    ac.check("pilot_sha_matches_csv", pj["gamma_csv_sha256"] == _sha(res_dir / "gamma_all.csv"), "gamma_all csv sha256")
    ac.check("C4", pj["C4"]["verdict"] == "PASS", f"estimated {pj['est_cpu_hours']:.4f} CPU h (<= 100): {pj['C4']['verdict']}")
    if w is None:      # no window: gate 5 is a reference value (no verdict). Only check that the verdict is REFERENCE
        ac.check("C5_reference_only", pj["C5"]["verdict"] == "REFERENCE",
                 f"verdict={pj['C5']['verdict']} (reference) dE_max={pj['dE_max_fit_true_eV']} eV (reference value, no verdict); fit=True pairs={pj['n_fit_true_pairs']}")
    else:
        ac.check("C5", pj["C5"]["verdict"] == "PASS",
                 f"verdict={pj['C5']['verdict']} dE_max={pj['dE_max_fit_true_eV']} eV (<= 0.01); fit=True pairs={pj['n_fit_true_pairs']}")
    ac.info(f"s_bar={pj['s_bar_sec']:.4f}s p95={pj['p95_sec']:.4f}s P_all={pj['P_all']} P_W={pj['P_W']} "
            f"est={pj['est_cpu_hours']:.4f} CPU h")


def _sha(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check8(res_dir: Path, no_ref: bool, ac: AC, w=None):
    """CHECK8: deduplication."""
    res_dir = Path(res_dir)
    S = json.load(open(res_dir / "stage_d2_summary.json"))
    allr = _read_csv(res_dir / "gamma_all.csv")
    ded = _read_csv(res_dir / "gamma_dedup.csv")
    man = json.load(open(res_dir / "dedup_manifest.json"))
    rv = json.load(open(res_dir / "stage_d2_reverify_summary.json"))
    conv = [r for r in allr if r["converged"] == "True"]
    ac.check("exceptions_zero", S["n_errors"] == 0 and S["n_unresolved"] == 0 and S.get("status") != "FAIL",
             f"n_errors={S['n_errors']} n_unresolved={S['n_unresolved']} status={S.get('status')}")
    p_all, p_w, per, sizes = _brute_counts(allr, w)
    pk = (0.0, C.N_ATOMS - 4)
    n_p = sizes.get(pk, 0)
    base, base_name = (p_all, "P_all") if w is None else (p_w, "P_W")
    expect = base - per.get(pk, 0) + n_p * (n_p - 1) // 2     # the check group is replaced by all pairs (with no window the subtraction is 0 and expect=P_all)
    ac.check("window_eV_recorded", S["window_eV"] == w, f"summary window_eV={S['window_eV']} arg={w} (None = no window)")
    ac.check("total_pairs_eq_P_adjusted", S["n_pairs_total_expected"] == expect and S["n_pairs_read"] == expect,
             f"summary expected/read={S['n_pairs_total_expected']}/{S['n_pairs_read']}; independent "
             f"{base_name}({base}) - {base_name}(pilot group {per.get(pk, 0)}) + C({n_p},2)({n_p * (n_p - 1) // 2}) = {expect}")
    ac.check("rows_out_eq_conv_minus_removed", len(ded) == len(conv) - S["n_removed"] == S["n_rows_out"],
             f"dedup rows={len(ded)} converged_in={len(conv)} removed={S['n_removed']}")
    ac.check("dedup_all_converged", all(r["converged"] == "True" for r in ded), "all rows converged=True")
    k_all = {(r["cnid"], r["k"], r["seed"]): r for r in allr}
    ac.check("dedup_subset_of_all", all((r["cnid"], r["k"], r["seed"]) in k_all and k_all[(r["cnid"], r["k"], r["seed"])] == r
                                        for r in ded), "every dedup row identical to a gamma_all row")
    # the representative is the lexicographic minimum of (seed,k,cnid), and the removed rows are not in dedup
    kept = {(r["cnid"], r["k"], r["seed"]) for r in ded}
    bad_rep, bad_removed, n_rem = [], [], 0
    for c in man["components"]:
        mem = c["members"]
        rep = min(mem, key=lambda m: (m["seed"], m["k"], m["cnid"]))
        if (rep["cnid"], rep["k"], rep["seed"]) != (c["representative"]["cnid"], c["representative"]["k"], c["representative"]["seed"]):
            bad_rep.append(c["representative"])
        for m in mem:
            key = (m["cnid"], str(m["k"]), str(m["seed"]))
            is_rep = key == (rep["cnid"], str(rep["k"]), str(rep["seed"]))
            if is_rep != (key in kept):
                bad_removed.append(m)
            if not is_rep:
                n_rem += 1
    ac.check("representative_is_lexmin_seed_k_cnid", not bad_rep, f"violations={len(bad_rep)} {bad_rep[:2]}")
    ac.check("removed_rows_absent_kept_rows_present", not bad_removed and n_rem == S["n_removed"],
             f"violations={len(bad_removed)} removed_in_manifest={n_rem} summary_removed={S['n_removed']}")
    ref = [r for r in ded if r["cnid"] == "cnid_0_0" and r["k"] == "0" and r["converged"] == "True"
           and "/dx_p0.000/k0_s" in r["dir"]]
    if no_ref:
        ac.info(f"reference rows (cnid_0_0,k=0,dx_p0.000) = {len(ref)} (--no-reference-check: not enforced)")
    else:
        ac.check("reference_row_exactly_1", len(ref) == 1, f"rows={len(ref)}")
    # rerun check
    exp_v = sum(sum(1 for a in range(len(v)) for b in range(a + 1, len(v)) if w is None or abs(v[a] - v[b]) <= w)
                for v in _groups_energy(ded).values())
    ac.check("reverify_pairs_complete", rv["n_pairs_read"] == exp_v and rv["n_errors"] == 0 and rv.get("status") in ("PASS", "FAIL"),
             f"reverify pairs read={rv['n_pairs_read']} independent {'all-pair' if w is None else 'window'} count on dedup csv={exp_v} errors={rv['n_errors']}")
    ac.check("reverify_fit_edges_zero", rv.get("n_fit_edges", -1) == 0, f"n_fit_edges={rv.get('n_fit_edges')}")
    ac.info(f"n_fit_true={S['n_fit_true']} components(multi)={S['n_components_multi']} removed={S['n_removed']} "
            f"non_clique={S['n_non_clique_components']} dE_max_fit_true={S['dE_max_fit_true_eV']}")


def _groups_energy(rows):
    g = {}
    for r in rows:
        if r["converged"] == "True":
            g.setdefault((float(r["dx_A"]), int(r["natoms"])), []).append(float(r["energy_eV"]))
    return g


# ---------------------------------------------------------------- stage E (CHECK9 and CHECK10. The training smoke tests of CHECK11 and CHECK12 come after the production data)
E_STAT_TOL = 1e-12
E_SPLIT_SEED = 42
E_CHUNK_SEED_BASE = 20301000


def _e_expected_counts(n: int, explicit: str | None) -> tuple[int, int, int]:
    """Independent implementation (stage_e_common is not imported). n=31 is fixed to 25/3/3 as in the plan. Otherwise n_val=n_test=max(1, n//10)."""
    if explicit:
        a, b, c = (int(x) for x in explicit.split(","))
        return a, b, c
    if n == 31:
        return 25, 3, 3
    nv = max(1, n // 10)
    return n - 2 * nv, nv, nv


def check9(res_dir: Path, expect_n: int, expect_counts: str | None, regression_json: str | None, ac: AC):
    """CHECK9: split (per template, 25/3/3, mutually disjoint) + regression of dataset.py."""
    res_dir = Path(res_dir)
    sp = json.load(open(res_dir / "split.json"))
    rows = _read_csv(res_dir / "gamma_dedup.csv")
    tr, va, te = sp["templates_train"], sp["templates_val"], sp["templates_test"]
    exp = _e_expected_counts(expect_n, expect_counts)
    ac.check("template_counts", (len(tr), len(va), len(te)) == exp, f"train/val/test={len(tr)}/{len(va)}/{len(te)} expected {exp} (n_templates={expect_n})")
    union = set(tr) | set(va) | set(te)
    ac.check("templates_disjoint_union", len(union) == len(tr) + len(va) + len(te) == expect_n,
             f"|union|={len(union)} sum={len(tr) + len(va) + len(te)} expected {expect_n}")
    # independent re-derivation of the permutation (seed 42, names sorted in lexicographic order)
    names = sorted(union)
    perm = np.random.default_rng(E_SPLIT_SEED).permutation(len(names))
    order = [names[int(i)] for i in perm]
    rt = (sorted(order[:exp[0]]), sorted(order[exp[0]:exp[0] + exp[1]]), sorted(order[exp[0] + exp[1]:]))
    ac.check("permutation_seed42_reproduced", (sorted(tr), sorted(va), sorted(te)) == rt and sp.get("seed") == E_SPLIT_SEED,
             f"seed={sp.get('seed')} recorded templates equal default_rng(42).permutation({len(names)}) assignment")
    side_of = {c: "train" for c in tr} | {c: "val" for c in va} | {c: "test" for c in te}
    # agreement of the side of each row (independently from the whole CSV) with the row positions of the split JSON
    pos_side: dict[int, str] = {}
    for name, key in (("train", "train_indices"), ("val", "val_indices"), ("test", "test_indices")):
        for p in sp[key]:
            pos_side[p] = name
    row_sides: dict[str, set[str]] = {}
    for p, r in enumerate(rows):
        row_sides.setdefault(r["cnid"], set()).add(pos_side.get(p, "NONE"))
    multi = {c: sorted(s) for c, s in row_sides.items() if len(s) != 1}
    ac.check("one_side_per_cnid", not multi, f"cnids with rows on != 1 side: {multi}")
    wrong = {c: (sorted(s)[0], side_of.get(c)) for c, s in row_sides.items() if len(s) == 1 and sorted(s)[0] != side_of.get(c)}
    ac.check("rows_side_matches_template_side", not wrong, f"mismatches (index side, template side): {wrong}")
    n_rows = len(rows)
    lens = (len(sp["train_indices"]), len(sp["val_indices"]), len(sp["test_indices"]))
    ac.check("n_sum_eq_rows", sp["n_train"] + sp["n_val"] + sp["n_test"] == n_rows and lens == (sp["n_train"], sp["n_val"], sp["n_test"]),
             f"n_train+n_val+n_test={sp['n_train'] + sp['n_val'] + sp['n_test']} rows={n_rows} lens={lens}")
    allpos = sorted(sp["train_indices"] + sp["val_indices"] + sp["test_indices"])
    ac.check("indices_partition_rows", allpos == list(range(n_rows)), f"sorted union == range({n_rows}): {allpos == list(range(n_rows))}")
    ac.check("val_dx_indices_empty", sp.get("val_dx_indices") == [], f"val_dx_indices={sp.get('val_dx_indices')}")
    ac.info(f"cnid_0_0_side={sp.get('cnid_0_0_side')}")
    # regression of dataset.py (the checks that need torch are done by dataset_regression.py on another node, and its JSON is verified)
    if regression_json is None:
        ac.info("dataset regression: not checked (--no-regression-check)")
        return
    rj = json.load(open(regression_json))
    ac.check("regression_json_verdict_PASS", rj.get("verdict") == "PASS" and not rj.get("failed"), f"verdict={rj.get('verdict')} failed={rj.get('failed')}")
    lens_old = {k: (v["len_before"], v["len_after"], v["identical"]) for k, v in rj["r1"].items()}
    ac.check("old_json_3489_33_356_identical_before_after",
             lens_old == {"train": (3489, 3489, True), "val": (33, 33, True), "val_dx": (356, 356, True)}, f"{lens_old}")
    sha_now = _sha(res_dir / "split.json")
    ac.check("regression_used_this_split_json", rj.get("new_json_sha256") == sha_now,
             f"regression new_json_sha256={str(rj.get('new_json_sha256'))[:12]} this split.json sha256={sha_now[:12]}")
    ac.check("new_json_train_val_test_readable",
             all(rj["r2"].get(s, {}).get("equal_to_json") for s in ("train", "val", "test")) and
             [rj["r2"][s]["len"] for s in ("train", "val", "test")] == list(lens),
             f"r2={ {s: rj['r2'][s]['len'] for s in rj['r2']} }")


def _pct(vals_sorted, p):
    """Linear-interpolation percentile by an independent implementation (the same definition as the default of numpy.percentile)."""
    k = (len(vals_sorted) - 1) * p / 100.0
    f = math.floor(k)
    c = min(f + 1, len(vals_sorted) - 1)
    return vals_sorted[f] + (vals_sorted[c] - vals_sorted[f]) * (k - f)


def check10(res_dir: Path, ac: AC):
    """CHECK10: statistics and evaluation targets (train rows only)."""
    res_dir = Path(res_dir)
    sp = json.load(open(res_dir / "split.json"))
    rows = _read_csv(res_dir / "gamma_dedup.csv")
    train = [rows[p] for p in sp["train_indices"]]
    dx = [float(r["dx_A"]) for r in train]
    n = len(dx)
    mu = math.fsum(dx) / n
    s2 = math.fsum((x - mu) ** 2 for x in dx) / n          # ddof=0
    ds = json.load(open(res_dir / "dxvpsde_stats.json"))
    ac.check("dx_mu", abs(ds["mu_uncond"] - mu) <= E_STAT_TOL, f"json={ds['mu_uncond']!r} independent={mu!r} diff={abs(ds['mu_uncond'] - mu):.2e}")
    ac.check("dx_sigma2_ddof0", abs(ds["sigma2"] - s2) <= E_STAT_TOL, f"json={ds['sigma2']!r} independent={s2!r} diff={abs(ds['sigma2'] - s2):.2e}")
    ac.check("dx_n_rows_train_only", ds["n_rows"] == n == sp["n_train"], f"n_rows={ds['n_rows']} train rows={n}")
    all_dx = [float(r["dx_A"]) for r in rows]
    mu_all = math.fsum(all_dx) / len(all_dx)
    ac.info(f"mu(train)={mu!r} mu(all rows)={mu_all!r}")
    # evaluation targets
    g = sorted(float(r["gamma_Jm2"]) for r in train)
    q10, q90, gmin, gmax = _pct(g, 10), _pct(g, 90), g[0], g[-1]
    et = json.load(open(res_dir / "eval_targets.json"))
    t = et["targets"]
    ac.check("eval_q10_q90_min_max", all(abs(et[k] - v) <= E_STAT_TOL for k, v in (("q10", q10), ("q90", q90), ("min", gmin), ("max", gmax))),
             f"json q10/q90/min/max={et['q10']!r}/{et['q90']!r}/{et['min']!r}/{et['max']!r} independent={q10!r}/{q90!r}/{gmin!r}/{gmax!r}")
    delta = (q90 - q10) / 4.0
    exp7 = [q10 + i * delta for i in range(5)] + [gmin - delta, gmax + delta]
    ac.check("eval_7_values_formula", len(t) == 7 and all(abs(a - b) <= E_STAT_TOL for a, b in zip(t, exp7)),
             f"targets={t} expected={exp7}")
    inner = t[:5]
    steps = [inner[i + 1] - inner[i] for i in range(4)]
    ac.check("eval_inner_ascending_equispaced", all(s > 0 for s in steps) and max(steps) - min(steps) <= E_STAT_TOL and abs(abs(steps[0]) - delta) <= E_STAT_TOL,
             f"steps={steps} delta={delta!r}")
    ac.check("eval_extrapolation_outside", t[5] < inner[0] and t[6] > inner[4], f"extrap_low={t[5]!r} < inner0={inner[0]!r}; extrap_high={t[6]!r} > inner4={inner[4]!r}")
    ch = et["chunks"]
    ok_ch = len(ch) == 14 and all(c["chunk"] == i and c["target_index"] == i // 2 and c["seed"] == E_CHUNK_SEED_BASE + i
                                  and abs(c["target_gamma_Jm2"] - t[i // 2]) <= E_STAT_TOL for i, c in enumerate(ch))
    ac.check("eval_chunks_14_seed_20301000_plus_c", ok_ch, f"n_chunks={len(ch)} seeds={[c['seed'] for c in ch][:3]}...{[c['seed'] for c in ch][-1:]}")
    # distribution of the number of atoms
    nd = json.load(open(res_dir / "num_atoms_dist_train.json"))
    cnt = Counter(int(r["natoms"]) for r in train)
    ent = {e["n"]: (e["count"], e["p"]) for e in nd["entries"]}
    ok_nd = set(ent) == set(cnt) and all(ent[k][0] == cnt[k] and abs(ent[k][1] - cnt[k] / n) <= E_STAT_TOL for k in cnt) \
        and abs(sum(p for _, p in ent.values()) - 1.0) <= E_STAT_TOL and nd["n_train"] == n
    ac.check("natoms_dist_counts", ok_nd, f"counts={dict(sorted(cnt.items()))} (json entries {len(ent)})")
    ac.check("natoms_block_has_key", "AL_S5_210_TRAIN" in nd["num_atoms_distribution_py_block"], "block starts with the training-split key")
    # average density (n/volume computed independently with ase)
    ad = json.load(open(res_dir / "average_density.json"))
    dens = []
    for r in train:
        at = ase_read(str(Path(r["dir"]) / "POSCAR_relaxed"), format="vasp")
        dens.append(len(at) / at.get_volume())
    avg = math.fsum(dens) / len(dens)
    ac.check("average_density_recomputed", abs(ad["average_density_train_split"] - avg) <= E_STAT_TOL and ad["n_train"] == n,
             f"json={ad['average_density_train_split']!r} independent(ase)={avg!r}; card value used in configs={ad['card_value_used_in_configs']!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B", "C", "D1", "D2", "E"])
    ap.add_argument("--cnids", default="")
    ap.add_argument("--results-dir", default=str(C.RESULTS))
    ap.add_argument("--structures-dir", default=str(C.STRUCTURES))
    ap.add_argument("--log-dir", default=str(C.VERIFICATION))
    ap.add_argument("--log-tag", default="")
    ap.add_argument("--no-recompute-audit", action="store_true")
    ap.add_argument("--inputs-dir", default=str(C.RESULTS / "relax_inputs"))
    ap.add_argument("--window-ev", type=float, default=None, help="D1, D2: window of the matching (default: no window = all pairs. Make it the same as the value passed to the stage script)")
    ap.add_argument("--no-reference-check", action="store_true", help="D2: for small tests without reference rows (treated as INFO)")
    ap.add_argument("--limit", type=int, default=None, help="for the stage C smoke test: the first N structures of each template")
    ap.add_argument("--expect-n-templates", type=int, default=C.N_TEMPLATES, help="stage E: total number of templates (31 in production. 3 in a small test)")
    ap.add_argument("--expect-counts", default=None, help="stage E: state the numbers of templates 'train,val,test' explicitly (the default follows the rule: 31 -> 25,3,3)")
    ap.add_argument("--regression-json", default=None, help="stage E: output JSON of dataset_regression.py (the dataset.py regression of CHECK9)")
    ap.add_argument("--no-regression-check", action="store_true", help="stage E: treat the dataset regression check as INFO (for the validation)")
    args = ap.parse_args()
    res_dir, st_dir, log_dir = Path(args.results_dir), Path(args.structures_dir), Path(args.log_dir)
    tm = C.read_template_map(res_dir if (res_dir / "template_map.csv").exists() else C.RESULTS)
    sel = C.select_cnids([r["cnid"] for r in tm], args.cnids)
    acs = []
    if args.stage == "A":
        a = AC(1, log_dir, args.log_tag, "template"); check1(tm, a); a.close(); acs.append(a)
        a = AC(2, log_dir, args.log_tag, f"dx structure (cnids={len(sel)})"); check2(tm, sel, st_dir, a); a.close(); acs.append(a)
        a = AC(3, log_dir, args.log_tag, f"refit of the symmetry operation, gate 1 (cnids={len(sel)})")
        check3(sel, res_dir, st_dir, a); a.close(); acs.append(a)
    elif args.stage == "D1":
        a = AC(6, log_dir, args.log_tag, "identity of the formula of gamma (stage D1)"); check6(res_dir, Path(args.inputs_dir), a, args.window_ev); a.close(); acs.append(a)
        a = AC(7, log_dir, args.log_tag, "matching cost and misses, gates 4 and 5 (stage D1)"); check7(res_dir, a, args.window_ev); a.close(); acs.append(a)
    elif args.stage == "D2":
        a = AC(8, log_dir, args.log_tag, "deduplication (stage D2)"); check8(res_dir, args.no_reference_check, a, args.window_ev); a.close(); acs.append(a)
    elif args.stage == "E":
        if not args.no_regression_check and not args.regression_json:
            raise SystemExit("stage E: --regression-json is required (or --no-regression-check for a validation run)")
        reg = None if args.no_regression_check else args.regression_json
        a = AC(9, log_dir, args.log_tag, "split (stage E)"); check9(res_dir, args.expect_n_templates, args.expect_counts, reg, a); a.close(); acs.append(a)
        a = AC(10, log_dir, args.log_tag, "statistics and targets (stage E)"); check10(res_dir, a); a.close(); acs.append(a)
    elif args.stage == "C":
        a = AC(5, log_dir, args.log_tag, f"relaxation (cnids={len(sel)}, limit={args.limit})")
        check5(sel, res_dir, st_dir, args.limit, a); a.close(); acs.append(a)
    else:
        a = AC(4, log_dir, args.log_tag, f"removal, gate 2 (cnids={len(sel)})")
        check4(sel, res_dir, st_dir, not args.no_recompute_audit, a); a.close(); acs.append(a)
    failed = [f"{x.n}:{n}" for x in acs for n in x.failed_names()]
    print("RESULT " + " ".join(f"check-{x.n}={'PASS' if x.passed else 'FAIL'}" for x in acs) + f" failed={failed}")
    sys.exit(0 if all(x.passed for x in acs) else 1)


if __name__ == "__main__":
    main()
