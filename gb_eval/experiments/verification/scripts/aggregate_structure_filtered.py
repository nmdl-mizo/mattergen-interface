#!/usr/bin/env python3
"""aggregate_structure_filtered.py -- metrics restricted to valid structures (two grain boundaries, equivalent).

Plan: see the experiment description (later differences take precedence).
Method: imports the existing aggregate_relax_novelty.py (AG) and, after AG.load_inputs, with apply_filter(records, X)
  replaces converged := converged AND X_post and valid_pre := valid_pre AND X_pre, and passes AG.compute_components, AG.compute_all and AG.differences **unchanged**.
  The existing scripts and outputs are not changed. Formulas and settings are not copied.
Predicates: G2_stage = status of gb_symfit_main.csv in {fitted,no_gb,multi_gb,disordered} AND gb0_status == "two_gb".
           E_stage  = G2_stage AND gb_symfit_max_A <= THETA_A. Any other status (excluded_pre, axis_mismatch, non_al, error) is False for both G2 and E (miss).
Usage (compute node of the CPU partition, python of the evaluation environment. CPU only. No MLIP is used):
  python aggregate_structure_filtered.py --mode validation [--out-dir DIR] [--procs N]   -> DIR/validation.json (F1 to F6. exit 3 unless all PASS)
  python aggregate_structure_filtered.py --mode main --selected proposed:2.0,baseline:2.0 [--out-dir DIR] → DIR/{summary.json,per_target.csv,report.md}
    At start, main checks that DIR/validation.json has all_pass=true and that the sha256 of the input scripts agrees, and otherwise exits with 2 (refusal).
Exit codes: 0 = success / 2 = failure of the inputs or checks / 3 = validation FAIL
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import aggregate_relax_novelty as AG  # noqa: E402

# threshold of equivalence [A] frozen before the analysis.
# also fixed to this value in the later differences (not changed). Source: decision note
# ${GBREL_WORK}/workspace/an internal note that is not included
THETA_A = 0.064

SENS_A = 1.0                     # sensitivity (number of baseline structures with gb_symfit_max_A < 1 A). theta is not changed
STAGES = ("pre", "post")
GB0_STATUS_OK = ("fitted", "no_gb", "multi_gb", "disordered")           # any other status is False for both G2 and E
KNOWN_STATUS = GB0_STATUS_OK + ("axis_mismatch", "non_al", "excluded_pre", "error")
REASONS = ("no_gb", "multi_gb", "disordered", "excluded_pre", "other")    # reasons why a structure is not G2 (other = axis_mismatch, non_al, error, others)
SETS_SHOWN = ("inner5", "all7")
XS = ("G2", "E")
# reference values of F5 and F6 (measured values; counts without imposing converged; main, all 7 targets, 350 per arm)
EXPECT_COUNTS = {("proposed", "G2"): 350, ("baseline", "G2"): 74, ("proposed", "E"): 350, ("baseline", "E"): 0}
SYMFIT_REQUIRED = ["layer", "arm", "w", "chunk", "cif_index", "stage", "status", "gb0_status", "gb_symfit_max_A"]
SCRIPT_NAMES = ("aggregate_structure_filtered.py", "aggregate_relax_novelty.py", "common.py")

RESULTS = Path(K.P(f"{K.RUN}/results"))
DEFAULT_SYMFIT = RESULTS / "gb_equivalence" / "gb_symfit_main.csv"
DEFAULT_OUT = Path(K.P(f"{K.VRUN}/results/structure_filtered"))   # nothing is written to results/ of the relax_novelty experiment (default output destination of the verification version)
EXISTING_SUMMARY = RESULTS / "relax_novelty_summary.json"


# ---------------------------------------------------------------------------
# loading of gb_symfit_main.csv and the join key (w is normalized in one place to f"{float(w):.1f}")
# ---------------------------------------------------------------------------
def norm_w(w) -> str:
    return f"{float(w):.1f}"


def skey(arm, w, chunk, cif_index, stage):
    return (arm, norm_w(w), int(chunk), int(cif_index), stage)


def load_symfit_rows(path):
    rows = K.read_csv_rows(path)
    if not rows:
        raise K.InputError(f"{path} is empty")
    missing = [c for c in SYMFIT_REQUIRED if c not in rows[0]]
    if missing:
        raise K.InputError(f"{path}: missing columns {missing}")
    return rows


def index_symfit(rows):
    """Row list -> (key -> {status, gb0_status, max_A}), number of duplicate keys."""
    table, dup = {}, 0
    for r in rows:
        k = skey(r["arm"], r["w"], r["chunk"], r["cif_index"], r["stage"])
        if k in table:
            dup += 1
            continue
        table[k] = {"layer": r["layer"], "status": r["status"], "gb0_status": r["gb0_status"], "max_A": K.to_float(r["gb_symfit_max_A"])}
    return table, dup


def is_g2(row) -> bool:
    return row["status"] in GB0_STATUS_OK and row["gb0_status"] == "two_gb"


def is_e(row) -> bool:
    return is_g2(row) and row["max_A"] is not None and row["max_A"] <= THETA_A


def non_g2_reason(row):
    if is_g2(row):
        return None
    return row["status"] if row["status"] in ("no_gb", "multi_gb", "disordered", "excluded_pre") else "other"


# ---------------------------------------------------------------------------
# check F1: join
# ---------------------------------------------------------------------------
def check_f1(records, symfit_rows):
    """All records x {pre,post} must be joined one to one with the rows of gb_symfit_main.csv. K.InputError if violated. Returns: the measured values of the check."""
    table, dup = index_symfit(symfit_rows)
    info = {"n_symfit_rows": len(symfit_rows), "n_records": len(records), "n_keys_expected": 2 * len(records), "dup_keys": dup}
    if dup:
        raise K.InputError(f"F1: {dup} duplicated keys in gb_symfit_main.csv")
    bad_layer = sorted({v["layer"] for v in table.values() if v["layer"] != "main"})
    if bad_layer:
        raise K.InputError(f"F1: unexpected layer(s) in gb_symfit_main.csv: {bad_layer}")
    bad_status = sorted({v["status"] for v in table.values() if v["status"] not in KNOWN_STATUS})
    if bad_status:
        raise K.InputError(f"F1: unknown status value(s): {bad_status}")
    bad_stage = sorted({k[4] for k in table if k[4] not in STAGES})
    if bad_stage:
        raise K.InputError(f"F1: unknown stage value(s): {bad_stage}")
    rec_keys = {skey(r["arm"], r["w"], r["chunk"], r["cif_index"], st) for r in records for st in STAGES}
    info["n_record_keys"] = len(rec_keys)
    if len(rec_keys) != 2 * len(records):
        raise K.InputError(f"F1: record keys {len(rec_keys)} != 2 x {len(records)} (duplicate keys in records)")
    un_rec = sorted(rec_keys - set(table))
    un_sym = sorted(set(table) - rec_keys)
    info["unmatched_records"] = len(un_rec)
    info["unmatched_symfit"] = len(un_sym)
    if un_rec or un_sym:
        raise K.InputError(f"F1: unmatched records {len(un_rec)} (e.g. {un_rec[:3]}) / unmatched symfit rows {len(un_sym)} (e.g. {un_sym[:3]})")
    n_bad = sum(1 for v in table.values() if v["gb0_status"] == "two_gb" and v["status"] != "fitted")
    info["two_gb_not_fitted"] = n_bad
    if n_bad:
        raise K.InputError(f"F1(m1): {n_bad} rows have gb0_status == two_gb but status != fitted")
    n_nomax = sum(1 for v in table.values() if v["gb0_status"] == "two_gb" and v["max_A"] is None)
    info["two_gb_without_max_A"] = n_nomax
    if n_nomax:
        raise K.InputError(f"F1: {n_nomax} two_gb rows have no finite gb_symfit_max_A")
    return info, table


# ---------------------------------------------------------------------------
# apply_filter(r4-1)
# ---------------------------------------------------------------------------
def apply_filter(records, X, symfit=None, extra_false=frozenset()):
    """X in {all_true, G2, E}. Returns a copy of records in which converged := converged AND X_post and valid_pre := valid_pre AND X_pre.
    extra_false: a set of keys in skey form (with stage). Forces X of that (structure, stage) to False (for the FAIL validation of F3)."""
    if X not in ("all_true", "G2", "E"):
        raise ValueError(f"unknown X {X!r}")
    if X != "all_true" and symfit is None:
        raise ValueError("symfit table is required for X in {G2, E}")
    out = []
    for r in records:
        r2 = dict(r)
        xs = {}
        for st in STAGES:
            k = skey(r["arm"], r["w"], r["chunk"], r["cif_index"], st)
            if k in extra_false:
                xs[st] = False
            elif X == "all_true":
                xs[st] = True
            else:
                row = symfit[k]
                xs[st] = is_g2(row) if X == "G2" else is_e(row)
        r2["converged"] = bool(r["converged"] and xs["post"])
        r2["valid_pre"] = bool(r["valid_pre"] and xs["pre"])
        out.append(r2)
    return out


def run_metrics(records, procs, train_gamma_min, log):
    comps, stats = AG.compute_components(records, procs, log)
    per_target, sets, cells = AG.compute_all(records, comps, train_gamma_min)
    diffs = AG.differences(sets)
    return {"comps": comps, "stats": stats, "per_target": per_target, "sets": sets, "diffs": diffs}


def jnorm(x):
    return json.loads(json.dumps(x))


def flat_sets(sets):
    return {f"{k[0]}|{k[1]}|w{k[2]}|{k[3]}": v for k, v in sets.items()}


def flat_diffs(diffs):
    return {f"{k[0]}|w{k[1]}|{k[2]}": v for k, v in diffs.items()}


# ---------------------------------------------------------------------------
# check F2: exact agreement with the existing relax_novelty_summary.json
# ---------------------------------------------------------------------------
def _leaf_compare(a, b, path, mism, counter):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                mism.append(f"{path}/{k}: key only in {'ours' if k in a else 'existing'}")
                continue
            _leaf_compare(a[k], b[k], f"{path}/{k}", mism, counter)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _leaf_compare(x, y, f"{path}[{i}]", mism, counter)
    else:
        counter[0] += 1
        if a != b or type(a) != type(b):
            mism.append(f"{path}: ours={a!r} existing={b!r}")


def compare_to_existing(sets, diffs, existing, selected):
    """Cross-checks all keys of sets of the main and SELECTED section (layer, w of proposed/baseline, target sets inner5, extrap2, all7) and the proposed - baseline differences with the existing summary."""
    layer, ws = selected["layer"], selected["w"]
    mism, counter = [], [0]
    ours_sets, ours_diffs = flat_sets(sets), flat_diffs(diffs)
    for arm in K.ARMS:
        for sname in AG.TARGET_SETS:
            key = f"{layer}|{arm}|w{ws[arm]}|{sname}"
            if key not in existing["sets"]:
                mism.append(f"{key}: missing in existing summary")
                continue
            if key not in ours_sets:
                mism.append(f"{key}: missing in ours")
                continue
            _leaf_compare(jnorm(ours_sets[key]), existing["sets"][key], key, mism, counter)
    if ws["proposed"] == ws["baseline"]:
        for sname in AG.TARGET_SETS:
            key = f"{layer}|w{ws['proposed']}|{sname}"
            if key not in existing["proposed_minus_baseline"] or key not in ours_diffs:
                mism.append(f"proposed_minus_baseline/{key}: missing")
                continue
            _leaf_compare(jnorm(ours_diffs[key]), existing["proposed_minus_baseline"][key], f"proposed_minus_baseline/{key}", mism, counter)
    return {"n_leaves_compared": counter[0], "n_mismatch": len(mism), "mismatches": mism[:20]}


# ---------------------------------------------------------------------------
# check F3: the value must change when one row is made X=False (FAIL validation)
# ---------------------------------------------------------------------------
def probe_isolated_vertices(records, log):
    """Measures whether each group of S_post of proposed and the 5 inner targets has edges (n_edges) by calling the connected-component computation of AG unchanged (procs=1, S of pre is empty and only the post groups)."""
    sub = [dict(r, valid_pre=False) for r in records if r["arm"] == "proposed" and r["is_inner"]]
    members = AG.s_members(sub)
    orig = AG._comp_worker
    seen = {}

    def probing(task):
        key, structs = task
        out = orig(task)
        seen[key] = {"n_vertices": len(structs), "n_comp": out[1], "n_edges": out[2]}
        return out
    AG._comp_worker = probing
    try:
        AG.compute_components(sub, 1, log)
    finally:
        AG._comp_worker = orig
    return members, seen


def pick_f3_vertex(members, seen):
    """Among the S_post vertices of degree 0 (the group has 0 edges), the one with the lexicographically smallest (chunk, cif_index). None if there is none."""
    iso = []
    for key, recs in members.items():
        if key[4] != "post":
            continue
        if seen[key]["n_edges"] == 0 and seen[key]["n_vertices"] == len(recs):
            iso += [(r["chunk"], r["cif_index"], r) for r in recs]
    allv = [(r["chunk"], r["cif_index"]) for key, recs in members.items() if key[4] == "post" for r in recs]
    summ = {"n_S_post_vertices_proposed_inner5": len(allv), "n_groups": sum(1 for k in members if k[4] == "post"),
            "n_edges_total": sum(v["n_edges"] for k, v in seen.items() if k[4] == "post"),
            "n_degree0_vertices": len(iso), "global_min_vertex": min(allv) if allv else None}
    if not iso:
        return None, summ
    iso.sort(key=lambda t: (t[0], t[1]))
    return iso[0][2], summ


# ---------------------------------------------------------------------------
# check F4: number of components of the induced subgraph (synthetic graph)
# ---------------------------------------------------------------------------
def f4_synthetic():
    """a-b-c (edges a-b, b-c) + isolated vertex d. The number of components is 3 when only b is X=False and 2 when all vertices are X=True. Goes through the same apply_filter, AG.s_members and K.count_components_by_fit as the implementation."""
    names = {"a": 0, "b": 1, "c": 2, "d": 3}
    edges = {frozenset((0, 1)), frozenset((1, 2))}
    recs = [{"layer": "main", "arm": "proposed", "w": "2.0", "chunk": 0, "cif_index": i, "target_id": 0, "is_inner": True, "gamma_target": 1.0,
             "valid_pre": False, "gamma_pre": None, "novel_pre": None, "converged": True, "novel_post": True, "gamma_relaxed": 1.0}
            for i in names.values()]
    sym = {}
    for i in names.values():
        for st in STAGES:
            sym[skey("proposed", "2.0", 0, i, st)] = {"layer": "main", "status": "fitted", "gb0_status": "two_gb", "max_A": 0.0}
    for st in STAGES:                                     # only b is X=False (not G2)
        sym[skey("proposed", "2.0", 0, 1, st)] = {"layer": "main", "status": "no_gb", "gb0_status": "no_gb", "max_A": None}

    def induced(records):
        mem = AG.s_members(records).get(("main", "proposed", "2.0", 0, "post"), [])
        n_comp, e, _ = K.count_components_by_fit(len(mem), lambda i, j: frozenset((mem[i]["cif_index"], mem[j]["cif_index"])) in edges)
        return n_comp, [m["cif_index"] for m in mem]
    n_all, v_all = induced(apply_filter(recs, "all_true"))
    n_b, v_b = induced(apply_filter(recs, "G2", sym))
    # the original way of counting (count the components of the graph of all vertices that contain a vertex of X)
    full = AG.s_members(apply_filter(recs, "all_true"))[("main", "proposed", "2.0", 0, "post")]
    _, e_full, _ = K.count_components_by_fit(len(full), lambda i, j: frozenset((full[i]["cif_index"], full[j]["cif_index"])) in edges)
    labels, _ = K.connected_components(len(full), e_full)
    xtrue = {full[i]["cif_index"] for i in range(len(full)) if sym[skey("proposed", "2.0", 0, full[i]["cif_index"], "post")]["gb0_status"] == "two_gb"}
    n_orig = len({labels[i] for i in range(len(full)) if full[i]["cif_index"] in xtrue})
    return {"pass": bool(n_all == 2 and n_b == 3), "n_components_all_X_true": n_all, "vertices_all": v_all,
            "n_components_b_X_false": n_b, "vertices_b_false": v_b, "n_components_if_counting_original_components_containing_X": n_orig,
            "expected": {"all_X_true": 2, "b_X_false": 3}}


# ---------------------------------------------------------------------------
# F5, F6 and the stratification per target (records and symfit only. UNI.fit not needed)
# ---------------------------------------------------------------------------
def count_tables(records, table):
    """Counts per arm x X x stage (main, all 7 targets, converged not imposed), counts with converged imposed, stratification per target, sensitivity."""
    c = {}
    for arm in K.ARMS:
        for X in XS:
            for st in STAGES:
                c[(arm, X, st)] = 0
    conv = {(arm, X): 0 for arm in K.ARMS for X in XS}
    tgt = {}
    sens = {(arm, st): {"all7": 0, "inner5": 0, "all7_converged": 0} for arm in K.ARMS for st in STAGES}
    for r in records:
        e = tgt.setdefault((r["arm"], r["target_id"]), {
            "n_req": 0, "gamma_target": r["gamma_target"], "g2_pre": 0, "g2_post": 0, "e_pre": 0, "e_post": 0,
            "g2_post_converged": 0, "e_post_converged": 0,
            "reasons_pre": {k: 0 for k in REASONS}, "reasons_post": {k: 0 for k in REASONS}, "n_unconditional_near_post": 0, "n_unconditional_near_post_not_g2": 0})
        e["n_req"] += 1
        for st in STAGES:
            row = table[skey(r["arm"], r["w"], r["chunk"], r["cif_index"], st)]
            g, ee = is_g2(row), is_e(row)
            c[(r["arm"], "G2", st)] += int(g)
            c[(r["arm"], "E", st)] += int(ee)
            e["g2_" + st] += int(g)
            e["e_" + st] += int(ee)
            reason = non_g2_reason(row)
            if reason is not None:
                e["reasons_" + st][reason] += 1
            if row["status"] == "fitted" and row["max_A"] is not None and row["max_A"] < SENS_A:
                s = sens[(r["arm"], st)]
                s["all7"] += 1
                s["inner5"] += int(r["is_inner"])
                s["all7_converged"] += int(r["converged"])
        post = table[skey(r["arm"], r["w"], r["chunk"], r["cif_index"], "post")]
        if r["converged"]:
            for X, fn in (("G2", is_g2), ("E", is_e)):
                if fn(post):
                    conv[(r["arm"], X)] += 1
                    e[("g2" if X == "G2" else "e") + "_post_converged"] += 1
            if r["gamma_relaxed"] is not None and abs(r["gamma_relaxed"] - r["gamma_target"]) <= AG.NEAR:
                e["n_unconditional_near_post"] += 1
                e["n_unconditional_near_post_not_g2"] += int(not is_g2(post))
    return c, conv, tgt, sens


def inner_count(tgt, arm, key):
    return sum(v[key] for (a, t), v in tgt.items() if a == arm and t <= K.INNER_MAX_TARGET_ID)


def all_count(tgt, arm, key):
    return sum(v[key] for (a, t), v in tgt.items() if a == arm)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def write_new(path, text):
    with open(path, "x") as f:                  # FileExistsError if it exists (overwriting is prohibited)
        f.write(text)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def script_hashes():
    d = Path(__file__).resolve().parent
    return {n: sha256_of(d / n) for n in SCRIPT_NAMES}


def ci_txt(k, n):
    lo, hi = AG.wilson(k, n)
    return f"{AG.fnum(k / n)} [{AG.fnum(lo)}, {AG.fnum(hi)}] ({k}/{n})"


def build_report(path, args_str, selected, theta, res, counts, conv, tgt, sens, f_info, existing_ref):
    sl, sw = selected["layer"], selected["w"]
    L = []
    A = L.append
    proposed_g2, van_g2 = counts[("proposed", "G2", "post")], counts[("baseline", "G2", "post")]
    proposed_e, van_e = counts[("proposed", "E", "post")], counts[("baseline", "E", "post")]
    van_in_g2 = inner_count(tgt, "baseline", "g2_post")
    van_all_g2_conv = conv[("baseline", "G2")]
    van_near = inner_count(tgt, "baseline", "n_unconditional_near_post")
    van_near_not_g2 = inner_count(tgt, "baseline", "n_unconditional_near_post_not_g2")
    van_e_min = f_info["baseline_post_min_max_A"]
    A("# Report of the metrics restricted to valid structures (two grain boundaries, equivalent)")
    A("")
    A(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} / layer: main (w=2.0) / command: `{args_str}`")
    A(f"- Main comparison: proposed w{sw['proposed']} vs baseline w{sw['baseline']}. 5 inner targets (n_req=250) and all 7 targets (n_req=350). The denominator is the requested number. Exclusions, unconverged structures and structures that are not X count as misses.")
    A(f"- **G2** = the status of `gb_symfit_main.csv` is one of fitted, no_gb, multi_gb, disordered, and gb0_status == two_gb (judged to have two grain boundaries at stage 0). Judged separately for stage = post (after relaxation) and pre (before relaxation).")
    A(f"- **E** = G2 and gb_symfit_max_A <= theta. **theta = {theta} A** (frozen: chosen as 0.064 A after looking at the distribution and not changed in this aggregation). The tolerance within which the two grain boundaries overlap under the 2_1 screw symmetry.")
    A("- Metrics: (7a)(7b) = number of converged AND X_post / n_req. (2'-X) = converged AND X_post AND |gamma_relaxed-gamma_target| <= 0.10. (4-X) = converged AND novel AND X_post. "
      "(6-post-X) = sum of the numbers of connected components of the induced subgraph whose vertices are restricted to S_post intersect X_post / n_req. (1'-X) = median |gamma_relaxed-gamma_target| of the structures with converged AND X_post only (count shown). "
      "The pre side (for reference, not used for claims) is given with the same replacement.")
    A("")
    A("## Notes (preregistered; the wording of the manuscript follows them)")
    A("")
    A(f"1. **Reason for restricting to G2**: the formula of gamma has the denominator 2A and presupposes that \"there are two grain boundaries\". The gamma of a structure without two grain boundaries is not a quantity that measures the tracking of the target gamma. "
      f"(Measured: the unconditional numerator of (2') of baseline is {van_near}/250 for the 5 inner targets, of which {van_near_not_g2} do not have two grain boundaries (G2_post=False).)")
    A(f"2. **Asymmetry**: G2 and E of proposed are 100% by construction (measured G2_post {proposed_g2}/350, E_post {proposed_e}/350), so all changes of the proposed - baseline difference in the G2 and E versions come from the decrease on the baseline side. "
      f"E of baseline is {van_e}/350 (measured; the minimum of gb_symfit_max_A of the {van_g2} structures judged to have two grain boundaries is {AG.fnum(van_e_min, 3)} A for post), so the tracking of baseline is read in the G2 version.")
    A(f"3. **Wording**: do not write \"baseline cannot keep the structure\" but give counts (main, all 7 targets {van_g2}/350 = {100.0 * van_g2 / 350:.0f}%, 5 inner targets {van_in_g2}/250; both are counts of G2_post, before imposing converged). "
      f"As a sensitivity check, the number of baseline structures with gb_symfit_max_A < {SENS_A:g} A is {sens[('baseline', 'post')]['all7']} for post over all 7 targets ({sens[('baseline', 'post')]['inner5']} for the 5 inner targets; of the 7 targets, {sens[('baseline', 'post')]['all7_converged']} are converged) and "
      f"{sens[('baseline', 'pre')]['all7']} for pre over all 7 targets (all only in rows with status=fitted). theta stays frozen.")
    A("4. **Stratification by target**: the number of baseline structures with two grain boundaries is given per target (table below). Grain boundaries disappear more at lower-gamma targets. "
      "A pooled conditional rate (e.g. the near-target rate among structures with two grain boundaries) is not given because its denominator is biased by target.")
    A("")
    A("### Number of structures with two grain boundaries per target (post, converged not imposed; the denominator is 50 each)")
    A("")
    A("| target | gamma_target [J/m2] | proposed G2 | baseline G2 | proposed E | baseline E | baseline G2 (pre) | reasons why baseline is not G2 (post): no_gb / multi_gb / disordered / excluded_pre / other |")
    A("|---|---|---|---|---|---|---|---|")
    for t in range(K.N_TARGETS):
        h, v = tgt[("proposed", t)], tgt[("baseline", t)]
        rs = v["reasons_post"]
        A(f"| {t}{' (extrapolation)' if t > K.INNER_MAX_TARGET_ID else ''} | {AG.fnum(v['gamma_target'], 4)} | {h['g2_post']}/50 | {v['g2_post']}/50 | {h['e_post']}/50 | {v['e_post']}/50 | {v['g2_pre']}/50 | "
          f"{rs['no_gb']} / {rs['multi_gb']} / {rs['disordered']} / {rs['excluded_pre']} / {rs['other']} |")
    A("")
    A("### Rate of two grain boundaries per target (G2_post / 50. Wilson 95% CI)")
    A("")
    A("| target | gamma_target [J/m2] | proposed G2 rate | baseline G2 rate | baseline E rate |")
    A("|---|---|---|---|---|")
    for t in range(K.N_TARGETS):
        h, v = tgt[("proposed", t)], tgt[("baseline", t)]
        A(f"| {t} | {AG.fnum(v['gamma_target'], 4)} | {ci_txt(h['g2_post'], 50)} | {ci_txt(v['g2_post'], 50)} | {ci_txt(v['e_post'], 50)} |")
    A("")
    A("### Cross-check and confirmation of reference values (F5, F6)")
    A("")
    A("| item | proposed | baseline |")
    A("|---|---|---|")
    for X in XS:
        for st in ("post", "pre"):
            A(f"| ({'7a' if X == 'G2' else '7b'}{'-pre' if st == 'pre' else ''}) numerator = #{{{X}_{st}}} (main, all 7 targets, converged not imposed) | {counts[('proposed', X, st)]}/350 | {counts[('baseline', X, st)]}/350 |")
    for X in XS:
        A(f"| (7{'a' if X == 'G2' else 'b'}) numerator = #{{converged AND {X}_post}} (value with converged imposed) | {conv[('proposed', X)]}/350 | {conv[('baseline', X)]}/350 |")
    A("")
    # table of the main comparison
    metric_rows = [("survive", "(7{x}) converged AND {X}_post"), ("near_post", "(2'-{X}) converged AND {X}_post AND near the target (gamma after relaxation)"),
                   ("novel_post", "(4-{X}) converged AND novel AND {X}_post"), ("nun_post", "(6-post-{X}) novel, unique, near the target (connected components of the induced subgraph)")]
    for sname, title, nreq in (("inner5", "5 inner targets", 250), ("all7", "all 7 targets", 350)):
        A(f"## Main comparison: {title} (n_req={nreq}). proposed w{sw['proposed']} vs baseline w{sw['baseline']}")
        A("")
        for X in XS:
            sets, diffs = res[X]["sets"], res[X]["diffs"]
            h, v = sets[(sl, "proposed", sw["proposed"], sname)], sets[(sl, "baseline", sw["baseline"], sname)]
            dd = diffs[(sl, sw["proposed"], sname)]
            xl = "a" if X == "G2" else "b"
            A(f"### X = {X}")
            A("")
            A("| metric | proposed | baseline | proposed - baseline [Newcombe 95% CI] |")
            A("|---|---|---|---|")
            for mid, label in metric_rows:
                d = dd[mid]
                A(f"| {label.format(x=xl, X=X)} | {AG.fci(h[mid])} | {AG.fci(v[mid])} | {AG.fnum(d['diff'])} [{AG.fnum(d['lo'])}, {AG.fnum(d['hi'])}] |")
            hv, vv = h["median_abs_dgamma_post"], v["median_abs_dgamma_post"]
            A(f"| (1'-{X}) median abs(gamma_relaxed-gamma_target) (converged AND {X}_post only; count shown) | {AG.fnum(hv['value'])} (n={hv['n_used']}) | {AG.fnum(vv['value'])} (n={vv['n_used']}) | "
              f"{AG.fnum(dd['median_abs_dgamma_post']['diff'])} |")
            A("")
    A("## Reference (pre side; not used for claims)")
    A("")
    A("| target set | X | metric | proposed | baseline | proposed - baseline [Newcombe 95% CI] |")
    A("|---|---|---|---|---|---|")
    for sname in ("inner5", "all7"):
        for X in XS:
            sets, diffs = res[X]["sets"], res[X]["diffs"]
            h, v = sets[(sl, "proposed", sw["proposed"], sname)], sets[(sl, "baseline", sw["baseline"], sname)]
            dd = diffs[(sl, sw["proposed"], sname)]
            kh, kv, n = h["counts"]["n_valid_pre"], v["counts"]["n_valid_pre"], h["n_req"]
            lo, hi = AG.newcombe(kh, n, kv, n)
            xl = "a" if X == "G2" else "b"
            A(f"| {sname} | {X} | (7{xl}-pre) valid_pre ∧ {X}_pre | {ci_txt(kh, n)} | {ci_txt(kv, n)} | {AG.fnum((kh - kv) / n)} [{AG.fnum(lo)}, {AG.fnum(hi)}] |")
            for mid, label in (("near_pre", "(2-X) valid_pre AND X_pre AND near the target (gamma before relaxation)"), ("novel_pre", "(4-X-pre) valid_pre AND novel AND X_pre"), ("nun_pre", "(6-pre-X)")):
                d = dd[mid]
                A(f"| {sname} | {X} | {label.replace('X', X)} | {AG.fci(h[mid])} | {AG.fci(v[mid])} | {AG.fnum(d['diff'])} [{AG.fnum(d['lo'])}, {AG.fnum(d['hi'])}] |")
    A("")
    A("## Stratified exclusions (why a structure is not G2. arm x target x stage. The denominator is 50 each.)")
    A("")
    A("| arm | target | stage | G2 | no_gb | multi_gb | disordered | excluded_pre | other |")
    A("|---|---|---|---|---|---|---|---|---|")
    for arm in K.ARMS:
        for t in range(K.N_TARGETS):
            e = tgt[(arm, t)]
            for st in ("post", "pre"):
                rs = e["reasons_" + st]
                A(f"| {arm} | {t} | {st} | {e['g2_' + st]} | {rs['no_gb']} | {rs['multi_gb']} | {rs['disordered']} | {rs['excluded_pre']} | {rs['other']} |")
    A("")
    A(f"- Calls of UNI.fit (connected components): G2 version {res['G2']['stats']['n_groups']} groups, {res['G2']['stats']['n_calls']} calls, {res['G2']['stats']['n_edges']} edges / "
      f"E version {res['E']['stats']['n_groups']} groups, {res['E']['stats']['n_calls']} calls, {res['E']['stats']['n_edges']} edges.")
    A(f"- The existing metrics and outputs (`{EXISTING_SUMMARY}`) are unchanged. For the exact agreement of F2 see validation.json.")
    A("")
    write_new(path, "\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# mode
# ---------------------------------------------------------------------------
def load_records(args, log):
    train_info = AG.load_train_info(K.P(f"{K.RUN}/results/train_ref_manifest.csv"))
    ns = argparse.Namespace(manifest_dir=None, relax_root=None, novelty_root=None, train_manifest=None)
    records = AG.load_inputs(["main"], ns, train_info)          # r4-1: the loading and the consistency check are left to the existing load_inputs (paths are not rewritten)
    log(f"inputs complete: {len(records)} records (layer main)")
    return records, train_info


def mode_validation(args, out_dir, log):
    wpath = out_dir / "validation.json"
    if wpath.exists():
        raise FileExistsError(f"refusing to overwrite {wpath}")
    records, train_info = load_records(args, log)
    gmin = train_info["gamma_min"]
    existing = json.loads(EXISTING_SUMMARY.read_text())
    selected = {"layer": "main", "w": {"proposed": "2.0", "baseline": "2.0"}}
    W = {"all_pass": False, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "args": " ".join(sys.argv), "theta_A": THETA_A,
         "script_sha256": script_hashes(), "symfit_csv": str(args.symfit_csv), "symfit_csv_sha256": sha256_of(args.symfit_csv),
         "existing_summary": str(EXISTING_SUMMARY), "existing_summary_sha256": sha256_of(EXISTING_SUMMARY)}
    passes = {}

    # F1
    f1 = {"pass": False}
    table = None
    try:
        rows = load_symfit_rows(args.symfit_csv)
        info, table = check_f1(records, rows)
        f1.update(info)
        f1["pass"] = bool(info["n_keys_expected"] == 1400 and info["unmatched_records"] == 0 and info["unmatched_symfit"] == 0 and info["dup_keys"] == 0)
        # FAIL validation: F1 gives InputError (exit 2 in main) for an input from which one row of gb_symfit_main.csv was removed
        try:
            check_f1(records, rows[:-1])
            f1["fail_validation_truncated_symfit"] = {"raised": False}
        except K.InputError as e:
            f1["fail_validation_truncated_symfit"] = {"raised": True, "message": str(e)[:200]}
        f1["pass"] = bool(f1["pass"] and f1["fail_validation_truncated_symfit"]["raised"])
    except (K.InputError, FileNotFoundError) as e:
        f1["error"] = str(e)[:400]
    W["F1"] = f1
    passes["F1"] = f1["pass"]
    log(f"F1 pass={f1['pass']}")

    # F2 (X=all_true agrees exactly with the existing one) and F3
    f2 = {"pass": False}
    f3 = {"pass": False}
    base_recs = apply_filter(records, "all_true")
    base = run_metrics(base_recs, args.procs, gmin, log)
    cmpres = compare_to_existing(base["sets"], base["diffs"], existing, selected)
    f2.update(cmpres)
    f2["pass"] = bool(cmpres["n_mismatch"] == 0 and cmpres["n_leaves_compared"] > 0)
    W["F2"] = f2
    passes["F2"] = f2["pass"]
    log(f"F2 pass={f2['pass']} leaves={cmpres['n_leaves_compared']} mismatches={cmpres['n_mismatch']}")

    members, seen = probe_isolated_vertices(records, log)
    vtx, probe = pick_f3_vertex(members, seen)
    f3["probe"] = probe
    if vtx is None:
        f3["error"] = "S_post (proposed, main, inner5) has no vertex of degree 0. Stopping and reporting according to check F3 of the plan"
    else:
        key = skey(vtx["arm"], vtx["w"], vtx["chunk"], vtx["cif_index"], "post")
        f3["vertex"] = {"arm": vtx["arm"], "chunk": vtx["chunk"], "cif_index": vtx["cif_index"], "target_id": vtx["target_id"],
                        "is_global_min_of_S_post": bool((vtx["chunk"], vtx["cif_index"]) == probe["global_min_vertex"])}
        mod = run_metrics(apply_filter(records, "all_true", extra_false=frozenset([key])), args.procs, gmin, log)
        checks = {}
        okall = True
        for sname in ("inner5", "all7"):
            for arm in K.ARMS:
                b, m = base["sets"][("main", arm, "2.0", sname)], mod["sets"][("main", arm, "2.0", sname)]
                for mid in ("novel_post", "near_post", "nun_post", "survive"):
                    delta = b[mid]["k"] - m[mid]["k"]
                    want = 1 if arm == "proposed" and mid != "survive" else (1 if arm == "proposed" else 0)
                    checks[f"{arm}|{sname}|{mid}"] = {"base_k": b[mid]["k"], "modified_k": m[mid]["k"], "delta": delta, "expected_delta": want}
                    okall = okall and (delta == want)
                for mid in ("novel_pre", "nun_pre", "near_pre"):
                    delta = b[mid]["k"] - m[mid]["k"]
                    checks[f"{arm}|{sname}|{mid}"] = {"base_k": b[mid]["k"], "modified_k": m[mid]["k"], "delta": delta, "expected_delta": 0}
                    okall = okall and (delta == 0)
        f3["checks"] = checks
        f3["pass"] = bool(okall)
    W["F3"] = f3
    passes["F3"] = f3["pass"]
    log(f"F3 pass={f3['pass']}")

    # F4
    f4 = f4_synthetic()
    W["F4"] = f4
    passes["F4"] = f4["pass"]

    # F5 and F6
    f5, f6 = {"pass": False}, {"pass": False}
    if table is not None:
        counts, conv, tgt, sens = count_tables(records, table)
        f5["counts_post"] = {f"{a}|{X}": counts[(a, X, "post")] for a in K.ARMS for X in XS}
        f5["expected"] = {f"{a}|{X}": v for (a, X), v in EXPECT_COUNTS.items()}
        f5["converged_imposed"] = {f"{a}|{X}": conv[(a, X)] for a in K.ARMS for X in XS}
        f5["pass"] = bool(all(counts[(a, X, "post")] == v for (a, X), v in EXPECT_COUNTS.items())
                          and all(conv[(a, X)] <= counts[(a, X, "post")] for a in K.ARMS for X in XS))
        f6["counts_pre"] = {f"{a}|{X}": counts[(a, X, "pre")] for a in K.ARMS for X in XS}
        f6["pass"] = bool(all(counts[(a, X, "pre")] == counts[(a, X, "post")] for a in K.ARMS for X in XS))
    W["F5"] = f5
    W["F6"] = f6
    passes["F5"] = f5["pass"]
    passes["F6"] = f6["pass"]

    W["all_pass"] = bool(all(passes.values()))
    W["passes"] = passes
    out_dir.mkdir(parents=True, exist_ok=True)
    write_new(wpath, json.dumps(W, indent=2, ensure_ascii=False) + "\n")
    log(f"wrote {wpath}: {passes}")
    return 0 if W["all_pass"] else 3


def check_validation_gate(out_dir, symfit_csv):
    wpath = out_dir / "validation.json"
    if not wpath.is_file():
        raise K.InputError(f"{wpath} not found (run --mode validation first)")
    W = json.loads(wpath.read_text())
    if W.get("all_pass") is not True:
        raise K.InputError(f"{wpath} is not all_pass=true (the production computation is refused): passes={W.get('passes')}")
    if W.get("theta_A") != THETA_A:
        raise K.InputError(f"validation theta_A {W.get('theta_A')} != {THETA_A}")
    if W.get("symfit_csv_sha256") != sha256_of(symfit_csv):
        raise K.InputError("symfit_csv_sha256 of validation.json does not agree with the current gb_symfit_main.csv")
    cur = script_hashes()
    if W.get("script_sha256") != cur:
        raise K.InputError("script_sha256 of validation.json does not agree with the current script (the script changed after the validation)")
    return W


def mode_main(args, out_dir, log):
    paths = {n: out_dir / n for n in ("summary.json", "per_target.csv", "report.md")}
    for n, p in paths.items():
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
    W = check_validation_gate(out_dir, args.symfit_csv)
    if not args.selected:
        raise K.InputError("--selected proposed:<W>,baseline:<W> is required in main mode")
    selected = AG.parse_selected(args.selected, ["main"])
    records, train_info = load_records(args, log)
    ws = {r["w"] for r in records}
    if ws != {selected["w"]["proposed"]} or selected["w"]["proposed"] != selected["w"]["baseline"]:
        raise K.InputError(f"selected w {selected['w']} does not match main records w {sorted(ws)}")
    gmin = train_info["gamma_min"]
    rows = load_symfit_rows(args.symfit_csv)
    f1_info, table = check_f1(records, rows)
    counts, conv, tgt, sens = count_tables(records, table)
    for (a, X), v in EXPECT_COUNTS.items():                    # F5 and F6 (if they differ, suspect an error of the cross-check and stop)
        if counts[(a, X, "post")] != v:
            raise K.InputError(f"F5: {a} {X}_post count {counts[(a, X, 'post')]} != {v}")
    for a in K.ARMS:
        for X in XS:
            if counts[(a, X, "pre")] != counts[(a, X, "post")]:
                raise K.InputError(f"F6: {a} {X} pre count {counts[(a, X, 'pre')]} != post {counts[(a, X, 'post')]}")
    for a in K.ARMS:
        for X in XS:
            if conv[(a, X)] > counts[(a, X, "post")]:
                raise K.InputError(f"F5: converged-imposed count exceeds raw count for {a} {X}")
    van_g2_vals = [v["max_A"] for k, v in table.items() if k[0] == "baseline" and k[4] == "post" and is_g2(v)]
    f_info = {"baseline_post_min_max_A": min(van_g2_vals) if van_g2_vals else None}
    res = {}
    for X in XS:
        recs = apply_filter(records, X, table)
        res[X] = run_metrics(recs, args.procs, gmin, log)
        log(f"X={X}: metrics done")
        if len(res[X]["per_target"]) != 2 * K.N_TARGETS:
            raise K.InputError(f"per_target rows {len(res[X]['per_target'])} != 14 for X={X}")
    existing = json.loads(EXISTING_SUMMARY.read_text())
    neg = compare_to_existing(res["G2"]["sets"], res["G2"]["diffs"], existing, selected)      # negative control of F2: the G2 version should disagree with the unconditional one (confirmation that the comparator fails)
    if neg["n_mismatch"] == 0:
        raise K.InputError("F2 negative control: the G2 version agreed exactly with the unconditional existing summary (the comparator or the restriction to G2 is not effective). Stopping and reporting")
    # per_target.csv
    pt_rows = []
    for X in XS:
        for r in res[X]["per_target"]:
            e = tgt[(r["arm"], r["target_id"])]
            row = {"X": X}
            row.update(r)
            row.update({"n_X_pre": e["g2_pre"] if X == "G2" else e["e_pre"], "n_X_post": e["g2_post"] if X == "G2" else e["e_post"],
                        "n_X_post_converged": e["g2_post_converged"] if X == "G2" else e["e_post_converged"]})
            for st in ("post", "pre"):
                for k in REASONS:
                    row[f"not_g2_{st}_{k}"] = e["reasons_" + st][k]
            pt_rows.append(row)
    fields = list(pt_rows[0].keys())
    K.write_csv_rows(paths["per_target.csv"], fields, pt_rows)
    summary = {
        "mode": "main", "date": time.strftime("%Y-%m-%d %H:%M:%S"), "args": " ".join(sys.argv), "theta_A": THETA_A, "selected": selected,
        "definitions": {"G2": "status in {fitted,no_gb,multi_gb,disordered} and gb0_status == two_gb (per stage)", "E": "G2 and gb_symfit_max_A <= theta_A",
                        "converged_replacement": "converged := converged and X_post", "valid_pre_replacement": "valid_pre := valid_pre and X_pre",
                        "n_req": "50 x number of targets (inner5=250, all7=350); X-false / excluded / unconverged are misses (C45)"},
        "validation": {"path": str(out_dir / "validation.json"), "all_pass": W["all_pass"], "date": W["date"]},
        "F1": f1_info,
        "F5_counts_post_raw": {f"{a}|{X}": counts[(a, X, "post")] for a in K.ARMS for X in XS},
        "F6_counts_pre_raw": {f"{a}|{X}": counts[(a, X, "pre")] for a in K.ARMS for X in XS},
        "converged_imposed_counts": {f"{a}|{X}": conv[(a, X)] for a in K.ARMS for X in XS},
        "f2_negative_control_G2_vs_unfiltered_existing": {"n_mismatch": neg["n_mismatch"], "n_leaves_compared": neg["n_leaves_compared"], "first_mismatches": neg["mismatches"][:5]},
        "per_target_counts": {f"{a}|{t}": {k: v for k, v in e.items()} for (a, t), e in sorted(tgt.items())},
        "sensitivity_gb_symfit_max_A_lt_1A": {f"{a}|{st}": v for (a, st), v in sens.items()},
        "baseline_post_min_gb_symfit_max_A_among_G2": f_info["baseline_post_min_max_A"],
        "X": {X: {"uni_fit_stats": res[X]["stats"], "sets": flat_sets(res[X]["sets"]), "proposed_minus_baseline": flat_diffs(res[X]["diffs"])} for X in XS},
    }
    write_new(paths["summary.json"], json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    build_report(paths["report.md"], " ".join(sys.argv), selected, THETA_A, res, counts, conv, tgt, sens, f_info, existing)
    log(f"wrote {paths['summary.json']}, {paths['per_target.csv']} ({len(pt_rows)} rows), {paths['report.md']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["validation", "main"])
    ap.add_argument("--selected", default=None, help="main only. Example: proposed:2.0,baseline:2.0")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--symfit-csv", default=str(DEFAULT_SYMFIT))
    ap.add_argument("--procs", type=int, default=1)
    args = ap.parse_args()
    args.symfit_csv = Path(K.P(args.symfit_csv))
    out_dir = Path(K.P(args.out_dir))
    t0 = time.time()

    def log(msg):
        print(f"[SF {time.time() - t0:7.1f}s] {msg}", flush=True)
    try:
        rc = mode_validation(args, out_dir, log) if args.mode == "validation" else mode_main(args, out_dir, log)
    except (K.InputError, FileExistsError) as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(2)
    if rc != 0:
        print(f"FAIL: validation did not pass (exit {rc}). Check {out_dir / 'validation.json'}", file=sys.stderr)
        sys.exit(rc)
    log("SF DONE")


if __name__ == "__main__":
    main()
