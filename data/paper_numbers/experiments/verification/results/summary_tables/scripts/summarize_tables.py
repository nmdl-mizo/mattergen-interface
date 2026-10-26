#!/usr/bin/python3
"""summarize_tables.py -- re-aggregation of the existing data for the review of the outline of the Al paper (M1/M2/M3/M7).

Run: /usr/bin/python3 -I summarize_tables.py [--out-dir DIR]
Standard library only. Read only (the output is only the CSV files under --out-dir). No computation and no job submission.
Target: main layer, w=2.0, the 7 points of target Expt.1 (proposed and baseline). Values in J/m^2.

Inputs (all existing masters):
  RES = experiments/relax_novelty/results
    relax/main/{arm}_w2.0/chunk*/relax_rows.csv   per structure gamma_pre / gamma_relaxed / converged / energy_eV
    gb_equivalence/gb_symfit_main.csv             stage 0 (G2/no_gb), stage 1 (gb_symfit_max_A)  stage=pre|post
    train_ref_manifest.csv                        gamma_Jm2 of the 4,419 training structures
  VRES = experiments/verification/results
    gb_stage2/gb_stage2_main.csv                  stage 2. eq12_0.064 (= E12) has rows only for structures of G2. post only
    dft_recalc/gamma_vasp.csv, dft_recalc/verdict.json

Definitions (the definitions of the masters are used as they are; sources in the summary document):
  G2   = symfit.status in {fitted,no_gb,multi_gb,disordered} and gb0_status == two_gb
  equivalent (post) = G2 and stage2.eq12_0.064 == 1  (E12 = maximum deviation of stage 1 or stage 2 <= theta=0.064 A)
  equivalent (pre)  = G2 and gb_symfit_max_A <= 0.064  (stage 1 only; stage 2 exists only for post)
  G2 non-equivalent  = G2 and not equivalent (including thin_grain)
  No GB      = symfit.status == no_gb (not G2)
  excluded before relaxation = relax_rows.excluded_pre != '' (or symfit.status == excluded_pre)
  near       = abs(gamma - gamma_target) <= 0.10
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from statistics import NormalDist

RES = '<ORIGINAL_ROOT>/workspace/experiments/relax_novelty/results'
VRES = '<ORIGINAL_ROOT>/workspace/experiments/verification/results'
DEFAULT_OUT = f"{VRES}/summary_tables"
THETA_COL = "eq12_0.064"
THETA = 0.064
NEAR_TOL = 0.10
G2_STATUS = {"fitted", "no_gb", "multi_gb", "disordered"}
INNER = [0, 1, 2, 3, 4]
ALL7 = [0, 1, 2, 3, 4, 5, 6]
GROUP = {0: "inner", 1: "inner", 2: "inner", 3: "inner", 4: "inner", 5: "extrap_low", 6: "extrap_high"}
ARMS = ("proposed", "baseline")


# ---------------------------------------------------------------- basic functions
def rd(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt(x, nd=6):
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_csv(path, fields, rows, nd=6):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for r in rows:
            w.writerow([fmt(r.get(k), nd) if not isinstance(r.get(k), str) else r.get(k) for k in fields])


def quantile(sorted_vals, q):
    """Same formula as the numpy.percentile default (linear interpolation, method 7). q is in 0..1."""
    n = len(sorted_vals)
    if n == 0:
        return None
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def mean(v):
    return sum(v) / len(v) if v else None


def pearson(x, y):
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def avg_rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 2:
        return None
    return pearson(avg_rank(x), avg_rank(y))


def theil_sen(x, y, alpha=0.95):
    """Same formula as scipy.stats.theilslopes(y, x, alpha=0.95, method='separate') (scipy 1.15.x, _stats_mstats_common.py:130-170).
    Slope = median of the slopes of all pairs (x_i > x_j). The CI is the slope of the ranks Rl, Ru determined from the variance of Sen (1968) equation 2.6 (with tie correction for x and y) (not a bootstrap).
    x has many ties (50 per target), so only pairs with deltax>0 are used.
    Returns: (slope, ci_lo, ci_hi, n_pairs)"""
    n = len(x)
    slopes = []
    for i in range(n):
        xi, yi = x[i], y[i]
        for j in range(n):
            dx = xi - x[j]
            if dx > 0:
                slopes.append((yi - y[j]) / dx)
    if not slopes:
        return None, None, None, 0
    slopes.sort()
    nt = len(slopes)
    med = statistics.median(slopes)
    a = 1.0 - alpha if alpha > 0.5 else alpha
    z = NormalDist().inv_cdf(a / 2.0)  # negative value (-1.959964)

    def ties(v):
        c = Counter(v)
        return [k for k in c.values() if k > 1]

    sigsq = (1 / 18.0) * (n * (n - 1) * (2 * n + 5)
                          - sum(k * (k - 1) * (2 * k + 5) for k in ties(x))
                          - sum(k * (k - 1) * (2 * k + 5) for k in ties(y)))
    sigma = math.sqrt(sigsq)
    ru = min(int(round((nt - z * sigma) / 2.0)), nt - 1)   # Python round uses round-half-even, the same as np.round
    rl = max(int(round((nt + z * sigma) / 2.0)) - 1, 0)
    return med, slopes[rl], slopes[ru], nt


# ---------------------------------------------------------------- data loading
def load_main():
    """arm -> {(chunk,cif): dict(target_id, gamma_target, relax_status, excluded_pre, converged, gamma_pre, gamma_relaxed,
    energy_eV, natoms, cls_pre, cls_post, ...)}"""
    sym = {"pre": {}, "post": {}}
    for r in rd(f"{RES}/gb_equivalence/gb_symfit_main.csv"):
        if r["layer"] == "main" and float(r["w"]) == 2.0:
            sym[r["stage"]][(r["arm"], int(r["chunk"]), int(r["cif_index"]))] = r
    st2 = {}
    for r in rd(f"{VRES}/gb_stage2/gb_stage2_main.csv"):
        if r["layer"] == "main" and float(r["w"]) == 2.0:
            st2[(r["arm"], int(r["chunk"]), int(r["cif_index"]))] = r
    data = {}
    for arm in ARMS:
        d = {}
        for f in sorted(glob.glob(f"{RES}/relax/main/{arm}_w2.0/chunk*/relax_rows.csv")):
            for r in rd(f):
                assert r["arm"] == arm and r["layer"] == "main" and float(r["w"]) == 2.0
                key = (int(r["chunk"]), int(r["cif_index"]))
                assert key not in d, ("duplicate", arm, key)
                d[key] = r
        data[arm] = d
    return data, sym, st2


def classify(arm, key, stage, relax, sym, st2):
    """Returns: (class, detail). class in equivalent / g2_noneq / no_gb / excluded_pre / unconverged / other.
    Order of decisions: excluded_pre -> (post only) unconverged -> stage 0 / stage 1 / stage 2."""
    rr = relax[arm][key]
    sr = sym[stage].get((arm,) + key)
    if sr is None:
        raise RuntimeError(f"symfit row missing: {arm} {key} {stage}")
    if rr["excluded_pre"] != "" or sr["status"] == "excluded_pre":
        return "excluded_pre", rr["excluded_pre"] or "excluded_pre"
    if stage == "post" and rr["status"] != "ok_converged":
        return "unconverged", rr["status"]
    st = sr["status"]
    g2 = (st in G2_STATUS) and (sr["gb0_status"] == "two_gb")
    if g2:
        mx = sr["gb_symfit_max_A"]
        e1 = mx != "" and float(mx) <= THETA
        if stage == "pre":
            return ("equivalent" if e1 else "g2_noneq"), ("E1only" if e1 else "")
        s2 = st2.get((arm,) + key)
        if s2 is None:
            raise RuntimeError(f"stage2 row missing for post G2: {arm} {key}")
        thin = s2["status"] == "thin_grain"
        if s2[THETA_COL] == "1":
            return "equivalent", "E1only" if e1 else "E2only"
        assert s2[THETA_COL] == "0", (arm, key, s2[THETA_COL])
        return "g2_noneq", ("thin_grain" if thin else "") + ("|E1" if e1 else "")
    if st == "no_gb":
        return "no_gb", ""
    return "other", f"{st}/{sr['gb0_status']}"


# ---------------------------------------------------------------- M1
def build_m1(data, sym, st2):
    fields = ["arm", "stage", "scope", "gamma_target", "group", "n_req", "equivalent", "g2_noneq", "no_gb", "excluded_pre",
              "unconverged", "other", "sum_check", "g2_total", "g2_noneq_thin_grain", "equiv_E1only_criterion_count",
              "equivalent_criterion"]
    rows = []
    classes = {}
    for arm in ARMS:
        for stage in ("pre", "post"):
            per_t = defaultdict(Counter)
            gt = {}
            for key, rr in data[arm].items():
                t = int(rr["target_id"])
                gt[t] = float(rr["gamma_target"])
                cl, det = classify(arm, key, stage, data, sym, st2)
                classes[(arm, stage, key)] = (cl, det)
                c = per_t[t]
                c["n_req"] += 1
                c[cl] += 1
                if cl == "g2_noneq" and "thin_grain" in det:
                    c["thin"] += 1
                if stage == "post":
                    sr = sym["post"][(arm,) + key]
                    if sr["status"] in G2_STATUS and sr["gb0_status"] == "two_gb":
                        mx = sr["gb_symfit_max_A"]
                        if mx != "" and float(mx) <= THETA:
                            c["e1only"] += 1
                else:
                    if cl == "equivalent":
                        c["e1only"] += 1
            crit = "G2 and E12(theta=0.064, stage1 or stage2)" if stage == "post" else \
                "G2 and E1 only(theta=0.064, stage1 only; stage2 does not exist for pre)"

            def mk(scope, ts, group, gtv):
                c = Counter()
                for t in ts:
                    c.update(per_t[t])
                r = {"arm": arm, "stage": stage, "scope": scope, "gamma_target": gtv, "group": group, "n_req": c["n_req"],
                     "equivalent": c["equivalent"], "g2_noneq": c["g2_noneq"], "no_gb": c["no_gb"],
                     "excluded_pre": c["excluded_pre"], "unconverged": c["unconverged"], "other": c["other"],
                     "g2_total": c["equivalent"] + c["g2_noneq"], "g2_noneq_thin_grain": c["thin"],
                     "equiv_E1only_criterion_count": c["e1only"], "equivalent_criterion": crit}
                r["sum_check"] = (r["equivalent"] + r["g2_noneq"] + r["no_gb"] + r["excluded_pre"] + r["unconverged"] + r["other"])
                assert r["sum_check"] == r["n_req"], r
                return r
            for t in ALL7:
                rows.append(mk(f"t{t}", [t], GROUP[t], gt[t]))
            rows.append(mk("inner5", INNER, "inner5", None))
            rows.append(mk("all7", ALL7, "all7", None))
    return fields, rows, classes


# ---------------------------------------------------------------- M2
def metrics_block(vals_by_t, gt, ts, n_req_per_t, ts_fit=True):
    """vals_by_t: t -> list of gamma. ts: list of the target ids concerned."""
    xs, ys = [], []
    for t in ts:
        for v in vals_by_t.get(t, []):
            xs.append(gt[t]); ys.append(v)
    n = len(ys)
    n_req = n_req_per_t * len(ts)
    out = {"n_req": n_req, "n": n}
    if n == 0:
        return out
    sv = sorted(ys)
    dev = [v - t for v, t in zip(ys, xs)]
    out["median"] = statistics.median(sv)
    out["q1"] = quantile(sv, 0.25)
    out["q3"] = quantile(sv, 0.75)
    out["iqr"] = out["q3"] - out["q1"]
    out["mean"] = mean(ys)
    out["mae"] = mean([abs(d) for d in dev])
    out["rmse"] = math.sqrt(mean([d * d for d in dev]))
    k = sum(1 for d in dev if abs(d) <= NEAR_TOL)
    out["near_k"] = k
    out["near_rate_req"] = k / n_req
    out["near_rate_n"] = k / n
    out["mean_signed_dev"] = mean(dev)
    if len(ts) > 1:
        out["pearson_r"] = pearson(xs, ys)
        out["spearman_rho"] = spearman(xs, ys)
        if ts_fit:
            slope, lo, hi, npairs = theil_sen(xs, ys)
            out["ts_slope"], out["ts_ci_lo"], out["ts_ci_hi"], out["ts_n_pairs"] = slope, lo, hi, npairs
    return out


M2_FIELDS = ["population", "stage", "scope", "gamma_target", "group", "n_req", "n", "n_equivalent_E12", "median", "q1", "q3", "iqr",
             "mean", "mae", "rmse", "near_k", "near_rate_req", "near_rate_n", "mean_signed_dev", "ts_slope", "ts_ci_lo",
             "ts_ci_hi", "ts_n_pairs", "pearson_r", "spearman_rho"]


def build_m2(data, classes, arm, population, only_g2):
    gt = {}
    for rr in data[arm].values():
        gt[int(rr["target_id"])] = float(rr["gamma_target"])
    rows = []
    for stage in ("pre", "post"):
        vals = defaultdict(list)
        n_eq = defaultdict(int)
        for key, rr in data[arm].items():
            t = int(rr["target_id"])
            if rr["excluded_pre"] != "":
                continue
            if stage == "post" and rr["status"] != "ok_converged":
                continue
            if only_g2:
                cpost = classes[(arm, "post", key)][0]
                if cpost not in ("equivalent", "g2_noneq"):
                    continue
            vals[t].append(float(rr["gamma_pre"] if stage == "pre" else rr["gamma_relaxed"]))
            if classes[(arm, "post", key)][0] == "equivalent":
                n_eq[t] += 1
        for scope, ts, group in ([(f"t{t}", [t], GROUP[t]) for t in ALL7] + [("inner5", INNER, "inner5"), ("all7", ALL7, "all7")]):
            m = metrics_block(vals, gt, ts, 50)
            m.update({"population": population, "stage": stage, "scope": scope, "group": group,
                      "gamma_target": gt[ts[0]] if len(ts) == 1 else None,
                      "n_equivalent_E12": sum(n_eq[t] for t in ts)})
            rows.append(m)
    return rows


# ---------------------------------------------------------------- M3
def build_m3(data, m2_proposed):
    train = sorted(float(r["gamma_Jm2"]) for r in rd(f"{RES}/train_ref_manifest.csv"))
    n = len(train)
    assert n == 4419, n
    gt = {}
    for rr in data["proposed"].values():
        gt[int(rr["target_id"])] = float(rr["gamma_target"])
    by = {(r["stage"], r["scope"]): r for r in m2_proposed}
    fields = ["row_type", "id", "gamma_target", "group", "n_train_within_0p10", "train_percentile_below", "proposed_n_pre",
              "proposed_near_k_pre", "proposed_near_rate_req_pre", "proposed_near_rate_n_pre", "proposed_n_post", "proposed_near_k_post",
              "proposed_near_rate_req_post", "proposed_near_rate_n_post", "value", "n"]
    rows = []
    # per target
    near_pre, near_post, ntrain = [], [], []
    for t in ALL7:
        tv = gt[t]
        cnt = sum(1 for g in train if abs(g - tv) <= NEAR_TOL)
        below = sum(1 for g in train if g < tv)
        pre = by[("pre", f"t{t}")]
        post = by[("post", f"t{t}")]
        rows.append({"row_type": "target", "id": f"t{t}", "gamma_target": tv, "group": GROUP[t], "n_train_within_0p10": cnt,
                     "train_percentile_below": 100.0 * below / n, "proposed_n_pre": pre["n"], "proposed_near_k_pre": pre["near_k"],
                     "proposed_near_rate_req_pre": pre["near_rate_req"], "proposed_near_rate_n_pre": pre["near_rate_n"],
                     "proposed_n_post": post["n"], "proposed_near_k_post": post["near_k"],
                     "proposed_near_rate_req_post": post["near_rate_req"], "proposed_near_rate_n_post": post["near_rate_n"]})
        near_pre.append(pre["near_rate_req"]); near_post.append(post["near_rate_req"]); ntrain.append(cnt)
    # training quantiles
    rows.append({"row_type": "train_summary", "id": "n_train", "value": n, "n": n})
    rows.append({"row_type": "train_summary", "id": "min", "value": train[0], "n": n})
    rows.append({"row_type": "train_summary", "id": "max", "value": train[-1], "n": n})
    rows.append({"row_type": "train_summary", "id": "mean", "value": mean(train), "n": n})
    for q in (5, 10, 25, 50, 75, 90, 95):
        rows.append({"row_type": "train_quantile", "id": f"p{q}", "value": quantile(train, q / 100.0), "n": n})
    # Spearman (n=7, reference value)
    rows.append({"row_type": "spearman_n7", "id": "near_rate_req_pre_vs_n_train_within_0p10", "value": spearman(near_pre, ntrain), "n": 7})
    rows.append({"row_type": "spearman_n7", "id": "near_rate_req_post_vs_n_train_within_0p10", "value": spearman(near_post, ntrain), "n": 7})
    # reference for the 5 inner points only (n=5)
    rows.append({"row_type": "spearman_n5_inner", "id": "near_rate_req_pre_vs_n_train_within_0p10", "value": spearman(near_pre[:5], ntrain[:5]), "n": 5})
    rows.append({"row_type": "spearman_n5_inner", "id": "near_rate_req_post_vs_n_train_within_0p10", "value": spearman(near_post[:5], ntrain[:5]), "n": 5})
    return fields, rows, train


# ---------------------------------------------------------------- M7
def dft_stats(d, m, label):
    """d = DFT, m = MACE. R^2 relative to y=x regards DFT as the observed value and MACE (+offset) as the predicted value: 1 - SS_res/SS_tot (SS_tot around the mean of DFT)."""
    n = len(d)
    diff = [a - b for a, b in zip(d, m)]
    off = statistics.median(diff)
    mu = mean(d)
    ss_tot = sum((a - mu) ** 2 for a in d)
    ss_res = sum((a - b) ** 2 for a, b in zip(d, m))
    ss_res_off = sum((a - (b + off)) ** 2 for a, b in zip(d, m))
    r = pearson(m, d)
    return {"n": n, "mae_raw": mean([abs(x) for x in diff]), "offset_median_dft_minus_mace": off,
            "mae_offset_removed": mean([abs(x - off) for x in diff]), "pearson_r": r, "spearman_rho": spearman(m, d),
            "ols_r2": r * r, "r2_yx_raw": 1 - ss_res / ss_tot, "r2_yx_offset_removed": 1 - ss_res_off / ss_tot,
            "max_abs_err_raw": max(abs(x) for x in diff), "max_abs_err_offset_removed": max(abs(x - off) for x in diff)}


def build_m7(data):
    rows = rd(f"{VRES}/dft_recalc/gamma_vasp.csv")
    assert len(rows) == 26, len(rows)
    sub = {"proposed": [r for r in rows if r["arm"] == "proposed"], "baseline": [r for r in rows if r["arm"] == "baseline"], "all": rows}
    fields = ["arm", "offset_scope", "n", "mae_raw", "offset_median_dft_minus_mace", "mae_offset_removed", "pearson_r", "spearman_rho",
              "ols_r2", "r2_yx_raw", "r2_yx_offset_removed", "max_abs_err_raw", "max_abs_err_offset_removed"]
    out = []
    res = {}
    for label in ("proposed", "baseline", "all"):
        rs = sub[label]
        d = [float(r["gamma_dft"]) for r in rs]
        m = [float(r["gamma_mace"]) for r in rs]
        s = dft_stats(d, m, label)
        s.update({"arm": label, "offset_scope": "own" if label != "all" else "pooled_single_offset(26)"})
        out.append(s)
        res[label] = s
    # version evaluated on 26 points after subtracting the offset per arm (the median for proposed and for baseline separately) from all
    d_all, m_all, off_by = [], [], {}
    for lab in ("proposed", "baseline"):
        off_by[lab] = res[lab]["offset_median_dft_minus_mace"]
    adj = []
    for r in rows:
        dd, mm = float(r["gamma_dft"]), float(r["gamma_mace"])
        adj.append(dd - (mm + off_by[r["arm"]]))
    d = [float(r["gamma_dft"]) for r in rows]
    m = [float(r["gamma_mace"]) for r in rows]
    mu = mean(d)
    ss_tot = sum((a - mu) ** 2 for a in d)
    ss_res_arm = sum(x * x for x in adj)
    out.append({"arm": "all", "offset_scope": "armwise_offsets_proposed_and_baseline_separately", "n": 26,
                "mae_raw": res["all"]["mae_raw"], "offset_median_dft_minus_mace": None,
                "mae_offset_removed": mean([abs(x) for x in adj]), "pearson_r": res["all"]["pearson_r"],
                "spearman_rho": res["all"]["spearman_rho"], "ols_r2": res["all"]["ols_r2"], "r2_yx_raw": res["all"]["r2_yx_raw"],
                "r2_yx_offset_removed": 1 - ss_res_arm / ss_tot, "max_abs_err_raw": res["all"]["max_abs_err_raw"],
                "max_abs_err_offset_removed": max(abs(x) for x in adj)})
    return fields, out, res


def build_m7_energy(data):
    """Comparison of the DFT E_toten with the MACE total energy of the same structure (relax_rows.csv energy_eV: total energy after MACE relaxation, directly recorded value).
    The gamma is not back-calculated. The join key is (arm, chunk, cif_index) (columns of gamma_vasp.csv)."""
    rows = rd(f"{VRES}/dft_recalc/gamma_vasp.csv")
    pairs = []
    checks = {"n_joined": 0, "natoms_mismatch": 0, "max_abs_gamma_mismatch": 0.0, "target_mismatch": 0}
    for r in rows:
        key = (int(r["chunk"]), int(r["cif_index"]))
        rr = data[r["arm"]].get(key)
        if rr is None or rr["energy_eV"] == "":
            raise RuntimeError(f"no MACE energy record for {r['name']}")
        checks["n_joined"] += 1
        if int(rr["natoms"]) != int(r["N"]):
            checks["natoms_mismatch"] += 1
        if int(rr["target_id"]) != int(r["target"]):
            checks["target_mismatch"] += 1
        checks["max_abs_gamma_mismatch"] = max(checks["max_abs_gamma_mismatch"], abs(float(rr["gamma_relaxed"]) - float(r["gamma_mace"])))
        N = int(r["N"])
        pairs.append((r["arm"], float(r["E_toten"]) / N, float(rr["energy_eV"]) / N, float(r["E_toten"]), float(rr["energy_eV"]), N))
    fields = ["arm", "n", "mae_raw_eV_per_atom", "offset_median_dft_minus_mace_eV_per_atom", "mae_offset_removed_eV_per_atom",
              "pearson_r", "max_abs_err_raw_eV_per_atom", "mae_raw_eV_per_cell", "offset_median_eV_per_cell", "source_mace_energy",
              "source_dft_energy"]
    out = []
    for label in ("proposed", "baseline", "all"):
        ps = [p for p in pairs if label == "all" or p[0] == label]
        d = [p[1] for p in ps]; m = [p[2] for p in ps]
        diff = [a - b for a, b in zip(d, m)]
        off = statistics.median(diff)
        dc = [p[3] - p[4] for p in ps]
        out.append({"arm": label, "n": len(ps), "mae_raw_eV_per_atom": mean([abs(x) for x in diff]),
                    "offset_median_dft_minus_mace_eV_per_atom": off,
                    "mae_offset_removed_eV_per_atom": mean([abs(x - off) for x in diff]), "pearson_r": pearson(m, d),
                    "max_abs_err_raw_eV_per_atom": max(abs(x) for x in diff), "mae_raw_eV_per_cell": mean([abs(x) for x in dc]),
                    "offset_median_eV_per_cell": statistics.median(dc),
                    "source_mace_energy": "relax/main/{arm}_w2.0/chunk*/relax_rows.csv:energy_eV (MACE-MH-1 omat_pbe relaxed total energy)",
                    "source_dft_energy": "dft_recalc/gamma_vasp.csv:E_toten (VASP single-point on MACE-relaxed structure)"})
    return fields, out, checks


# ---------------------------------------------------------------- reproduction check
def close(a, b, tol):
    return a is not None and abs(a - b) <= tol


def build_repro(m1_rows, m2_proposed, m2_van, classes, m7res):
    items = []
    by = {(r["stage"], r["scope"]): r for r in m2_proposed}

    def add(item, source, documented, recomputed, tol):
        ok = close(recomputed, documented, tol) if isinstance(recomputed, (int, float)) else False
        items.append({"item": item, "source": source, "documented": documented, "recomputed": recomputed, "tolerance": tol,
                      "abs_diff": None if recomputed is None else abs(recomputed - documented), "match": "YES" if ok else "NO"})

    TR = "experiments/range_targets/results/tracking_report.md"
    RN = "experiments/relax_novelty/results/relax_novelty_report.md"
    pre5, pre7 = by[("pre", "inner5")], by[("pre", "all7")]
    add("proposed pre inner5 TS slope", f"{TR}:15", 0.638, pre5["ts_slope"], 0.0005)
    add("proposed pre inner5 TS CI lo", f"{TR}:15", 0.587, pre5["ts_ci_lo"], 0.0005)
    add("proposed pre inner5 TS CI hi", f"{TR}:15", 0.688, pre5["ts_ci_hi"], 0.0005)
    add("proposed pre all7 TS slope", f"{TR}:20", 0.404, pre7["ts_slope"], 0.0005)
    add("proposed pre all7 TS CI lo", f"{TR}:20", 0.367, pre7["ts_ci_lo"], 0.0005)
    add("proposed pre all7 TS CI hi", f"{TR}:20", 0.439, pre7["ts_ci_hi"], 0.0005)
    add("proposed pre inner5 MAE", f"{TR}:15", 0.2735, pre5["mae"], 0.00005)
    add("proposed pre inner5 mean_dev", f"{TR}:15", -0.2138, pre5["mean_signed_dev"], 0.00005)
    add("proposed pre inner5 near_k", f"{TR}:15", 97, pre5["near_k"], 0)
    post5, post7 = by[("post", "inner5")], by[("post", "all7")]
    add("proposed post inner5 MAE", f"{RN}:33", 0.288, post5["mae"], 0.0005)
    add("proposed post inner5 near_k", f"{RN}:33", 91, post5["near_k"], 0)
    add("proposed pre all7 MAE", f"{RN}:63", 0.491, pre7["mae"], 0.0005)
    add("proposed post all7 MAE", f"{RN}:63", 0.503, post7["mae"], 0.0005)
    add("proposed pre all7 near_k", f"{RN}:63", 97, pre7["near_k"], 0)
    add("proposed post all7 near_k", f"{RN}:63", 91, post7["near_k"], 0)
    # recompute the median |dev| (inner5 pre/post, all7) separately
    return items


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    data, sym, st2 = load_main()
    for arm in ARMS:
        assert len(data[arm]) == 350, (arm, len(data[arm]))
    # identity of the target values (same gamma_target for the same target_id)
    for arm in ARMS:
        tv = defaultdict(set)
        for rr in data[arm].values():
            tv[int(rr["target_id"])].add(rr["gamma_target"])
        assert all(len(v) == 1 for v in tv.values()), "gamma_target not unique per target_id"
    tvh = {int(r["target_id"]): r["gamma_target"] for r in data["proposed"].values()}
    tvv = {int(r["target_id"]): r["gamma_target"] for r in data["baseline"].values()}
    assert tvh == tvv, "proposed/baseline targets differ"

    # M1
    m1_fields, m1_rows, classes = build_m1(data, sym, st2)
    write_csv(f"{out}/m1_classes.csv", m1_fields, m1_rows)

    # M2
    m2_proposed = build_m2(data, classes, "proposed", "proposed all structures (main, w=2.0)", only_g2=False)
    write_csv(f"{out}/m2_proposed_tracking.csv", M2_FIELDS, m2_proposed)
    m2_van = build_m2(data, classes, "baseline",
                      "baseline G2 structures only (gamma = 2-GB cell average; E12 equivalent = 0; no-GB structures excluded)", only_g2=True)
    write_csv(f"{out}/m2_baseline_g2_tracking.csv", M2_FIELDS, m2_van)

    # M3
    m3_fields, m3_rows, train = build_m3(data, m2_proposed)
    write_csv(f"{out}/m3_support.csv", m3_fields, m3_rows)

    # M7
    m7_fields, m7_rows, m7res = build_m7(data)
    write_csv(f"{out}/m7_dft_stats.csv", m7_fields, m7_rows)
    e_fields, e_rows, e_checks = build_m7_energy(data)
    write_csv(f"{out}/m7_total_energy.csv", e_fields, e_rows, nd=8)

    # reproduction check (agreement with existing values). Additionally the median |dev| and the existing values of M1/M7 are checked
    items = build_repro(m1_rows, m2_proposed, m2_van, classes, m7res)
    # median |dev| of proposed (median_abs_dgamma_pre/post of relax_novelty_report.md:33, :63)
    def med_abs(stage, ts):
        v = []
        for key, rr in data["proposed"].items():
            if int(rr["target_id"]) in ts:
                v.append(abs(float(rr["gamma_pre" if stage == "pre" else "gamma_relaxed"]) - float(rr["gamma_target"])))
        return statistics.median(v)
    RN = "experiments/relax_novelty/results/relax_novelty_report.md"
    TR = "experiments/range_targets/results/tracking_report.md"
    for item, src, doc, val in [("proposed pre inner5 median |gamma-target|", f"{RN}:33", 0.179, med_abs("pre", INNER)),
                                ("proposed post inner5 median |gamma-target|", f"{RN}:33", 0.174, med_abs("post", INNER)),
                                ("proposed pre all7 median |gamma-target|", f"{RN}:63", 0.398, med_abs("pre", ALL7)),
                                ("proposed post all7 median |gamma-target|", f"{RN}:63", 0.405, med_abs("post", ALL7))]:
        items = [i for i in items if i["item"] != item]
        items.append({"item": item, "source": src, "documented": doc, "recomputed": val, "tolerance": 0.0005,
                      "abs_diff": abs(val - doc), "match": "YES" if abs(val - doc) <= 0.0005 else "NO"})
    # median/Q1/Q3 of proposed pre per target (tracking_report.md:27-33)
    doc_pt = {0: (0.7180, 0.6700, 0.8611), 1: (1.0537, 0.9129, 1.1213), 2: (1.4774, 1.2857, 1.5697), 3: (1.6845, 1.4890, 1.8032),
              4: (1.7423, 1.6079, 1.8552), 5: (0.6750, 0.6375, 0.7206), 6: (1.5323, 1.2910, 1.6190)}
    for r in m2_proposed:
        if r["stage"] == "pre" and r["scope"].startswith("t"):
            t = int(r["scope"][1:])
            for nm, dv, rv in zip(("median", "q1", "q3"), doc_pt[t], (r["median"], r["q1"], r["q3"])):
                items.append({"item": f"proposed pre t{t} {nm}", "source": f"{TR}:{27 + t}", "documented": dv, "recomputed": rv,
                              "tolerance": 0.00005, "abs_diff": abs(rv - dv), "match": "YES" if abs(rv - dv) <= 0.00005 else "NO"})
            near_doc = {0: 35, 1: 25, 2: 28, 3: 9, 4: 0, 5: 0, 6: 0}[t]
            items.append({"item": f"proposed pre t{t} near_k", "source": "experiments/relax_novelty/results/relax_novelty_per_target.csv (near_pre, main proposed w2.0; also table 9-B of an earlier check)",
                          "documented": near_doc, "recomputed": r["near_k"], "tolerance": 0, "abs_diff": abs(r["near_k"] - near_doc),
                          "match": "YES" if r["near_k"] == near_doc else "NO"})
    # existing values of M1 (table 1-A): baseline post, totals over layers 0/74/256/20, proposed 350/0/0/0
    FA = "an internal note that is not included:59-60"
    tot = {(r["arm"], r["stage"]): r for r in m1_rows if r["scope"] == "all7"}
    for (arm, exp) in (("proposed", (350, 0, 0, 0)), ("baseline", (0, 74, 256, 20))):
        r = tot[(arm, "post")]
        got = (r["equivalent"], r["g2_noneq"], r["no_gb"], r["excluded_pre"] + r["unconverged"] + r["other"])
        items.append({"item": f"M1 post {arm} all7 E/I/N/X", "source": FA, "documented": "/".join(map(str, exp)),
                      "recomputed": "/".join(map(str, got)), "tolerance": 0, "abs_diff": "", "match": "YES" if tuple(got) == exp else "NO"})
    exp_v = {0: (0, 3, 43, 4), 1: (0, 8, 41, 1), 2: (0, 11, 39, 0), 3: (0, 18, 30, 2), 4: (0, 22, 27, 1), 5: (0, 1, 38, 11), 6: (0, 11, 38, 1)}
    for t, exp in exp_v.items():
        r = [x for x in m1_rows if x["arm"] == "baseline" and x["stage"] == "post" and x["scope"] == f"t{t}"][0]
        got = (r["equivalent"], r["g2_noneq"], r["no_gb"], r["excluded_pre"] + r["unconverged"] + r["other"])
        items.append({"item": f"M1 post baseline t{t} E/I/N/X", "source": FA, "documented": "/".join(map(str, exp)),
                      "recomputed": "/".join(map(str, got)), "tolerance": 0, "abs_diff": "", "match": "YES" if tuple(got) == exp else "NO"})
    # whether the G2 set is the same before and after relaxation (baseline): table 3: main baseline moved 0
    isg2 = lambda c: c in ("equivalent", "g2_noneq")
    moved = sum(1 for k in data["baseline"] if isg2(classes[("baseline", "pre", k)][0]) != isg2(classes[("baseline", "post", k)][0]))
    items.append({"item": "baseline main G2 membership pre vs post: n moved", "source": "an internal note that is not included:96",
                  "documented": 0, "recomputed": moved, "tolerance": 0, "abs_diff": abs(moved - 0), "match": "YES" if moved == 0 else "NO"})
    # M3: (1.9347 -> 85 structures, 2.3361 -> 0 structures, 99.7/99.98 pct), p values of train
    FA5 = "an internal note that is not included:42"
    m3t = {r["id"]: r for r in m3_rows if r["row_type"] == "target"}
    items.append({"item": "M3 n_train within 0.10 of 1.9347 (t3)", "source": FA5, "documented": 85, "recomputed": m3t["t3"]["n_train_within_0p10"],
                  "tolerance": 0, "abs_diff": abs(m3t["t3"]["n_train_within_0p10"] - 85), "match": "YES" if m3t["t3"]["n_train_within_0p10"] == 85 else "NO"})
    items.append({"item": "M3 n_train within 0.10 of 2.3361 (t4)", "source": FA5, "documented": 0, "recomputed": m3t["t4"]["n_train_within_0p10"],
                  "tolerance": 0, "abs_diff": m3t["t4"]["n_train_within_0p10"], "match": "YES" if m3t["t4"]["n_train_within_0p10"] == 0 else "NO"})
    for tid, doc in (("t3", 99.7), ("t4", 99.98)):
        v = m3t[tid]["train_percentile_below"]
        items.append({"item": f"M3 train percentile of {tid}", "source": FA5, "documented": doc, "recomputed": v, "tolerance": 0.005,
                      "abs_diff": abs(v - doc), "match": "YES" if abs(v - doc) <= 0.05 else "NO"})
    q = {r["id"]: r["value"] for r in m3_rows if r["row_type"] == "train_quantile"}
    for qn, doc in (("p5", 0.6630), ("p10", 0.7192), ("p25", 0.8908), ("p50", 1.0800), ("p75", 1.3151), ("p90", 1.5580), ("p95", 1.6994)):
        items.append({"item": f"M3 train {qn}", "source": FA5, "documented": doc, "recomputed": q[qn], "tolerance": 0.00005,
                      "abs_diff": abs(q[qn] - doc), "match": "YES" if abs(q[qn] - doc) <= 0.00005 else "NO"})
    # M7: verdict.json report_only
    vj = json.load(open(f"{VRES}/dft_recalc/verdict.json"))["report_only"]
    V = "experiments/verification/results/dft_recalc/verdict.json:381-399"
    chk = [("M7 spearman all", vj["spearman_rho"]["all"], m7res["all"]["spearman_rho"]),
           ("M7 spearman proposed", vj["spearman_rho"]["proposed"], m7res["proposed"]["spearman_rho"]),
           ("M7 spearman baseline", vj["spearman_rho"]["baseline"], m7res["baseline"]["spearman_rho"]),
           ("M7 offset proposed", vj["offset_median_dft_minus_mace_J_m2"]["proposed"], m7res["proposed"]["offset_median_dft_minus_mace"]),
           ("M7 offset baseline", vj["offset_median_dft_minus_mace_J_m2"]["baseline"], m7res["baseline"]["offset_median_dft_minus_mace"]),
           ("M7 offset-removed MAE proposed", vj["offset_removed_mae_J_m2"]["proposed"], m7res["proposed"]["mae_offset_removed"]),
           ("M7 offset-removed MAE baseline", vj["offset_removed_mae_J_m2"]["baseline"], m7res["baseline"]["mae_offset_removed"])]
    for nm, doc, val in chk:
        items.append({"item": nm, "source": V, "documented": doc, "recomputed": val, "tolerance": 1e-9, "abs_diff": abs(val - doc),
                      "match": "YES" if abs(val - doc) <= 1e-9 else "NO"})
    for lab in ("proposed", "baseline", "all"):
        pass
    # soundness of the energy join of M7
    for k, v in e_checks.items():
        items.append({"item": f"M7 total-energy join check: {k}", "source": "relax_rows.csv x gamma_vasp.csv", "documented": "",
                      "recomputed": (f"{v:.3e}" if isinstance(v, float) else v), "tolerance": "", "abs_diff": "", "match": "INFO"})
    r_fields = ["item", "source", "documented", "recomputed", "tolerance", "abs_diff", "match"]
    for it in items:
        for k in ("documented", "recomputed", "abs_diff"):
            if isinstance(it.get(k), float) and it.get("match") != "INFO":
                it[k] = fmt(it[k], 6 if k != "abs_diff" else 9)
    write_csv(f"{out}/repro_checks.csv", r_fields, items)
    print("wrote:", sorted(os.listdir(out)))
    n_no = sum(1 for i in items if i["match"] == "NO")
    print(f"repro_checks: {len(items)} items, NO={n_no}")


if __name__ == "__main__":
    main()
