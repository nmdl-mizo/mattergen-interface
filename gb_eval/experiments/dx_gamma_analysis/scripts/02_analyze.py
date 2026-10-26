#!/usr/bin/env python
"""Part 1 analyses A/B/C/D. The inputs are only results/merged_proposed_w2.csv and results/train_ref.csv (outputs of 01_merge_proposed_w2.py).
Run (<LOGIN_NODE>, mace-mh env):
  ssh <LOGIN_NODE> 'ulimit -v 8000000; export OMP_NUM_THREADS=1; timeout 900 \
    ${MACE_PY} -I ${GBREL_CODE}/gb_eval/experiments/dx_gamma_analysis/scripts/02_analyze.py'
Output: results/A_*.csv/json, B_*.csv/json, C_*.csv, D_*.csv/json (all in the same directory). The random seed is fixed (SEED).
Choice of statistics (fixed beforehand; not changed after seeing the results):
  - Width of the neighbourhood 0.10 J/m^2 = manuscript P17 = the value assumed by the requester (they agree). Training dx range [-0.75, +0.60] A.
  - Intervals: proportions use Wilson 95%. The CI of Pearson uses Fisher z. The CI of Spearman uses a percentile bootstrap (5000, resampling rows with replacement).
  - Tests of contingency tables: the asymptotic chi2 and a permutation test with fixed marginals (chi2 statistic, 200000 permutations) are given together. The permutation test is primary (the number of cells with expected count <5 is recorded in the output cells_expected_lt5: measured 0). The lower limit of the permutation p is 1/200001 = 5e-6.
  - Stratification by target: Pearson with centring within target (dx and y minus the mean of each target), Spearman of ranks within target. The p value permutes dx within target (20000 times).
  - Linear model: OLS. y ~ target dummies + N (+ dx). Both the classical SE and HC3 are given. The addition of dx is a partial F test.
  - Multiplicity: tests per target are given with Holm correction (exploratory; not promoted to a main finding).
"""
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)
import csv
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

ROOT = _gbrel.WORK + '/workspace' if os.path.exists(_gbrel.WORK) else _gbrel.WORK + '/workspace'
RES = f"{ROOT}/experiments/dx_gamma_analysis/results"
MERGED = f"{RES}/merged_proposed_w2.csv"
TRAIN = f"{RES}/train_ref.csv"
SEED = 20261015
DX_LO, DX_HI = -0.75, 0.60
NEAR = 0.10
FACTOR = 16.0218
E_BULK = -3.7569042674314765
rng = np.random.default_rng(SEED)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ----------------------------------------------------------------------------- helpers
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        v = (m - i) * p[idx]
        run = max(run, v)
        adj[idx] = min(1.0, run)
    return adj


def desc(x):
    x = np.asarray(x, float)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return dict(n=len(x), mean=x.mean(), sd=x.std(ddof=1) if len(x) > 1 else float("nan"), median=med, q1=q1, q3=q3, iqr=q3 - q1,
                min=x.min(), max=x.max())


def perm_chi2(table, nperm=200000):
    """Permutation test of an r x c contingency table (marginals fixed). chi2 statistic. Returns (stat, p)."""
    t = np.asarray(table)
    R, C = t.shape
    rows = np.repeat(np.arange(R), t.sum(axis=1))
    cols = np.concatenate([np.repeat(np.arange(C), t[i]) for i in range(R)])
    rm = t.sum(1).astype(float)
    cm = t.sum(0).astype(float)
    exp = np.outer(rm, cm) / t.sum()
    def chi(cc):
        tab = np.bincount(rows * C + cc, minlength=R * C).reshape(R, C)
        return float(((tab - exp) ** 2 / exp).sum())
    obs = chi(cols)
    cnt = 0
    for _ in range(nperm):
        cnt += chi(rng.permutation(cols)) >= obs - 1e-9
    return obs, (cnt + 1) / (nperm + 1)


def fisher_z_ci(r, n):
    if n < 4 or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    return (float(np.tanh(z - 1.959963984540054 * se)), float(np.tanh(z + 1.959963984540054 * se)))


def boot_spearman(x, y, nb=5000):
    n = len(x)
    out = np.empty(nb)
    for b in range(nb):
        idx = rng.integers(0, n, n)
        xs, ys = x[idx], y[idx]
        out[b] = stats.spearmanr(xs, ys)[0] if (np.ptp(xs) > 0 and np.ptp(ys) > 0) else np.nan
    return tuple(np.nanpercentile(out, [2.5, 97.5]))


def safe_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(pearson_r=np.nan, pearson_p=np.nan, spearman_rho=np.nan, spearman_p=np.nan)
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    return dict(pearson_r=pr[0], pearson_p=pr[1], spearman_rho=sr[0], spearman_p=sr[1])


def ols(X, y, names):
    """OLS. Returns dict(coef, se, se_hc3, p, p_hc3, r2, rss, df_resid)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    res = y - X @ b
    rss = float(res @ res)
    df = n - k
    s2 = rss / df
    se = np.sqrt(np.diag(XtXi) * s2)
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    w = (res / (1 - h)) ** 2
    cov3 = XtXi @ (X.T * w) @ X @ XtXi
    se3 = np.sqrt(np.diag(cov3))
    tss = float(((y - y.mean()) ** 2).sum())
    return dict(names=names, coef=b, se=se, se_hc3=se3, p=2 * stats.t.sf(np.abs(b / se), df), p_hc3=2 * stats.t.sf(np.abs(b / se3), df),
                r2=1 - rss / tss, rss=rss, df=df, n=n)


def design(d, target_ids, extra_cols):
    cols = [np.ones(len(d))]
    names = ["const"]
    ts = sorted(target_ids)
    for t in ts[1:]:
        cols.append((d["target_id"].values == t).astype(float))
        names.append(f"target_{t}")
    for c in extra_cols:
        cols.append(d[c].values.astype(float))
        names.append(c)
    return np.column_stack(cols), names


def partial_f(r_small, r_big):
    q = r_small["df"] - r_big["df"]
    f = ((r_small["rss"] - r_big["rss"]) / q) / (r_big["rss"] / r_big["df"])
    return f, stats.f.sf(f, q, r_big["df"]), q


def within_target_stats(d, xcol, ycol, nperm=20000):
    """Pearson centred within target / Spearman of ranks within target, and the permutation p within target."""
    tg = d["target_id"].values
    x = d[xcol].values.astype(float)
    y = d[ycol].values.astype(float)
    groups = [np.where(tg == t)[0] for t in np.unique(tg)]
    def centered(v):
        out = np.empty_like(v)
        for g in groups:
            out[g] = v[g] - v[g].mean()
        return out
    def ranked(v):
        out = np.empty_like(v)
        for g in groups:
            out[g] = stats.rankdata(v[g]) - (len(g) + 1) / 2
        return out
    def corr(a, b):
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 0 else float("nan")
    xc, yc = centered(x), centered(y)
    xr, yr = ranked(x), ranked(y)
    r_c, r_s = corr(xc, yc), corr(xr, yr)
    cnt_c = cnt_s = 0
    for _ in range(nperm):
        perm = x.copy()
        for g in groups:
            perm[g] = x[rng.permutation(g)]
        cnt_c += abs(corr(centered(perm), yc)) >= abs(r_c) - 1e-12
        cnt_s += abs(corr(ranked(perm), yr)) >= abs(r_s) - 1e-12
    return dict(centered_pearson=r_c, centered_pearson_perm_p=(cnt_c + 1) / (nperm + 1),
                rankwithin_spearman=r_s, rankwithin_spearman_perm_p=(cnt_s + 1) / (nperm + 1), n=len(d), n_strata=len(groups))


def round_floats(o, nd=6):
    if isinstance(o, dict):
        return {k: round_floats(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(v, nd) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and np.isnan(o)) else round(float(o), nd if abs(o) >= 1e-3 or o == 0 else 8)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return round_floats(o.tolist(), nd)
    return o


def dump(name, obj):
    with open(f"{RES}/{name}", "w", encoding="utf-8") as f:
        json.dump(round_floats(obj), f, ensure_ascii=False, indent=1)


# ----------------------------------------------------------------------------- load
d = pd.read_csv(MERGED)
assert len(d) == 350 and d["struct_id"].is_unique
d["dx"] = d["dx_A"]
d["N"] = d["natoms"]
d["gam"] = d["gamma_relaxed_Jm2"]
d["dg"] = d["dgamma_Jm2"]
d["adg"] = d["abs_dgamma_Jm2"]
d["out"] = (d["dx_side"] != "inside").astype(int)
tr = pd.read_csv(TRAIN)
tr = tr[tr["split"] == "train"].copy()
assert len(tr) == 4419
inner = d[d["target_id"] <= 4].copy()
TLAB = {0: "t0 (0.731)", 1: "t1 (1.132)", 2: "t2 (1.533)", 3: "t3 (1.935)", 4: "t4 (2.336)", 5: "t5 (0.129)", 6: "t6 (2.938)"}
tgam = d.groupby("target_id")["gamma_target_Jm2"].first().to_dict()
log = {}

# ============================================================================= A
rows = []
def add_group(label, sub):
    dd = desc(sub["dx"].values)
    nb, ni, na = int((sub.dx_side == "below").sum()), int((sub.dx_side == "inside").sum()), int((sub.dx_side == "above").sum())
    lo, hi = wilson(nb + na, len(sub))
    rows.append(dict(group=label, **dd, n_below=nb, n_inside=ni, n_above=na, frac_outside=(nb + na) / len(sub), wilson_lo=lo, wilson_hi=hi))
add_group("all350", d)
add_group("inner5(t0-t4)", inner)
for t in range(7):
    add_group(f"target_{t}", d[d.target_id == t])
for n_ in sorted(d.N.unique()):
    add_group(f"N={n_}", d[d.N == n_])
A_sum = pd.DataFrame(rows)
A_sum.to_csv(f"{RES}/A_dx_summary.csv", index=False)

# train dx reference
trdx = desc(tr["dx_A"].values)
dump("A_train_dx_summary.json", dict(train_n=len(tr), **trdx, level_counts={str(k): int(v) for k, v in tr["dx_A"].round(2).value_counts().sort_index().items()}))

# contingency tables and tests
ct_side = pd.crosstab(d["target_id"], d["dx_side"]).reindex(columns=["below", "inside", "above"], fill_value=0)
ct_side.to_csv(f"{RES}/A_target_by_side_table.csv")
ct_out = pd.DataFrame({"inside": ct_side["inside"], "outside": ct_side["below"] + ct_side["above"]})
ct_out.to_csv(f"{RES}/A_target_by_out_table.csv")
chi2_out = stats.chi2_contingency(ct_out.values)
chi2_side = stats.chi2_contingency(ct_side.values + 0)
perm_out = perm_chi2(ct_out.values)
perm_side = perm_chi2(ct_side.values)
n_cell_exp_lt5_out = int((chi2_out[3] < 5).sum())
n_cell_exp_lt5_side = int((chi2_side[3] < 5).sum())
kw = stats.kruskal(*[d[d.target_id == t]["dx"].values for t in range(7)])
kw5 = stats.kruskal(*[d[d.target_id == t]["dx"].values for t in range(5)])
rho_tg = stats.spearmanr(d["gamma_target_Jm2"], d["dx"])
rho_tg5 = stats.spearmanr(inner["gamma_target_Jm2"], inner["dx"])
def cramers_v(chi2, tab):
    return float(np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1))))
# Fisher of target vs the rest (Holm)
pw = []
for t in range(7):
    a = int(ct_out.loc[t, "outside"]); b = int(ct_out.loc[t, "inside"])
    c = int(ct_out["outside"].sum() - a); e = int(ct_out["inside"].sum() - b)
    orr, p = stats.fisher_exact([[a, b], [c, e]])
    pw.append(dict(target=t, outside=a, inside=b, outside_rest=c, inside_rest=e, odds_ratio=orr, fisher_p=p))
padj = holm([x["fisher_p"] for x in pw])
for x, pa in zip(pw, padj):
    x["fisher_p_holm"] = pa
pd.DataFrame(pw).to_csv(f"{RES}/A_target_vs_rest_fisher.csv", index=False)
# by N
ctN = pd.crosstab(d["N"], d["out"])
ctN.columns = ["inside", "outside"]
ctN.to_csv(f"{RES}/A_N_by_out_table.csv")
permN = perm_chi2(ctN.values, nperm=100000)
rhoN_dx = stats.spearmanr(d["N"], d["dx"])
rhoN_tg = stats.spearmanr(d["N"], d["gamma_target_Jm2"])
# comparison with the training distribution of N
trN = tr["natoms"].value_counts(normalize=True).sort_index()
genN = d["N"].value_counts(normalize=True).sort_index()
pd.DataFrame({"train_frac": trN, "gen_frac": genN}).to_csv(f"{RES}/A_N_distribution_train_vs_gen.csv")
log["A"] = dict(
    n=350, n_below=int((d.dx_side == "below").sum()), n_inside=int((d.dx_side == "inside").sum()), n_above=int((d.dx_side == "above").sum()),
    n_within_1e6_of_bound=int(((d.dx - DX_LO).abs() < 1e-6).sum() + ((d.dx - DX_HI).abs() < 1e-6).sum()),
    chi2_out_target=dict(chi2=chi2_out[0], dof=chi2_out[2], p_asymptotic=chi2_out[1], cells_expected_lt5=n_cell_exp_lt5_out, cramers_v=cramers_v(chi2_out[0], ct_out.values),
                         perm_chi2=perm_out[0], perm_p=perm_out[1], perm_n=200000),
    chi2_side_target=dict(chi2=chi2_side[0], dof=chi2_side[2], p_asymptotic=chi2_side[1], cells_expected_lt5=n_cell_exp_lt5_side, cramers_v=cramers_v(chi2_side[0], ct_side.values),
                          perm_chi2=perm_side[0], perm_p=perm_side[1], perm_n=200000),
    kruskal_dx_by_target_all7=dict(H=kw[0], p=kw[1]), kruskal_dx_by_target_inner5=dict(H=kw5[0], p=kw5[1]),
    spearman_target_gamma_vs_dx_all7=dict(rho=rho_tg[0], p=rho_tg[1]), spearman_target_gamma_vs_dx_inner5=dict(rho=rho_tg5[0], p=rho_tg5[1]),
    N_by_out=dict(perm_chi2=permN[0], perm_p=permN[1], perm_n=100000), spearman_N_vs_dx=dict(rho=rhoN_dx[0], p=rhoN_dx[1]),
    spearman_N_vs_target_gamma=dict(rho=rhoN_tg[0], p=rhoN_tg[1]),
    test_assumptions="The tests of the contingency tables assume that the 50 structures within a target of the 350 structures are independent of each other (ignoring the shared seed within the same chunk and the same generation run of the model). The number of cells with expected count <5 is 0 (cells_expected_lt5). Even so, the permutation test is primary and the asymptotic chi2 is given together (the asymptotic p appears as 0 by rounding, so the lower limit is taken as the permutation p of 5e-6). The permutation test has fixed marginals (consistent with the design in which the requested number is fixed to 50 per target).",
)

# ============================================================================= B
ys = {"gamma_relaxed": "gam", "dgamma": "dg", "abs_dgamma": "adg"}
brow = []
for t in range(7):
    sub = d[d.target_id == t]
    for yn, yc in ys.items():
        c = safe_corr(sub["dx"], sub[yc])
        lo_r, hi_r = fisher_z_ci(c["pearson_r"], len(sub)) if not np.isnan(c["pearson_r"]) else (np.nan, np.nan)
        blo, bhi = boot_spearman(sub["dx"].values, sub[yc].values) if not np.isnan(c["spearman_rho"]) else (np.nan, np.nan)
        # OLS slope
        X = np.column_stack([np.ones(len(sub)), sub["dx"].values])
        fit = ols(X, sub[yc].values, ["const", "dx"])
        tcrit = stats.t.ppf(0.975, fit["df"])
        brow.append(dict(target=t, target_gamma=tgam[t], outcome=yn, n=len(sub), dx_median=sub["dx"].median(), **c, pearson_ci_lo=lo_r, pearson_ci_hi=hi_r,
                         spearman_ci_lo=blo, spearman_ci_hi=bhi, ols_slope_per_A=fit["coef"][1], ols_slope_se=fit["se"][1],
                         ols_slope_ci_lo=fit["coef"][1] - tcrit * fit["se"][1], ols_slope_ci_hi=fit["coef"][1] + tcrit * fit["se"][1], ols_r2=fit["r2"]))
B_t = pd.DataFrame(brow)
# Holm: Spearman p per outcome (7 targets)
for yn in ys:
    m = B_t.outcome == yn
    B_t.loc[m, "spearman_p_holm7"] = holm(B_t.loc[m, "spearman_p"].values)
    B_t.loc[m, "pearson_p_holm7"] = holm(B_t.loc[m, "pearson_p"].values)
B_t.to_csv(f"{RES}/B_per_target_corr.csv", index=False)

# overall (pooled) and stratified
orow = []
for scope, dd_ in (("all7", d), ("inner5", inner)):
    for yn, yc in ys.items():
        raw = safe_corr(dd_["dx"], dd_[yc])
        wt = within_target_stats(dd_, "dx", yc)
        orow.append(dict(scope=scope, outcome=yn, n=len(dd_), pooled_pearson_r=raw["pearson_r"], pooled_pearson_p=raw["pearson_p"],
                         pooled_spearman_rho=raw["spearman_rho"], pooled_spearman_p=raw["spearman_p"],
                         within_target_centered_pearson=wt["centered_pearson"], within_target_centered_pearson_perm_p=wt["centered_pearson_perm_p"],
                         within_target_rank_spearman=wt["rankwithin_spearman"], within_target_rank_spearman_perm_p=wt["rankwithin_spearman_perm_p"], n_strata=wt["n_strata"]))
pd.DataFrame(orow).to_csv(f"{RES}/B_overall_corr.csv", index=False)

# neighbourhood achievement: inside vs outside the range
def near_table(dd_, by_target=True):
    r = []
    groups = [("all", dd_)]
    if by_target:
        groups = [(f"target_{t}", dd_[dd_.target_id == t]) for t in sorted(dd_.target_id.unique())] + groups
    for lab, g in groups:
        for side_lab, mask in (("inside", g.dx_side == "inside"), ("outside", g.dx_side != "inside"), ("below", g.dx_side == "below"), ("above", g.dx_side == "above")):
            s = g[mask]
            k = int(s.near_P17.sum())
            lo, hi = wilson(k, len(s)) if len(s) else (np.nan, np.nan)
            r.append(dict(stratum=lab, dx_class=side_lab, n=len(s), n_near=k, rate=k / len(s) if len(s) else np.nan, wilson_lo=lo, wilson_hi=hi))
    return pd.DataFrame(r)
nt_all = near_table(d)
nt_all["scope"] = "all7"
nt_in = near_table(inner, by_target=False)
nt_in["scope"] = "inner5_pooled"
pd.concat([nt_all, nt_in]).to_csv(f"{RES}/B_near_by_range.csv", index=False)

# in/out comparison within target (Fisher) and CMH
def cmh(dd_, nperm=20000):
    num = den = 0.0
    strata = []
    for t in sorted(dd_.target_id.unique()):
        g = dd_[dd_.target_id == t]
        a = int(((g.out == 0) & (g.near_P17 == 1)).sum()); b = int(((g.out == 0) & (g.near_P17 == 0)).sum())
        c = int(((g.out == 1) & (g.near_P17 == 1)).sum()); e = int(((g.out == 1) & (g.near_P17 == 0)).sum())
        n_ = a + b + c + e
        num += a * e / n_
        den += b * c / n_
        if min(a + b, c + e) > 0 and (a + c) > 0 and (b + e) > 0:
            p = stats.fisher_exact([[a, b], [c, e]])[1]
            orr = stats.fisher_exact([[a, b], [c, e]])[0]
        else:
            p, orr = np.nan, np.nan
        strata.append(dict(target=t, near_inside=a, notnear_inside=b, near_outside=c, notnear_outside=e, fisher_p=p, odds_ratio_inside_vs_outside=orr,
                           informative=bool(min(a + b, c + e) > 0 and (a + c) > 0 and (b + e) > 0)))
    or_mh = num / den if den > 0 else float("inf")
    # 95% CI of ln(OR_MH) with the Robins-Breslow-Greenland variance
    Rs = Ss = 0.0; v1 = v2 = v3 = 0.0
    for t in sorted(dd_.target_id.unique()):
        g = dd_[dd_.target_id == t]
        a = ((g.out == 0) & (g.near_P17 == 1)).sum(); b = ((g.out == 0) & (g.near_P17 == 0)).sum()
        c = ((g.out == 1) & (g.near_P17 == 1)).sum(); e = ((g.out == 1) & (g.near_P17 == 0)).sum()
        n_ = a + b + c + e
        P_, Q_, R_, S_ = (a + e) / n_, (b + c) / n_, a * e / n_, b * c / n_
        Rs += R_; Ss += S_; v1 += P_ * R_; v2 += P_ * S_ + Q_ * R_; v3 += Q_ * S_
    if Rs > 0 and Ss > 0:
        var = v1 / (2 * Rs ** 2) + v2 / (2 * Rs * Ss) + v3 / (2 * Ss ** 2)
        mh_ci = (float(np.exp(np.log(or_mh) - 1.96 * np.sqrt(var))), float(np.exp(np.log(or_mh) + 1.96 * np.sqrt(var))))
    else:
        mh_ci = (float("nan"), float("nan"))
    # permutation: shuffle the out label within target, and use the total number of near and inside as the statistic
    tg = dd_.target_id.values
    near = dd_.near_P17.values
    outv = dd_.out.values
    groups = [np.where(tg == t)[0] for t in np.unique(tg)]
    obs = int(((outv == 0) & (near == 1)).sum())
    cnt_ge = cnt_le = 0
    for _ in range(nperm):
        o2 = outv.copy()
        for g in groups:
            o2[g] = outv[rng.permutation(g)]
        v = int(((o2 == 0) & (near == 1)).sum())
        cnt_ge += v >= obs
        cnt_le += v <= obs
    p2 = min(1.0, 2 * min((cnt_ge + 1) / (nperm + 1), (cnt_le + 1) / (nperm + 1)))
    return dict(mh_odds_ratio_inside_vs_outside=or_mh, mh_or_ci95=mh_ci, obs_near_inside=obs, perm_p_two_sided=p2, perm_n=nperm), strata
cmh_all, st_all = cmh(d)
cmh_in, st_in = cmh(inner)
pd.DataFrame(st_all).to_csv(f"{RES}/B_near_in_vs_out_by_target_fisher.csv", index=False)
# difference between the positive and negative sides (within inner5, only the pairs that can be compared with the target fixed)
sides_rows = []
for t in range(5):
    g = inner[inner.target_id == t]
    for a_lab, b_lab in (("below", "inside"), ("above", "inside"), ("below", "above")):
        ga, gb = g[g.dx_side == a_lab], g[g.dx_side == b_lab]
        if len(ga) == 0 or len(gb) == 0:
            sides_rows.append(dict(target=t, a=a_lab, b=b_lab, n_a=len(ga), n_b=len(gb), note="one side has 0 cases (cannot be compared)"))
            continue
        ka, kb = int(ga.near_P17.sum()), int(gb.near_P17.sum())
        orr, p = stats.fisher_exact([[ka, len(ga) - ka], [kb, len(gb) - kb]])
        sides_rows.append(dict(target=t, a=a_lab, b=b_lab, n_a=len(ga), n_b=len(gb), near_a=ka, near_b=kb, rate_a=ka / len(ga), rate_b=kb / len(gb), fisher_p=p, odds_ratio=orr))
pd.DataFrame(sides_rows).to_csv(f"{RES}/B_near_side_pairs_by_target.csv", index=False)
bi, ai = inner[inner.dx_side == "below"], inner[inner.dx_side == "above"]
f_ba = stats.fisher_exact([[int(bi.near_P17.sum()), len(bi) - int(bi.near_P17.sum())], [int(ai.near_P17.sum()), len(ai) - int(ai.near_P17.sum())]])

# linear model
mrows = []
def model_block(dd_, scope):
    tids = sorted(dd_.target_id.unique())
    for yn, yc in ys.items():
        y = dd_[yc].values
        dd_ = dd_.copy()
        dd_["above"] = (dd_.dx_side == "above").astype(float)
        dd_["below"] = (dd_.dx_side == "below").astype(float)
        dd_["Nc"] = dd_["N"] - 116.0
        X0, n0 = design(dd_, tids, [])
        X1, n1 = design(dd_, tids, ["Nc"])
        X2, n2 = design(dd_, tids, ["Nc", "dx"])
        X3, n3 = design(dd_, tids, ["Nc", "above", "below"])
        r0, r1, r2, r3 = ols(X0, y, n0), ols(X1, y, n1), ols(X2, y, n2), ols(X3, y, n3)
        f21 = partial_f(r1, r2)
        f31 = partial_f(r1, r3)
        for mname, fit, ref in (("M1_target", r0, None), ("M2_target+N", r1, None), ("M3_target+N+dx", r2, f21), ("M4_target+N+above+below", r3, f31)):
            for i, nm in enumerate(fit["names"]):
                if nm.startswith("target_") or nm == "const":
                    continue
                mrows.append(dict(scope=scope, outcome=yn, model=mname, term=nm, n=fit["n"], coef=fit["coef"][i], se=fit["se"][i], p=fit["p"][i], se_hc3=fit["se_hc3"][i], p_hc3=fit["p_hc3"][i],
                                  r2=fit["r2"], partial_F_vs_M2=(ref[0] if ref else np.nan), partial_F_p=(ref[1] if ref else np.nan), partial_F_df_num=(ref[2] if ref else np.nan), df_resid=fit["df"]))
            if fit is r0:
                mrows.append(dict(scope=scope, outcome=yn, model=mname, term="(R2 only)", n=fit["n"], r2=fit["r2"], df_resid=fit["df"]))
        mrows.append(dict(scope=scope, outcome=yn, model="delta_R2", term="dx added to M2", n=r2["n"], r2=r2["r2"] - r1["r2"]))
        mrows.append(dict(scope=scope, outcome=yn, model="delta_R2", term="above+below added to M2", n=r3["n"], r2=r3["r2"] - r1["r2"]))
model_block(d, "all7")
model_block(inner, "inner5")
pd.DataFrame(mrows).to_csv(f"{RES}/B_linear_models.csv", index=False)

log["B"] = dict(cmh_all7=cmh_all, cmh_inner5=cmh_in,
                near_overall=dict(k=int(d.near_P17.sum()), n=350, inner5_k=int(inner.near_P17.sum()), inner5_n=250),
                fisher_below_vs_above_inner5_pooled=dict(odds_ratio=f_ba[0], p=f_ba[1], below_n=len(bi), below_near=int(bi.near_P17.sum()), above_n=len(ai), above_near=int(ai.near_P17.sum())),
                near_0p10_equals_P17=bool((d["near_P17"] == d["near_0p10_flag_eq_P17"]).all()),
                n_unconverged=int((d.converged == 0).sum()))

# auxiliary on the relation between the neighbourhood achievement of a target and dx: number of near and inside/outside the range (all 350)
log["B"]["near_by_side_all7"] = {s: dict(n=int((d.dx_side == s).sum()), near=int(d[d.dx_side == s].near_P17.sum())) for s in ("below", "inside", "above")}
log["B"]["near_by_side_inner5"] = {s: dict(n=int((inner.dx_side == s).sum()), near=int(inner[inner.dx_side == s].near_P17.sum())) for s in ("below", "inside", "above")}

# ============================================================================= C
cmet = [("converged", "converged"), ("fmax_final_eVA", "fmax_final_eVA"), ("min_dist_pre_A", "min_dist_pre_A"), ("min_dist_relaxed_A", "min_dist_relaxed_A"),
        ("max_disp_free_A", "max_disp_free_A"), ("max_disp_fixed_A", "max_disp_fixed_A"), ("dE_relax_eV_derived", "dE_relax_eV_derived"),
        ("dgamma_relax_Jm2", "dgamma_relax_Jm2"), ("n_gb_post", "n_gb_post"), ("symfit_max_A_post", "symfit_max_A_post"), ("stage2_s2_max_A", "stage2_s2_max_A"),
        ("n_steps", "n_steps"), ("natoms", "natoms")]
crow = []
for lab, col in cmet:
    v = pd.to_numeric(d[col], errors="coerce")
    ok = v.notna()
    row = dict(metric=lab, n_valid=int(ok.sum()), n_missing=int((~ok).sum()))
    if ok.sum():
        row.update(min=v[ok].min(), median=v[ok].median(), max=v[ok].max(), n_unique=int(v[ok].nunique()))
    if ok.sum() > 3 and v[ok].nunique() > 1:
        c = safe_corr(d.loc[ok, "dx"], v[ok])
        row.update(spearman_rho_vs_dx_pooled=c["spearman_rho"], spearman_p_pooled=c["spearman_p"])
        dd2 = d[ok].copy()
        dd2["_v"] = v[ok]
        wt = within_target_stats(dd2, "dx", "_v", nperm=5000)
        row.update(within_target_rank_spearman=wt["rankwithin_spearman"], within_target_perm_p=wt["rankwithin_spearman_perm_p"])
        for s in ("below", "inside", "above"):
            row[f"median_{s}"] = v[ok & (d.dx_side == s)].median()
    else:
        row["note"] = "the value is constant (or there are few valid values), so the correlation is undefined"
    crow.append(row)
pd.DataFrame(crow).to_csv(f"{RES}/C_validity_metrics.csv", index=False)
cat = []
for col in ("status", "gb0_status_pre", "gb0_status_post", "stage2_status", "symfit_op_post", "stage2_s2_op", "excluded_pre"):
    for k, v in d[col].fillna("NA").value_counts().items():
        cat.append(dict(column=col, value=k, count=int(v)))
for thr in ("stage2_eq12_0.1", "stage2_eq12_0.032"):
    for k, v in d[thr].value_counts().items():
        cat.append(dict(column=thr, value=k, count=int(v)))
pd.DataFrame(cat).to_csv(f"{RES}/C_categorical_counts.csv", index=False)
# structures with the lowest minimum interatomic distance and dx
low = d.nsmallest(10, "min_dist_relaxed_A")[["struct_id", "target_id", "dx_A", "natoms", "min_dist_pre_A", "min_dist_relaxed_A", "gamma_relaxed_Jm2", "dgamma_Jm2"]]
low.to_csv(f"{RES}/C_lowest_min_dist_relaxed_top10.csv", index=False)

# ============================================================================= D
d["Vcell"] = d.cell_a_A * d.cell_b_A * d.cell_c_A
d["V_per_atom"] = d["Vcell"] / d["N"]
d["Ecell_minus_NEbulk"] = d["energy_relaxed_eV"] - d["N"] * E_BULK
d["gamma_from_E"] = d["Ecell_minus_NEbulk"] / (2 * d["area_A2"]) * FACTOR
res_id = float((d["gamma_from_E"] - d["gam"]).abs().max())
d["gamma_pre_from_E"] = (d["energy_pre_eV_derived"] - d["N"] * E_BULK) / (2 * d["area_A2"]) * FACTOR
res_pre = float((d["gamma_pre_from_E"] - d["gamma_pre_Jm2"]).abs().max())
area_u = d["area_A2"].round(9).unique().tolist()
eV_per_Jm2 = 2 * d["area_A2"].iloc[0] / FACTOR
drow = []
for lab, col in (("V_per_atom [A^3]", "V_per_atom"), ("Ecell-N*Ebulk [eV]", "Ecell_minus_NEbulk")):
    c = safe_corr(d["dx"], d[col])
    wt = within_target_stats(d, "dx", col, nperm=5000)
    c2 = safe_corr(d[col], d["gam"])
    wt2 = within_target_stats(d, col, "gam", nperm=5000)
    drow.append(dict(quantity=lab, spearman_vs_dx_pooled=c["spearman_rho"], within_target_rank_spearman_vs_dx=wt["rankwithin_spearman"],
                     spearman_vs_gamma_relaxed_pooled=c2["spearman_rho"], within_target_rank_spearman_vs_gamma_relaxed=wt2["rankwithin_spearman"]))
# regression of V/atom on dx: do dx and N alone nearly determine V/atom (algebraic)
Xv, nv = design(d, [0], ["dx", "N"])
fv = ols(Xv, d["V_per_atom"].values, nv)
pd.DataFrame(drow).to_csv(f"{RES}/D_energy_volume_relations.csv", index=False)

# dx dependence on the training side (reference. Not regarded as the same relation as in the generation)
trr = tr.copy()
trr["Nc"] = trr["natoms"] - 116.0
cn = sorted(trr["cnid"].unique())
cols = [np.ones(len(trr))]
names = ["const"]
for c in cn[1:]:
    cols.append((trr["cnid"].values == c).astype(float))
    names.append(f"cnid_{c}")
cols.append(trr["Nc"].values)
names.append("Nc")
cols.append(trr["dx_A"].values)
names.append("dx")
ft = ols(np.column_stack(cols), trr["gamma_Jm2"].values, names)
i_dx = names.index("dx")
i_N = names.index("Nc")
tcrit = stats.t.ppf(0.975, ft["df"])
tr_by_dx = trr.groupby(trr["dx_A"].round(2)).agg(n=("gamma_Jm2", "size"), gamma_median=("gamma_Jm2", "median"), gamma_q1=("gamma_Jm2", lambda s: s.quantile(.25)), gamma_q3=("gamma_Jm2", lambda s: s.quantile(.75)), N_median=("natoms", "median"))
tr_by_dx.to_csv(f"{RES}/D_train_gamma_by_dx_level.csv")
# training: correlation between dx and gamma within target (within template)
tmp = trr.rename(columns={"dx_A": "dx", "gamma_Jm2": "g", "cnid": "target_id"})
wt_tr = within_target_stats(tmp[["dx", "g", "target_id"]].reset_index(drop=True), "dx", "g", nperm=2000)
log["D"] = dict(
    area_A2_unique=area_u, eV_per_1Jm2_whole_cell=eV_per_Jm2, near_tol_in_eV=eV_per_Jm2 * NEAR,
    max_abs_gamma_identity_residual_relaxed_Jm2=res_id, max_abs_gamma_identity_residual_pre_Jm2=res_pre,
    cell_b_unique=sorted(set(d.cell_b_A.round(9))), cell_c_unique=sorted(set(d.cell_c_A.round(9))),
    V_per_atom_range=[d.V_per_atom.min(), d.V_per_atom.max()], V_per_atom_vs_dx_N_ols=dict(coef=dict(zip(nv, fv["coef"])), r2=fv["r2"]),
    train_gamma_ols=dict(n=len(trr), dx_slope_per_A=ft["coef"][i_dx], dx_slope_se=ft["se"][i_dx], dx_slope_ci=[ft["coef"][i_dx] - tcrit * ft["se"][i_dx], ft["coef"][i_dx] + tcrit * ft["se"][i_dx]],
                         N_slope_per_atom=ft["coef"][i_N], N_slope_se=ft["se"][i_N], r2=ft["r2"], n_templates=len(cn), note="gamma ~ cnid fixed effects + N + dx (training: 4419 rows of train; dx has 10 levels -0.75..0.60)"),
    train_within_template_dx_gamma=dict(centered_pearson=wt_tr["centered_pearson"], rank_spearman=wt_tr["rankwithin_spearman"], n_strata=wt_tr["n_strata"]),
)
# summary of the within-target OLS slope on the generation side (B_per_target_corr is shown again: gamma_relaxed vs dx)
log["D"]["generated_ols_slope_gamma_vs_dx_by_target"] = {int(r.target): dict(slope=r.ols_slope_per_A, ci=[r.ols_slope_ci_lo, r.ols_slope_ci_hi]) for r in B_t[B_t.outcome == "gamma_relaxed"].itertuples()}

# ============================================================================= additional analyses (added after seeing the first results. State in the report that they are exploratory)
# (1) relation between N x dx_side and N x dx (coupling of dx and N)
ctNs = pd.crosstab(d["N"], d["dx_side"]).reindex(columns=["below", "inside", "above"], fill_value=0)
ctNs.to_csv(f"{RES}/A_N_by_side_table.csv")
permNs = perm_chi2(ctNs.values, nperm=100000)
wt_Ndx = within_target_stats(d, "N", "dx", nperm=5000)
trN_dx = stats.spearmanr(tr["natoms"], tr["dx_A"])
tmp2 = tr.rename(columns={"natoms": "N", "dx_A": "dx", "cnid": "target_id"})[["N", "dx", "target_id"]].reset_index(drop=True)
wt_tr_Ndx = within_target_stats(tmp2, "N", "dx", nperm=2000)
# (2) volume/atom models M5: y ~ target + V/atom ; M6: y ~ target + V/atom + N
mrows2 = []
for scope, dd_ in (("all7", d), ("inner5", inner.assign(Vcell=lambda x: x.cell_a_A * x.cell_b_A * x.cell_c_A))):
    dd_ = dd_.copy()
    dd_["Vcell"] = dd_.cell_a_A * dd_.cell_b_A * dd_.cell_c_A
    dd_["V_per_atom"] = dd_["Vcell"] / dd_["N"]
    dd_["Nc"] = dd_["N"] - 116.0
    tids = sorted(dd_.target_id.unique())
    for yn, yc in ys.items():
        y = dd_[yc].values
        fits = {}
        for mname, cols_ in (("M2_target+N", ["Nc"]), ("M3_target+N+dx", ["Nc", "dx"]), ("M5_target+V/atom", ["V_per_atom"]), ("M6_target+V/atom+N", ["V_per_atom", "Nc"]), ("M7_target+V/atom+dx", ["V_per_atom", "dx"])):
            X, nm = design(dd_, tids, cols_)
            f_ = ols(X, y, nm)
            fits[mname] = f_
            for i, name_ in enumerate(nm):
                if name_.startswith("target_") or name_ == "const":
                    continue
                mrows2.append(dict(scope=scope, outcome=yn, model=mname, term=name_, n=f_["n"], coef=f_["coef"][i], se=f_["se"][i], se_hc3=f_["se_hc3"][i], p_hc3=f_["p_hc3"][i], r2=f_["r2"], df_resid=f_["df"],
                                   aic=f_["n"] * np.log(f_["rss"] / f_["n"]) + 2 * (len(nm) + 1)))
pd.DataFrame(mrows2).to_csv(f"{RES}/D_models_volume_per_atom.csv", index=False)
# (5) training side: dx of the train structures within +-0.10 of each target
trsup = []
for t in range(7):
    m_ = (tr["gamma_Jm2"] - tgam[t]).abs() <= NEAR
    s_ = tr[m_]
    if len(s_):
        q = s_["dx_A"].quantile([0, .25, .5, .75, 1.0]).values
        trsup.append(dict(target=t, target_gamma=tgam[t], n_train_within_0p10=len(s_), dx_min=q[0], dx_q1=q[1], dx_median=q[2], dx_q3=q[3], dx_max=q[4], N_median=s_["natoms"].median(),
                          n_dx_gt_0p30=int((s_["dx_A"] > 0.30).sum())))
    else:
        trsup.append(dict(target=t, target_gamma=tgam[t], n_train_within_0p10=0))
pd.DataFrame(trsup).to_csv(f"{RES}/D_train_support_dx_by_target.csv", index=False)
# (6) structures with large relaxation
big = d.reindex(d["dgamma_relax_Jm2"].abs().sort_values(ascending=False).index).head(10)[["struct_id", "target_id", "dx_A", "dx_side", "natoms", "min_dist_pre_A", "min_dist_relaxed_A", "max_disp_free_A", "dgamma_relax_Jm2", "dE_relax_eV_derived", "gamma_relaxed_Jm2", "near_P17"]]
big.to_csv(f"{RES}/C_largest_relaxation_top10.csv", index=False)
nbig = int((d["dgamma_relax_Jm2"].abs() > 0.05).sum())
nbig_side = d[d["dgamma_relax_Jm2"].abs() > 0.05]["dx_side"].value_counts().to_dict()
nmd = int((d["min_dist_pre_A"] < 2.3).sum())
log["extra"] = dict(
    N_by_side_perm=dict(chi2=permNs[0], perm_p=permNs[1], perm_n=100000),
    within_target_rank_spearman_N_vs_dx=dict(rho=wt_Ndx["rankwithin_spearman"], perm_p=wt_Ndx["rankwithin_spearman_perm_p"]),
    train_spearman_N_vs_dx=dict(rho=trN_dx[0], p=trN_dx[1]), train_within_template_rank_spearman_N_vs_dx=dict(rho=wt_tr_Ndx["rankwithin_spearman"]),
    n_abs_dgamma_relax_gt_0p05=nbig, side_of_those=nbig_side, n_min_dist_pre_lt_2p3=nmd,
    near_hits_outside_range=dict(k=int(((d.out == 1) & (d.near_P17 == 1)).sum()), of=int(d.near_P17.sum())),
)

dump("analysis_log.json", log)
print("done")
