#!/usr/bin/env python
"""Figures A1/A2/B1/B2/B3. Inputs: results/merged_proposed_w2.csv, train_ref.csv, A_dx_summary.csv, B_per_target_corr.csv, B_near_by_range.csv, analysis_log.json.
Uses the paper-figures skill (a copy of paperfig.py v1.1.0 is placed in the same directory). PNG only.
Run: ssh <LOGIN_NODE> 'ulimit -v 8000000; export OMP_NUM_THREADS=1; timeout 900 ${MACE_PY} -I ${GBREL_CODE}/gb_eval/experiments/dx_gamma_analysis/scripts/03_figures.py'
"""
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paperfig import COL_2, OKABE_ITO, save_fig, setup_style

ROOT = _gbrel.WORK + '/workspace' if os.path.exists(_gbrel.WORK) else _gbrel.WORK + '/workspace'
RES = f"{ROOT}/experiments/dx_gamma_analysis/results"
FIG = f"{ROOT}/experiments/dx_gamma_analysis/figures"
DX_LO, DX_HI = -0.75, 0.60
setup_style()
d = pd.read_csv(f"{RES}/merged_proposed_w2.csv")
tr = pd.read_csv(f"{RES}/train_ref.csv")
tr = tr[tr["split"] == "train"]
Bt = pd.read_csv(f"{RES}/B_per_target_corr.csv")
Bn = pd.read_csv(f"{RES}/B_near_by_range.csv")
log = json.load(open(f"{RES}/analysis_log.json"))
tg = d.groupby("target_id")["gamma_target_Jm2"].first()
ORDER = list(tg.sort_values().index)          # t5(0.129) < t0 < t1 < t2 < t3 < t4 < t6(2.938)
COL = {t: OKABE_ITO[i] for i, t in enumerate([5, 0, 1, 2, 3, 4, 6])}
COL[5] = OKABE_ITO[7]; COL[0] = OKABE_ITO[5]; COL[1] = OKABE_ITO[2]; COL[2] = OKABE_ITO[3]; COL[3] = OKABE_ITO[1]; COL[4] = OKABE_ITO[6]; COL[6] = OKABE_ITO[0]
rng = np.random.default_rng(1)
def tlabel(t, nl=True):
    return f"t{t}{chr(10) if nl else ' '}{tg[t]:.3f}"
def band(ax):
    ax.axhspan(DX_LO, DX_HI, color="#dddddd", alpha=0.6, lw=0, zorder=0)

# ------------------------------------------------------------------ A1
fig, ax = plt.subplots(figsize=(COL_2, COL_2 * 0.5))
band(ax)
for i, t in enumerate(ORDER):
    s = d[d.target_id == t]
    ax.boxplot([s.dx_A], positions=[i], widths=0.5, showfliers=False, medianprops=dict(color="black", lw=1.0), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    ins = s.dx_side == "inside"
    xj = i + rng.uniform(-0.18, 0.18, len(s))
    ax.scatter(xj[ins], s.dx_A[ins], s=5, color="#0072B2", alpha=0.8, lw=0, zorder=3, rasterized=True, label="inside range" if i == 0 else None)
    ax.scatter(xj[~ins], s.dx_A[~ins], s=5, color="#D55E00", alpha=0.8, lw=0, zorder=3, rasterized=True, label="outside range" if i == 0 else None)
    ax.text(i, 2.12, f"out {int((~ins).sum())}/{len(s)}\n(below {int((s.dx_side=='below').sum())}, above {int((s.dx_side=='above').sum())})", ha="center", va="top", fontsize=6)
ax.axhline(DX_LO, color="k", lw=0.5, ls="--"); ax.axhline(DX_HI, color="k", lw=0.5, ls="--")
ax.text(-0.45, DX_HI + 0.04, f"train max +{DX_HI:.2f}", ha="left", va="bottom", fontsize=6)
ax.text(-0.45, DX_LO - 0.04, f"train min {DX_LO:.2f}", ha="left", va="top", fontsize=6)
ax.set_xticks(range(7)); ax.set_xticklabels([tlabel(t) for t in ORDER])
ax.set_xlabel("target index and target $\\gamma$ [J/m$^2$] (ascending)")
ax.set_ylabel("dx = (a $-$ a$_t$)/2 [$\\AA$]")
ax.set_ylim(-1.8, 2.15)
ax.legend(loc="lower right", ncol=2, fontsize=6, handletextpad=0.2)
ax.set_title("proposed, w=2.0, main run: generated dx per target (n=50 each, N=350; grey = training range)", fontsize=8)
save_fig(fig, "A1_dx_by_target", FIG); plt.close(fig)

# ------------------------------------------------------------------ A2
fig, ax = plt.subplots(figsize=(COL_2 * 0.7, COL_2 * 0.4))
centers = np.round(np.arange(-10, 14) * 0.15, 2)
edges = np.append(centers - 0.075, centers[-1] + 0.075)
gen_c, _ = np.histogram(d.dx_A, bins=edges)
tr_lv = tr["dx_A"].round(2).value_counts().sort_index()
ax.bar(centers - 0.0, gen_c / len(d), width=0.13, color="#D55E00", alpha=0.85, label=f"generated (proposed, w=2.0), N={len(d)}")
ax.bar(tr_lv.index.values, tr_lv.values / len(tr), width=0.13, color="none", edgecolor="#0072B2", lw=1.0, label=f"training (10 levels), N={len(tr)}")
ax.axvline(DX_LO - 0.075, color="k", lw=0.5, ls="--"); ax.axvline(DX_HI + 0.075, color="k", lw=0.5, ls="--")
ax.set_xlabel("dx [$\\AA$] (bins of 0.15 $\\AA$ centred on the training grid)")
ax.set_ylabel("fraction of structures")
ax.legend(loc="upper left", fontsize=6)
nb, ni, na = [(d.dx_side == s).sum() for s in ("below", "inside", "above")]
ax.text(0.98, 0.95, f"generated: below {nb}, inside {ni}, above {na}\n(outside {nb+na}/350 = {100*(nb+na)/350:.1f}%)\ndashed = training bounds ($-$0.75, +0.60)", transform=ax.transAxes, ha="right", va="top", fontsize=6)
save_fig(fig, "A2_dx_train_vs_generated", FIG); plt.close(fig)

# ------------------------------------------------------------------ B1
fig, ax = plt.subplots(figsize=(COL_2, COL_2 * 0.55))
ax.axvspan(DX_LO, DX_HI, color="#dddddd", alpha=0.6, lw=0, zorder=0)
for t in ORDER:
    s = d[d.target_id == t]
    ax.scatter(s.dx_A, s.gamma_relaxed_Jm2, s=6, color=COL[t], lw=0, alpha=0.85, rasterized=True, label=f"t{t}: target {tg[t]:.3f}")
    ax.axhline(tg[t], color=COL[t], lw=0.5, ls=":")
ax.set_xlabel("dx [$\\AA$]")
ax.set_ylabel("$\\gamma_{\\rm relaxed}$ [J/m$^2$] (dotted = each target)")
ax.legend(loc="upper left", fontsize=6, ncol=2)
ax.text(0.99, 0.02, "N=350 (50/target). Grey = training dx range. Pooled across targets: Spearman 0.72 (confounded by target);\nwithin-target rank Spearman $-$0.013 (perm. p=0.74) -- see B_overall_corr.csv", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
save_fig(fig, "B1_dx_vs_gamma_relaxed", FIG); plt.close(fig)

# ------------------------------------------------------------------ B2
fig, axs = plt.subplots(1, 2, figsize=(COL_2, COL_2 * 0.42), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]
ax.axvspan(DX_LO, DX_HI, color="#dddddd", alpha=0.6, lw=0, zorder=0)
ax.axhspan(-0.10, 0.10, color="#009E73", alpha=0.18, lw=0, zorder=0)
for t in ORDER:
    s = d[d.target_id == t]
    ax.scatter(s.dx_A, s.dgamma_Jm2, s=5, color=COL[t], lw=0, alpha=0.85, rasterized=True, label=f"t{t}")
ax.set_xlabel("dx [$\\AA$]"); ax.set_ylabel("$\\Delta\\gamma=\\gamma_{\\rm relaxed}-\\gamma_{\\rm target}$ [J/m$^2$]")
ax.legend(loc="lower left", fontsize=6, ncol=4, columnspacing=0.6, handletextpad=0.1)
ax.set_title("(a) pooled, N=350; green = near (|$\\Delta\\gamma$|$\\leq$0.10); grey = train dx range", fontsize=7)
ax = axs[1]
bb = Bt[Bt.outcome == "dgamma"].set_index("target")
for i, t in enumerate(ORDER):
    r = bb.loc[t]
    sig = r.spearman_p_holm7 < 0.05
    ax.errorbar(i, r.spearman_rho, yerr=[[r.spearman_rho - r.spearman_ci_lo], [r.spearman_ci_hi - r.spearman_rho]], fmt="o", color=COL[t], mfc=COL[t] if sig else "white", ms=4, capsize=2, lw=0.8)
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(range(7)); ax.set_xticklabels([tlabel(t) for t in ORDER], fontsize=6)
ax.set_xlabel("target index and target $\\gamma$ [J/m$^2$]"); ax.set_ylabel("Spearman $\\rho$(dx, $\\Delta\\gamma$), within target")
ax.set_ylim(-0.8, 0.5)
ax.set_title("(b) per target, n=50 each; bars = 95% bootstrap CI;\nfilled = Holm-adjusted p<0.05 (7 tests, exploratory)", fontsize=7)
save_fig(fig, "B2_dx_vs_dgamma", FIG); plt.close(fig)

# ------------------------------------------------------------------ B3
fig, axs = plt.subplots(1, 2, figsize=(COL_2, COL_2 * 0.42), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]
a = Bn[(Bn.scope == "all7")]
w = 0.38
for i, t in enumerate(ORDER):
    for j, (cls, colr) in enumerate((("inside", "#0072B2"), ("outside", "#D55E00"))):
        r = a[(a.stratum == f"target_{t}") & (a.dx_class == cls)].iloc[0]
        x = i + (j - 0.5) * w
        ax.bar(x, r.rate if r.n else 0, width=w * 0.95, color=colr, alpha=0.85, label=("dx inside training range" if cls == "inside" else "dx outside") if i == 0 else None)
        if r.n:
            ax.errorbar(x, r.rate, yerr=[[max(0.0, r.rate - r.wilson_lo)], [max(0.0, r.wilson_hi - r.rate)]], color="k", lw=0.6, capsize=1.5)
        ax.text(x, 1.0, f"{int(r.n_near)}/{int(r.n)}", ha="center", va="bottom", fontsize=5, rotation=90)
ax.set_xticks(range(7)); ax.set_xticklabels([tlabel(t) for t in ORDER], fontsize=6)
ax.set_ylim(0, 1.18); ax.set_ylabel("near rate (|$\\Delta\\gamma$|$\\leq$0.10 J/m$^2$, converged)")
ax.set_xlabel("target index and target $\\gamma$ [J/m$^2$]")
ax.legend(loc="upper right", fontsize=6, ncol=2, bbox_to_anchor=(1.0, 0.74))
c = log["B"]["cmh_all7"]
ax.set_title(f"(a) by target; bars = Wilson 95% CI; labels = near/n. CMH OR(in/out)={c['mh_odds_ratio_inside_vs_outside']:.2f} [{c['mh_or_ci95'][0]:.2f}, {c['mh_or_ci95'][1]:.2f}]", fontsize=6.5)
ax = axs[1]
p = Bn[(Bn.scope == "inner5_pooled") & (Bn.stratum == "all") & (Bn.dx_class.isin(["below", "inside", "above"]))].set_index("dx_class")
for i, cls in enumerate(["below", "inside", "above"]):
    r = p.loc[cls]
    ax.bar(i, r.rate, color=["#CC79A7", "#0072B2", "#D55E00"][i], alpha=0.85, width=0.7)
    ax.errorbar(i, r.rate, yerr=[[max(0.0, r.rate - r.wilson_lo)], [max(0.0, r.wilson_hi - r.rate)]], color="k", lw=0.6, capsize=2)
    ax.text(i, 1.0, f"{int(r.n_near)}/{int(r.n)}", ha="center", va="bottom", fontsize=6)
ax.set_xticks(range(3)); ax.set_xticklabels(["below\n(<$-$0.75)", "inside", "above\n(>+0.60)"])
ax.set_ylim(0, 1.18); ax.set_ylabel("near rate"); ax.set_xlabel("dx class (pooled over t0-t4, N=250)")
ax.set_title("(b) pooled inner5, NOT target-adjusted\n(below is mostly t0)", fontsize=6.5)
save_fig(fig, "B3_near_rate_by_dx_range", FIG); plt.close(fig)
print("figs done")
