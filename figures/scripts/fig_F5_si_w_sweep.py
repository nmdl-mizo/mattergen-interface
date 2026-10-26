#!/usr/bin/env python3
"""fig_F5_si_w_sweep.py -- SI figure F5 "tracking of gamma before and after relaxation for each guidance strength w" (2 rows x 4 columns).

Adapted from the brief of the SI w-sweep figure.
Inputs (read only): experiments/relax_novelty/results/relax/additional/proposed_w{1.0,2.0,3.0,4.0}/chunk{00..13}/relax_rows.csv
                  experiments/data_rebuild/results/{gamma_dedup.csv,split.json} (training gamma range)
Output: figures/png/F5_si_w_sweep_tracking.png (paperfig 1.1.0 save_fig, double width, 600 dpi)
      figure_sources/si_w_sweep/F5_si_w_sweep_tracking.{svg,pdf} (for visual inspection)
      figures/F5_w_sweep_counts.csv (28 rows + summary; source of the numbers F5NDIFF and F5CLIPN)
Stops without writing the CSV or the figure if the near-target counts differ in even one cell from the reference values.
Run: with the evaluation python on a login node:  ulimit -v 8000000; export OMP_NUM_THREADS=1; timeout 900 ${MACE_PY} <this file>
"""
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _gbrel.WORK if os.path.isdir(_gbrel.WORK + '/workspace') else _gbrel.WORK
GC = ROOT + "/workspace"
PA = GC + "/paper"
sys.path.insert(0, PA + "/figures/scripts")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import paperfig
from paperfig import COL_2, OKABE_ITO, save_fig, setup_style

assert paperfig.__version__ == "1.1.0", paperfig.__version__

RELAX = GC + "/experiments/relax_novelty/results/relax/additional"
DATA = GC + "/experiments/data_rebuild/results"
DRAFT = PA + "/figure_sources/si_w_sweep"
CSV_OUT = PA + "/figures/F5_w_sweep_counts.csv"
PNG_DIR = PA + "/figures/png"
NAME = "F5_si_w_sweep_tracking"

WS = [1.0, 2.0, 3.0, 4.0]
BAND = 0.10          # P17. |gamma - target| <= BAND (inclusive, not rounded)
N_REQ = 50
YMAX = 3.0
TARGETS_REF = {0: 0.71918, 1: 0.92888, 2: 1.13858, 3: 1.34828, 4: 1.55798, 5: 0.32047, 6: 2.74640}
# reference values (near_pre, near_post), in the order target_id 0..6. Do not rewrite.
REF = {
    1.0: [(25, 26), (18, 21), (20, 17), (14, 13), (10, 10), (0, 0), (0, 0)],
    2.0: [(34, 34), (21, 22), (24, 25), (21, 21), (23, 21), (0, 0), (0, 0)],
    3.0: [(40, 39), (24, 23), (30, 30), (24, 25), (27, 26), (0, 0), (0, 0)],
    4.0: [(35, 35), (29, 30), (31, 30), (30, 26), (26, 28), (0, 0), (0, 0)],
}

# ============================================================ training gamma range (read in the same way as in the Fig. 3 script)
with open(DATA + "/gamma_dedup.csv", newline="", encoding="utf-8") as f:
    allrows = list(csv.DictReader(f))
rows_c = [r for r in allrows if r["converged"] == "True"]
sp = json.load(open(DATA + "/split.json"))
pos = sp["train_indices"]
assert sp["n_train"] == len(pos) and len(set(pos)) == len(pos) and all(0 <= p < len(rows_c) for p in pos)
train = np.array([float(rows_c[p]["gamma_Jm2"]) for p in pos])
TR_MIN, TR_MAX = float(train.min()), float(train.max())
assert round(TR_MIN, 4) == 0.5302 and round(TR_MAX, 4) == 2.5367, (TR_MIN, TR_MAX)   # N03a, N03b

# ============================================================ input loading
DF = {}
for w in WS:
    files = sorted(glob.glob("%s/proposed_w%.1f/chunk*/relax_rows.csv" % (RELAX, w)))
    assert len(files) == 14, (w, len(files))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert len(d) == 350, (w, len(d))
    assert (d["arm"] == "proposed").all() and (d["w"] == w).all()
    assert not d.duplicated(["chunk", "cif_index"]).any()
    assert (d.groupby("target_id").size() == N_REQ).all() and sorted(d["target_id"].unique()) == list(range(7))
    st = d["status"].value_counts().to_dict()
    if w == 4.0:
        assert st == {"ok_converged": 345, "excluded_pre": 5}, st
        ex = d[d["status"] == "excluded_pre"]
        assert (ex["target_id"] == 5).all() and ex["gamma_relaxed"].isna().all()
    else:
        assert st == {"ok_converged": 350}, (w, st)
    for t, g in TARGETS_REF.items():
        gt = d.loc[d["target_id"] == t, "gamma_target"]
        assert gt.nunique() == 1 and abs(float(gt.iloc[0]) - g) < 6e-6, (w, t, gt.unique())
    assert (d.loc[d["target_id"] <= 4, "is_inner"] == 1).all() and (d.loc[d["target_id"] >= 5, "is_inner"] == 0).all()
    DF[w] = d

# ============================================================ aggregation (only the points to draw; excluded rows enter neither the near-target counts nor the mean)
TID_ORDER = sorted(TARGETS_REF, key=lambda t: TARGETS_REF[t])      # target_id in ascending target gamma: 5,0,1,2,3,4,6
G = np.array([TARGETS_REF[t] for t in TID_ORDER])
PRE, POST = {}, {}       # (w, target_id) -> gamma array of the points to draw
rows = []
for w in WS:
    d = DF[w]
    ok = d[d["status"] == "ok_converged"].sort_values(["target_id", "chunk", "cif_index"])
    assert np.isfinite(ok["gamma_pre"]).all() and np.isfinite(ok["gamma_relaxed"]).all()
    for t in range(7):
        g = TARGETS_REF[t]
        s = ok[ok["target_id"] == t]
        pre, post = s["gamma_pre"].to_numpy(float), s["gamma_relaxed"].to_numpy(float)
        PRE[(w, t)], POST[(w, t)] = pre, post
        n_ex = int((d[d["target_id"] == t]["status"] == "excluded_pre").sum())
        assert len(pre) == len(post) == N_REQ - n_ex
        gt = float(d.loc[d["target_id"] == t, "gamma_target"].iloc[0])
        npre = int((np.abs(pre - gt) <= BAND).sum())
        npost = int((np.abs(post - gt) <= BAND).sum())
        rows.append(dict(
            w=w, target_id=t, gamma_target=round(gt, 6), is_inner=int(t <= 4), n_req=N_REQ, n_plotted=len(pre), n_excluded_pre=n_ex,
            near_pre=npre, near_post=npost, abs_diff_near=abs(npre - npost),
            mean_pre=round(float(pre.mean()), 6), mean_post=round(float(post.mean()), 6),
            median_pre=round(float(np.median(pre)), 6), median_post=round(float(np.median(post)), 6),
            n_above_ymax_pre=int((pre > YMAX).sum()), n_above_ymax_post=int((post > YMAX).sum())))
assert len(rows) == 28

# comparison with the reference values (all 28 cells)
bad = []
for r in rows:
    ref = REF[r["w"]][r["target_id"]]
    if (r["near_pre"], r["near_post"]) != ref:
        bad.append((r["w"], r["target_id"], (r["near_pre"], r["near_post"]), ref))
print("near-count cells checked: 28, mismatches:", len(bad))
if bad:
    print("MISMATCH (w, target_id, got(pre,post), ref(pre,post)):", bad)
    raise SystemExit("the near-target counts disagree with the reference values. Stopped without writing the CSV or the figure.")

ndiff = max(r["abs_diff_near"] for r in rows)
clip_pre = sum(r["n_above_ymax_pre"] for r in rows)
clip_post = sum(r["n_above_ymax_post"] for r in rows)
print("max |near_pre - near_post| =", ndiff, " (reference 4)")
print("n_above_ymax: pre total =", clip_pre, " post total =", clip_post, " F5CLIPN =", clip_pre + clip_post)
for r in rows:
    if r["n_above_ymax_pre"] or r["n_above_ymax_post"]:
        print("  above YMAX: w=%.1f t%d pre=%d post=%d" % (r["w"], r["target_id"], r["n_above_ymax_pre"], r["n_above_ymax_post"]))
        for nm, dct in (("pre", PRE), ("post", POST)):
            v = dct[(r["w"], r["target_id"])]
            print("     %s values > YMAX: %s" % (nm, [round(float(x), 3) for x in v[v > YMAX]]))

COLS = ["w", "target_id", "gamma_target", "is_inner", "n_req", "n_plotted", "n_excluded_pre", "near_pre", "near_post", "abs_diff_near",
        "mean_pre", "mean_post", "median_pre", "median_post", "n_above_ymax_pre", "n_above_ymax_post"]
summary = {c: "" for c in COLS}
summary.update(w="all", target_id="summary", abs_diff_near=ndiff, n_above_ymax_pre=clip_pre, n_above_ymax_post=clip_post)
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader()
    for r in rows:
        r = dict(r); r["w"] = "%.1f" % r["w"]
        wr.writerow(r)
    wr.writerow(summary)
print("wrote", CSV_OUT, " (header L1, rows L2-L29, summary L30)")

# ============================================================ drawing
setup_style()
matplotlib.rcParams.update({
    "figure.constrained_layout.use": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.labelsize": 8, "axes.titlesize": 8,
})
C_BLUE = OKABE_ITO[5]            # blue of the proposed model (same as C_P of fig_F3_tracking.py)
C_OUT = "#e6e6e6"                # background outside the training range (same as gray_out of Fig. 3)
C_GRAY = "0.35"
W_MM, H_MM = 178.0, 94.0
ML, MR, MT, MB, VGAP, HGAP = 15.0, 2.0, 7.0, 22.0, 5.0, 2.0
assert abs(COL_2 * 25.4 - W_MM) < 1e-9
XR = (0.0, 3.1)
YLIM = (0.0, YMAX)
JW = 0.12
rng = np.random.default_rng(20261020)
JIT = {t: rng.uniform(-JW / 2, JW / 2, N_REQ) for t in range(7)}   # fixed offset per target (common to all panels, deterministic)
TICKS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(W_MM / 25.4, H_MM / 25.4))
fig.subplots_adjust(left=ML / W_MM, right=1 - MR / W_MM, bottom=MB / H_MM, top=1 - MT / H_MM,
                    wspace=HGAP / ((W_MM - ML - MR - 3 * HGAP) / 4.0), hspace=VGAP / ((H_MM - MT - MB - VGAP) / 2.0))
LETTERS = "abcdefgh"
drawn_clip = {}
for ri, (dct, key) in enumerate(((PRE, "pre"), (POST, "post"))):
    for ci, w in enumerate(WS):
        ax = axs[ri, ci]
        ax.axhspan(TR_MIN, TR_MAX, color="#ffffff", lw=0, zorder=0)
        ax.axhspan(YLIM[0], TR_MIN, color=C_OUT, lw=0, zorder=0)
        ax.axhspan(TR_MAX, YLIM[1], color=C_OUT, lw=0, zorder=0)
        ax.plot(XR, XR, color=C_GRAY, lw=0.7, ls=(0, (4, 2)), zorder=1)
        for t in range(7):
            g = TARGETS_REF[t]
            y = dct[(w, t)]
            x = g + JIT[t][:len(y)]
            inside = t <= 4
            under = y <= YMAX
            if inside:
                ax.scatter(x[under], y[under], s=3.5, facecolors=[C_BLUE], edgecolors="none", alpha=0.85, zorder=4, clip_on=False)
            else:
                ax.scatter(x[under], y[under], s=3.5, facecolors="none", edgecolors=[C_BLUE], linewidths=0.45, zorder=4, clip_on=False)
            nclip = int((~under).sum())
            drawn_clip[(w, key, t)] = nclip
            if nclip:
                xc = x[~under]
                kw = dict(facecolors=[C_BLUE], edgecolors="none") if inside else dict(facecolors="none", edgecolors=[C_BLUE], linewidths=0.5)
                ax.scatter(xc, np.full(len(xc), YMAX), marker="^", s=9, zorder=5, clip_on=False, **kw)
                ax.text(g + 0.10, YMAX, str(nclip), fontsize=5.5, ha="left", va="center", color="0.15", zorder=8, clip_on=False)
            # near-target count k/50 (just below YMAX; vertical text so that it does not overlap its neighbors)
            near = int((np.abs(y - g) <= BAND).sum())
            ax.text(g, YMAX - 0.06, "%d/%d" % (near, N_REQ), fontsize=5.5, rotation=90, ha="center", va="top", color="0.15", zorder=8)
        # mean (taken over the drawn points only). The position is the target value
        means = np.array([float(dct[(w, t)].mean()) for t in TID_ORDER])
        assert (means < YMAX).all(), means
        ax.scatter(G, means, marker="D", s=14, c="black", edgecolors="white", linewidths=0.5, zorder=7, clip_on=False)
        ax.set_xlim(*XR)
        ax.set_ylim(*YLIM)
        ax.set_xticks(G)
        ax.set_xticklabels(["%.2f" % g for g in G], rotation=90)
        ax.set_yticks(TICKS)
        ax.set_facecolor("none")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.text(0.0, 1.03, "(%s)" % LETTERS[ri * 4 + ci], transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")
        if ri == 0:
            ax.set_title("w = %.1f" % w, fontsize=8, pad=9)
        if ci == 0:
            ax.set_ylabel(("Before relaxation" if ri == 0 else "After relaxation") + "\nGenerated γ [J/m²]")
        if ri == 1:
            ax.set_xlabel("Target γ [J/m²]")
            ax.tick_params(axis="x", labelsize=6.5)

# the number of drawn triangles must agree with the aggregation
for w in WS:
    for t in range(7):
        r = next(r for r in rows if r["w"] == w and r["target_id"] == t)
        assert drawn_clip[(w, "pre", t)] == r["n_above_ymax_pre"] and drawn_clip[(w, "post", t)] == r["n_above_ymax_post"]

handles = [
    Line2D([], [], marker="o", ls="none", mfc=C_BLUE, mec="none", ms=3.2, alpha=0.85),
    Line2D([], [], marker="o", ls="none", mfc="none", mec=C_BLUE, mew=0.7, ms=3.2),
    Line2D([], [], marker="D", ls="none", mfc="black", mec="white", mew=0.5, ms=4.2),
    Line2D([], [], color=C_GRAY, lw=0.7, ls=(0, (4, 2))),
    Patch(facecolor=C_OUT, edgecolor="0.6", linewidth=0.4),
    Line2D([], [], marker="^", ls="none", mfc=C_BLUE, mec="none", ms=3.8),
]
labels = ["Target inside training range", "Target outside training range", "Mean", "y = x", "Outside training energy range", "Above axis limit"]
fig.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
           handlelength=1.6, columnspacing=1.6, labelspacing=0.4, fontsize=7)

# saving: PNG with paperfig.save_fig, SVG/PDF for visual inspection
_, png = save_fig(fig, NAME, PNG_DIR)
fig.savefig(os.path.join(DRAFT, NAME + ".svg"), format="svg", metadata={"Date": None})
fig.savefig(os.path.join(DRAFT, NAME + ".pdf"), metadata={"Creator": "paperfig %s / fig_F5_si_w_sweep.py" % paperfig.__version__})
plt.close(fig)
print("wrote", png)
print("wrote", os.path.join(DRAFT, NAME + ".svg"), "and .pdf")
print("DONE")
