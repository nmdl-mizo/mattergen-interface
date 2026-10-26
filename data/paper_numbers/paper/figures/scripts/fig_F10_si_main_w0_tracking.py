#!/usr/bin/env python3
"""fig_F10_si_main_w0_tracking.py -- SI figure F10 "tracking of gamma before and after relaxation for the main experiment (w = 2.0) and unconditional generation (w = 0)" (2 rows x 2 columns).

Adapted from fig_F5_si_w_sweep.py.
Inputs (read only): experiments/relax_novelty/results/relax/main/proposed_w2.0/chunk{00..13}/relax_rows.csv
                  experiments/verification/results/relax/proposed_w0/proposed_w0.0/chunk{00..13}/relax_rows.csv
                  experiments/data_rebuild/results/{gamma_dedup.csv,split.json} (training gamma range)
Output: figures/png/F10_si_main_w0_tracking.png (paperfig 1.1.0 save_fig, double width, 600 dpi)
      figure_sources/si_main_w0/F10_si_main_w0_tracking.{svg,pdf} (for visual inspection)
      figures/F10_main_w0_counts.csv (14 rows + summary; source of the number F10NDIFF)
Stops without writing the CSV or the figure if the near-target counts differ in even one cell from the reference values.
Run: with the evaluation python (environment-eval.yml): ulimit -v 8000000; export OMP_NUM_THREADS=1; timeout 900 python <this file>
"""
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '<ORIGINAL_ROOT>' if os.path.isdir('<ORIGINAL_ROOT>/workspace') else '<ORIGINAL_ROOT>'
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

DATA = GC + "/experiments/data_rebuild/results"
DRAFT = PA + "/figure_sources/si_main_w0"
CSV_OUT = PA + "/figures/F10_main_w0_counts.csv"
PNG_DIR = PA + "/figures/png"
NAME = "F10_si_main_w0_tracking"

# columns = (column name, w, input directory, column heading). Left = main experiment, proposed model w = 2.0; right = unconditional generation w = 0 (same seed)
COLS_DEF = [
    ("main_w2.0", 2.0, GC + "/experiments/relax_novelty/results/relax/main/proposed_w2.0",
     "Proposed model, main experiment (w = 2.0)"),
    ("uncond_w0", 0.0, GC + "/experiments/verification/results/relax/proposed_w0/proposed_w0.0",
     "Unconditional generation (w = 0, same seed)"),
]
CN = [c[0] for c in COLS_DEF]
BAND = 0.10          # P17. |gamma - target| <= BAND (inclusive, not rounded)
N_REQ = 50
YMAX = 3.0
# target gamma values (common to both columns)
TARGETS_REF = {0: 0.7308270, 1: 1.1321329, 2: 1.5334388, 3: 1.9347447, 4: 2.3360505, 5: 0.1288682, 6: 2.9380093}
# reference values (near_pre, near_post), in the order target_id 0..6. Do not rewrite.
REF = {
    "main_w2.0": [(35, 35), (25, 22), (28, 25), (9, 9), (0, 0), (0, 0), (0, 0)],
    "uncond_w0": [(12, 12), (8, 7), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0)],
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
for cname, w, dpath, _ in COLS_DEF:
    files = sorted(glob.glob("%s/chunk*/relax_rows.csv" % dpath))
    assert len(files) == 14, (cname, len(files))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert len(d) == 350, (cname, len(d))
    assert (d["w"] == w).all(), (cname, d["w"].unique())
    print("column %s: arm values = %s, w values = %s" % (cname, sorted(d["arm"].unique()), sorted(d["w"].unique())))
    assert not d.duplicated(["chunk", "cif_index"]).any()
    assert (d.groupby("target_id").size() == N_REQ).all() and sorted(d["target_id"].unique()) == list(range(7))
    st = d["status"].value_counts().to_dict()
    assert st == {"ok_converged": 350}, (cname, st)
    for t, g in TARGETS_REF.items():
        gt = d.loc[d["target_id"] == t, "gamma_target"]
        assert gt.nunique() == 1 and abs(float(gt.iloc[0]) - g) < 6e-6, (cname, t, gt.unique())
    assert (d.loc[d["target_id"] <= 4, "is_inner"] == 1).all() and (d.loc[d["target_id"] >= 5, "is_inner"] == 0).all()
    DF[cname] = d

# ============================================================ aggregation (only the points to draw)
TID_ORDER = sorted(TARGETS_REF, key=lambda t: TARGETS_REF[t])      # target_id in ascending target gamma: 5,0,1,2,3,4,6
G = np.array([TARGETS_REF[t] for t in TID_ORDER])
PRE, POST = {}, {}       # (column, target_id) -> gamma array of the points to draw
rows = []
for cname in CN:
    d = DF[cname]
    ok = d[d["status"] == "ok_converged"].sort_values(["target_id", "chunk", "cif_index"])
    assert np.isfinite(ok["gamma_pre"]).all() and np.isfinite(ok["gamma_relaxed"]).all()
    for t in range(7):
        g = TARGETS_REF[t]
        s = ok[ok["target_id"] == t]
        pre, post = s["gamma_pre"].to_numpy(float), s["gamma_relaxed"].to_numpy(float)
        PRE[(cname, t)], POST[(cname, t)] = pre, post
        n_ex = int((d[d["target_id"] == t]["status"] == "excluded_pre").sum())
        assert n_ex == 0 and len(pre) == len(post) == N_REQ - n_ex
        gt = float(d.loc[d["target_id"] == t, "gamma_target"].iloc[0])
        npre = int((np.abs(pre - gt) <= BAND).sum())
        npost = int((np.abs(post - gt) <= BAND).sum())
        rows.append(dict(
            column=cname, target_id=t, gamma_target=round(gt, 6), is_inner=int(t <= 4), n_req=N_REQ, n_plotted=len(pre), n_excluded_pre=n_ex,
            near_pre=npre, near_post=npost, abs_diff_near=abs(npre - npost),
            mean_pre=round(float(pre.mean()), 6), mean_post=round(float(post.mean()), 6),
            median_pre=round(float(np.median(pre)), 6), median_post=round(float(np.median(post)), 6),
            n_above_ymax_pre=int((pre > YMAX).sum()), n_above_ymax_post=int((post > YMAX).sum())))
assert len(rows) == 14

# comparison with the reference values (all 14 cells)
bad = []
for r in rows:
    ref = REF[r["column"]][r["target_id"]]
    if (r["near_pre"], r["near_post"]) != ref:
        bad.append((r["column"], r["target_id"], (r["near_pre"], r["near_post"]), ref))
print("near-count cells checked: %d, mismatches: %d" % (len(rows), len(bad)))
if bad:
    print("MISMATCH (column, target_id, got(pre,post), ref(pre,post)):", bad)
    raise SystemExit("the near-target counts disagree with the reference values. Stopped without writing the CSV or the figure.")

ndiff = max(r["abs_diff_near"] for r in rows)
clip_pre = sum(r["n_above_ymax_pre"] for r in rows)
clip_post = sum(r["n_above_ymax_post"] for r in rows)
print("max |near_pre - near_post| =", ndiff, " (reference 3)")
print("n_above_ymax: pre total =", clip_pre, " post total =", clip_post, " (reference 0)")
assert ndiff == 3, ndiff
assert clip_pre == 0 and clip_post == 0, (clip_pre, clip_post)

CSV_COLS = ["column", "target_id", "gamma_target", "is_inner", "n_req", "n_plotted", "n_excluded_pre", "near_pre", "near_post", "abs_diff_near",
            "mean_pre", "mean_post", "median_pre", "median_post", "n_above_ymax_pre", "n_above_ymax_post"]
summary = {c: "" for c in CSV_COLS}
summary.update(column="all", target_id="summary", abs_diff_near=ndiff, n_above_ymax_pre=clip_pre, n_above_ymax_post=clip_post)
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=CSV_COLS)
    wr.writeheader()
    for r in rows:
        wr.writerow(r)
    wr.writerow(summary)
print("wrote", CSV_OUT, " (header L1, rows L2-L15, summary L16)")

# ============================================================ drawing
setup_style()
matplotlib.rcParams.update({
    "figure.constrained_layout.use": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.labelsize": 8, "axes.titlesize": 8,
})
C_BLUE = OKABE_ITO[5]            # blue of the proposed model (same as in F5 and Fig. 3; the w = 0 column has the same color)
C_OUT = "#e6e6e6"                # background outside the training range (same as gray_out of Fig. 3)
C_GRAY = "0.35"
W_MM, H_MM = 178.0, 112.0
ML, MR, MT, MB, VGAP, HGAP = 23.0, 2.0, 9.0, 24.0, 5.0, 3.0
NCOL = len(COLS_DEF)
assert abs(COL_2 * 25.4 - W_MM) < 1e-9
XR = (0.0, 3.1)
YLIM = (0.0, YMAX)
JW = 0.12
rng = np.random.default_rng(20261020)
JIT = {t: rng.uniform(-JW / 2, JW / 2, N_REQ) for t in range(7)}   # fixed offset per target (common to all panels, deterministic)
TICKS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

fig, axs = plt.subplots(2, NCOL, sharex=True, sharey=True, figsize=(W_MM / 25.4, H_MM / 25.4))
fig.subplots_adjust(left=ML / W_MM, right=1 - MR / W_MM, bottom=MB / H_MM, top=1 - MT / H_MM,
                    wspace=HGAP / ((W_MM - ML - MR - (NCOL - 1) * HGAP) / float(NCOL)),
                    hspace=VGAP / ((H_MM - MT - MB - VGAP) / 2.0))
LETTERS = "abcd"
drawn_clip = {}
for ri, (dct, key) in enumerate(((PRE, "pre"), (POST, "post"))):
    for ci, (cname, w, _, ctitle) in enumerate(COLS_DEF):
        ax = axs[ri, ci]
        ax.axhspan(TR_MIN, TR_MAX, color="#ffffff", lw=0, zorder=0)
        ax.axhspan(YLIM[0], TR_MIN, color=C_OUT, lw=0, zorder=0)
        ax.axhspan(TR_MAX, YLIM[1], color=C_OUT, lw=0, zorder=0)
        ax.plot(XR, XR, color=C_GRAY, lw=0.7, ls=(0, (4, 2)), zorder=1)
        for t in range(7):
            g = TARGETS_REF[t]
            y = dct[(cname, t)]
            x = g + JIT[t][:len(y)]
            inside = t <= 4
            under = y <= YMAX
            if inside:
                ax.scatter(x[under], y[under], s=3.5, facecolors=[C_BLUE], edgecolors="none", alpha=0.85, zorder=4, clip_on=False)
            else:
                ax.scatter(x[under], y[under], s=3.5, facecolors="none", edgecolors=[C_BLUE], linewidths=0.45, zorder=4, clip_on=False)
            nclip = int((~under).sum())
            drawn_clip[(cname, key, t)] = nclip
            if nclip:
                xc = x[~under]
                kw = dict(facecolors=[C_BLUE], edgecolors="none") if inside else dict(facecolors="none", edgecolors=[C_BLUE], linewidths=0.5)
                ax.scatter(xc, np.full(len(xc), YMAX), marker="^", s=9, zorder=5, clip_on=False, **kw)
                ax.text(g + 0.10, YMAX, str(nclip), fontsize=5.5, ha="left", va="center", color="0.15", zorder=8, clip_on=False)
            # near-target count k/50 (just below YMAX; vertical text so that it does not overlap its neighbors)
            near = int((np.abs(y - g) <= BAND).sum())
            ax.text(g, YMAX - 0.06, "%d/%d" % (near, N_REQ), fontsize=5.5, rotation=90, ha="center", va="top", color="0.15", zorder=8)
        # mean (taken over the drawn points only). The position is the target value
        means = np.array([float(dct[(cname, t)].mean()) for t in TID_ORDER])
        assert (means < YMAX).all(), means
        ax.scatter(G, means, marker="D", s=14, c="black", edgecolors="white", linewidths=0.5, zorder=7, clip_on=False)
        ax.set_xlim(*XR)
        ax.set_ylim(*YLIM)
        ax.set_xticks(G)
        ax.set_xticklabels(["%.2f" % g for g in G], rotation=90)
        ax.set_yticks(TICKS)
        ax.set_facecolor("none")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.text(0.0, 1.03, "(%s)" % LETTERS[ri * NCOL + ci], transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")
        if ri == 0:
            ax.set_title(ctitle, fontsize=8, pad=9)
        if ci == 0:
            ax.set_ylabel(("Before relaxation" if ri == 0 else "After relaxation") + "\nγ of generated\nstructure [J/m²]")
        if ri == 1:
            ax.set_xlabel("Target γ [J/m²]")
            ax.tick_params(axis="x", labelsize=6.5)

# the number of drawn triangles must agree with the aggregation
for cname in CN:
    for t in range(7):
        r = next(r for r in rows if r["column"] == cname and r["target_id"] == t)
        assert drawn_clip[(cname, "pre", t)] == r["n_above_ymax_pre"] and drawn_clip[(cname, "post", t)] == r["n_above_ymax_post"]

# legend (a point above the limit is 0, so "Above axis limit" is not listed)
handles = [
    Line2D([], [], marker="o", ls="none", mfc=C_BLUE, mec="none", ms=3.2, alpha=0.85),
    Line2D([], [], marker="o", ls="none", mfc="none", mec=C_BLUE, mew=0.7, ms=3.2),
    Line2D([], [], marker="D", ls="none", mfc="black", mec="white", mew=0.5, ms=4.2),
    Line2D([], [], color=C_GRAY, lw=0.7, ls=(0, (4, 2))),
    Patch(facecolor=C_OUT, edgecolor="0.6", linewidth=0.4),
]
labels = ["Target inside training range", "Target outside training range", "Mean", "y = x", "Outside training energy range"]
fig.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
           handlelength=1.6, columnspacing=1.6, labelspacing=0.4, fontsize=7)

# saving: PNG with paperfig.save_fig, SVG/PDF for visual inspection
_, png = save_fig(fig, NAME, PNG_DIR)
fig.savefig(os.path.join(DRAFT, NAME + ".svg"), format="svg", metadata={"Date": None})
fig.savefig(os.path.join(DRAFT, NAME + ".pdf"), metadata={"Creator": "paperfig %s / fig_F10_si_main_w0_tracking.py" % paperfig.__version__})
plt.close(fig)
print("wrote", png)
print("wrote", os.path.join(DRAFT, NAME + ".svg"), "and .pdf")
print("DONE")
