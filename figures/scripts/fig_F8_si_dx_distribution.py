#!/usr/bin/env python3
"""fig_F8_si_dx_distribution.py -- SI figure F8 "distribution of the generated dx and the dx range of the training data" (single 86 mm, 1 panel).

Adapted from the plan of the SI figures F6-F9.
Input (read only): experiments/dx_gamma_analysis/results/merged_proposed_w2.csv (proposed model, main experiment w=2.0, dx before relaxation, 350 rows)
                  the paper's numbers table (reference values for the assertions; read only)
Output: figures/png/F8_si_dx_distribution.png (paperfig 1.1.0 save_fig, 600 dpi)
      figure_sources/si_f6_f9/F8_si_dx_distribution.{svg,pdf} (for visual inspection), F8_counts.csv (7 targets + summary row)
Stops without writing the CSV or the figure if even one value disagrees with the table (the table is not rewritten).
Run: with the evaluation python on a login node:  ulimit -v 8000000; export OMP_NUM_THREADS=1; timeout 900 ${MACE_PY} <this file>
"""
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)
import csv
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
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import paperfig
from paperfig import COL_1, OKABE_ITO, save_fig, setup_style

assert paperfig.__version__ == "1.1.0", paperfig.__version__

SRC = GC + "/experiments/dx_gamma_analysis/results/merged_proposed_w2.csv"
DRAFT = PA + "/figure_sources/si_f6_f9"
PNG_DIR = PA + "/figures/png"
NAME = "F8_si_dx_distribution"
CSV_OUT = DRAFT + "/F8_counts.csv"

# ============================================================ numbers table
NUM = {e["id"]: e["value"] for e in yaml.safe_load(open(_gbrel.CODE + "/data/paper_numbers/ledger_subset.yaml", encoding="utf-8"))["numbers"]}
LO, HI, ST = float(NUM["DSDXLO"]), float(NUM["DSDXHI"]), float(NUM["DSDXST"])
assert (LO, HI, ST) == (-0.75, 0.60, 0.15), (LO, HI, ST)
LEVELS = np.round(np.arange(LO, HI + 1e-9, ST), 10)
assert len(LEVELS) == 10 and abs(LEVELS[-1] - HI) < 1e-9, LEVELS
# 51 / 156 are the values in the definition field of the number DXOUT (dx < -0.75 is 51, dx > +0.60 is 156)
N_BELOW_REF, N_ABOVE_REF = 51, 156
DX_CELL_REF_A = 54.15601152562611        # DXOUT definition: dx = (cell_a_A - 54.15601152562611) / 2
N_REQ = 50

# ============================================================ inputs and assertions
d = pd.read_csv(SRC)
assert len(d) == int(NUM["DXN"]) == 350, len(d)
assert not d.duplicated(["chunk", "cif_index"]).any()
assert (d.groupby("target_id").size() == N_REQ).all() and sorted(d["target_id"].unique()) == list(range(7))
assert (d.groupby("target_id")["gamma_target_Jm2"].nunique() == 1).all()
assert np.isfinite(d["dx_A"]).all()
# definition of dx (recomputed from the cell a) and agreement of the column
dx_re = (d["cell_a_A"].to_numpy(float) - DX_CELL_REF_A) / 2.0
assert np.abs(dx_re - d["dx_A"].to_numpy(float)).max() < 1e-6, np.abs(dx_re - d["dx_A"]).max()
# agreement of the in/out-of-range column with the recomputation from the thresholds (tolerance 1e-9 at the boundary)
tol = 1e-9
below = (d["dx_A"] < LO - tol).to_numpy()
above = (d["dx_A"] > HI + tol).to_numpy()
out_re = below | above
assert (out_re == (d["dx_in_train_range"].to_numpy() == 0)).all(), "the dx_in_train_range column and the recomputation from the thresholds disagree"
assert (d["dx_side"][below] == "below").all() and (d["dx_side"][above] == "above").all() and (d["dx_side"][~out_re] == "inside").all()
n_out = int(out_re.sum())
assert n_out == int(NUM["DXOUT"]) == 207, n_out
assert int(below.sum()) == N_BELOW_REF and int(above.sum()) == N_ABOVE_REF, (int(below.sum()), int(above.sum()))
inner = d["is_inner"].to_numpy() == 1
near = d["near_P17"].to_numpy() == 1
assert (inner == (d["target_id"].to_numpy() <= 4)).all()
assert not (near & ~inner).any()
n_near = int((near & inner).sum())
n_near_out = int((near & inner & out_re).sum())
n_near_in = int((near & inner & ~out_re).sum())
assert n_near == int(NUM["NVNEAR"]) == 91, n_near
assert n_near_in == int(NUM["DXRIN"]) == 48 and n_near_out == int(NUM["DXROUT"]) == 43, (n_near_in, n_near_out)
# near_P17 agrees with |gamma_relaxed - gamma_target| <= P17 (inclusive, not rounded)
P17 = float(NUM["P17"])
near_re = inner & (np.abs(d["gamma_relaxed_Jm2"].to_numpy(float) - d["gamma_target_Jm2"].to_numpy(float)) <= P17 + 1e-12)
assert (near_re == near).all(), int((near_re != near).sum())

# ============================================================ aggregation CSV
TID = sorted(range(7), key=lambda t: float(d.loc[d["target_id"] == t, "gamma_target_Jm2"].iloc[0]))   # ascending target gamma
G = np.array([float(d.loc[d["target_id"] == t, "gamma_target_Jm2"].iloc[0]) for t in TID])
rows = []
for t, g in zip(TID, G):
    m = (d["target_id"] == t).to_numpy()
    rows.append(dict(target_id=t, gamma_target=round(float(g), 6), n=int(m.sum()), n_out=int((m & out_re).sum()),
                     n_below=int((m & below).sum()), n_above=int((m & above).sum()),
                     n_near=int((m & near).sum()), n_near_in=int((m & near & ~out_re).sum()), n_near_out=int((m & near & out_re).sum())))
tot = {k: sum(r[k] for r in rows) for k in ("n", "n_out", "n_below", "n_above", "n_near", "n_near_in", "n_near_out")}
assert tot == dict(n=350, n_out=207, n_below=51, n_above=156, n_near=91, n_near_in=48, n_near_out=43), tot
COLS = ["target_id", "gamma_target", "n", "n_out", "n_below", "n_above", "n_near", "n_near_in", "n_near_out"]
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader()
    for r in rows:
        wr.writerow(r)
    wr.writerow(dict(target_id="summary", gamma_target="", **tot))
print("wrote", CSV_OUT, "(header + 7 target rows + summary)")
print("assert OK: DXN=350 DXOUT=%d (below %d + above %d) NVNEAR=%d DXRIN=%d DXROUT=%d DSDXLO/HI/ST=%.2f/%.2f/%.2f levels=%d"
      % (n_out, int(below.sum()), int(above.sum()), n_near, n_near_in, n_near_out, LO, HI, ST, len(LEVELS)))
for r in rows:
    print("  t%d  target %.4f  out %2d/50 (below %d, above %d)  near %2d (in %d, out %d)"
          % (r["target_id"], r["gamma_target"], r["n_out"], r["n_below"], r["n_above"], r["n_near"], r["n_near_in"], r["n_near_out"]))

# ============================================================ drawing
setup_style()
matplotlib.rcParams.update({
    "figure.constrained_layout.use": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.labelsize": 8, "axes.titlesize": 7,
})
C_BLUE = OKABE_ITO[5]            # blue of the proposed model (same as C_P of fig_F3_tracking.py)
C_BAND = "#e6e6e6"
C_LVL = "0.55"
W_MM, H_MM = 86.0, 96.0
ML, MR, MT, MB = 12.5, 2.0, 6.5, 33.0
assert abs(COL_1 * 25.4 - W_MM) < 1e-9
XR = (0.0, 3.1)                  # same horizontal axis range as in Fig. 3
YLIM = (-1.7, 2.4)
Y_K = 2.12                       # display height of k/50 (above all points)
assert d["dx_A"].max() < Y_K - 0.1 and d["dx_A"].min() > YLIM[0]
JW = 0.22                        # full width of the horizontal offset per target (visualization only; not used in the aggregation)
rng = np.random.default_rng(20261020)
JIT = {t: rng.uniform(-JW / 2, JW / 2, N_REQ) for t in range(7)}      # fixed offset per target (deterministic)

fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
ax = fig.add_axes([ML / W_MM, MB / H_MM, 1 - (ML + MR) / W_MM, 1 - (MB + MT) / H_MM])
ax.axhspan(LO, HI, color=C_BAND, lw=0, zorder=0)
for lv in LEVELS:
    ax.axhline(lv, color=C_LVL, lw=0.4, zorder=1)
for t, g in zip(TID, G):
    s = d[d["target_id"] == t].sort_values(["chunk", "cif_index"])
    x = g + JIT[t]
    y = s["dx_A"].to_numpy(float)
    nr = (s["near_P17"].to_numpy() == 1)
    if t <= 4:
        ax.scatter(x[~nr], y[~nr], s=4.5, facecolors=[C_BLUE], edgecolors="none", alpha=0.35, zorder=3)
        ax.scatter(x[nr], y[nr], s=4.5, facecolors=[C_BLUE], edgecolors="none", alpha=0.9, zorder=4)
    else:
        ax.scatter(x, y, s=4.5, facecolors="none", edgecolors=[C_BLUE], linewidths=0.45, alpha=0.6, zorder=3)
    k = int((~s["dx_in_train_range"].to_numpy().astype(bool)).sum())
    ax.text(g, Y_K, "%d/%d" % (k, N_REQ), fontsize=6, ha="center", va="bottom", color="0.15", zorder=8)
ax.set_xlim(*XR)
ax.set_ylim(*YLIM)
ax.set_xticks(G)
ax.set_xticklabels(["%.2f" % g for g in G])
ax.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
ax.set_xlabel("Target γ [J/m²]")
ax.set_ylabel("Generated dx [Å]")
ax.set_title("k/50: structures with dx outside the training range", loc="left", fontsize=6.5, pad=2.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles = [
    Line2D([], [], marker="o", ls="none", mfc=C_BLUE, mec="none", ms=3.4, alpha=0.9),
    Line2D([], [], marker="o", ls="none", mfc=C_BLUE, mec="none", ms=3.4, alpha=0.35),
    Line2D([], [], marker="o", ls="none", mfc="none", mec=C_BLUE, mew=0.6, ms=3.4, alpha=0.6),
    Patch(facecolor=C_BAND, edgecolor="none"),
    Line2D([], [], color=C_LVL, lw=0.6),
]
labels = ["Near target (inner targets, ±0.10 J/m²)", "Not near target", "Target outside training range (not near)",
          "Training dx range (−0.75 to +0.60 Å)", "Training dx levels (0.15 Å steps)"]
fig.legend(handles=handles, labels=labels, loc="lower left", bbox_to_anchor=(ML / W_MM - 0.02, 0.0), ncol=1, frameon=False,
           handlelength=1.6, labelspacing=0.35, fontsize=6.5)

_, png = save_fig(fig, NAME, PNG_DIR)
fig.savefig(os.path.join(DRAFT, NAME + ".svg"), format="svg", metadata={"Date": None})
fig.savefig(os.path.join(DRAFT, NAME + ".pdf"), metadata={"Creator": "paperfig %s / fig_F8_si_dx_distribution.py" % paperfig.__version__})
plt.close(fig)
print("wrote", png)
print("wrote", os.path.join(DRAFT, NAME + ".svg"), "and .pdf")
print("DONE")
