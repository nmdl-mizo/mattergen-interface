#!/usr/bin/env python3
"""make_fig3.py -- Fig. 3 final (fine adjustment of v2 for the paper). Left: horizontal histogram of the training gamma / right: relaxed gamma vs target gamma scatter (main variant = scatter).
Changes from the earlier version: hatching outside the training range (no <pattern>/clipPath; line segments computed geometrically into a LineCollection),
4 legend entries, left short-line width 1.2 pt and length 2/3, title wording, figure width 178 mm, average diamonds 1.3 times larger, explicit margins by subplots_adjust.
Data loading, aggregation and checking against the fact table are the same code as in the earlier version. verification_final.csv must agree with the earlier verification csv in all columns (otherwise stop without drawing).
Inputs (read only) are the same as in the earlier version. The earlier version and figures/ are never overwritten.
Run: with the evaluation python (environment-eval.yml) on a login node:  ulimit -v 8000000; export OMP_NUM_THREADS=1; timeout 900 python <this file> [--bg {hatch_out,band_only,white_hatch}] [--left-ticks {color,gray}] [--all]
  Default (hatch_out, color): writes all main products Fig3.{svg,pdf,png,emf}, Fig3_left/right.svg, Fig3_cmp_bg_hatch_out.png.
  Other combinations: comparison PNG only (Fig3_cmp_bg_<bg>.png / Fig3_cmp_leftticks_gray.png). --all: main products + all comparison PNGs + Fig3_cmp_backgrounds.png.
"""
import argparse
import csv
import dataclasses
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ap = argparse.ArgumentParser()
ap.add_argument("--bg", choices=("gray_out", "hatch_out", "band_only", "white_hatch"), default="gray_out", help="background: white inside + plain light gray outside the training range, no hatching (adopted, default) / hatching outside the training range (earlier main variant) / light gray band only (as in the earlier version) / white inside + hatching outside")
ap.add_argument("--left-ticks", dest="left_ticks", choices=("color", "gray"), default="color", help="color of the target short lines of the left panel: target color (main variant) / single color #808080")
ap.add_argument("--all", action="store_true", help="main variant + all comparison PNGs")
ARGS = ap.parse_args()


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))
ROOT = '<ORIGINAL_ROOT>' if os.path.isdir('<ORIGINAL_ROOT>/workspace') else '<ORIGINAL_ROOT>'
GC = ROOT + "/workspace"
PA = GC + "/paper"
sys.path.insert(0, PA + "/figures/scripts")      # copy of paperfig.py (identical to the master copy by diff)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import paperfig
from paperfig import COL_1, save_fig, setup_style

assert paperfig.__version__ == "1.1.0", paperfig.__version__

RELAX = GC + "/experiments/relax_novelty/results"
VERIF = GC + "/experiments/verification/results"
DATA = GC + "/experiments/data_rebuild/results"
DXG = GC + "/experiments/dx_gamma_analysis/results"
FACT = PA + "/figure_sources/fig3/reference_table.md"
BAND = 0.10          # near the target = target +-0.10 J/m2 (inclusive; same <= as in the earlier tracking script)
N_PER = 50

# ============================================================ data loading (once)
rr = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(RELAX + "/relax/main/proposed_w2.0/chunk*/relax_rows.csv"))], ignore_index=True)
n_rows_raw = len(rr)
assert n_rows_raw == 350, n_rows_raw
assert (rr["status"] == "ok_converged").all() and (rr["arm"] == "proposed").all() and (rr["w"] == 2.0).all()
assert not rr.duplicated(["chunk", "cif_index"]).any()
rr = rr.sort_values(["target_id", "chunk", "cif_index"]).reset_index(drop=True)
assert (rr.groupby("target_id").size() == N_PER).all()
gtar = rr.groupby("target_id")["gamma_target"].first().to_numpy()          # target_id 0..6
assert (rr.groupby("target_id")["gamma_target"].nunique() == 1).all()
order = np.argsort(gtar)                                                    # target_id in ascending target gamma
TID = list(order)
G = gtar[order]                                                             # the 7 target values in ascending order
Y = [rr.loc[rr["target_id"] == t, "gamma_relaxed"].to_numpy() for t in TID]  # 50 per target
yall = np.concatenate(Y)

# training gamma (as in the earlier tracking script: position of the rows with converged==True x train_indices)
with open(DATA + "/gamma_dedup.csv", newline="", encoding="utf-8") as f:
    allrows = list(csv.DictReader(f))
rows_c = [r for r in allrows if r["converged"] == "True"]
sp = json.load(open(DATA + "/split.json"))
pos = sp["train_indices"]
assert sp["n_train"] == len(pos) and len(set(pos)) == len(pos) and all(0 <= p < len(rows_c) for p in pos)
train = np.array([float(rows_c[p]["gamma_Jm2"]) for p in pos])
TR_MIN, TR_MAX, TR_MED = float(train.min()), float(train.max()), float(np.median(train))
IN_RANGE = (G >= TR_MIN) & (G <= TR_MAX)                                    # targets inside the training range
assert IN_RANGE.sum() == 5 and (~IN_RANGE).sum() == 2 and (~IN_RANGE)[[0, 6]].all()
ntr = np.array([int((np.abs(train - g) <= BAND).sum()) for g in G])
near = np.array([int((np.abs(y - g) <= BAND).sum()) for y, g in zip(Y, G)])

# ============================================================ reading the fact table (with line numbers)
fact_lines = open(FACT, encoding="utf-8").read().split("\n")


def fnum(s):
    return float(s.replace("−", "-").replace("+", ""))


def paren(s):
    m = re.search(r"\(([-+−]?\d+\.\d+)\)", s)
    return fnum(m.group(1)) if m else None


# look up the table of section E (around lines 131-137) by its first cell
FE = {}
sec = None
for i, ln in enumerate(fact_lines, 1):
    if ln.startswith("## "):
        sec = ln[3:5].strip()
    if sec == "E." and ln.startswith("| ") and re.match(r"\| \d\.\d{4} \|", ln):
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        FE[c[0]] = dict(line=i, cells=c)
assert len(FE) == 7, list(FE)
fact_target_label = ["%.4f" % g for g in G]
assert sorted(FE) == sorted(fact_target_label), (sorted(FE), fact_target_label)

# constant rows other than section 0
line_train = next(i for i, l in enumerate(fact_lines, 1) if l.startswith("| train gamma |"))
line_tgt = next(i for i, l in enumerate(fact_lines, 1) if l.startswith("| target gamma (7 levels)"))
line_split = next(i for i, l in enumerate(fact_lines, 1) if l.startswith("| data split |"))
line_sum = next(i for i, l in enumerate(fact_lines, 1) if "five inner targets: **91**" in l)

# ============================================================ verification.csv (written before drawing)
V = []   # item, target, value, fact_value, fact_loc, match, note


def add(item, target, value, fact=None, loc="", match=None, note=""):
    V.append(dict(item=item, target_gamma=target, value=value, reference_table_value=fact, reference_table_loc=loc, match=match, note=note))


def cmp_tol(v, f, tol):
    return bool(abs(v - f) <= tol)


for k, g in enumerate(G):
    lab = "%.4f" % g
    fe = FE[lab]
    c, L = fe["cells"], "reference_table.md:L%d" % fe["line"]
    y = Y[k]
    n = len(y)
    mean, med = float(y.mean()), float(np.median(y))
    sd1, sd0 = float(y.std(ddof=1)), float(y.std(ddof=0))
    kin = int(near[k])
    fn, fd = [int(x) for x in c[5].split("/")]
    add("n_structures", lab, n, fd, L + "(near-target denominator)", n == fd, "requested number=50, all converged, 0 excluded")
    add("mean_gamma_relaxed", lab, round(mean, 6), paren(c[2]), L + " col3", cmp_tol(mean, paren(c[2]), 5e-7), "")
    add("median_gamma_relaxed", lab, round(med, 6), paren(c[3]), L + " col4", cmp_tol(med, paren(c[3]), 5e-7), "")
    if c[4].startswith("0."):
        add("std_gamma_relaxed_ddof1", lab, round(sd1, 6), float(c[4]), L + " col5", round(sd1, 3) == float(c[4]), "ddof=1 (sample standard deviation). The fact table has 3 digits")
    else:
        add("std_gamma_relaxed_ddof1", lab, round(sd1, 6), None, L + " col5", "n/a", "no value in the fact table (not used in the text). Computed from the CSV for reference")
    add("std_gamma_relaxed_ddof0", lab, round(sd0, 6), None, "", "n/a", "reference (population standard deviation). The fact table gives 0.158-0.292 for the inner 5")
    add("n_near_within_0p10", lab, kin, fn, L + " col6", kin == fn, "|gamma_relaxed-target|<=0.10")
    add("n_train_within_0p10", lab, int(ntr[k]), int(c[7]), L + " col8", int(ntr[k]) == int(c[7]), "|train gamma-target|<=0.10, train N=4419")
    add("mean_minus_target", lab, round(mean - g, 6), paren(c[8]), L + " col9", cmp_tol(mean - g, paren(c[8]), 5e-7), "")
# target values (precise values in line 97)
m = re.search(r"precise values: ([\d\., ]+)", fact_lines[line_tgt - 1])
prec = [float(x) for x in re.findall(r"\d\.\d{6}", fact_lines[line_tgt - 1])]
assert len(prec) == 7, prec
for g in G:
    j = int(np.argmin(np.abs(np.array(prec) - g)))
    add("target_gamma_precise", "%.4f" % g, round(float(g), 6), prec[j], "reference_table.md:L%d" % line_tgt, cmp_tol(g, prec[j], 5e-7), "relax_rows.csv gamma_target column (rounded to 6 digits)")
# totals
fsum = int(re.search(r"five inner targets: \*\*(\d+)\*\*", fact_lines[line_sum - 1]).group(1))
add("sum_near_inner5", "inner5", int(near[IN_RANGE].sum()), fsum, "reference_table.md:L%d" % line_sum, int(near[IN_RANGE].sum()) == fsum, "total number of near-target structures of the 5 inner targets")
add("sum_near_all7", "all7", int(near.sum()), 91, "reference_table.md:L%d(91/350)" % line_sum, int(near.sum()) == 91, "0 for the 2 outer targets")
sum_ntr_fact = sum(int(FE[("%.4f" % g)]["cells"][7]) for g in G[IN_RANGE])
add("sum_n_train_near_inner5", "inner5", int(ntr[IN_RANGE].sum()), sum_ntr_fact, "reference_table.md:L131-135 arithmetic sum of col8",
    int(ntr[IN_RANGE].sum()) == sum_ntr_fact, "the fact table gives no total (714+985+427+85+0). Checked against the sum of the per-target values")
# training summary
lt = fact_lines[line_train - 1]
f_min, f_max = [float(x) for x in re.findall(r"minimum (\d\.\d+), maximum (\d\.\d+)", lt)[0]]
f_med = float(re.search(r"median \d\.\d+ \((\d\.\d+)\)", lt).group(1))
add("train_n", "train", int(len(train)), 4419, "reference_table.md:L%d" % line_split, len(train) == 4419, "")
add("train_min", "train", round(TR_MIN, 6), f_min, "reference_table.md:L%d" % line_train, cmp_tol(TR_MIN, f_min, 5e-7), "")
add("train_max", "train", round(TR_MAX, 6), f_max, "reference_table.md:L%d" % line_train, cmp_tol(TR_MAX, f_max, 5e-7), "")
add("train_median", "train", round(TR_MED, 6), f_med, "reference_table.md:L%d" % line_train, cmp_tol(TR_MED, f_med, 5e-7), "")
add("n_targets_in_train_range", "all7", int(IN_RANGE.sum()), 5, "request (5 inside, 2 outside the range)/L%d" % line_tgt, int(IN_RANGE.sum()) == 5, "outside = 0.1289, 2.9380")
# cross-checks (between different files)
m2 = pd.read_csv(VERIF + "/summary_tables/m2_proposed_tracking.csv")
m2 = m2[(m2["stage"] == "post") & m2["group"].isin(["inner", "extrap_low", "extrap_high"])]
m2 = m2.reset_index(drop=True)
m3 = pd.read_csv(VERIF + "/summary_tables/m3_support.csv")
m3t = m3[m3["row_type"] == "target"]
for k, g in enumerate(G):
    r2 = m2.iloc[(m2["gamma_target"].astype(float) - g).abs().argsort().iloc[0]]
    r3 = m3t.iloc[(m3t["gamma_target"] - g).abs().argsort().iloc[0]]
    ok = cmp_tol(Y[k].mean(), r2["mean"], 5e-7) and cmp_tol(np.median(Y[k]), r2["median"], 5e-7) and int(near[k]) == int(r2["near_k"]) == int(r3["proposed_near_k_post"]) and int(ntr[k]) == int(r3["n_train_within_0p10"])
    add("crosscheck_vs_m2_m3(mean,median,near_k,n_train)", "%.4f" % g, bool(ok), True, "summary_tables/m2_proposed_tracking.csv(post), m3_support.csv", bool(ok), "")
mh = pd.read_csv(DXG + "/merged_proposed_w2.csv")
mh = mh.sort_values(["target_id", "chunk", "cif_index"]).reset_index(drop=True)
dmax = float(np.abs(mh["gamma_relaxed_Jm2"].to_numpy() - rr["gamma_relaxed"].to_numpy()).max()) if len(mh) == 350 else float("nan")
add("crosscheck_merged_proposed_w2_vs_relax_rows_maxabsdiff", "all", dmax, 0.0, "dx_gamma_analysis/results/merged_proposed_w2.csv gamma_relaxed_Jm2 (350 rows)", bool(dmax < 1e-9), "row order aligned by (target_id,chunk,cif_index)")
tref = pd.read_csv(DXG + "/train_ref.csv")
tref_tr = np.sort(tref.loc[tref["split"] == "train", "gamma_Jm2"].to_numpy())
dtr = float(np.abs(tref_tr - np.sort(train)).max()) if len(tref_tr) == len(train) else float("nan")
add("crosscheck_train_ref_vs_gamma_dedup_split_maxabsdiff", "train", dtr, 0.0, "dx_gamma_analysis/results/train_ref.csv (split==train, gamma_Jm2, %d rows)" % len(tref_tr), bool(dtr < 1e-9), "compared after sorting")
add("rows_raw_relax_rows", "all", n_rows_raw, 350, "relax_rows.csv 14 chunks", n_rows_raw == 350, "all status=ok_converged, 0 excluded, 0 missing")
add("y_range_all350", "all", "%.6f..%.6f" % (yall.min(), yall.max()), None, "", "n/a", "to check the axis range")


# ============================================================ verification_final.csv (written before drawing; compared with the earlier verification csv files)
PREV = PA + "/figure_sources/fig3/verification_comparison.csv"
V2CSV = PA + "/figure_sources/fig3/verification_second.csv"
vdf = pd.DataFrame(V)
vnew = pd.read_csv(__import__("io").StringIO(vdf.to_csv(index=False)), dtype=str, keep_default_na=False)   # make the string representation the same as in the earlier run by a CSV round trip
vprev = pd.read_csv(PREV, dtype=str, keep_default_na=False)
assert len(vnew) == len(vprev) and not vnew.duplicated(["item", "target_gamma"]).any(), (len(vnew), len(vprev))
mg = vnew.merge(vprev[["item", "target_gamma", "value", "match"]].rename(columns={"value": "prev_value", "match": "prev_match_fact"}),
                on=["item", "target_gamma"], how="outer", indicator=True)
assert (mg["_merge"] == "both").all(), mg[mg["_merge"] != "both"]
mg["match"] = (mg["value"] == mg["prev_value"]).map({True: "True", False: "False"})        # string agreement with the value of the earlier verification.csv
vout = pd.DataFrame({"item": mg["item"], "target_gamma": mg["target_gamma"], "value": mg["value"], "prev_value": mg["prev_value"],
                     "match": mg["match"], "reference_table_value": mg["reference_table_value"], "reference_table_loc": mg["reference_table_loc"],
                     "match_fact": vnew["match"], "note": mg["note"]})
vout = vout.set_index(["item", "target_gamma"]).loc[list(zip(vnew["item"], vnew["target_gamma"]))].reset_index()
bad_prev = vout[vout["match"] != "True"]
bad_fact = vout[vout["match_fact"] == "False"]
print("verification_final rows:", len(vout), " mismatch_vs_prev:", len(bad_prev), " mismatch_vs_reference_table:", len(bad_fact))
if len(bad_prev) or len(bad_fact):
    print(bad_prev.to_string()); print(bad_fact.to_string())
    raise SystemExit("disagreement with the earlier verification.csv or with the fact table. Drawing aborted.")
v2df = pd.read_csv(V2CSV, dtype=str, keep_default_na=False)
vchk = pd.read_csv(__import__("io").StringIO(vout.to_csv(index=False)), dtype=str, keep_default_na=False)
assert vchk.shape == v2df.shape and (vchk.columns == v2df.columns).all() and (vchk.values == v2df.values).all(), "disagreement with the earlier verification csv"
print("verification_final == verification_v2 (all %d rows x %d cols, string-exact)" % v2df.shape)
vout.to_csv(OUT + "/verification_final.csv", index=False, encoding="utf-8")

# ============================================================ font (Arial; if the server has none, register the Liberation Sans file under the name "Arial")
import matplotlib.font_manager as fm
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_hex

if not any(e.name == "Arial" for e in fm.fontManager.ttflist):
    for f in sorted(glob.glob("/usr/share/fonts/liberation-sans/LiberationSans-*.ttf")):
        fm.fontManager.addfont(f)
    for e in [e for e in fm.fontManager.ttflist if e.name == "Liberation Sans"]:
        fm.fontManager.ttflist.append(dataclasses.replace(e, name="Arial"))
FONT_FILE = fm.findfont("Arial", fallback_to_default=False)
print("Arial resolves to:", FONT_FILE)

# ============================================================ drawing conventions
setup_style()
matplotlib.rcParams.update({
    "font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.constrained_layout.use": False, "axes.prop_cycle": matplotlib.cycler(color=["#000000"]),
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
})
from matplotlib.legend_handler import HandlerBase

W_MM, H_MM = 178.0, 90.0                                # two-column width 178 mm (check 178+-2)
ML, MR, MB, MT, GAP = 11.5, 1.5, 11.0, 6.0, 2.5         # margins [mm] (explicit via subplots_adjust)
WAX = W_MM - ML - MR - GAP
WL, WR = WAX / 4.0, WAX * 3.0 / 4.0                     # left:right = 1:3
HP = H_MM - MB - MT
assert abs(ML + WL + GAP + WR + MR - W_MM) < 1e-9 and abs(WR / WL - 3.0) < 1e-9
YLIM = (0.0, 3.1)
XR = (0.0, 3.1)
assert yall.min() > YLIM[0] and yall.max() < YLIM[1] and G.min() > XR[0] and G.max() < XR[1]
JW = 0.05                                               # full width of the horizontal jitter (visualization only; not used in the aggregation)
rng = np.random.default_rng(20261019)
JIT = [rng.uniform(-JW / 2, JW / 2, N_PER) for _ in range(7)]
CMAP = matplotlib.colormaps["viridis"]
COLS = [CMAP(0.0 + 0.85 * (g - G.min()) / (G.max() - G.min())) for g in G]     # same rule as in the earlier version
C_BAND = "#e6e6e6"
C_HATCH = "#c8c8c8"
C_OUT = "#e6e6e6"                                       # gray_out: plain light gray outside the training range (vertical axis gamma)
LW_HATCH = 0.4
HATCH_DY = 0.06                                         # spacing of the hatch lines in the y direction [J/m2] (45 degrees in the mm space of the drawing)
C_HIST, C_HIST_E = "#9A9A9A", "#555555"
C_GRAY = "0.35"
C_LEFT_GRAY = "#808080"
LEFT_TICK_LW = 1.2
LEFT_TICK_FRAC = 0.16 * 2.0 / 3.0                       # 16% of the axis width in the earlier version (0.84*XMAX to XMAX); here 2/3 of that
DIAMOND_SCALE = 1.3                                     # average diamond: side 1.3 times (earlier s=22 -> s=22*1.3**2)
BINS = np.arange(0.5, 2.6001, 0.05)
HCNT, _ = np.histogram(train, bins=BINS)
assert HCNT.sum() == len(train) == 4419
XMAX_L = int(np.ceil(HCNT.max() / 0.78 / 200.0) * 200)          # same axis range as in the earlier version
assert HCNT.max() < 0.78 * XMAX_L
print("train hist max/bin:", int(HCNT.max()), " left xlim:", XMAX_L)
print("COLORS:", [(float(round(g, 4)), to_hex(c)) for g, c in zip(G, COLS)])


def hatch_segments(w_mm, xrange_):
    """Returns, by geometric computation, the 45-degree hatch line segments placed outside the training range (below: 0 to TR_MIN, above: TR_MAX to the top of YLIM).
    The face is the mm space [0,w_mm]x[0,HP]. Lines y = x + c, c = k*sp (k integer; same phase in the lower and upper bands). End points are cut inside the rectangle, so no clip is needed.
    Returns: (low_segments, high_segments); each segment = [(x_data, y_data), (x_data, y_data)]"""
    sp = HATCH_DY * HP / YLIM[1]                        # spacing in the y direction [mm]
    def region(ya, yb):
        out = []
        for k in range(int(np.floor((ya - w_mm) / sp)) - 1, int(np.ceil(yb / sp)) + 2):
            c = k * sp
            xa, xb = max(0.0, ya - c), min(w_mm, yb - c)
            if xb - xa > 1e-6:
                out.append([(xa / w_mm * xrange_, (xa + c) / HP * YLIM[1]), (xb / w_mm * xrange_, (xb + c) / HP * YLIM[1])])
        return out
    return region(0.0, TR_MIN / YLIM[1] * HP), region(TR_MAX / YLIM[1] * HP, HP)


def draw_bg(ax, bg, w_mm, xrange_, tag):
    """Background (lowest zorder): the band of the training range and the hatching outside it."""
    if bg == "gray_out":
        # white inside the range; plain light gray only outside it (gamma<TR_MIN, gamma>TR_MAX). No hatching. zorder 0 (behind points, lines and histograms)
        ax.axhspan(TR_MIN, TR_MAX, color="#ffffff", lw=0, zorder=0).set_gid("train_band_" + tag)
        ax.axhspan(YLIM[0], TR_MIN, color=C_OUT, lw=0, zorder=0).set_gid("gray_outside_low_" + tag)
        ax.axhspan(TR_MAX, YLIM[1], color=C_OUT, lw=0, zorder=0).set_gid("gray_outside_high_" + tag)
        return
    band = "#ffffff" if bg == "white_hatch" else C_BAND
    ax.axhspan(TR_MIN, TR_MAX, color=band, lw=0, zorder=0).set_gid("train_band_" + tag)
    if bg in ("hatch_out", "white_hatch"):
        lo, hi = hatch_segments(w_mm, xrange_)
        for nm, sg in (("low", lo), ("high", hi)):
            h = LineCollection(sg, colors=C_HATCH, linewidths=LW_HATCH, capstyle="butt", zorder=0.5)
            h.set_gid("hatch_outside_%s_%s" % (nm, tag))
            ax.add_collection(h)
        print("hatch %s: %d + %d line segments (dy=%.3f J/m2 = %.3f mm)" % (tag, len(lo), len(hi), HATCH_DY, HATCH_DY * HP / YLIM[1]))


def draw_left(ax, show_y, bg, left_ticks):
    pc = PolyCollection([[(0, e0), (c, e0), (c, e1), (0, e1)] for c, e0, e1 in zip(HCNT, BINS[:-1], BINS[1:]) if c > 0],
                        facecolors=C_HIST, edgecolors=C_HIST_E, linewidths=0.3, zorder=2)
    pc.set_gid("train_hist")
    ax.add_collection(pc)
    draw_bg(ax, bg, WL, XMAX_L, "left")
    x0 = (1.0 - LEFT_TICK_FRAC) * XMAX_L
    segs = [[(x0, g), (XMAX_L, g)] for g in G]
    lc = LineCollection(segs, colors=(COLS if left_ticks == "color" else [C_LEFT_GRAY] * 7), linewidths=LEFT_TICK_LW, zorder=4,
                        linestyles=["solid" if IN_RANGE[k] else (0, (2.2, 1.6)) for k in range(7)])
    lc.set_gid("target_ticks")
    ax.add_collection(lc)
    ax.set_xlim(0, XMAX_L)
    ax.set_xticks(list(range(0, XMAX_L + 1, 200)))
    ax.set_xlabel("Count per 0.05 J/m² bin")
    ax.set_title("Training data (N = %s)" % format(len(train), ","), loc="left", fontsize=8, pad=3)
    ax.title.set_gid("title_left")
    ax.set_ylim(*YLIM)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    if show_y:
        ax.set_ylabel("γ [J/m²]")
    else:
        ax.tick_params(labelleft=False)
    ax.set_gid("ax_left")


class OutsideKey:
    def __init__(self, hatch, kind="marker"):
        self.hatch = hatch
        self.kind = kind          # "marker": hollow circle (earlier behavior) / "swatch": light gray rectangle (background legend of gray_out) / "hollow": hollow circle only (target legend of gray_out)


class HandlerOutside(HandlerBase):
    """Legend sample: a small 45-degree hatch piece + hollow circle for the hatched background, hollow circle only for band_only (built with Line2D instead of Patch.hatch=<pattern>)."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x0, y0 = -xdescent, -ydescent
        arts = []
        if orig_handle.kind == "swatch":
            arts.append(Rectangle((x0, y0 + 0.1 * height), width, 0.8 * height, facecolor=C_OUT, edgecolor="0.6", linewidth=0.4, transform=trans))
            return arts
        if orig_handle.kind == "hollow":
            arts.append(Line2D([x0 + width / 2], [y0 + height / 2], marker="o", ls="none", mfc="none", mec="0.4", mew=0.6, ms=3.0, transform=trans))
            return arts
        if orig_handle.hatch:
            sp = 2.2
            for c in np.arange(-width, height, sp):
                xa, xb = max(0.0, -c), min(width, height - c)
                if xb - xa > 1e-6:
                    arts.append(Line2D([x0 + xa, x0 + xb], [y0 + xa + c, y0 + xb + c], color=C_HATCH, lw=LW_HATCH, solid_capstyle="butt", transform=trans))
        arts.append(Line2D([x0 + width / 2], [y0 + height / 2], marker="o", ls="none", mfc="none", mec="0.4", mew=0.6, ms=3.0, transform=trans))
        return arts


def draw_right(ax, show_y, bg):
    draw_bg(ax, bg, WR, XR[1] - XR[0], "right")
    ln, = ax.plot(XR, XR, color=C_GRAY, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ln.set_gid("identity_line")
    lc = LineCollection([[(max(g - 0.19, XR[0]), g), (min(g + 0.19, XR[1]), g)] for g in G], colors=C_GRAY, linewidths=0.9, zorder=1.5)
    lc.set_gid("target_levels")
    ax.add_collection(lc)
    for k in range(7):
        x = G[k] + JIT[k]
        if IN_RANGE[k]:
            sc = ax.scatter(x, Y[k], s=4.0, facecolors=[COLS[k]], edgecolors="none", alpha=0.85, zorder=4)
        else:
            sc = ax.scatter(x, Y[k], s=4.0, facecolors="none", edgecolors=[COLS[k]], linewidths=0.5, zorder=4)
        sc.set_gid("scatter_points_t%d" % k)
    mean = [float(y.mean()) for y in Y]
    mm = ax.scatter(G, mean, marker="D", s=22 * DIAMOND_SCALE ** 2, c="black", edgecolors="white", linewidths=0.8, zorder=7)
    mm.set_gid("mean_markers")
    ax.set_xlim(*XR)
    ax.set_ylim(*YLIM)
    ax.set_xticks(G)
    ax.set_xticklabels(["%.2f" % g for g in G])
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_xlabel("Target γ [J/m²]")
    ax.set_title("Generated structures after relaxation (n = 50 per target)", loc="left", fontsize=8, pad=3)
    ax.title.set_gid("title_right")
    if show_y:
        ax.set_ylabel("γ [J/m²]")
    else:
        ax.tick_params(labelleft=False)
    handles = [Line2D([], [], marker="o", ls="none", mfc="0.4", mec="none", ms=2.6, label="Each structure"),
               Line2D([], [], marker="D", ls="none", mfc="black", mec="white", mew=0.8, ms=4.5 * DIAMOND_SCALE, label="Mean per target")]
    labels = ["Each structure", "Mean per target"]
    if bg == "gray_out":
        handles += [OutsideKey(False, "swatch"), OutsideKey(False, "hollow")]
        labels += ["Outside training energy range", "Target outside training range"]
    else:
        handles += [OutsideKey(bg != "band_only")]
        labels += ["Outside training range"]
    handles += [Line2D([], [], color=C_GRAY, lw=0.7, ls=(0, (4, 2)), label="y = x")]
    labels += ["y = x"]
    leg = ax.legend(handles=handles, labels=labels, handler_map={OutsideKey: HandlerOutside()}, loc="upper left", bbox_to_anchor=(0.0, 0.79), frameon=False,
                    handlelength=1.8, labelspacing=0.35, borderpad=0.3, borderaxespad=0.2, fontsize=7)
    leg.set_gid("legend")
    ax.set_gid("ax_right")


def finish_axes(*axs):
    for ax in axs:
        ax.set_facecolor("none")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


TICK_ROT = {}


def xtick_policy(fig, ax):
    """Measures the overlap of the x tick labels (2 digits, horizontal). Rotated by 30 degrees if they overlap."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bb = [t.get_window_extent(rend) for t in ax.get_xticklabels()]
    gaps = [bb[i + 1].x0 - bb[i].x1 for i in range(len(bb) - 1)]
    TICK_ROT["min_gap_px"] = min(gaps)
    TICK_ROT["min_gap_mm"] = min(gaps) / fig.dpi * 25.4
    if min(gaps) < 0.5 * fig.dpi / 25.4:                 # labels closer than 0.5 mm count as overlapping
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        TICK_ROT["rotated"] = True
    else:
        TICK_ROT["rotated"] = False


def build(panels, bg, left_ticks):
    """panels: 'both' | 'left' | 'right'. The physical dimensions of the axes are common to all figures (left WL, right WR, height HP). Margins are explicit via subplots_adjust."""
    wmm = {"both": W_MM, "left": ML + WL + MR, "right": ML + WR + MR}[panels]
    fig_kw = dict(figsize=(wmm / 25.4, H_MM / 25.4))
    if panels == "both":
        fig, (axl, axr) = plt.subplots(1, 2, sharey=True, gridspec_kw=dict(width_ratios=[1, 3]), **fig_kw)
        fig.subplots_adjust(left=ML / wmm, right=1 - MR / wmm, bottom=MB / H_MM, top=1 - MT / H_MM, wspace=GAP / (WAX / 2.0))
        draw_left(axl, True, bg, left_ticks); draw_right(axr, False, bg); finish_axes(axl, axr)
        axes = [(axl, WL), (axr, WR)]
    else:
        fig, ax = plt.subplots(1, 1, **fig_kw)
        fig.subplots_adjust(left=ML / wmm, right=1 - MR / wmm, bottom=MB / H_MM, top=1 - MT / H_MM)
        if panels == "left":
            draw_left(ax, True, bg, left_ticks); axes = [(ax, WL)]
        else:
            draw_right(ax, True, bg); axes = [(ax, WR)]
        finish_axes(ax)
    for ax, w in axes:                                   # measured physical dimensions (mm)
        p = ax.get_position()
        assert abs(p.width * wmm - w) < 1e-6 and abs(p.height * H_MM - HP) < 1e-6, (p.width * wmm, w, p.height * H_MM, HP)
    if panels in ("both", "right"):
        xtick_policy(fig, axes[-1][0])
    return fig

# ============================================================ SVG post-processing (materialize <use>, remove clipPath, remove unneeded transforms)
SVGNS, XLNS = "http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"
for p_, u_ in [("", SVGNS), ("xlink", XLNS), ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
               ("cc", "http://creativecommons.org/ns#"), ("dc", "http://purl.org/dc/elements/1.1/")]:
    ET.register_namespace(p_, u_)


def _parse_style(s):
    d = {}
    for part in (s or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def _shift_path(d, dx, dy):
    toks = re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            assert t in "MLCQz" or t == "Z", "unsupported path command %r" % t
            out.append(t); i += 1
            continue
        out.append("%.5f %.5f" % (float(toks[i]) + dx, float(toks[i + 1]) + dy)); i += 2
    return " ".join(out)


def postprocess_svg(path):
    tree = ET.parse(path)
    root = tree.getroot()
    q = lambda t: "{%s}%s" % (SVGNS, t)
    parent = {c: p for p in root.iter() for c in p}
    defs_el = {e.get("id"): e for e in root.iter() if e.tag == q("path") and e.get("id") and parent[e].tag == q("defs")}
    n_use = 0
    used_ids = set()
    for u in [e for e in root.iter() if e.tag == q("use")]:
        ref = u.get("{%s}href" % XLNS)
        assert ref and ref.startswith("#") and ref[1:] in defs_el, ref
        assert u.get("transform") is None, "use with transform"
        r = defs_el[ref[1:]]
        used_ids.add(ref[1:])
        st = _parse_style(r.get("style")); st.update(_parse_style(u.get("style")))
        new = ET.Element(q("path"))
        new.set("d", _shift_path(r.get("d"), float(u.get("x", 0)), float(u.get("y", 0))))
        if st:
            new.set("style", "; ".join("%s: %s" % kv for kv in st.items()))
        p = parent[u]
        p.insert(list(p).index(u), new)
        p.remove(u)
        n_use += 1
    for i in used_ids:
        d_el = defs_el[i]
        parent[d_el].remove(d_el)
    n_clip = 0
    for e in list(root.iter()):
        if e.tag == q("clipPath"):
            parent[e].remove(e); n_clip += 1
    for e in root.iter():
        if "clip-path" in e.attrib:
            del e.attrib["clip-path"]
        if e.tag == q("text") and re.fullmatch(r"rotate\(-?0 [\d.\-]+ [\d.\-]+\)", e.get("transform", "")):
            del e.attrib["transform"]
    for e in root.iter():                             # opacity 0 may be ignored by EMF/PowerPoint -> set explicitly to none
        if e.get("style"):
            st = _parse_style(e.get("style"))
            ch = False
            if st.get("fill-opacity") == "0":
                st["fill"] = "none"; del st["fill-opacity"]; ch = True
            if st.get("stroke-opacity") == "0":
                st["stroke"] = "none"; del st["stroke-opacity"]; ch = True
            if ch:
                e.set("style", "; ".join("%s: %s" % kv for kv in st.items()))
    for e in list(root.iter()):                       # remove <defs> that became empty (keep those containing <style>)
        if e.tag == q("defs") and len(e) == 0:
            parent[e].remove(e)
    tree.write(path, xml_declaration=True, encoding="utf-8")
    return n_use, n_clip


def inkscape(args):
    r = subprocess.run(["inkscape", "-z"] + args, capture_output=True, text=True, timeout=200)
    return r.returncode, (r.stderr or "")[-400:]


def save_all(fig, name, png=True, pdf=True, svg=True, emf=False, check=None):
    if png:
        save_fig(fig, name, OUT, pdf=False)
    if pdf:
        fig.savefig(os.path.join(OUT, name + ".pdf"), metadata={"Creator": "paperfig %s / make_fig3_final.py" % paperfig.__version__})
    if svg:
        sv = os.path.join(OUT, name + ".svg")
        fig.savefig(sv, format="svg", facecolor="none", metadata={"Date": None})
        raw = None
        if check is not None:
            raw = os.path.join(check, name + "_raw.svg"); shutil.copy(sv, raw)
        n_use, n_clip = postprocess_svg(sv)
        print("%s.svg: baked <use> %d, removed clipPath %d" % (name, n_use, n_clip))
        if raw is not None:                           # render before and after post-processing with the same renderer (inkscape) and measure the pixel difference
            from PIL import Image
            a, b = os.path.join(check, name + "_raw.png"), os.path.join(check, name + "_post.png")
            inkscape([raw, "--export-png=" + a, "--export-dpi=150"]); inkscape([sv, "--export-png=" + b, "--export-dpi=150"])
            ia, ib = np.asarray(Image.open(a).convert("L"), float), np.asarray(Image.open(b).convert("L"), float)
            print("%s: inkscape raster raw vs post-processed SVG: shape %s, mean|diff|=%.4f /255, max|diff|=%d, pixels differing >32: %d"
                  % (name, ia.shape, np.abs(ia - ib).mean(), int(np.abs(ia - ib).max()), int((np.abs(ia - ib) > 32).sum())))
        if emf:
            ep = os.path.join(OUT, name + ".emf")
            rc, err = inkscape([sv, "--export-emf=" + ep])
            print("%s.emf: inkscape rc=%d size=%s %s" % (name, rc, os.path.getsize(ep) if os.path.exists(ep) else None, err.strip()[:200]))
            if check is not None and os.path.exists(ep):
                rc2, err2 = inkscape([ep, "--export-png=" + os.path.join(check, name + "_emf.png"), "--export-dpi=150"])
                print("%s.emf re-import to png (inkscape): rc=%d" % (name, rc2))



# ============================================================ run
def compose_backgrounds():
    from PIL import Image
    names = [("hatch_out", "bg = hatch_out (recommended)"), ("band_only", "bg = band_only (v2-like)"), ("white_hatch", "bg = white_hatch")]
    ims = [Image.open(os.path.join(OUT, "Fig3_cmp_bg_%s.png" % n)).convert("RGB") for n, _ in names]
    f = plt.figure(figsize=(3 * 6.4, 3.4))
    for i, ((n, t), im) in enumerate(zip(names, ims)):
        a = f.add_axes([i / 3.0, 0.0, 1 / 3.0, 0.93])
        a.imshow(im, interpolation="lanczos"); a.axis("off")
        f.text(i / 3.0 + 0.005, 0.95, t, fontsize=10, va="bottom", ha="left")
    f.savefig(os.path.join(OUT, "Fig3_cmp_backgrounds.png"), dpi=220, facecolor="white")
    plt.close(f)


chk = tempfile.mkdtemp(prefix="fig3final_check_")
is_main = (ARGS.bg == "gray_out" and ARGS.left_ticks == "color")
if ARGS.all or is_main:
    fig = build("both", "gray_out", "color")
    save_all(fig, "Fig3", png=True, pdf=True, svg=True, emf=True, check=chk)
    shutil.copy(os.path.join(OUT, "Fig3.png"), os.path.join(OUT, "Fig3_cmp_bg_gray_out.png"))
    plt.close(fig)
    for panels, name in (("left", "Fig3_left"), ("right", "Fig3_right")):
        fig = build(panels, "gray_out", "color")
        save_all(fig, name, png=False, pdf=False, svg=True)
        plt.close(fig)
if ARGS.all:
    for bg in ("band_only", "white_hatch"):
        fig = build("both", bg, "color"); save_all(fig, "Fig3_cmp_bg_" + bg, png=True, pdf=False, svg=False); plt.close(fig)
    fig = build("both", "hatch_out", "gray"); save_all(fig, "Fig3_cmp_leftticks_gray", png=True, pdf=False, svg=False); plt.close(fig)
    compose_backgrounds()
elif not is_main:
    name = "Fig3_cmp_bg_" + ARGS.bg if ARGS.left_ticks == "color" else "Fig3_cmp_leftticks_gray"
    fig = build("both", ARGS.bg, ARGS.left_ticks); save_all(fig, name, png=True, pdf=False, svg=False); plt.close(fig)

print("x tick labels (2-digit, horizontal) check:", TICK_ROT)
print("check dir:", chk)
print("train range %.6f..%.6f median %.6f" % (TR_MIN, TR_MAX, TR_MED))
print("DONE")
