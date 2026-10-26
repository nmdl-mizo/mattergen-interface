#!/usr/bin/env python3
"""make_fig4.py -- Fig. 4 final fine adjustment.
Copied and modified from the earlier version (make_fig4_v3.py). Numbers, ECDF data, layout and wording are the same as in the earlier version. No new computation.
Output: directly under the output directory (Fig4.{svg,pdf,png,emf}, _panel_a/_panel_b.{svg,png}, verification.csv).
Run: with the evaluation python (environment-eval.yml) on a login node: python <this file>
"""
import dataclasses
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.abspath(os.path.join(HERE, ".."))
ROOT = '<ORIGINAL_ROOT>' if os.path.isdir('<ORIGINAL_ROOT>/workspace') else '<ORIGINAL_ROOT>'
GC = ROOT + "/workspace"
PA = GC + "/paper"
PD = PA + "/figure_sources/fig4"
S2 = PD + "/stage2"
S3 = PD + "/stage3"
assert os.path.realpath(T) == os.path.realpath(PD + "/render")
sys.path.insert(0, PD + "/scripts")          # svgpost.py (unchanged)
sys.path.insert(0, PA + "/figures/scripts")  # paperfig.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import paperfig
from paperfig import OKABE_ITO, setup_style
import svgpost

assert paperfig.__version__ == "1.1.0"

if not any(e.name == "Arial" for e in fm.fontManager.ttflist):
    for f in sorted(glob.glob("/usr/share/fonts/liberation-sans/LiberationSans-*.ttf")):
        fm.fontManager.addfont(f)
    for e in [e for e in fm.fontManager.ttflist if e.name == "Liberation Sans"]:
        fm.fontManager.ttflist.append(dataclasses.replace(e, name="Arial"))
print("Arial resolves to:", fm.findfont("Arial", fallback_to_default=False))

setup_style()
matplotlib.rcParams.update({
    "font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.constrained_layout.use": False, "axes.prop_cycle": matplotlib.cycler(color=["#000000"]),
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 7, "axes.titlesize": 8, "legend.fontsize": 6.5,
})

BLUE = OKABE_ITO[5]
GREY_F = "#C9C9C9"
NEU = "#555555"
LEAD = "#1F4E79"   # dark blue of leader lines and labels
FS = 7.0        # body text
FS_S = 6.5      # auxiliary
FS_LAB = 9.0    # panel labels
FS_HEAD = 8.0   # headings
LW_AXIS = 0.6
LW_ECDF = 1.0
BAND = 0.10
MARGIN_TOP = 1.0      # mm (same value as Fig. 2)
MARGIN_BOT = 1.5      # mm
LEADER_LEN = 2.5      # mm
TICK_HALF = 0.035     # half length of the p10/p90 ticks (y units, same as in the earlier version)
LABEL_OFF = 0.72      # mm. 0.72 mm each in x and y from the tick end -> straight-line distance 1.02 mm (>= 1 mm)

# ------------------------------------------------------------ data (same as in the earlier version)
P = pd.read_csv(PD + "/data/fig4_rows.csv", dtype={"class": str})
assert len(P) == 250
C = pd.read_csv(PA + "/figures/F4_classification_counts.csv")
Call = C[C["target_id"].astype(str) == "all5"].iloc[0]
DR = pd.read_csv(S2 + "/data/Fig4_distance_rows.csv")
DS = pd.read_csv(S2 + "/data/Fig4_distance_stats.csv").set_index("quantity")
N_WU = int((P["class"] == "Within target window / Train-unmatched").sum())
N_WM = int((P["class"] == "Within target window / Train-matched").sum())
N_OU = int((P["class"] == "Outside target window / Train-unmatched").sum())
N_OM = int((P["class"] == "Outside target window / Train-matched").sum())
N_W, N_O = N_WU + N_WM, N_OU + N_OM
N_ALL = N_W + N_O

V = []


def shown(fig, text, value, expected, source):
    ok = (value == expected)
    V.append(dict(stage="fig", item="%s | %s" % (fig, text), value=value, expected=expected, source=source,
                  match=ok, status="PASS" if ok else "FAIL"))
    assert ok, (fig, text, value, expected)


def info(stage, item, value, source):
    V.append(dict(stage=stage, item=item, value=value, expected="", source=source, match=True, status="INFO"))


def atleast(fig, text, value, lo, source):
    ok = bool(value >= lo)
    V.append(dict(stage="fig", item="%s | %s" % (fig, text), value="%.3f" % value, expected=">= %g" % lo, source=source,
                  match=ok, status="PASS" if ok else "FAIL"))
    assert ok, (fig, text, value, lo)


def near(fig, text, value, target, tol, source):
    ok = bool(abs(value - target) <= tol)
    V.append(dict(stage="fig", item="%s | %s" % (fig, text), value="%.3f" % value, expected="%g +- %g" % (target, tol), source=source,
                  match=ok, status="PASS" if ok else "FAIL"))
    assert ok, (fig, text, value, target)


shown("common", "counts WU/WM/OU/OM", "%d/%d/%d/%d" % (N_WU, N_WM, N_OU, N_OM),
      "%d/%d/%d/%d" % (Call["near_nomatch"], Call["near_match"], Call["outside_nomatch"], Call["outside_match"]),
      "fig4_rows.csv class vs figures/F4_classification_counts.csv all5 (near_nomatch/near_match/outside_nomatch/outside_match)")
shown("common", "total 250", N_ALL, 250, "fig4_rows.csv; C all5 n_req=%s" % Call["n_req"])
shown("common", "within/outside 91/159", "%d/%d" % (N_W, N_O), "91/159", "C all5 (near_*; outside_*)")
shown("common", "within-window unmatched/matched 90/1", "%d/%d" % (N_WU, N_WM), "90/1", "C all5 near_nomatch/near_match")
shown("common", "outside unmatched/matched 155/4 (caption only)", "%d/%d" % (N_OU, N_OM), "155/4", "C all5 outside_nomatch/outside_match")
shown("common", "window threshold", BAND, 0.10, "fig4_rows.csv in_window definition (|dgamma|<=0.10 J/m2)")
dP = P[P["class"] == "Within target window / Train-unmatched"][["chunk", "cif_index", "d_rms_min_A"]].sort_values(["chunk", "cif_index"]).reset_index(drop=True)
dD = DR[["chunk", "cif_index", "d_rms_min_A"]].sort_values(["chunk", "cif_index"]).reset_index(drop=True)
shown("common", "d_rms_min_A rows.csv == fig4_rows.csv (90 rows)",
      bool(len(dP) == 90 and (dP[["chunk", "cif_index"]].values == dD[["chunk", "cif_index"]].values).all()
           and np.allclose(dP["d_rms_min_A"].values, dD["d_rms_min_A"].values, rtol=0, atol=1e-9)), True,
      "stage2/data/Fig4_distance_rows.csv vs data/fig4_rows.csv d_rms_min_A")

d_rms = np.sort(DR["d_rms_min_A"].values.astype(float))
n90 = len(d_rms)
shown("b", "n", n90, 90, "stage2/data/Fig4_distance_rows.csv")
q = lambda p: float(np.percentile(d_rms, p))
P50, P10, P90 = q(50), q(10), q(90)
for nm, val, exp, col in (("median", P50, 0.1651, "p50"), ("p10", P10, 0.0802, "p10"), ("p90", P90, 0.4446, "p90"), ("max", float(d_rms.max()), 0.5730, "max")):
    shown("b", "%s recomputed (4 digits) == spec" % nm, round(val, 4), exp, "numpy percentile linear on Fig4_distance_rows.csv; spec stage3/02_v3_spec_ja.md §3")
    shown("b", "%s == Fig4_distance_stats.csv (4 digits)" % nm, round(val, 4), round(float(DS.loc["d_rms_min_A", col]), 4),
          "stage2/data/Fig4_distance_stats.csv d_rms_min_A %s" % col)


# ------------------------------------------------------------ canvas
class Cv:
    def __init__(self, w_mm, h_mm):
        self.W, self.H = w_mm, h_mm
        self.fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4))
        self.ov = self.fig.add_axes([0, 0, 1, 1], zorder=20)
        self.ov.set_xlim(0, w_mm); self.ov.set_ylim(0, h_mm); self.ov.axis("off")
        self.reg = {}

    def axmm(self, x0, y0, w, h, **kw):
        return self.fig.add_axes([x0 / self.W, y0 / self.H, w / self.W, h / self.H], **kw)

    def text(self, x, y, s, fs=FS, ha="left", va="center", color="black", bold=False, z=6, **kw):
        assert "\n" not in s
        return self.ov.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, fontweight="bold" if bold else "normal", zorder=z, **kw)

    def line(self, xs, ys, color="black", lw=0.5, z=5, **kw):
        l = Line2D(xs, ys, color=color, linewidth=lw, zorder=z, solid_capstyle="butt", **kw)
        self.ov.add_line(l)
        return l

    def bbox_mm(self, t):
        self.fig.canvas.draw()
        bb = t.get_window_extent(self.fig.canvas.get_renderer())
        k = 25.4 / self.fig.dpi
        return (bb.x0 * k, bb.y0 * k, bb.x1 * k, bb.y1 * k)


def hatch_lines(x0, y0, x1, y1, step=1.0, phase=0.0):
    segs = []
    kmin = int(np.floor((y0 - x1) / step)) - 1
    kmax = int(np.ceil((y1 - x0) / step)) + 1
    for k in range(kmin, kmax + 1):
        c = k * step + phase
        lo, hi = max(x0, y0 - c), min(x1, y1 - c)
        if hi - lo > 1e-6:
            segs.append([(lo, lo + c), (hi, hi + c)])
    return segs


# (face, edge, hatch). WM = blue-edged hatching (hatching drawn as line segments, no <pattern>)
STY = {"WU": (BLUE, BLUE, None), "WM": ("white", BLUE, BLUE), "OU": (GREY_F, GREY_F, None),
       "NU": (NEU, NEU, None), "NM": ("white", NEU, "#555555")}


def rect(cv, x0, y0, x1, y1, kind, gid, edge_lw=0.0):
    """Filled bars have no edge (edge_lw=0). Only the hatched cells (WM, NM) get an explicit edge line of 0.6 pt."""
    fc, ec, hat = STY[kind]
    cv.ov.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc, edgecolor=ec, linewidth=edge_lw, zorder=3, gid=gid))
    if hat is not None:
        segs = hatch_lines(x0, y0, x1, y1, step=1.0)
        if segs:
            cv.ov.add_collection(LineCollection(segs, colors=hat, linewidths=0.4, zorder=3.5, gid=gid + "_hatch", capstyle="butt"))


def check_layout(cv, name):
    cv.fig.canvas.draw()
    r = cv.fig.canvas.get_renderer()
    k = 25.4 / cv.fig.dpi
    items = []
    for t in cv.fig.findobj(mtext.Text):
        s = t.get_text().strip()
        if not s or not t.get_visible():
            continue
        bb = t.get_window_extent(r)
        items.append((s, bb.x0 * k, bb.y0 * k, bb.x1 * k, bb.y1 * k))
    prob = []
    for s, x0, y0, x1, y1 in items:
        if x0 < -0.05 or y0 < -0.05 or x1 > cv.W + 0.05 or y1 > cv.H + 0.05:
            prob.append("OUT  %r (%.1f,%.1f)-(%.1f,%.1f) canvas %.0fx%.0f" % (s, x0, y0, x1, y1, cv.W, cv.H))
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            ox = min(a[3], b[3]) - max(a[1], b[1]); oy = min(a[4], b[4]) - max(a[2], b[2])
            if ox > 0.15 and oy > 0.15:
                prob.append("OVER %r x %r (%.2f x %.2f mm)" % (a[0], b[0], ox, oy))
    # top/bottom margin: topmost and bottommost points of the text bounding rectangles and of the overlay Rectangle
    ymax = max(it[4] for it in items)
    ymin = min(it[2] for it in items)
    for p in cv.ov.patches:
        ymin = min(ymin, p.get_y())
    top_m, bot_m = cv.H - ymax, ymin
    print("[layout %s] texts=%d problems=%d  top margin %.3f mm  bottom margin %.3f mm" % (name, len(items), len(prob), top_m, bot_m))
    for p in prob:
        print("   ", p)
    return prob, top_m, bot_m


def rect_poly_dist(rc, pts, step=0.01):
    """Minimum distance (dense sampling) between a rectangle rc=(x0,y0,x1,y1) and a polyline pts (mm)."""
    best = 1e9
    for p, qq in zip(pts[:-1], pts[1:]):
        p = np.asarray(p, float); qq = np.asarray(qq, float)
        n = max(2, int(np.hypot(*(qq - p)) / step) + 1)
        t = np.linspace(0, 1, n)[:, None]
        xy = p + (qq - p) * t
        dx = np.maximum.reduce([rc[0] - xy[:, 0], np.zeros(n), xy[:, 0] - rc[2]])
        dy = np.maximum.reduce([rc[1] - xy[:, 1], np.zeros(n), xy[:, 1] - rc[3]])
        best = min(best, float(np.hypot(dx, dy).min()))
    return best


LAYOUT = {}
OUT_SVGS = []


def special_checks(cv, name):
    """(a) horizontal distance of the leader-line labels/notes, (b) distance of the p10/p90 labels from the ticks and curves."""
    R = cv.reg
    if "leader_label" in R:
        bl = cv.bbox_mm(R["leader_label"]); bn = cv.bbox_mm(R["note"])
        gap = bl[0] - bn[2]
        atleast(name, "note(left) to '1 train-matched' label(right) horizontal gap [mm]", gap, 10.0,
                "text window extents (note x1 -> label x0)")
        ld = R["leader_line"]
        xs, ys = ld.get_xdata(), ld.get_ydata()
        near(name, "leader length [mm]", abs(ys[1] - ys[0]), LEADER_LEN, 0.001, "Line2D ydata")
        shown(name, "leader vertical (x equal)", bool(xs[0] == xs[1]), True, "Line2D xdata")
        shown(name, "leader color/width", "%s/%.1f" % (ld.get_color(), ld.get_linewidth()), "%s/0.6" % LEAD, "Line2D props")
        shown(name, "leader starts at center of hatched segment",
              bool(abs(xs[0] - R["wm_center"]) < 1e-9), True, "xs vs lower-bar WM rect center")
        shown(name, "leader label fontsize/weight", "%.1f/%s" % (R["leader_label"].get_fontsize(), R["leader_label"].get_fontweight()), "7.0/bold", "Text props")
        # gap between the top of a label and the end of its line (INFO)
        info("fig", "%s | leader end to label top [mm]" % name, "%.3f" % (ys[1] - bl[3]), "text window extent")
    if "ecdf_pts" in R:
        for qn in ("p10", "p90"):
            t = R[qn]; tick = R[qn + "_tick"]
            rc = cv.bbox_mm(t)
            d_curve = rect_poly_dist(rc, R["ecdf_pts"])
            d_tick = rect_poly_dist(rc, tick)
            atleast(name, "%s label to ECDF curve min distance [mm]" % qn, d_curve, 1.0, "dense sampling (0.01 mm) rect vs step polyline, mm coords")
            atleast(name, "%s label to its tick min distance [mm]" % qn, d_tick, 1.0, "dense sampling rect vs tick segment")
        # orientation of the placement
        tx10, ty10 = R["p10_tick"][1]; tx90, ty90 = R["p90_tick"][0]
        b10 = cv.bbox_mm(R["p10"]); b90 = cv.bbox_mm(R["p90"])
        shown(name, "p10 label is upper-left of tick", bool(b10[2] <= tx10 and b10[1] >= ty10), True, "bbox vs tick upper end")
        shown(name, "p90 label is lower-right of tick", bool(b90[0] >= tx90 and b90[3] <= ty90), True, "bbox vs tick lower end")
        # minimum distance between the median label and the curve (INFO)
        info("fig", "%s | median label to ECDF curve min distance [mm]" % name, "%.3f" % rect_poly_dist(cv.bbox_mm(R["med"]), R["ecdf_pts"]), "dense sampling")
        shown(name, "ECDF linewidth 1.0 / axis linewidth 0.6",
              "%.1f/%.1f" % (R["ecdf_line"].get_linewidth(), R["ax"].spines["left"].get_linewidth()), "1.0/0.6", "Line2D / spine props")


def render(cv, name, formats=("svg", "png")):
    fig = cv.fig
    prob, top_m, bot_m = check_layout(cv, name)
    LAYOUT[name] = (prob, top_m, bot_m)
    special_checks(cv, name)
    info("fig", "%s | margin top / bottom [mm] (text bbox & rect extents)" % name, "%.3f / %.3f" % (top_m, bot_m), "check_layout")
    if name == "Fig4":
        near(name, "top margin [mm]", top_m, MARGIN_TOP, 0.1, "text bbox top (panel labels) vs canvas")
        near(name, "bottom margin [mm]", bot_m, MARGIN_BOT, 0.1, "text bbox / legend swatch bottom vs canvas")
    sv = os.path.join(T, name + ".svg")
    fig.savefig(sv, format="svg", facecolor="none", metadata={"Date": None})
    n_use, n_clip = svgpost.postprocess_svg(sv)
    OUT_SVGS.append((name, sv))
    if "pdf" in formats:
        fig.savefig(os.path.join(T, name + ".pdf"), metadata={"Creator": "paperfig %s / make_fig4_render.py" % paperfig.__version__})
    r = subprocess.run(["rsvg-convert", "-w", "2400", "-b", "white", "-o", os.path.join(T, name + ".png"), sv], capture_output=True, text=True, timeout=200)
    assert r.returncode == 0, r.stderr
    if "emf" in formats:
        rc, err = svgpost.inkscape([sv, "--export-emf=" + os.path.join(T, name + ".emf")])
        assert rc == 0 and os.path.getsize(os.path.join(T, name + ".emf")) > 1000, (rc, err)
    print("%s: baked <use> %d, removed clipPath %d" % (name, n_use, n_clip))
    plt.close(fig)


# ============================================================ (a)
def panel_a(cv, x0, y0, W, H, fig_id):
    xa = x0 + 0.5
    BW = W - 1.0                       # full width = 250 structures
    ytop = y0 + H
    wu = BW * N_W / N_ALL              # width of the upper Within bar
    shown(fig_id, "upper bar all/within/outside", "%d/%d/%d" % (N_ALL, N_W, N_O), "250/91/159", "C all5")
    shown(fig_id, "lower bar unmatched/matched", "%d/%d" % (N_WU, N_WM), "90/1", "C all5 near_nomatch/near_match")
    shown(fig_id, "center '%d / %d train-unmatched'" % (N_WU, N_W), "%d/%d" % (N_WU, N_W), "90/91", "C all5 near_nomatch/(near_nomatch+near_match)")
    shown(fig_id, "stage labels n", "%d/%d" % (N_ALL, N_W), "250/91", "C all5")
    # upper row
    cv.text(xa, ytop - 2.0, "All generated (n = %d)" % N_ALL, fs=FS, bold=True)
    uy0, uy1 = ytop - 9.0, ytop - 4.2
    rect(cv, xa, uy0, xa + wu, uy1, "WU", "a_up_w")
    rect(cv, xa + wu, uy0, xa + BW, uy1, "OU", "a_up_o")
    cv.text(xa + wu / 2, (uy0 + uy1) / 2, "Within: %d" % N_W, fs=FS_S, ha="center", color="white", bold=True)
    cv.text(xa + wu + (BW - wu) / 2, (uy0 + uy1) / 2, "Outside: %d" % N_O, fs=FS_S, ha="center", color="black")
    # lower row (rescaled version, full width = 91)
    ly0, ly1 = ytop - 24.8, ytop - 20.0
    lw_ = BW
    bu = lw_ * N_WU / N_W
    cv.line([xa, xa], [uy0, ly1], color="#9A9A9A", lw=0.4)                              # guide line
    cv.line([xa + wu, xa + lw_], [uy0, ly1], color="#9A9A9A", lw=0.4)
    rect(cv, xa, ly0, xa + bu, ly1, "WU", "a_lo_u")                                     # filled bars: no edge
    rect(cv, xa + bu, ly0, xa + lw_, ly1, "WM", "a_lo_m", edge_lw=0.6)                  # blue-edged hatched cells
    cv.text(xa + bu / 2, (ly0 + ly1) / 2, "%d" % N_WU, fs=FS, ha="center", color="white", bold=True)
    cv.text(xa + 1.2, ytop - 14.8, "Within target window (n = %d)" % N_W, fs=FS, bold=True, color=BLUE)
    # leader line: 2.5 mm straight down from the middle of the hatched interval (0.6 pt, #1F4E79). The 7 pt bold label is below the end of the line, the note is at the same height at the left end
    xs = xa + bu + (lw_ - bu) / 2
    cv.reg["wm_center"] = xs
    cv.reg["leader_line"] = cv.line([xs, xs], [ly0, ly0 - LEADER_LEN], color=LEAD, lw=0.6)
    ytxt = ly0 - LEADER_LEN - 0.45
    cv.reg["leader_label"] = cv.text(xs + 0.3, ytxt, "%d train-matched" % N_WM, fs=FS, ha="right", va="top", color=LEAD, bold=True)
    cv.reg["note"] = cv.text(xa, ytxt, "Lower bar rescaled to n = %d" % N_W, fs=FS_S, va="top", color="#707070")
    # central result (largest text in the figure, 11 pt)
    cv.text(xa, y0 + 11.6, "%d / %d train-unmatched" % (N_WU, N_W), fs=11, bold=True, color=BLUE)
    # legend (1 row). Bottom edge = margin MARGIN_BOT (referenced to the bottom of the swatch)
    ly = y0 + MARGIN_BOT + 1.2
    for i, (kind, lab) in enumerate((("NU", "Train-unmatched"), ("NM", "Train-matched"))):
        xx = xa + i * 27.0
        rect(cv, xx, ly - 1.2, xx + 4.0, ly + 1.2, kind, "legend_" + kind, edge_lw=(0.6 if kind == "NM" else 0.0))
        cv.text(xx + 5.2, ly, lab, fs=FS_S)


# ============================================================ (b)
def ecdf_xy(vals, xmax):
    xs = np.concatenate([[0.0], vals, [xmax]])
    ys = np.concatenate([[0.0], np.arange(1, len(vals) + 1) / len(vals), [1.0]])
    return xs, ys


def Fat(vals, x):
    return np.searchsorted(vals, x, side="right") / len(vals)


MB_B = 9.5   # (b) margin below the axis (overwritten by calibrate_mb)


def panel_b(cv, x0, y0, W, H, fig_id="b", MB=None):
    ML, MR, MT = 10.5, 2.5, 5.5
    MB = MB_B if MB is None else MB
    ytop = y0 + H
    cv.text(x0 + 0.5, ytop - 2.0, "Within-window, train-unmatched (n = %d)" % n90, fs=FS)
    shown(fig_id, "top text n", n90, 90, "stage2/data/Fig4_distance_rows.csv")
    axx0, axy0, axw, axh = x0 + ML, y0 + MB, W - ML - MR, H - MB - MT
    ax = cv.axmm(axx0, axy0, axw, axh)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(LW_AXIS)
    ax.tick_params(width=LW_AXIS)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=0); ax.set_axisbelow(True)
    xmax = 0.65
    xs, ys = ecdf_xy(d_rms, xmax)
    line, = ax.step(xs, ys, where="post", color=BLUE, linewidth=LW_ECDF, zorder=4)
    xs2, ys2 = ecdf_xy(np.sort(DR["d_rms_min_A"].values.astype(float)), 0.6)
    shown(fig_id, "ECDF step data == stage2 (x[:-1], y)", bool(np.array_equal(line.get_xdata()[:-1], xs2[:-1]) and np.array_equal(line.get_ydata(), ys2)), True,
          "stage2/scripts/make_fig4_stage2.py ecdf_xy on stage2 rows.csv (only terminal x 0.6 -> 0.65)")
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(0, 0.601, 0.1)); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels(["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"])
    ax.set_xlabel("Minimum RMS displacement to training data (Å)", labelpad=2, fontsize=FS)
    ax.set_ylabel("Cumulative fraction", labelpad=2, fontsize=FS)
    ax.axvline(P50, color="#888888", linewidth=0.6, linestyle=(0, (3, 2)), zorder=3)
    lab50 = "median %.3f Å" % P50
    shown(fig_id, "median label", lab50, "median 0.165 Å", "Fig4_distance_stats.csv p50=0.165125; spec stage3 §3")
    med = ax.text(P50 + 0.012, 0.95, lab50, fontsize=FS, ha="left", va="center", color="#222222", zorder=6)
    # mm coordinate transform (canvas coordinates)
    X = lambda x: axx0 + x / xmax * axw
    Y = lambda y: axy0 + y * axh
    # p10/p90: the tick is a short vertical line on the curve (as in the earlier version). The label is placed 1.02 mm from the tick end (placed in mm coordinates on the overlay)
    for qn, val, side in (("p10", P10, "L"), ("p90", P90, "R")):
        f = Fat(d_rms, val)
        ax.plot([val, val], [f - TICK_HALF, f + TICK_HALF], color="black", linewidth=0.9, zorder=5)
        tick_mm = [(X(val), Y(f - TICK_HALF)), (X(val), Y(f + TICK_HALF))]
        cv.reg[qn + "_tick"] = tick_mm
        if side == "L":   # upper left of the tick: the lower right corner of the label is (-0.72, +0.72) mm from the upper end of the tick
            cv.reg[qn] = cv.text(tick_mm[1][0] - LABEL_OFF, tick_mm[1][1] + LABEL_OFF, qn, fs=FS_S, ha="right", va="bottom")
        else:             # lower right of the tick: the upper left corner of the label is (+0.72, -0.72) mm from the lower end of the tick
            cv.reg[qn] = cv.text(tick_mm[0][0] + LABEL_OFF, tick_mm[0][1] - LABEL_OFF, qn, fs=FS_S, ha="left", va="top")
    # mm polyline of the step line (post)
    pts = [(X(xs[0]), Y(ys[0]))]
    for i in range(1, len(xs)):
        pts.append((X(xs[i]), Y(ys[i - 1]))); pts.append((X(xs[i]), Y(ys[i])))
    cv.reg.update(ecdf_pts=pts, ecdf_line=line, ax=ax, med=med)
    return ax


def calibrate_mb():
    """Finds MB for which the bottom of the x-axis label of (b) sits at MARGIN_BOT (linear, because xlabel is placed relative to the bottom of the axis)."""
    global MB_B
    cv = Cv(82.0, 48.0)
    panel_b(cv, 0, 0, 82.0, 48.0, "calib", MB=9.5)
    cv.fig.canvas.draw()
    xl = cv.reg["ax"].xaxis.label
    y0_mm = cv.bbox_mm(xl)[1]
    plt.close(cv.fig)
    del V[-len([v for v in V if v["item"].startswith("calib")]):]   # discard the temporary rows used for calibration
    MB_B = 9.5 + (MARGIN_BOT - y0_mm)
    print("calibrate: xlabel bottom at MB=9.5 -> %.3f mm ; MB_B = %.3f" % (y0_mm, MB_B))


# ============================================================ output
HEAD_H = 7.0
BODY_H = 48.0
W_A, X_B, W_B = 86.0, 96.0, 82.0


def make_full():
    W, H = 178.0, HEAD_H + BODY_H
    cv = Cv(W, H)
    # panel labels: left edge = left edge of the panel, top edge = 1 mm top margin (aligned by the bounding rectangle). Headings on the same baseline
    yb = H - 3.5
    labs = []
    for lab, x, head in (("(a)", 0.0, "Matching to training structures"), ("(b)", X_B, "Geometrical distance to training structures")):
        tl = cv.text(x, yb, lab, fs=FS_LAB, bold=True, va="baseline")
        labs.append((tl, x, head))
    top = max(cv.bbox_mm(tl)[3] for tl, _, _ in labs)
    yb += (H - MARGIN_TOP) - top
    for tl, x, head in labs:
        tl.set_y(yb)
        cv.text(x + 6.2, yb, head, fs=FS_HEAD, va="baseline")
    cv.reg["yb"] = yb
    panel_a(cv, 0.0, 0.0, W_A, BODY_H, "FULL_a")
    panel_b(cv, X_B, 0.0, W_B, BODY_H, "FULL_b")
    # assertion that the left edges agree
    for tl, x, _ in labs:
        bb = cv.bbox_mm(tl)
        info("fig", "Fig4 | panel label '%s' left edge / panel left [mm]" % tl.get_text(), "%.3f / %.3f" % (bb[0], x), "text bbox x0 (glyph side bearing included)")
        assert abs(bb[0] - x) < 0.3
    info("fig", "Fig4 | label & heading baseline y [mm from bottom]", "%.3f (H=%.0f)" % (yb, H), "Text.get_position")
    render(cv, "Fig4", formats=("svg", "pdf", "png", "emf"))


def make_panels():
    cv = Cv(W_A, BODY_H); panel_a(cv, 0, 0, W_A, BODY_H, "PA"); render(cv, "Fig4_panel_a")
    cv = Cv(W_B, BODY_H); panel_b(cv, 0, 0, W_B, BODY_H, "PB"); render(cv, "Fig4_panel_b")


FORBID = ("new GB phase", "novel structural unit", "non-memorization", "thermodynamically stable")


def svg_checks():
    for name, sv in OUT_SVGS:
        txt = open(sv, encoding="utf-8").read()
        for tag in ("<pattern", "<tspan", "<clipPath", "<use", "<image"):
            shown("svg:" + name, "forbidden %s count" % tag, txt.count(tag), 0, "grep on %s" % os.path.basename(sv))
        fams = sorted(set(m.strip("' ") for m in re.findall(r"font-family:\s*([^;\"]+)", txt)))
        shown("svg:" + name, "font-family values", fams, ["Arial"], "regex font-family in %s" % os.path.basename(sv))
        shown("svg:" + name, "font-family attr other than Arial", len([m for m in re.findall(r"font-family[:=]\s*[\"']?([^;\"']+)", txt) if m.strip("' ") != "Arial"]), 0,
              "grep font-family in %s" % os.path.basename(sv))
        bad = [m for m in re.findall(r"<text[^>]*>([^<]*)</text>", txt, flags=re.S) if "\n" in m]
        shown("svg:" + name, "<text> with newline", len(bad), 0, "regex <text>..</text> in %s" % os.path.basename(sv))
        shown("svg:" + name, "forbidden expressions %s" % list(FORBID), sum(txt.lower().count(f.lower()) for f in FORBID), 0, "grep -i on %s" % os.path.basename(sv))
        w = re.search(r'<svg[^>]*\bwidth="([\d.]+)pt"', txt); h = re.search(r'<svg[^>]*\bheight="([\d.]+)pt"', txt)
        wm, hm = float(w.group(1)) * 25.4 / 72, float(h.group(1)) * 25.4 / 72
        info("svg", "%s | size mm" % name, "%.2f x %.2f" % (wm, hm), "SVG width/height attribute (pt -> mm)")
        if name == "Fig4":
            shown("svg:" + name, "width 178 mm (+-0.05)", abs(wm - 178.0) < 0.05, True, "SVG width attr")
            shown("svg:" + name, "height 55 mm (+-0.05)", abs(hm - 55.0) < 0.05, True, "SVG height attr (%.2f mm)" % hm)
        # set of font sizes (pt)
        sizes = sorted(set(float(m) for m in re.findall(r"font-size:\s*(\d+(?:\.\d+)?)px", txt)))
        info("svg", "%s | font sizes (px in SVG = pt)" % name, str(sizes), "regex font: ... px")


def v3_consistency():
    """Checks that, among the fig rows of the earlier verification csv, those with the same item string have the same value."""
    v3 = pd.read_csv(S3 + "/verification_v3.csv", dtype=str)
    v3 = v3[(v3["stage"] == "fig") & (v3["status"] == "PASS")]
    mine = {v["item"]: str(v["value"]) for v in V if v["stage"] == "fig"}
    n_cmp, bad = 0, []
    for _, r in v3.iterrows():
        if r["item"] in mine:
            n_cmp += 1
            if mine[r["item"]] != str(r["value"]):
                bad.append((r["item"], r["value"], mine[r["item"]]))
    miss = [i for i in v3["item"] if i not in mine and "PA_alt" not in i and not i.startswith("svg:")]
    shown("v3-consistency", "v3 items (excl. alt panel, svg checks) absent here", len(miss), 0, "stage3/verification_v3.csv; %s" % miss[:3])
    shown("v3-consistency", "number of v3 PASS items compared (same item string)", n_cmp == 30, True, "stage3/verification_v3.csv (%d items compared)" % n_cmp)
    shown("v3-consistency", "mismatching values vs v3", len(bad), 0, "stage3/verification_v3.csv; %s" % (bad[:3],))


if __name__ == "__main__":
    calibrate_mb()
    make_full(); make_panels(); svg_checks(); v3_consistency()
    vf = pd.DataFrame(V).astype(str)
    vf.to_csv(T + "/verification_render.csv", index=False)
    print("verification_render rows: %d ; FAIL %d ; INFO %d" % (len(vf), int((vf["status"] == "FAIL").sum()), int((vf["status"] == "INFO").sum())))
    print("layout problems:", {k: len(v[0]) for k, v in LAYOUT.items()})
