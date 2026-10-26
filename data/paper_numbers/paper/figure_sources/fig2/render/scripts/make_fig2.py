#!/usr/bin/env python3
"""make_fig2.py -- Fig. 2 final fine adjustment (label fixes and horizontal centroid alignment of panels (a) and (b)).
Copied from the earlier version and modified (the earlier version is unchanged). Imports scripts/common.py (unchanged; no __pycache__ is written).
Output goes to the output directory only. Everything is drawn by hand in one coordinate system (pt, y downwards, 1 unit = 1 pt).
mode="v2" reproduces the earlier layout to measure the centroid "before" (not used for the output); mode="new" is the final version.
Run: with the evaluation python (environment-eval.yml) on a login node: python <this file>"""
import sys, os, re, json, subprocess
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from common import *      # D = parent directory (used read-only)
mpl_setup()
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from ase.io import read
from PIL import Image

WD = D + "/render"
assert os.path.isdir(WD)
SCRIPT = "make_fig2_render.py"
MM = 72.0 / 25.4
SC = 5.0
RAD, LWA = 1.0, 0.4
GRAY_TXT = "#555555"
IDS = ["proposed_c02_i15", "baseline_c06_i16", "baseline_c00_i07"]
FRAMES = {"proposed_c02_i15": 15, "baseline_c06_i16": 14, "baseline_c00_i07": 7}
NATOMS = {"proposed_c02_i15": 114, "baseline_c06_i16": 112, "baseline_c00_i07": 116}
CAND = pd.read_csv(D + "/candidates/candidates.csv").set_index("struct_id")
E1 = load_E1()
SHIFT_V1 = json.load(open(D + "/shift/shift_primary.json"))
LET = {"a": 0, "b": 1, "c": 2}
V = []


def ok(item, exp, obs, cond):
    V.append((SCRIPT, item, str(exp), str(obs), "PASS" if cond else "FAIL"))
    assert cond, (item, exp, obs)


def parse_axes(s):
    d = dict(kv.split("=") for kv in s.split(","))
    return LET[d["n"]], LET[d["t"]], LET[d["p"]]


def load(sid_):
    c = CAND.loc[sid_]
    ok("frame_index == v1 notes %s" % sid_, FRAMES[sid_], int(c.frame_index), int(c.frame_index) == FRAMES[sid_])
    at = read(c.relax_dir + "/relaxed.extxyz", index=int(c.frame_index))
    e1n = int(E1[(E1.arm == c.arm) & (E1.chunk == c.chunk) & (E1.cif_index == c.cif_index) & (E1.stage == "post")].natoms.iloc[0])
    ok("natoms(extxyz)==E1==candidates==v1 %s" % sid_, "%d/%d/%d" % (e1n, int(c.natoms), NATOMS[sid_]), len(at),
       len(at) == e1n == int(c.natoms) == NATOMS[sid_])
    at.wrap()
    n, t, p = parse_axes(c["axes"])
    sc = at.get_scaled_positions(wrap=True)
    Ln, Lt, Lp = at.cell[n, n], at.cell[t, t], at.cell[p, p]
    sv = SHIFT_V1[sid_]
    ok("Ln==v1 %s" % sid_, "%.6f" % sv["Ln"], "%.6f" % Ln, abs(Ln - sv["Ln"]) < 1e-4)
    ok("Lp==v1 %s" % sid_, "%.6f" % sv["Lp"], "%.6f" % Lp, abs(Lp - sv["Lp"]) < 1e-4)
    x = sc[:, n] * Ln
    y = sc[:, p] * Lp
    layer = (sc[:, t] % 1.0 >= 0.5).astype(int)
    s = dict(c=c, id=sid_, x=x, y=y, layer=layer, Ln=Ln, Lp=Lp, dx=0.0, zg=None)
    if c.category != "no_orientation_switch":
        zAB, zBA = float(c.z_AB), float(c.z_BA)
        L = (zBA - zAB) % 1.0
        mid = (zAB + L / 2) % 1.0
        dxf = (0.5 - mid) % 1.0
        dx = dxf * Ln
        s["x"] = (x + dx) % Ln
        g1, g2 = (zAB + dxf) % 1.0, (zBA + dxf) % 1.0
        s.update(dx=dx, dxf=dxf, L=L, zg=[g1, g2])
        edge = min(min(g * Ln, (1 - g) * Ln) for g in (g1, g2))
        s["edge"] = edge
        ok("shift dx == v1 %s [A]" % sid_, "%.6f" % sv["dx"], "%.6f" % dx, abs(dx - sv["dx"]) < 1e-6)
        ok("shift dxf == v1 %s" % sid_, "%.8f" % sv["dxf"], "%.8f" % dxf, abs(dxf - sv["dxf"]) < 1e-8)
        ok("GB1 position == v1 %s [A]" % sid_, "%.6f" % sv["g"][0], "%.6f" % (g1 * Ln), abs(g1 * Ln - sv["g"][0]) < 1e-6)
        ok("GB2 position == v1 %s [A]" % sid_, "%.6f" % sv["g"][1], "%.6f" % (g2 * Ln), abs(g2 * Ln - sv["g"][1]) < 1e-6)
        ok("GB-to-cell-edge == v1 and >=6 A %s" % sid_, "%.4f" % sv["edge"], "%.4f" % edge, abs(edge - sv["edge"]) < 1e-6 and edge >= 6.0)
        ok("interval GB1->GB2 preserved %s" % sid_, "%.6f" % L, "%.6f" % ((g2 - g1) % 1.0), abs(((g2 - g1) % 1.0) - L) < 1e-9)
        ok("GB midpoint at a/2 %s" % sid_, "0.5", "%.6f" % ((g1 + L / 2) % 1.0), abs(((g1 + L / 2) % 1.0) - 0.5) < 1e-9)
        ok("x within [0,Ln) %s" % sid_, "[0,%.2f)" % Ln, "%.3f..%.3f" % (s["x"].min(), s["x"].max()), s["x"].min() >= 0 and s["x"].max() < Ln)
    else:
        ok("no shift (no_gb) == v1 %s" % sid_, "0.0", "0.0", sv["dx"] == 0.0)
    return s


S = [load(i) for i in IDS]

SUP = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}


def sci2(x):
    m, e = ("%.1e" % x).split("e")
    return "%s × 10%s" % (m, "".join(SUP[ch] for ch in str(int(e))))


def right_labels(c, mode):
    if mode == "v2":      # for measuring the centroid "before" (earlier labels kept as they were)
        if c.category == "equivalent":
            return ["Proposed model", "Equivalent GBs", "d_max (Stage 2) = %s Å" % sci2(c.s2_max_A)]
        if c.category == "nonequivalent":
            return ["MatterGen baseline", "Non-equivalent GBs", "d_max (Stage 2) = %.2f Å" % c.s2_max_A]
        return ["MatterGen baseline", "No orientation switch detected", "frac_A = %.2f, frac_B = %.2f" % (c.frac_A, c.frac_B)]
    if c.category == "equivalent":
        return ["Proposed model", "Equivalent GBs", "d_max = %s Å" % sci2(c.s2_max_A)]
    if c.category == "nonequivalent":
        return ["MatterGen baseline", "Non-equivalent GBs", "d_max = %.2f Å" % c.s2_max_A]
    return ["MatterGen baseline", "No orientation switch detected"]


LABELS = [right_labels(s["c"], "new") for s in S]
ok("label row1", "Proposed model | Equivalent GBs | d_max = 1.7 × 10⁻⁶ Å", " | ".join(LABELS[0]),
   LABELS[0] == ["Proposed model", "Equivalent GBs", "d_max = 1.7 × 10⁻⁶ Å"])
ok("label row2", "MatterGen baseline | Non-equivalent GBs | d_max = 2.09 Å", " | ".join(LABELS[1]),
   LABELS[1] == ["MatterGen baseline", "Non-equivalent GBs", "d_max = 2.09 Å"])
ok("label row3", "MatterGen baseline | No orientation switch detected (2 lines)", " | ".join(LABELS[2]),
   LABELS[2] == ["MatterGen baseline", "No orientation switch detected"])


def txt(ax, x, y, s, fs=7, bold=False, ha="left", va="top", color="#000000"):
    return ax.text(x, y, s, fontsize=fs, fontweight="bold" if bold else "normal", ha=ha, va=va, color=color)


def atoms(ax, xs, ys, layer, scale):
    d = 2 * RAD * scale
    for ly, fc in [(0, OKABE_ITO[0]), (1, "#FFFFFF")]:
        m = layer == ly
        if m.any():
            ax.scatter(xs[m], ys[m], s=d * d, marker="o", facecolors=fc, edgecolors="#000000", linewidths=LWA, zorder=3)


OV = RAD * SC
LAB_H = 13.5
GAP = 2.0 * MM
X0 = 8.0
RLAB_X = X0 + max(s["Ln"] for s in S) * SC + OV + 12.0
GEOM = {}
ART = {}


def new_art():
    ART.clear()
    ART.update(rlab=[], browlab=[], bars=[], struct_l=[], struct_r=[])


def draw_row(ax, s, y, labels, mode):
    two = s["zg"] is not None
    y_ct = y + (LAB_H if two else OV)
    Ln, Lp = s["Ln"], s["Lp"]
    wc, hc = Ln * SC, Lp * SC
    y_cb = y_ct + hc
    ax.add_patch(Rectangle((X0, y_ct), wc, hc, fill=False, edgecolor="#000000", linewidth=0.8, zorder=1))
    xs = X0 + s["x"] * SC
    atoms(ax, xs, y_cb - s["y"] * SC, s["layer"], SC)
    ART["struct_l"].append(min(X0, xs.min() - OV))
    ART["struct_r"].append(max(X0 + wc, xs.max() + OV))
    if two:
        for i, g in enumerate(s["zg"]):
            xx = X0 + g * Ln * SC
            ax.plot([xx, xx], [y_ct - OV - 1.0, y_cb + 3], color=OKABE_ITO[6], linewidth=1.0, linestyle=(0, (3, 2)), zorder=4)
            txt(ax, xx, y_ct - OV - 1.5, "GB%d" % (i + 1), fs=7, ha="center", va="bottom", color=OKABE_ITO[6])
    yc = (y_ct + y_cb) / 2
    if len(labels) == 3:
        ys_ = [yc - 11.0, yc, yc + 10.5]
    else:                  # two lines (row 3): two lines centered vertically
        ys_ = [yc - 5.5, yc + 5.5]
    for i, (lab, yy) in enumerate(zip(labels, ys_)):
        if i == 0:
            ART["rlab"].append(txt(ax, RLAB_X, yy, lab, fs=8, bold=True, va="center"))
        elif i == 1:
            ART["rlab"].append(txt(ax, RLAB_X, yy, lab, fs=8, va="center"))
        else:
            ART["rlab"].append(txt(ax, RLAB_X, yy, lab, fs=7, va="center", color=GRAY_TXT))
    GEOM[s["id"]] = dict(y_ct=y_ct, y_cb=y_cb, wc=wc, hc=hc)
    return y_cb + OV


B = pd.read_csv(B_PATH)
B = B[(B.stage == "post") & (B.scope == "all7")].set_index("arm")
EXP = {"proposed": dict(equivalent=350, g2_noneq=0, no_gb=0, excluded_pre=0, n_req=350, unconverged=0, other=0),
       "baseline": dict(equivalent=0, g2_noneq=74, no_gb=256, excluded_pre=20, n_req=350, unconverged=0, other=0)}
for arm_, d_ in EXP.items():
    for k_, v_ in d_.items():
        ok("(b) m1_classes %s %s" % (arm_, k_), v_, int(B.loc[arm_, k_]), int(B.loc[arm_, k_]) == v_)
CATS = [("equivalent", "Equivalent GBs", OKABE_ITO[3]), ("g2_noneq", "Non-equivalent GBs", OKABE_ITO[7]),
        ("no_gb", "No orientation switch detected", "#9A9A9A"), ("excluded_pre", "Cell-angle exclusion (before relaxation)", "hatch")]
ARMS = [("proposed", "Proposed model"), ("baseline", "MatterGen baseline")]
C_HATCH = "#555555"


def hatch_lines(x0, y0, x1, y1, d=2.55):
    segs = []
    for k in np.arange(np.floor((x0 - y1) / d), np.ceil((x1 - y0) / d) + 1):
        cc = k * d
        pts = []
        for u in (x0, x1):
            v = u - cc
            if y0 - 1e-9 <= v <= y1 + 1e-9: pts.append((u, v))
        for v in (y0, y1):
            u = v + cc
            if x0 + 1e-9 < u < x1 - 1e-9: pts.append((u, v))
        pts = sorted(set((round(a, 6), round(b, 6)) for a, b in pts))
        if len(pts) >= 2 and abs(pts[0][0] - pts[-1][0]) + abs(pts[0][1] - pts[-1][1]) > 1e-4:
            segs.append([pts[0], pts[-1]])
    return segs


def rect(ax, x0, y0, x1, y1, kind, lw=0.5, bar=False):
    if kind == "hatch":
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#FFFFFF", edgecolor="none", zorder=2))
        segs = [[(a, -b) for a, b in sg] for sg in hatch_lines(x0, -y1, x1, -y0)]
        ax.add_collection(LineCollection(segs, colors=C_HATCH, linewidths=0.4, zorder=2.5))
        fc = "none"
    else:
        fc = kind
    p = Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc, edgecolor="#000000", linewidth=lw, zorder=3)
    ax.add_patch(p)
    if bar: ART["bars"].append(p)


def bars_and_axis(ax, ys, BH, ax_l, ax_r, bar_rows=True):
    k = (ax_r - ax_l) / 350.0
    nbars = {}
    for (arm, lab), y0 in zip(ARMS, ys):
        left = 0
        for key, label, col in CATS:
            n = int(B.loc[arm, key])
            if n == 0: continue
            rect(ax, ax_l + left * k, y0, ax_l + (left + n) * k, y0 + BH, col, bar=True)
            fsz = 7 if n * k > 12 else 5.5
            ax.text(ax_l + (left + n / 2) * k, y0 + BH / 2 + 0.3, str(n), ha="center", va="center", fontsize=fsz, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", fc="#FFFFFF" if col == "hatch" else "none", ec="none", lw=0))
            nbars[(arm, key)] = n * k
            left += n
    y_axis = ys[1] + BH + 3
    ax.plot([ax_l, ax_r], [y_axis, y_axis], color="#000000", linewidth=0.5, zorder=3)
    for tv in (0, 100, 200, 300, 350):
        xx = ax_l + tv * k
        ax.plot([xx, xx], [y_axis, y_axis + 2.5], color="#000000", linewidth=0.5, zorder=3)
        txt(ax, xx, y_axis + 3.0, str(tv), fs=7, ha="center", va="top")
    txt(ax, (ax_l + ax_r) / 2, y_axis + 12.5, "Structures (of 350 requested)", fs=7, ha="center", va="top")
    return y_axis, nbars


def draw_b_v2(ax, x0, yt, width):
    """Reproduces panel (b) of the earlier version as it was (only for measuring the centroid "before")"""
    txt(ax, x0, yt, "(b)", fs=9, bold=True)
    BH = 11.0
    ax_l = x0 + 118.0
    ax_r = x0 + width
    ys = [yt + 10, yt + 10 + BH + 4]
    for (arm, lab), y0 in zip(ARMS, ys):
        ART["browlab"].append(txt(ax, ax_l - 6, y0 + BH / 2, lab, fs=7, ha="right", va="center"))
    y_axis, nbars = bars_and_axis(ax, ys, BH, ax_l, ax_r)
    y_leg = y_axis + 24.0
    sw, sh = 12.0, 6.0
    pos = [(ax_l, y_leg), (ax_l, y_leg + 9), (ax_l + 160, y_leg), (ax_l + 160, y_leg + 9)]
    for (key, label, col), (px, py) in zip(CATS, pos):
        rect(ax, px, py, px + sw, py + sh, col)
        txt(ax, px + sw + 4, py + sh / 2, label, fs=7, va="center")
    return y_leg + 9 + sh, nbars


# ---- final panel (b). Parameters (author's choice)
BH = 9.0             # bar height (earlier: 11), because of the total-height constraint of +-3 mm
G_LABEL = 1.0 * MM   # gap between a row label (bbox bottom) and the bar top = 1 mm (specification)
G_PREV = 3.5         # gap between the bottom of the previous bar and the next row label (bbox top)
G_HDR = 0.0          # gap between "(b)" and row label 1
LEG_DY = 8.0          # legend line spacing (earlier: 9)
LEG_COLGAP = 14.0


def draw_b_new(ax, yt, ax_l, ax_r, rend):
    tb = txt(ax, 2, yt, "(b)", fs=9, bold=True)
    hb = tb.get_window_extent(rend).height
    y_cur = yt + hb + G_HDR
    ys = []
    for (arm, lab) in ARMS:
        t = ax.text(ax_l, 0, lab, fontsize=7, ha="left", va="bottom")
        h = t.get_window_extent(rend).height
        y_bar = y_cur + h + G_LABEL
        t.set_y(y_bar - G_LABEL)
        ART["browlab"].append(t)
        ys.append(y_bar)
        y_cur = y_bar + BH + G_PREV
    y_axis, nbars = bars_and_axis(ax, ys, BH, ax_l, ax_r)
    y_leg = y_axis + 22.5
    sw, sh = 12.0, 6.0
    cols = [CATS[0:2], CATS[2:4]]
    widths = []
    for cc in cols:
        w = 0
        for key, label, col in cc:
            t = ax.text(0, 0, label, fontsize=7); w = max(w, t.get_window_extent(rend).width); t.remove()
        widths.append(sw + 4 + w)
    block = widths[0] + LEG_COLGAP + widths[1]
    xs0 = (ax_l + ax_r) / 2 - block / 2
    colx = [xs0, xs0 + widths[0] + LEG_COLGAP]
    for ci, cc in enumerate(cols):
        for ri, (key, label, col) in enumerate(cc):
            py = y_leg + ri * LEG_DY
            rect(ax, colx[ci], py, colx[ci] + sw, py + sh, col)
            ART.setdefault("legend", []).append(txt(ax, colx[ci] + sw + 4, py + sh / 2, label, fs=7, va="center"))
    ART["legend_span"] = (colx[0], colx[1] + widths[1])
    return y_leg + LEG_DY + sh, nbars


W = 178 * MM


def run_a(ax, mode):
    txt(ax, 2, 0, "(a)", fs=9, bold=True)
    y = 14.0 - LAB_H
    labs = [right_labels(s["c"], mode) for s in S]
    for s, lab in zip(S, labs):
        y = draw_row(ax, s, y, lab, mode) + GAP
    return y


def newfig(h):
    f = plt.figure(figsize=(W / 72, h / 72), dpi=72)
    a = f.add_axes([0, 0, 1, 1]); a.set_xlim(0, W); a.set_ylim(h, 0); a.axis("off")
    return f, a




def measure(fig, ax, mode):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    a_l, a_r = min(ART["struct_l"]), max(t.get_window_extent(rend).x1 for t in ART["rlab"])
    bl = [t.get_window_extent(rend).x0 for t in ART["browlab"]] + [p.get_x() for p in ART["bars"]]
    br = [p.get_x() + p.get_width() for p in ART["bars"]]
    b_l, b_r = min(bl), max(br)
    return dict(a_l=a_l, a_r=a_r, a_c=(a_l + a_r) / 2, b_l=b_l, b_r=b_r, b_c=(b_l + b_r) / 2)


# ---------------------------------------------------------------- reproduction of the earlier layout (centroid "before")
new_art()
f0, a0 = newfig(600)
ya0 = run_a(a0, "v2")
yb0, _ = draw_b_v2(a0, 2, ya0 + 2, W - 2 - 14)
M_V2 = measure(f0, a0, "v2")
H_V2 = yb0 + 2.5
plt.close(f0)
ok("v2 reproduction: height == v2 notes 99.6 mm", "99.6", "%.1f" % (H_V2 / MM), abs(H_V2 / MM - 99.6) < 0.06)

# ---------------------------------------------------------------- final version
new_art()
fd, axd = newfig(600)
ya = run_a(axd, "new")
a_l_new, a_r_new = min(ART["struct_l"]), None
fd.canvas.draw()
a_r_new = max(t.get_window_extent(fd.canvas.get_renderer()).x1 for t in ART["rlab"])
yb, _ = draw_b_new(axd, ya - GAP + 3.0, a_l_new, a_r_new, fd.canvas.get_renderer())
plt.close(fd)
H = yb + 2.0

new_art()
fig, ax = newfig(H)
GEOM.clear()
ya = run_a(ax, "new")
YB_TOP = ya - GAP + 3.0
fig.canvas.draw()
rend = fig.canvas.get_renderer()
yb, nb = draw_b_new(ax, YB_TOP, a_l_new, a_r_new, rend)
M_NEW = measure(fig, ax, "new")
over = []
for t in ax.texts:
    bb = t.get_window_extent(rend)
    if bb.x1 > W + 0.5 or bb.x0 < -0.5:
        over.append((t.get_text(), round(bb.x0, 1), round(bb.x1, 1)))
ok("no text beyond figure width (%.1f pt)" % W, "none", str(over) if over else "none", not over)
ok("height within v2 99.6 mm +- 3 mm", "96.6..102.6", "%.1f" % (H / MM), abs(H / MM - 99.6) <= 3.0)
V.append((SCRIPT, "size", "178 x 99.6+-3 mm", "%.1f x %.1f mm (%.1f x %.1f pt)" % (W / MM, H / MM, W, H), "INFO"))
V.append((SCRIPT, "(a) bottom y mm / (b) top..bottom mm", "-", "(a) ends %.1f mm; (b) %.1f..%.1f mm" % ((ya - GAP) / MM, YB_TOP / MM, yb / MM), "INFO"))
for sid_, g in GEOM.items():
    V.append((SCRIPT, "cell geom pt %s" % sid_, "-", "y_ct=%.1f y_cb=%.1f w=%.1f h=%.1f" % (g["y_ct"], g["y_cb"], g["wc"], g["hc"]), "INFO"))

for nm, M in [("v2 (before)", M_V2), ("render (after)", M_NEW)]:
    V.append((SCRIPT, "extent %s (a) left..right pt / mm" % nm, "-",
              "%.1f..%.1f pt / %.2f..%.2f mm; centre %.2f pt = %.2f mm" % (M["a_l"], M["a_r"], M["a_l"] / MM, M["a_r"] / MM, M["a_c"], M["a_c"] / MM), "INFO"))
    V.append((SCRIPT, "extent %s (b) left..right pt / mm" % nm, "-",
              "%.1f..%.1f pt / %.2f..%.2f mm; centre %.2f pt = %.2f mm" % (M["b_l"], M["b_r"], M["b_l"] / MM, M["b_r"] / MM, M["b_c"], M["b_c"] / MM), "INFO"))
    V.append((SCRIPT, "centroid diff %s (b)-(a) mm" % nm, "-", "%+.2f" % ((M["b_c"] - M["a_c"]) / MM), "INFO"))
dnew = abs(M_NEW["b_c"] - M_NEW["a_c"]) / MM
ok("AC centroid diff (b)-(a) <= 5 mm [final]", "<=5", "%+.2f mm" % ((M_NEW["b_c"] - M_NEW["a_c"]) / MM), dnew <= 5.0)
ok("AC bar left == structure left (<=0.5 pt)", "%.2f" % M_NEW["a_l"], "%.2f" % M_NEW["b_l"], abs(M_NEW["b_l"] - M_NEW["a_l"]) <= 0.5)
ok("AC bar right == right-label right (<=0.5 pt)", "%.2f" % M_NEW["a_r"], "%.2f" % M_NEW["b_r"], abs(M_NEW["b_r"] - M_NEW["a_r"]) <= 0.5)
V.append((SCRIPT, "(b) bar widths pt", "-", "; ".join("%s/%s=%.1f" % (a, k, v) for (a, k), v in nb.items()), "INFO"))
# whole ink extent (including legend and axis labels) INFO
inkL = min(t.get_window_extent(rend).x0 for t in ax.texts if t.get_position()[1] > YB_TOP - 1)
inkR = max(t.get_window_extent(rend).x1 for t in ax.texts if t.get_position()[1] > YB_TOP - 1)
V.append((SCRIPT, "(b) full-ink text extent incl. legend/axis labels pt", "-", "%.1f..%.1f; centre %.2f pt (a-centre %.2f)" % (inkL, inkR, (inkL + inkR) / 2, M_NEW["a_c"]), "INFO"))
V.append((SCRIPT, "(b) legend span pt", "-", "%.1f..%.1f (bar centre %.1f)" % (ART["legend_span"][0], ART["legend_span"][1], (a_l_new + a_r_new) / 2), "INFO"))
BAR_Y0 = min(p.get_y() for p in ART["bars"]); BAR_Y1 = BAR_Y0 + BH      # y extent of the first bar (for the PNG ink cross-check)
V.append((SCRIPT, "layout adopted", "-", "row label above bar (left-aligned at bar left, 1 mm above bar top); bars %.1f..%.1f pt" % (a_l_new, a_r_new), "INFO"))

# ---------------------------------------------------------------- output
base = WD + "/Fig2"
nu, nc = save_svg(fig, base + ".svg")
fig.savefig(base + ".pdf", facecolor="white")
rsvg(base + ".svg", base + ".png")
plt.close(fig)
print("svg baked use %d clip %d; %.1f x %.1f mm" % (nu, nc, W / MM, H / MM))
r = subprocess.run(["/usr/bin/inkscape", "--without-gui", "--file=" + base + ".svg", "--export-emf=" + base + ".emf"],
                   capture_output=True, text=True, timeout=200)
ok("emf written", ">0 B", os.path.getsize(base + ".emf") if os.path.exists(base + ".emf") else 0,
   os.path.exists(base + ".emf") and os.path.getsize(base + ".emf") > 0)

# per panel (same coordinates; panel a draws only (a), panel b only (b), with the height reduced)
new_art()
ha = ya - GAP + 2.5
fa, aa = newfig(ha)
run_a(aa, "new")
save_svg(fa, WD + "/Fig2_panel_a.svg"); rsvg(WD + "/Fig2_panel_a.svg", WD + "/Fig2_panel_a.png"); plt.close(fa)
new_art()
hb_ = yb - YB_TOP + 2.0
fb, ab = newfig(hb_)
fb.canvas.draw()
draw_b_new(ab, 0.0, a_l_new, a_r_new, fb.canvas.get_renderer())
save_svg(fb, WD + "/Fig2_panel_b.svg"); rsvg(WD + "/Fig2_panel_b.svg", WD + "/Fig2_panel_b.png"); plt.close(fb)

# ---------------------------------------------------------------- SVG check
for nm in ["Fig2", "Fig2_panel_a", "Fig2_panel_b"]:
    sv = open(WD + "/" + nm + ".svg", encoding="utf-8").read()
    for tag in ["<pattern", "<use", "clipPath", "<mask", "<filter", "tspan", "clip-path"]:
        ok("AC svg %s count of %s" % (nm, tag), 0, sv.count(tag), sv.count(tag) == 0)
    fams = set(re.findall(r"font-family:\s*([^;\"]+)", sv)) | set(re.findall(r"font:[^;\"]*?(?:px|pt)\s+([^;\"]+)", sv))
    fams = set(f.strip().strip("'\"") for f in fams)
    ok("AC svg %s font families" % nm, "Arial only", ",".join(sorted(fams)), fams == {"Arial"})
    texts = re.findall(r"<text[^>]*>(.*?)</text>", sv, flags=re.S)
    nl = sum(1 for t in texts if "\n" in t or "\r" in t)
    ok("AC svg %s <text> with newline (of %d)" % (nm, len(texts)), 0, nl, nl == 0)
    ok("AC svg %s <text> with child elements" % nm, 0, sum(1 for t in texts if "<" in t), all("<" not in t for t in texts))
    for pat in ["Stage", "frac", "J/m", "γ", "gamma"]:
        c_ = sv.count(pat) if pat != "gamma" else sv.lower().count("gamma")
        ok("AC svg %s occurrences of '%s'" % (nm, pat), 0, c_, c_ == 0)
    if nm == "Fig2":
        alltxt = [re.sub(r"&amp;", "&", t) for t in texts]
        ok("AC svg text strings count", "see list", len(alltxt), True)

# ---------------------------------------------------------------- agreement assertions with the earlier version (structures, shifts, natoms, values of (b))
v2v = pd.read_csv(D + "/v2/verification_v2.csv", dtype=str, keep_default_na=False)
v2map = {r.item: r for r in v2v.itertuples()}
KEYS = ("frame_index", "natoms", "Ln==", "Lp==", "shift ", "GB1 position", "GB2 position", "GB-to-cell-edge", "interval GB1", "GB midpoint", "x within", "no shift", "(b) m1_classes")
need = [i for i in v2map if i.startswith(KEYS)]
mine = {r[1]: r for r in V}
miss = [i for i in need if i not in mine]
ok("v2 items (structure/shift/natoms/(b)) all re-checked here", "0 missing of %d" % len(need), "%d missing" % len(miss), not miss)
bad = [i for i in need if mine[i][3] != v2map[i].observed or mine[i][4] != "PASS"]
ok("v2 observed values == render observed (structure/shift/natoms/(b)) over %d items" % len(need), "0 mismatch", "%d mismatch %s" % (len(bad), bad[:3]), not bad)

# pixel agreement of the structure part (left of the left edge of the right labels, rows of (a)): earlier PNG (read-only) and the final PNG
s_ = 2400.0 / W
x1 = int((RLAB_X - 4) * s_); y1 = int((ya - GAP) * s_)
# rsvg-convert rounds the output height to integer px, so the vertical scale differs slightly with H; compare by "agreement of dark pixels within 2 px" instead of exact pixel equality
dkA = np.array(Image.open(D + "/v2/Fig2_simplified_v2.png").convert("L"))[:y1, :x1] < 128
dkB = np.array(Image.open(base + ".png").convert("L"))[:y1, :x1] < 128
def near(M, R=2):
    out = np.zeros_like(M)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            out |= np.roll(np.roll(M, dy, 0), dx, 1)
    return out
missA = int((dkA & ~near(dkB)).sum()); missB = int((dkB & ~near(dkA)).sum())
ok("structure columns (x<%d px, y<%d px): dark px of v2 not within 2 px of final (and vice versa); of %d/%d dark px" % (x1, y1, dkA.sum(), dkB.sum()),
   "0 / 0", "%d / %d" % (missA, missB), missA == 0 and missB == 0)

# independent cross-check of the centroid with PNG ink (INFO): right edge only (right edge of the (a) right-label ink vs the bar ink)
img = np.array(Image.open(base + ".png").convert("L"))
def ink_cols(y0, y1_):
    sub = img[int(y0 * s_):int(y1_ * s_)]
    cols = np.where((sub < 200).any(axis=0))[0]
    return cols.min() / s_, (cols.max() + 1) / s_
def ink_range(y0, y1_, xa=0.0, xb=W):
    sub = img[int(y0 * s_):int(y1_ * s_), int(xa * s_):int(xb * s_)]
    cols = np.where((sub < 200).any(axis=0))[0]
    return xa + cols.min() / s_, xa + (cols.max() + 1) / s_
pa_l, _ = ink_range(14.0, ya - GAP, 0.0, RLAB_X - 2)
_, pa_r = ink_range(14.0, ya - GAP, RLAB_X - 2, W)
pb_l, pb_r = ink_range(BAR_Y0 + 1, BAR_Y1 - 1)
V.append((SCRIPT, "PNG ink cross-check (a) structure-left..label-right pt / mm", "-", "%.1f..%.1f pt / %.2f..%.2f mm; centre %.2f mm" % (pa_l, pa_r, pa_l / MM, pa_r / MM, (pa_l + pa_r) / 2 / MM), "INFO"))
V.append((SCRIPT, "PNG ink cross-check (b) bar-left..bar-right pt / mm", "-", "%.1f..%.1f pt / %.2f..%.2f mm; centre %.2f mm" % (pb_l, pb_r, pb_l / MM, pb_r / MM, (pb_l + pb_r) / 2 / MM), "INFO"))
ok("PNG ink centroid diff (b)-(a) <= 5 mm", "<=5", "%+.2f mm" % (((pb_l + pb_r) - (pa_l + pa_r)) / 2 / MM), abs(((pb_l + pb_r) - (pa_l + pa_r)) / 2 / MM) <= 5.0)

pd.DataFrame(V, columns=["script", "item", "expected", "observed", "result"]).to_csv(WD + "/verification_render.csv", index=False)
print("rows", len(V), "FAIL", sum(1 for r_ in V if r_[4] == "FAIL"))
print("V2  :", {k: round(v / MM, 2) for k, v in M_V2.items()}, "H %.1f" % (H_V2 / MM))
print("NEW :", {k: round(v / MM, 2) for k, v in M_NEW.items()}, "H %.1f" % (H / MM))
