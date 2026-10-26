#!/usr/bin/env python3
"""Manuscript version of the two-GB overview figure (3 panels).

Copy of plot_g2_overview.py with only the drawn labels replaced by reviewer-readable English.
Data reading, computation, coordinates and colors are unchanged. The data CSV is not written here
(it stays with the original script).

Outputs (results/figs_manuscript/): g2_overview.png / g2_overview.pdf. Refuses to overwrite.
Run: mace-mh env (matplotlib 3.10); cwd does not matter.

additions (an internal design note that is not included):
  --variant base    (default) identical drawing to the 2026-10-06 figure, name g2_overview.
  --variant reseed  panel C gets 3 regenerated points at w=2.0 and Equivalent = E12(theta=0.064);
                    name g2_overview_reseed (+ g2_overview_reseed_layout.json).
  --outdir DIR      output directory (default results/figs_manuscript).
  --self_check        run the input-consistency checks on real and perturbed inputs; writes no file.
"""
import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN.parent / "gen_eval" / "scripts"))  # paperfig.py (same as the original script)
from paperfig import COL_2, OKABE_ITO, save_fig, setup_style  # noqa: E402

# ==== BEGIN DATA KEYS ====
# Keys of the input JSON files. They are read, never drawn. Everything outside this block is label/figure code.
ARM_CON = "proposed"
ARM_UNC = "baseline"
LAYER_A = "additional"
LAYER_B = "main"
STAGE_FIN = "post"
K_NREQ = "n_req"
M_CONV = "survive"
M_NEAR = "near_post"
M_NOVEL = "novel_post"
M_NUN = "nun_post"
SET_B_FMT = "{layer}|{arm}|w2.0|inner5"
SET_C_FMT = "{layer}|{arm}|w2.0|all7"
ADDITIONAL = RUN / "results/gb_equivalence/gb_symfit_additional_g2_and_e.json"
MAIN = RUN / "results/structure_filtered/summary.json"
# (reseed variant only)
VRUN = RUN.parent / "verification"
RESEED_CSV = VRUN / "results/regeneration/g2_by_layer.csv"
STAGE2_CHECK = VRUN / "results/gb_stage2/stage2_check_inputs.json"
STAGE2_CHECK_VERDICT = VRUN / "results/gate_verdicts_140_20261008.md"
STAGE2_CHECK_PASS_LINE = "**C1 = PASS**"
THETA_KEY = "0.064"
RESEED_LAYERS = ("baseline_regen1", "baseline_regen2", "baseline_regen3")
# ==== END DATA KEYS ====

FIGDIR = RUN / "results/figs_manuscript"
NAME = "g2_overview"
EXTRAP_IDS = (5, 6)  # extrapolated targets (gamma_target outside the training range)
Z = 1.959963984540054

C_PROPOSED = OKABE_ITO[5]     # blue
C_VAN = OKABE_ITO[6]      # vermillion
VAN_W_COL = {1.0: OKABE_ITO[1], 2.0: OKABE_ITO[6], 3.0: OKABE_ITO[3], 4.0: OKABE_ITO[7]}
VAN_W_MK = {1.0: "o", 2.0: "s", 3.0: "^", 4.0: "D"}

# ---- drawn labels ----
LAB_CON = "Constrained"
LAB_UNC = "Unconstrained"


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, min(p, max(0.0, c - h)), max(p, min(1.0, c + h))  # clamp so rounding keeps p inside the interval


RESEED_DELTAS = (0.19, 0.37, 0.55)  # x offsets of the 3 regenerated points from w=2.0
RESEED_LABEL = "Unconstrained,\nExpt. 1 repeats (3 runs)"


def check_inputs(csv_rows, cells, C, mv, verdict_text):
    """Input-consistency checks of the reseed variant. Raises on any mismatch."""
    # 3-3-2 precondition : C1 must be PASS
    if STAGE2_CHECK_PASS_LINE not in verdict_text:
        raise RuntimeError(f"precondition missing: {STAGE2_CHECK_PASS_LINE!r} not in verdict text")
    rows = {(r["layer"], r["arm"]): r for r in csv_rows}
    # 3-3-3a main/baseline consistency with the existing figure data
    mrow = rows[("main", ARM_UNC)]
    assert int(mrow["k_g2_post"]) == mv["k"], ("main/baseline k", mrow["k_g2_post"], mv["k"])
    assert int(mrow["n_req"]) == mv["n"] == 350, ("main/baseline n", mrow["n_req"], mv["n"])
    # 3-3-3b regenerated rows
    for layer in RESEED_LAYERS:
        r = rows[(layer, ARM_UNC)]
        assert int(r["n_req"]) == 350, (layer, "n_req", r["n_req"])
        p, lo, hi = wilson(int(r["k_g2_post"]), int(r["n_req"]))
        assert abs(lo - float(r["wilson_lo"])) <= 1e-4, (layer, "wilson_lo", lo, r["wilson_lo"])
        assert abs(hi - float(r["wilson_hi"])) <= 1e-4, (layer, "wilson_hi", hi, r["wilson_hi"])
    # 3-3-3c / 3-3-3d old vs new population per w, E12 >= E1
    for w in (1.0, 2.0, 3.0, 4.0):
        cell = cells[f"{LAYER_A}/{ARM_UNC}/{w:.1f}"]
        assert cell["E1"][THETA_KEY] == C[(ARM_UNC, w)][5], (w, "E1 vs stage-1 e", cell["E1"][THETA_KEY], C[(ARM_UNC, w)][5])
        assert cell["G2"] == C[(ARM_UNC, w)][0], (w, "G2 vs g2", cell["G2"], C[(ARM_UNC, w)][0])
        assert cell["E12"][THETA_KEY] >= cell["E1"][THETA_KEY], (w, "E12 < E1", cell["E12"][THETA_KEY], cell["E1"][THETA_KEY])


def run_self_check(csv_rows, cells, C, mv, verdict_text):
    print("self_check: mv keys =", sorted(mv.keys()), "(using mv['k'], mv['n'])")
    ok = True
    try:
        check_inputs(csv_rows, cells, C, mv, verdict_text)
        print("real data: OK (no exception)")
    except Exception as e:  # noqa: BLE001
        ok = False
        print("real data: FAIL", type(e).__name__, e)

    def case(name, expect, **mod):
        nonlocal ok
        a = dict(csv_rows=copy.deepcopy(csv_rows), cells=copy.deepcopy(cells), C=C, mv=copy.deepcopy(mv), verdict_text=verdict_text)
        mod["apply"](a)
        try:
            check_inputs(a["csv_rows"], a["cells"], a["C"], a["mv"], a["verdict_text"])
            ok = False
            print(f"perturbed [{name}]: NOT DETECTED (FAIL)")
        except expect as e:
            print(f"perturbed [{name}]: detected {type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"perturbed [{name}]: wrong exception {type(e).__name__}: {e} (FAIL)")

    def m_e1(a):
        a["cells"][f"{LAYER_A}/{ARM_UNC}/2.0"]["E1"][THETA_KEY] += 1

    def m_lo(a):
        for r in a["csv_rows"]:
            if r["layer"] == "baseline_regen1":
                r["wilson_lo"] = str(float(r["wilson_lo"]) + 0.01)

    def m_k(a):
        for r in a["csv_rows"]:
            if r["layer"] == "main" and r["arm"] == ARM_UNC:
                r["k_g2_post"] = "75"

    def m_v(a):
        a["verdict_text"] = a["verdict_text"].replace(STAGE2_CHECK_PASS_LINE, "")

    case("E1[0.064] at w=2.0 +1", AssertionError, apply=m_e1)
    case("regen1 wilson_lo +0.01", AssertionError, apply=m_lo)
    case("main/baseline k_g2_post=75", AssertionError, apply=m_k)
    case("verdict without C1 PASS line", RuntimeError, apply=m_v)
    print("SELF_CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(args):
    reseed = args.variant == "reseed"
    outdir = Path(args.outdir) if args.outdir else FIGDIR
    name = NAME + "_reseed" if reseed else NAME
    out_png, out_pdf = (outdir / f"{name}.png", outdir / f"{name}.pdf")
    out_layout = outdir / f"{name}_layout.json"
    if not args.self_check:
        for f in (out_png, out_pdf) + ((out_layout,) if reseed else ()):
            if f.exists():
                raise FileExistsError(str(f))
    additional = json.loads(ADDITIONAL.read_text())
    mainj = json.loads(MAIN.read_text())
    assert additional["layer"] == LAYER_A and mainj["X"]["G2"]["sets"]

    # ---------- panel A data ----------
    pt = [r for r in additional["per_target"] if r["stage"] == STAGE_FIN]
    tids = sorted({r["target_id"] for r in pt}, key=lambda t: float(next(r["gamma_target"] for r in pt if r["target_id"] == t)))
    gam = {t: float(next(r["gamma_target"] for r in pt if r["target_id"] == t)) for t in tids}
    A = {}  # (arm,w) -> list of (tid, gamma, k, n, p, lo, hi)
    for arm in (ARM_CON, ARM_UNC):
        for w in (1.0, 2.0, 3.0, 4.0):
            L = []
            for t in tids:
                rr = [r for r in pt if r["arm"] == arm and float(r["w"]) == w and r["target_id"] == t]
                assert len(rr) == 1, (arm, w, t, len(rr))
                k, n = rr[0]["g2"], rr[0][K_NREQ]
                p, lo, hi = wilson(k, n)
                L.append((t, gam[t], k, n, p, lo, hi))
            A[(arm, w)] = L
            # consistency: per-target sum equals the (arm, w, final-stage) row total
            row = [r for r in additional["rows"] if r["arm"] == arm and float(r["w"]) == w and r["stage"] == STAGE_FIN][0]
            assert sum(x[2] for x in L) == row["g2"], (arm, w)

    # ---------- panel B data ----------
    sets = mainj["X"]["G2"]["sets"]
    metrics = [(M_CONV, "Two\nGBs"), (M_NEAR, "+ near\ntarget"),
               (M_NOVEL, "+ novel"), (M_NUN, "+ novel,\nunique,\nnear target")]
    B = {}
    for arm in (ARM_CON, ARM_UNC):
        s = sets[SET_B_FMT.format(layer=LAYER_B, arm=arm)]
        assert s[K_NREQ] == 250
        for mid, _ in metrics:
            m = s[mid]
            B[(arm, mid)] = (m["k"], m["n"], m["p"], m["lo"], m["hi"])

    # ---------- panel C data ----------
    ws = (1.0, 2.0, 3.0, 4.0)
    C = {}
    for arm in (ARM_CON, ARM_UNC):
        for w in ws:
            row = [r for r in additional["rows"] if r["arm"] == arm and float(r["w"]) == w and r["stage"] == STAGE_FIN][0]
            g = row["g2_wilson"]
            C[(arm, w)] = (row["g2"], row[K_NREQ], g["p"], g["lo"], g["hi"], row["e"])
    mv = sets[SET_C_FMT.format(layer=LAYER_B, arm=ARM_UNC)][M_CONV]
    assert mv["n"] == 350

    # ---------- reseed inputs ----------
    cells = None
    if reseed or args.self_check:
        with open(RESEED_CSV, newline="") as fh:
            csv_rows = list(csv.DictReader(fh))
        cells = json.loads(STAGE2_CHECK.read_text())["cells"]
        verdict_text = STAGE2_CHECK_VERDICT.read_text()
        if args.self_check:
            return run_self_check(csv_rows, cells, C, mv, verdict_text)
        check_inputs(csv_rows, cells, C, mv, verdict_text)
        rs_rows = {r["layer"]: r for r in csv_rows if r["arm"] == ARM_UNC}
        RS = [wilson(int(rs_rows[l]["k_g2_post"]), int(rs_rows[l]["n_req"])) for l in RESEED_LAYERS]  # (p, lo, hi) recomputed

    # ---------- drawing ----------
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(COL_2, COL_2 * 0.5), gridspec_kw={"width_ratios": [1, 1, 1]})
    axA, axB, axC = axes
    YT = [0, 0.25, 0.5, 0.75, 1.0]   # ticks only 0-1.0; the band above 1.0 is reserved for the legend
    C_YBOT = -0.10   # panel C only: room below 0 so the annotation sits clear of the data and the axis line
    E_Y = C_YBOT + 0.04   # annotation bottom edge: 0.04 above the lower axis end
    YTOP = 1.5   # taller legend band than the original (longer labels need more rows); figure height raised to compensate

    def legend_band(ax, ncol, fs, **kw):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=ncol, fontsize=fs, borderaxespad=0.0,
                  columnspacing=0.8, handlelength=1.3, handletextpad=0.3, labelspacing=0.2, borderpad=0.1, **kw)

    # A: lines connect the 5 interior targets (target_id 0-4) only; extrapolated (5,6) are isolated x markers
    dod = {1.0: -0.045, 2.0: -0.015, 3.0: 0.015, 4.0: 0.045}
    for w in ws:  # constrained arm: thin same-color lines (4 lines, w=1-4)
        L_ = A[(ARM_CON, w)]
        li = [x for x in L_ if x[0] not in EXTRAP_IDS]
        le = [x for x in L_ if x[0] in EXTRAP_IDS]
        axA.errorbar([x[1] for x in li], [x[4] for x in li], yerr=[[x[4] - x[5] for x in li], [x[6] - x[4] for x in li]],
                     color=C_PROPOSED, lw=0.5, elinewidth=0.4, capsize=0.8, marker="none", alpha=0.8, zorder=2,
                     label=f"{LAB_CON}, $w$ = 1–4" if w == 1.0 else None)
        axA.errorbar([x[1] for x in le], [x[4] for x in le], yerr=[[x[4] - x[5] for x in le], [x[6] - x[4] for x in le]],
                     color=C_PROPOSED, lw=0, elinewidth=0.4, capsize=0.8, marker="x", markersize=3.0, alpha=0.8, zorder=2)
    for w in ws:
        L_ = A[(ARM_UNC, w)]
        for (t, g, k, n, p, lo, hi) in L_:
            if t in EXTRAP_IDS:
                axA.errorbar([g + dod[w]], [p], yerr=[[p - lo], [hi - p]], color=VAN_W_COL[w], lw=0, elinewidth=0.6, capsize=1.0,
                             marker="x", markersize=3.6, zorder=3)
        li = [x for x in L_ if x[0] not in EXTRAP_IDS]
        axA.errorbar([x[1] + dod[w] for x in li], [x[4] for x in li], yerr=[[x[4] - x[5] for x in li], [x[6] - x[4] for x in li]],
                     color=VAN_W_COL[w], lw=0.6, elinewidth=0.6, capsize=1.0, zorder=3,
                     marker=VAN_W_MK[w], markersize=3.0, label=f"{LAB_UNC}, $w$ = {w:g}")
    axA.plot([], [], ls="none", marker="x", color="0.3", markersize=3.6, label="Extrapolated targets (×)")  # legend dummy
    ext_x = [gam[t] for t in EXTRAP_IDS]
    axA.axvspan(min(ext_x) - 0.12, min(ext_x) + 0.12, color="0.92", zorder=0)
    axA.axvspan(max(ext_x) - 0.12, max(ext_x) + 0.12, color="0.92", zorder=0)
    axA.set_xlabel("Target interface energy $\\gamma_{\\mathrm{target}}$ [J/m$^2$]")
    axA.set_ylabel("Fraction with two GBs")
    axA.set_ylim(-0.04, YTOP)
    axA.set_yticks(YT)
    axA.set_xlim(0.1, 2.95)
    legend_band(axA, 1, 6.0)

    # B
    bw = 0.36
    for ai, (arm, col, lab) in enumerate(((ARM_CON, C_PROPOSED, LAB_CON), (ARM_UNC, C_VAN, LAB_UNC))):
        xs, ps, yl, yh = [], [], [], []
        for mi, (mid, _) in enumerate(metrics):
            k, n, p, lo, hi = B[(arm, mid)]
            xs.append(mi + (ai - 0.5) * bw * 1.05)
            ps.append(p)
            yl.append(p - lo)
            yh.append(hi - p)
        axB.bar(xs, ps, width=bw, color=col, label=lab, zorder=2)
        axB.errorbar(xs, ps, yerr=[yl, yh], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.2, zorder=3)
        for x, mid in zip(xs, [m[0] for m in metrics]):
            k, n = B[(arm, mid)][:2]
            t_ = axB.text(x, B[(arm, mid)][4] + 0.02, f"{k}/{n}", ha="center", va="bottom", fontsize=4.8, rotation=90)
            t_.set_gid("count")
    axB.set_xticks(range(len(metrics)))
    axB.set_xticklabels([m[1] for m in metrics], fontsize=5.4)
    axB.set_xlim(-0.5, len(metrics) - 0.5)
    axB.set_ylim(-0.0, YTOP)
    axB.set_yticks(YT)
    axB.set_ylabel("Fraction")
    legend_band(axB, 2, 6.0)

    # C
    hx = list(ws)
    hp = [C[(ARM_CON, w)][2] for w in ws]
    axC.errorbar(hx, hp, yerr=[[C[(ARM_CON, w)][2] - C[(ARM_CON, w)][3] for w in ws], [C[(ARM_CON, w)][4] - C[(ARM_CON, w)][2] for w in ws]],
                 color=C_PROPOSED, lw=0.6, elinewidth=0.5, capsize=1.0, marker="o", markersize=2.2, label=LAB_CON, zorder=2)
    vp = [C[(ARM_UNC, w)][2] for w in ws]
    axC.errorbar(hx, vp, yerr=[[C[(ARM_UNC, w)][2] - C[(ARM_UNC, w)][3] for w in ws], [C[(ARM_UNC, w)][4] - C[(ARM_UNC, w)][2] for w in ws]],
                 color=C_VAN, lw=0.9, elinewidth=0.7, capsize=1.5, marker="o", markersize=3.2, label=LAB_UNC, zorder=3)
    axC.errorbar([2.0], [mv["p"]], yerr=[[mv["p"] - mv["lo"]], [mv["hi"] - mv["p"]]], color="black", lw=0, elinewidth=0.6, capsize=2.2,
                 marker="o", markersize=5.0, markerfacecolor="none", markeredgewidth=0.8, label=f"{LAB_UNC},\nExpt. 1 (separate run)", zorder=4)
    if reseed:
        for i, ((p_, lo_, hi_), d_) in enumerate(zip(RS, RESEED_DELTAS)):
            axC.errorbar([2.0 + d_], [p_], yerr=[[p_ - lo_], [hi_ - p_]], color="black", lw=0, elinewidth=0.5, capsize=1.6,
                         marker="D", markersize=3.2, markerfacecolor="none", markeredgewidth=0.7,
                         label=RESEED_LABEL if i == 0 else None, zorder=4)
    for w in ws:
        k_eq = cells[f"{LAYER_A}/{ARM_UNC}/{w:.1f}"]["E12"][THETA_KEY] if reseed else C[(ARM_UNC, w)][5]
        t_ = axC.text(w, E_Y, f"Equivalent\n= {k_eq}", ha="center", va="bottom", fontsize=4.8, color=C_VAN)
        t_.set_gid("E")
    axC.set_xticks(list(ws))
    axC.set_xticklabels(["1", "2", "3", "4"])
    axC.set_xlim(0.4, 4.6)
    axC.set_ylim(C_YBOT, YTOP)
    axC.set_yticks(YT)
    axC.set_xlabel("Guidance weight $w$")
    axC.set_ylabel("Fraction with two GBs")
    legend_band(axC, 1, 6.0)

    for ax, lab in zip(axes, "ABC"):
        t_ = ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")
        t_.set_gid("letter")
    fig.tight_layout(w_pad=0.6)

    if reseed:
        # layout record (pixels at fig.dpi, after draw): consumed by check_fig141.py
        fig.canvas.draw()
        s = 72.0
        rec_markers = []
        names = ["Constrained", "Unconstrained", "Expt1_main"] + [f"reseed_{l}" for l in RESEED_LAYERS]
        mlines = [ln for ln in axC.lines if ln.get_marker() in ("o", "D") and ln.get_linestyle() in ("-", "None", "none", "")]
        mlines = [ln for ln in mlines if len(ln.get_xdata()) > 0]
        assert len(mlines) == len(names), (len(mlines), names)
        for nm, ln in zip(names, mlines):
            ms, mew, mk = ln.get_markersize(), ln.get_markeredgewidth(), ln.get_marker()
            half = (ms * (math.sqrt(2) if mk == "D" else 1.0) / 2 + mew / 2) * fig.dpi / s  # path half-extent + edge half-width
            for x_, y_ in zip(ln.get_xdata(), ln.get_ydata()):
                cx, cy = axC.transData.transform((x_, y_))
                rec_markers.append(dict(name=nm, marker=mk, x=float(x_), y=float(y_), cx=float(cx), cy=float(cy),
                                        markersize=ms, mew=mew, bbox=[float(cx - half), float(cy - half), float(cx + half), float(cy + half)]))
        leg = axC.get_legend().get_window_extent()
        ab = axC.get_window_extent()
        rec_E = []
        for t in axC.texts:
            if t.get_gid() == "E":
                bb = t.get_window_extent()
                rec_E.append(dict(text=t.get_text(), x=float(t.get_position()[0]), bbox=[float(bb.x0), float(bb.y0), float(bb.x1), float(bb.y1)]))
        rec = dict(dpi=float(fig.dpi), fig_px=[float(v) for v in fig.bbox.size], deltas=list(RESEED_DELTAS),
                   axC_bbox=[float(ab.x0), float(ab.y0), float(ab.x1), float(ab.y1)],
                   legend_bbox=[float(leg.x0), float(leg.y0), float(leg.x1), float(leg.y1)],
                   markers=rec_markers, equivalent=rec_E)
        outdir.mkdir(parents=True, exist_ok=True)
        out_layout.write_text(json.dumps(rec, indent=1))
        print("wrote", out_layout)

    outdir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, name, str(outdir), pdf=True)
    plt.close(fig)
    print("wrote", out_png, out_pdf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("base", "reseed"), default="base")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--self_check", action="store_true")
    sys.exit(main(ap.parse_args()) or 0)
