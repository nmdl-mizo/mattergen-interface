#!/usr/bin/env python3
"""fig_F7_si_representative_structures.py -- SI figure F7 "representative structures near the target and not matching the training data (generated structure and its nearest training structure)" (double 178 mm, 4 rows x 2 columns).

Adapted from the plan of the SI figures F6-F9.
Selection (fixed by a rule before looking at the structures):
  (a)(b)(c) = among the 90 near-target structures of the proposed model, main experiment w=2.0, 5 inner targets (Fig4_distance_rows.csv), the structures whose d_RMS,min is closest to the table values GRP10, GRMG and GRP90 (for equal differences the smaller (chunk, cif_index))
  (d)       = rank 1 of low_gamma_novel_view/nn/nn_L.csv (lowest gamma; asserted against the table value LGHMIN)
  right column = nearest training structure of each structure by d_RMS (for (a)-(c): rms_nn_row_index of nn_G, for (d): nn_row_index of nn_L)
Drawing conventions taken from (adopted Fig. 2): figure_sources/fig2/render/scripts/make_fig2.py
  - L55-L61  correspondence of the axes (n = normal x, t = tilt axis [001] = normal to the paper, p = vertical z). Atom coordinates are wrapped; x in the n direction = frac_n * Ln, y in the p direction = frac_p * Lp
  - L61      coloring of the 2 layers along [001] (layers of frac_t. F2 uses the threshold frac_t % 1 >= 0.5. This script determines the phase of the 2 layers by a circular mean before assigning them, to avoid the misclassification of atoms near 0 that fall to 0.99999 by rounding errors)
  - L63-L81  periodic translation: x is shifted by dxf*Ln so that the midpoint of the grain-boundary interval GB1(z_AB)->GB2(z_BA) comes to the cell center a/2 (the two grain boundaries are symmetric about the cell center)
  - L124-L129 drawing of atoms (radius 1.0 A, layer 0 filled black, layer 1 open, edge 0.4 pt)
  - L152-L161 cell frame (black 0.8 pt) and the orange dashed lines at the grain-boundary positions (OKABE_ITO[6], 1.0 pt, dashes (3,2), labels "GB1"/"GB2")
Grain-boundary positions: for generated structures z_AB and z_BA of gb_stage2_main.csv (same source as F2). For training structures they are computed with the same instrument (count_gb of gb_symfit and gb_positions of gb_stage2_local,
  reference orientation W8 of gb_equivalence/gb_symfit_validation.json). It is asserted that the recomputed values of this instrument agree with z of gb_stage2_main.csv for generated structures before it is applied to training structures.
Inputs (read only): the CSV, extxyz and POSCAR files named in the plan, the paper's numbers table. Output: figures/png/F7_si_representative_structures.png (paperfig 1.1.0, 600 dpi),
  figure_sources/si_f6_f9/{F7_selection.csv, F7_si_representative_structures.svg, .pdf}
Run: with the evaluation python (environment-eval.yml): ulimit -v 8000000; export OMP_NUM_THREADS=1; timeout 900 python <this file>
"""
import csv
import glob
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '<ORIGINAL_ROOT>' if os.path.isdir('<ORIGINAL_ROOT>/workspace') else '<ORIGINAL_ROOT>'
GC = ROOT + "/workspace"
PA = GC + "/paper"
VRUN = GC + "/experiments/verification"
RRUN = GC + "/experiments/relax_novelty"
sys.path.insert(0, PA + "/figures/scripts")
sys.path.insert(0, VRUN + "/scripts")      # gb_symfit and gb_stage2_local (imported read-only; not copied)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ase.io import read
from matplotlib.patches import Circle, Rectangle

import paperfig
from paperfig import COL_2, OKABE_ITO, save_fig, setup_style

assert paperfig.__version__ == "1.1.0", paperfig.__version__
import gb_symfit as S          # noqa: E402
import gb_stage2_local as T    # noqa: E402

DRAFT = PA + "/figure_sources/si_f6_f9"
PNG_DIR = PA + "/figures/png"
NAME = "F7_si_representative_structures"
CSV_OUT = DRAFT + "/F7_selection.csv"


def P(p):
    """Convert so that it works whether the path is written with the compute-node or the login-node mount prefix, to the one that exists on this node."""
    p = str(p)
    for pre in ('/mnt/CLUSTER_ROOT_A', '/CLUSTER_ROOT_B'):
        if p.startswith(pre + "/"):
            return ROOT[:-len("/CLUSTER_ROOT_K")] + p[len(pre):]
    return p


# ============================================================ numbers table
NUM = {e["id"]: e["value"] for e in yaml.safe_load(open(PA + "/ledger_subset.yaml", encoding="utf-8"))["numbers"]}
GRP10, GRMG, GRP90, LGHMIN = float(NUM["GRP10"]), float(NUM["GRMG"]), float(NUM["GRP90"]), float(NUM["LGHMIN"])
assert (GRP10, GRMG, GRP90, LGHMIN) == (0.080227, 0.1651, 0.444568, 0.535), (GRP10, GRMG, GRP90, LGHMIN)

# ============================================================ selection
FD = pd.read_csv(PA + "/figure_sources/fig4/distance_audit/data/Fig4_distance_rows.csv")
assert len(FD) == 90 and not FD.duplicated(["chunk", "cif_index"]).any()
dr = FD["d_rms_min_A"].to_numpy(float)
assert abs(np.quantile(dr, 0.10) - GRP10) < 5e-7, np.quantile(dr, 0.10)
assert round(float(np.median(dr)), 4) == round(GRMG, 4), np.median(dr)
assert abs(np.quantile(dr, 0.90) - GRP90) < 5e-7, np.quantile(dr, 0.90)
print("assert OK: d_rms_min_A (n=90) p10=%.6f median=%.6f p90=%.6f  vs ledger GRP10=%.6f GRMG=%.4f GRP90=%.6f"
      % (np.quantile(dr, .1), np.median(dr), np.quantile(dr, .9), GRP10, GRMG, GRP90))
FD = FD.sort_values(["chunk", "cif_index"]).reset_index(drop=True)      # for equal differences the smaller (chunk, cif_index) = the first of the argmin


def pick(val):
    i = int(np.argmin(np.abs(FD["d_rms_min_A"].to_numpy(float) - val)))
    return FD.iloc[i]


SEL = [("a", "d_RMS,min ~ p10 (GRP10)", pick(GRP10)), ("b", "d_RMS,min ~ median (GRMG)", pick(GRMG)), ("c", "d_RMS,min ~ p90 (GRP90)", pick(GRP90))]
assert len({(int(r.chunk), int(r.cif_index)) for _l, _d, r in SEL}) == 3, "the 3 structures overlap"

# nn_G (row_index of the nearest neighbor by d_RMS; rows are looked up by (chunk, cif_index))
NNG = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(VRUN + "/results/grid_interp/nn_G_[0-9][0-9].csv"))], ignore_index=True)
assert len(NNG) == 90 and not NNG.duplicated(["chunk", "cif_index"]).any()
GL = pd.read_csv(VRUN + "/results/grid_interp/G_list.csv")
MAN = pd.read_csv(RRUN + "/results/train_ref_manifest.csv").set_index("row_index")
NNL = pd.read_csv(VRUN + "/results/low_gamma_novel_view/nn/nn_L.csv")
TBL = pd.read_csv(VRUN + "/results/low_gamma_novel_view/table.csv", comment="#", skip_blank_lines=True)   # there are train reference rows starting with "#" at the end
TBL = TBL.dropna(subset=["arm"])
TBL["rank"] = TBL["rank"].astype(int)
ST2 = pd.read_csv(VRUN + "/results/gb_stage2/gb_stage2_main.csv")
ST2 = ST2[(ST2["layer"] == "main") & (ST2["arm"] == "proposed") & (ST2["w"] == 2.0)].set_index(["chunk", "cif_index"])
assert len(ST2) == 350

items = []     # dictionary for drawing and CSV (4 elements)
for label, desc, r in SEL:
    ch, ci = int(r.chunk), int(r.cif_index)
    g = NNG[(NNG["chunk"] == ch) & (NNG["cif_index"] == ci)]
    assert len(g) == 1
    g = g.iloc[0]
    assert abs(float(g["d_rms_min_A"]) - float(r.d_rms_min_A)) < 1e-9, "d_rms_min_A of nn_G and of Fig4_distance_rows disagree"
    gl = GL[(GL["chunk"] == ch) & (GL["cif_index"] == ci)]
    assert len(gl) == 1
    gl = gl.iloc[0]
    assert abs(float(gl["gamma_relaxed"]) - float(r.gamma_relaxed)) < 1e-6 and int(gl["natoms"]) == int(r.natoms)
    nn_row = int(g["rms_nn_row_index"])
    nn_gamma = float(g["rms_nn_gamma"])
    assert abs(float(MAN.loc[nn_row, "gamma_Jm2"]) - nn_gamma) < 1e-9, "gamma of the manifest and rms_nn_gamma of nn_G disagree"
    assert str(MAN.loc[nn_row, "cnid"]) == str(g["rms_nn_cnid"]) and abs(float(MAN.loc[nn_row, "dx_A"]) - float(g["rms_nn_dx_A"])) < 1e-9
    items.append(dict(label=label, desc=desc, chunk=ch, cif_index=ci, target_id=int(r.target_id), gamma_relaxed=float(r.gamma_relaxed),
                      d_rms_min_A=float(r.d_rms_min_A), natoms=int(r.natoms), nn_row_index=nn_row, nn_gamma=nn_gamma,
                      path=P(gl["relaxed_extxyz"]), frame_index=int(gl["frame_index"])))
# (d) lowest gamma
l1 = NNL[NNL["rank"] == 1].iloc[0]
tb = TBL[TBL["rank"] == 1]
tb = tb[tb["arm"] == "proposed"].iloc[0]
assert (int(l1["chunk"]), int(l1["cif_index"])) == (int(tb["chunk"]), int(tb["cif_index"])) and int(l1["natoms"]) == int(tb["natoms"])
assert abs(float(l1["gamma_relaxed"]) - float(tb["gamma_relaxed"])) < 1e-9
assert round(float(l1["gamma_relaxed"]), 4) == round(LGHMIN, 4), (float(l1["gamma_relaxed"]), LGHMIN)       # table value LGHMIN 0.5350
nn_row_d = int(l1["nn_row_index"])
assert abs(float(MAN.loc[nn_row_d, "gamma_Jm2"]) - float(l1["nn_gamma_Jm2"])) < 1e-9 and str(MAN.loc[nn_row_d, "cnid"]) == str(l1["nn_cnid"])
items.append(dict(label="d", desc="lowest gamma (nn_L rank 1)", chunk=int(l1["chunk"]), cif_index=int(l1["cif_index"]), target_id=int(tb["target_id"]),
                  gamma_relaxed=float(l1["gamma_relaxed"]), d_rms_min_A=float(l1["d_rms_min_A"]), natoms=int(l1["natoms"]), nn_row_index=nn_row_d,
                  nn_gamma=float(l1["nn_gamma_Jm2"]), path=P(tb["relaxed_path"]), frame_index=int(tb["frame_index"])))
assert len({(it["chunk"], it["cif_index"]) for it in items}) == 4

# ============================================================ loading the structures and grain-boundary positions
REFS = S.load_validation_refs(Path(VRUN + "/results/gb_equivalence"))
print("reference orientations (W8): %.4f, %.4f deg" % tuple(REFS))
Z_TOL = 1e-6


def two_layer_labels(ft):
    """Assignment of the 2 layers along [001]. The phase phi is found by a circular mean with period 1/2; (ft - phi) near 0 is layer 0, near 1/2 is layer 1."""
    phi = (np.angle(np.exp(1j * 4.0 * np.pi * ft).sum()) / (4.0 * np.pi)) % 0.5
    u = (ft - phi) % 1.0
    return (((u + 0.25) % 1.0) >= 0.5).astype(int), float(phi)


def prepare(at, gb_z=None, tag=""):
    """at: ase Atoms. Returns: dict(x, y, layer, Ln, Lp, zg, ...). If gb_z=(z_AB, z_BA) is None it is computed with the instrument."""
    at = at.copy()
    at.wrap()
    cell = np.array(at.cell[:], dtype=float)
    axes = S.identify_axes(np.linalg.norm(cell, axis=1))
    assert S.axes_str(axes) == "n=a,t=b,p=c", (tag, S.axes_str(axes))
    n, t, p = axes
    frac = at.get_scaled_positions(wrap=True)
    c0 = S.count_gb(frac, cell, axes, REFS)
    assert c0["gb0_status"] == "two_gb", (tag, c0["gb0_status"], c0["gb_bin_seq"])
    zAB_c, zBA_c, _ = T.gb_positions(c0["gb_bin_seq"])
    if gb_z is None:
        zAB, zBA = zAB_c, zBA_c
    else:
        zAB, zBA = gb_z
        assert abs(zAB - zAB_c) < Z_TOL and abs(zBA - zBA_c) < Z_TOL, (tag, "z of gb_stage2_main.csv and the recomputed value of the instrument disagree", zAB, zAB_c, zBA, zBA_c)
    Ln, Lt, Lp = at.cell[n, n], at.cell[t, t], at.cell[p, p]
    assert abs(Lt - 4.04) < 0.15
    layer, phi = two_layer_labels(frac[:, t])
    n0, n1 = int((layer == 0).sum()), int((layer == 1).sum())
    assert n0 + n1 == len(at), (tag, n0, n1)
    u = (frac[:, t] - phi) % 1.0
    dev = float(np.minimum(np.minimum(u, 1.0 - u), np.abs(u - 0.5)).max())       # maximum deviation from the center of the layer (fraction; 0.25 is the upper limit of ambiguity)
    assert dev < 0.2, (tag, "there is an atom that does not belong to either of the 2 layers", dev)
    x = frac[:, n] * Ln
    y = frac[:, p] * Lp
    L = (zBA - zAB) % 1.0
    mid = (zAB + L / 2) % 1.0
    dxf = (0.5 - mid) % 1.0
    xs = (x + dxf * Ln) % Ln
    g1, g2 = (zAB + dxf) % 1.0, (zBA + dxf) % 1.0
    assert abs(((g1 + L / 2) % 1.0) - 0.5) < 1e-9 and abs(((g2 - g1) % 1.0) - L) < 1e-9
    assert xs.min() >= 0 and xs.max() < Ln
    return dict(dev=dev, x=xs, y=y, layer=layer, Ln=Ln, Lp=Lp, zg=[g1, g2], z_raw=(zAB, zBA), n0=n0, n1=n1, phi=phi, L=L, dxf=dxf, natoms=len(at),
                edge=min(min(g * Ln, (1 - g) * Ln) for g in (g1, g2)))


for it in items:
    at = read(it["path"], index=it["frame_index"])
    info = at.info
    assert int(info["chunk"]) == it["chunk"] and int(info["cif_index"]) == it["cif_index"] and len(at) == it["natoms"], (it["label"], info, len(at))
    assert abs(float(info["gamma_relaxed"]) - it["gamma_relaxed"]) < 1e-9, (it["label"], info["gamma_relaxed"], it["gamma_relaxed"])
    st = ST2.loc[(it["chunk"], it["cif_index"])]
    assert str(st["axes"]) == "n=a,t=b,p=c" and int(st["natoms"]) == it["natoms"] and np.isfinite(st["z_AB"]) and np.isfinite(st["z_BA"])
    it["gen"] = prepare(at, gb_z=(float(st["z_AB"]), float(st["z_BA"])), tag="gen_" + it["label"])
    mrow = MAN.loc[it["nn_row_index"]]
    it["nn_poscar_path"] = str(mrow["poscar_relaxed_path"])
    it["nn_cnid"] = str(mrow["cnid"])
    it["nn_dx"] = float(mrow["dx_A"])
    tat = read(P(it["nn_poscar_path"]), format="vasp")
    assert len(tat) == it["natoms"] == int(mrow["natoms"]), (it["label"], len(tat), it["natoms"])     # compare only structures with the same number of atoms
    it["tr"] = prepare(tat, tag="train_%d" % it["nn_row_index"])
    g_, t_ = it["gen"], it["tr"]
    print("(%s) gen chunk%02d idx%02d target %d gamma %.4f d_rms_min %.4f natoms %d | layers %d/%d (max dev %.3f) | GB z(raw) %.4f %.4f -> shifted %.4f %.4f | edge %.2f A"
          % (it["label"], it["chunk"], it["cif_index"], it["target_id"], it["gamma_relaxed"], it["d_rms_min_A"], it["natoms"], g_["n0"], g_["n1"], g_["dev"],
             g_["z_raw"][0], g_["z_raw"][1], g_["zg"][0], g_["zg"][1], g_["edge"]))
    print("    nn row %d %s dx %.2f gamma %.4f | %s | layers %d/%d (max dev %.3f) | GB z(raw) %.4f %.4f -> shifted %.4f %.4f | edge %.2f A"
          % (it["nn_row_index"], it["nn_cnid"], it["nn_dx"], it["nn_gamma"], it["nn_poscar_path"], t_["n0"], t_["n1"], t_["dev"], t_["z_raw"][0], t_["z_raw"][1],
             t_["zg"][0], t_["zg"][1], t_["edge"]))

COLS = ["label", "chunk", "cif_index", "target_id", "gamma_relaxed", "d_rms_min_A", "nn_row_index", "nn_poscar_path", "nn_gamma", "natoms"]
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader()
    for it in items:
        wr.writerow({k: (("%.9g" % it[k]) if isinstance(it[k], float) else it[k]) for k in COLS})
print("wrote", CSV_OUT)

# ============================================================ drawing (mm coordinates, y downwards; 1 unit = 1 mm)
setup_style()
matplotlib.rcParams.update({"figure.constrained_layout.use": False, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
SC = 1.45                      # mm / Å
RAD, LWA = 1.0, 0.4            # atom radius [A] and edge line [pt] (same as F2)
W_MM = 178.0
assert abs(COL_2 * 25.4 - W_MM) < 1e-9
X0L, X0R = 6.5, 91.5           # left edge of the cell of each column [mm]
HEAD = 12.0                    # from the start of the row to the top of the cell (2 heading lines + GB labels)
OV = RAD * SC                  # protrusion of atoms [mm]
ROW_PITCH = HEAD + 9.026 * SC + OV + 3.0
TOP = 1.0
H_MM = TOP + 4 * ROW_PITCH
GRAY_TXT = "#555555"
C_GB = OKABE_ITO[6]

fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_MM)
ax.set_ylim(H_MM, 0)
ax.axis("off")


def draw_cell(s, x0, y_ct):
    Ln, Lp = s["Ln"], s["Lp"]
    wc, hc = Ln * SC, Lp * SC
    y_cb = y_ct + hc
    ax.add_patch(Rectangle((x0, y_ct), wc, hc, fill=False, edgecolor="#000000", linewidth=0.8, zorder=1))
    for ly, fc in ((0, OKABE_ITO[0]), (1, "#FFFFFF")):
        m = s["layer"] == ly
        for xa, ya in zip(s["x"][m], s["y"][m]):
            ax.add_patch(Circle((x0 + xa * SC, y_cb - ya * SC), RAD * SC, facecolor=fc, edgecolor="#000000", linewidth=LWA, zorder=3))
    for i, g in enumerate(s["zg"]):
        xx = x0 + g * wc
        ax.plot([xx, xx], [y_ct - OV - 0.8, y_cb + 1.2], color=C_GB, linewidth=1.0, linestyle=(0, (3, 2)), zorder=4)
        ax.text(xx, y_ct - OV - 1.0, "GB%d" % (i + 1), fontsize=6, ha="center", va="bottom", color=C_GB)


for ri, it in enumerate(items):
    y0 = TOP + ri * ROW_PITCH
    y_ct = y0 + HEAD
    for x0, key in ((X0L, "gen"), (X0R, "tr")):
        draw_cell(it[key], x0, y_ct)
    ax.text(0.6, y_ct + 4.5, "(%s)" % it["label"], fontsize=8, fontweight="bold", ha="left", va="center")
    d_txt = {"a": "d$_{\\mathrm{RMS,min}}$ ≈ 10th percentile", "b": "d$_{\\mathrm{RMS,min}}$ ≈ median",
             "c": "d$_{\\mathrm{RMS,min}}$ ≈ 90th percentile", "d": "lowest γ"}[it["label"]]
    ax.text(X0L, y0 + 0.0, "Generated (proposed model): %s (chunk %d, index %d)" % (d_txt, it["chunk"], it["cif_index"]),
            fontsize=5.8, ha="left", va="top", color=GRAY_TXT)
    ax.text(X0L, y0 + 3.0, "γ = %.3f J/m²     d$_{\\mathrm{RMS,min}}$ = %.3f Å" % (it["gamma_relaxed"], it["d_rms_min_A"]),
            fontsize=6.5, ha="left", va="top")
    ax.text(X0R, y0 + 0.0, "Nearest training structure (row %d)" % it["nn_row_index"], fontsize=5.8, ha="left", va="top", color=GRAY_TXT)
    ax.text(X0R, y0 + 3.0, "γ = %.3f J/m²     %s, dx = %s Å" % (it["nn_gamma"], it["nn_cnid"], ("%+.2f" % it["nn_dx"]).replace("-", "\u2212")),
            fontsize=6.5, ha="left", va="top")

assert max(X0R + it["tr"]["Ln"] * SC + OV for it in items) < W_MM and max(X0L + it["gen"]["Ln"] * SC + OV for it in items) < X0R
_, png = save_fig(fig, NAME, PNG_DIR)
fig.savefig(os.path.join(DRAFT, NAME + ".svg"), format="svg", metadata={"Date": None})
fig.savefig(os.path.join(DRAFT, NAME + ".pdf"), metadata={"Creator": "paperfig %s / fig_F7_si_representative_structures.py" % paperfig.__version__})
plt.close(fig)
print("wrote", png)
print("wrote", os.path.join(DRAFT, NAME + ".svg"), "and .pdf")
print("DONE")
