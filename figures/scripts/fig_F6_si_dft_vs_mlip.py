#!/usr/bin/env python3
"""fig_F6_si_dft_vs_mlip.py -- SI figure F6 "gamma from DFT vs MACE (26 structures)" (1 panel, single 86 mm).

Adapted from the plan of the SI figures F6-F9.
Inputs (read only):
  experiments/verification/results/dft_recalc/gamma_vasp.csv                (26 rows)
  experiments/verification/results/summary_tables/m7_dft_stats.csv      (master of the statistics; row 4 is all,pooled_single_offset(26))
  the paper's numbers table (DFTNS, DFTNSH, DFTNSV, DFTNP, DFTPH, DFTPV, DFTK, DFTOFF, DFTRHO, DFTR)
Output: figures/png/F6_si_dft_vs_mlip.png (paperfig 1.1.0 save_fig, single width, 600 dpi)
      figure_sources/si_f6_f9/F6_si_dft_vs_mlip.{svg,pdf} (for visual inspection)
      figure_sources/si_f6_f9/F6_points_v2.csv, F6_stats_v2.txt
Stops without writing the CSV or the figure if the table values and the recomputed values do not agree to the rounding digits. The table is not rewritten.
Run: with the evaluation python on a login node:  ulimit -v 8000000; export OMP_NUM_THREADS=1; timeout 900 ${MACE_PY} <this file>
"""
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)
import os
import sys

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
from scipy.stats import spearmanr

import paperfig
from paperfig import COL_1, OKABE_ITO, save_fig, setup_style

assert paperfig.__version__ == "1.1.0", paperfig.__version__

DRAFT = PA + "/figure_sources/si_f6_f9"
PNG_DIR = PA + "/figures/png"
NAME = "F6_si_dft_vs_mlip"
CSV_IN = GC + "/experiments/verification/results/dft_recalc/gamma_vasp.csv"
M7_IN = GC + "/experiments/verification/results/summary_tables/m7_dft_stats.csv"

# ============================================================ numbers table
with open(_gbrel.CODE + "/data/paper_numbers/ledger_subset.yaml", encoding="utf-8") as f:
    LEDGER = {str(e["id"]): e for e in yaml.safe_load(f)["numbers"]}


def lv(i):
    return LEDGER[i]["value"]


# ============================================================ inputs and recomputation
d = pd.read_csv(CSV_IN)
assert len(d) == 26, len(d)
assert d["converged"].astype(str).eq("True").all() and d["ediff_reached"].astype(str).eq("True").all()
assert d["gamma_dft"].notna().all() and d["gamma_mace"].notna().all()
assert set(d["arm"]) == {"proposed", "baseline"} and set(d["role"]) == {"lo", "hi"}
assert not d.duplicated(["pair_id", "role"]).any()
d["diff"] = d["gamma_dft"] - d["gamma_mace"]

n_all = len(d)
n_h = int((d["arm"] == "proposed").sum())
n_v = int((d["arm"] == "baseline").sum())
pairs = d["pair_id"].unique()
pair_arm = d.groupby("pair_id")["arm"].first()
n_pairs = len(pairs)
n_ph = int((pair_arm == "proposed").sum())
n_pv = int((pair_arm == "baseline").sum())

piv_d = d.pivot(index="pair_id", columns="role", values="gamma_dft")
piv_m = d.pivot(index="pair_id", columns="role", values="gamma_mace")
assert piv_d.notna().all().all() and piv_m.notna().all().all() and len(piv_d) == n_pairs
sd = np.sign(piv_d["hi"] - piv_d["lo"])
sm = np.sign(piv_m["hi"] - piv_m["lo"])
assert (sd != 0).all() and (sm != 0).all()
k_agree = int((sd == sm).sum())

offset = float(np.median(d["diff"]))
rho = float(spearmanr(d["gamma_mace"], d["gamma_dft"]).statistic)
pear = float(np.corrcoef(d["gamma_mace"], d["gamma_dft"])[0, 1])

# row 4 of m7_dft_stats.csv (master)
m7 = pd.read_csv(M7_IN)
row = m7[(m7["arm"] == "all") & (m7["offset_scope"] == "pooled_single_offset(26)")]
assert len(row) == 1
row = row.iloc[0]
assert int(row["n"]) == 26

# comparison with the table: (id, recomputed value, rounding digits, table value)
checks = [
    ("DFTNS", n_all, None),
    ("DFTNSH", n_h, None),
    ("DFTNSV", n_v, None),
    ("DFTNP", n_pairs, None),
    ("DFTPH", n_ph, None),
    ("DFTPV", n_pv, None),
    ("DFTK", k_agree, None),
    ("DFTOFF", offset, 3),
    ("DFTRHO", rho, 3),
    ("DFTR", pear, 3),
]
lines = ["id\tledger_value\trecomputed\trounding\tstatus"]
bad = []
for i, val, nd in checks:
    ref = lv(i)
    ok = (val == ref) if nd is None else (round(val, nd) == ref)
    lines.append("%s\t%s\t%s\t%s\t%s" % (i, ref, ("%d" % val) if nd is None else "%.6f" % val, "integer" if nd is None else "round(.,%d)" % nd,
                                         "OK" if ok else "MISMATCH"))
    if not ok:
        bad.append((i, val, ref))
# comparison with the m7 master (does the recomputation from the CSV agree with the m7 numbers? tolerance 5e-6 = 6-digit rounding of m7)
m7cmp = [("m7.offset_median_dft_minus_mace", offset, float(row["offset_median_dft_minus_mace"])),
         ("m7.spearman_rho", rho, float(row["spearman_rho"])),
         ("m7.pearson_r", pear, float(row["pearson_r"]))]
for nm, a, b in m7cmp:
    ok = abs(a - b) < 5e-6
    lines.append("%s\t%.6f\t%.6f\tabs diff<5e-6\t%s" % (nm, b, a, "OK" if ok else "MISMATCH"))
    if not ok:
        bad.append((nm, a, b))
print("\n".join(lines))
if bad:
    print("MISMATCH:", bad)
    raise SystemExit("disagreement with the table values. Stopped without writing the CSV or the figure.")

os.makedirs(DRAFT, exist_ok=True)
with open(DRAFT + "/F6_stats_v2.txt", "w", encoding="utf-8") as f:
    f.write("# comparison with the table values of F6 (generated by fig_F6_si_dft_vs_mlip.py)\n")
    f.write("# input: %s (26 rows)\n# master: %s L4 (all,pooled_single_offset(26))\n" % (CSV_IN, M7_IN))
    f.write("# offset = median(gamma_dft - gamma_mace) over 26; rho/r = 26 points; k = number of pairs whose sign(hi-lo) agrees between DFT and MACE\n")
    f.write("\n".join(lines) + "\n")
d[["arm", "pair_id", "role", "gamma_mace", "gamma_dft", "diff"]].to_csv(DRAFT + "/F6_points_v2.csv", index=False, float_format="%.6f")
print("wrote F6_stats_v2.txt, F6_points_v2.csv")

# ============================================================ drawing
setup_style()
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
C_P = OKABE_ITO[5]          # blue of the proposed model (same as C_P of fig_F3_tracking.py)
C_B = OKABE_ITO[1]          # orange of the unconstrained baseline (F3 has no baseline color, so the orange of OKABE_ITO)
C_GRAY = "0.7"

allv = np.concatenate([d["gamma_mace"].to_numpy(), d["gamma_dft"].to_numpy()])
LO = np.floor((allv.min() - 0.05) * 10) / 10
HI = np.ceil((allv.max() + 0.05) * 10) / 10
print("axis range: %.2f - %.2f  (data %.4f - %.4f)" % (LO, HI, allv.min(), allv.max()))

fig, ax = plt.subplots(figsize=(COL_1, COL_1))
ax.set_aspect("equal", adjustable="box")
# connect lo-hi of the same pair with a thin gray line (13 lines)
for pid in pairs:
    p = d[d["pair_id"] == pid].set_index("role")
    ax.plot([p.loc["lo", "gamma_mace"], p.loc["hi", "gamma_mace"]], [p.loc["lo", "gamma_dft"], p.loc["hi", "gamma_dft"]],
            color=C_GRAY, lw=0.5, zorder=2)
xs = np.array([LO, HI])
ax.plot(xs, xs, color="0.25", lw=0.7, ls=(0, (4, 2)), zorder=1)
ax.plot(xs, xs + offset, color="0.25", lw=0.7, ls=(0, (1, 1.5)), zorder=1)
h = d[d["arm"] == "proposed"]
v = d[d["arm"] == "baseline"]
ax.scatter(h["gamma_mace"], h["gamma_dft"], s=14, facecolors=C_P, edgecolors="none", zorder=4)
ax.scatter(v["gamma_mace"], v["gamma_dft"], s=14, facecolors="none", edgecolors=C_B, linewidths=0.9, zorder=4)
ax.set_xlim(LO, HI)
ax.set_ylim(LO, HI)
ticks = np.arange(np.ceil(LO * 2) / 2, HI + 1e-9, 0.5)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlabel("γ (MLIP) [J/m²]")
ax.set_ylabel("γ (DFT) [J/m²]")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles = [
    Line2D([], [], marker="o", ls="none", mfc=C_P, mec="none", ms=4.2),
    Line2D([], [], marker="o", ls="none", mfc="none", mec=C_B, mew=0.9, ms=4.2),
    Line2D([], [], color="0.25", lw=0.7, ls=(0, (4, 2))),
    Line2D([], [], color="0.25", lw=0.7, ls=(0, (1, 1.5))),
]
labels = ["Proposed model (n = %d)" % n_h, "unconstrained baseline (n = %d)" % n_v, "y = x", "y = x + median offset"]
ax.legend(handles=handles, labels=labels, loc="upper left", frameon=False, handlelength=1.8, labelspacing=0.35, fontsize=6.5, borderaxespad=0.2)
ax.text(0.97, 0.04, "n = %d\nmedian(DFT − MLIP) = %s" % (n_all, LEDGER["DFTOFF"]["text"]), transform=ax.transAxes,
        ha="right", va="bottom", fontsize=6.0, color="0.15", linespacing=1.3)

_, png = save_fig(fig, NAME, PNG_DIR)
fig.savefig(os.path.join(DRAFT, NAME + ".svg"), format="svg", metadata={"Date": None})
fig.savefig(os.path.join(DRAFT, NAME + ".pdf"), metadata={"Creator": "paperfig %s / fig_F6_si_dft_vs_mlip.py" % paperfig.__version__})
plt.close(fig)
print("wrote", png)
print("wrote", os.path.join(DRAFT, NAME + ".svg"), "and .pdf")
print("DONE")
