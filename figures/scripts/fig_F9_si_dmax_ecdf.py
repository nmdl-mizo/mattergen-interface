#!/usr/bin/env python3
"""fig_F9_si_dmax_ecdf.py -- SI figure F9 "empirical cumulative distribution of the equivalence measure d_max and the threshold" (1 panel, single 86 mm).

Adapted from the plan of the SI figures F6-F9.
Inputs (read only):
  experiments/relax_novelty/results/gb_equivalence/gb_symfit_main.csv   (stage 1, before and after relaxation. 1400 rows)
  experiments/verification/results/gb_stage2/gb_stage2_main.csv         (stages 1 and 2, after relaxation. 424 rows)
  the paper's numbers table (EQ93, EQ91, EQ92, P11, EQ99lo, EQ99hi, B60)
Output: figures/png/F9_si_dmax_ecdf.png (paperfig 1.1.0 save_fig, single width, 600 dpi)
      figure_sources/si_f6_f9/F9_si_dmax_ecdf.{svg,pdf} (for visual inspection), F9_stats.csv
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
from matplotlib.patches import Patch

import paperfig
from paperfig import COL_1, OKABE_ITO, save_fig, setup_style

assert paperfig.__version__ == "1.1.0", paperfig.__version__

DRAFT = PA + "/figure_sources/si_f6_f9"
PNG_DIR = PA + "/figures/png"
NAME = "F9_si_dmax_ecdf"
S1_IN = GC + "/experiments/relax_novelty/results/gb_equivalence/gb_symfit_main.csv"
S2_IN = GC + "/experiments/verification/results/gb_stage2/gb_stage2_main.csv"

with open(_gbrel.CODE + "/data/paper_numbers/ledger_subset.yaml", encoding="utf-8") as f:
    LEDGER = {str(e["id"]): e for e in yaml.safe_load(f)["numbers"]}


def lv(i):
    return LEDGER[i]["value"]


# ============================================================ inputs
a = pd.read_csv(S1_IN)
assert len(a) == 1400, len(a)
b = pd.read_csv(S2_IN)
assert len(b) == 424, len(b)


def s1_series(arm, stage):
    s = a[(a["arm"] == arm) & (a["stage"] == stage) & (a["status"] == "fitted")]
    assert s["gb_symfit_max_A"].notna().all()
    return s.sort_values(["chunk", "cif_index"])


S1 = {("proposed", "pre"): s1_series("proposed", "pre"), ("proposed", "post"): s1_series("proposed", "post"),
      ("baseline", "post"): s1_series("baseline", "post")}
assert len(S1[("proposed", "pre")]) == 350 and len(S1[("proposed", "post")]) == 350 and len(S1[("baseline", "post")]) == 74

# stage 2 (after relaxation): empty values excluded
hs2 = b[b["arm"] == "proposed"]
vs2 = b[b["arm"] == "baseline"]
assert len(hs2) == 350 and len(vs2) == 74
assert hs2["s2_max_A"].notna().all()
n_v_s2 = int(vs2["s2_max_A"].notna().sum())
excl = vs2[vs2["s2_max_A"].isna()]
print("baseline stage 2 excluded (s2_max_A empty): %d rows -> %s" % (len(excl), excl[["chunk", "cif_index", "status", "note"]].to_dict("records")))

# the stage-1 values must agree in the two CSV files (after relaxation; the same structures)
for arm in ("proposed", "baseline"):
    m = S1[(arm, "post")][["chunk", "cif_index", "gb_symfit_max_A"]].merge(
        b[b["arm"] == arm][["chunk", "cif_index", "s1_max_A"]], on=["chunk", "cif_index"], how="inner")
    assert len(m) == len(S1[(arm, "post")]), (arm, len(m))
    rel = np.abs(m["gb_symfit_max_A"] - m["s1_max_A"]) / m["s1_max_A"]
    print("stage-1 consistency (%s, post, symfit csv vs stage2 csv s1_max_A): n=%d max rel diff = %.3g" % (arm, len(m), rel.max()))
    assert rel.max() < 1e-6, (arm, rel.max())

series = [
    ("proposed_s1_pre", "Proposed, stage 1, before relaxation", S1[("proposed", "pre")]["gb_symfit_max_A"].to_numpy(float), "proposed", ":"),
    ("proposed_s1_post", "Proposed, stage 1, after relaxation", S1[("proposed", "post")]["gb_symfit_max_A"].to_numpy(float), "proposed", "-"),
    ("proposed_s2_post", "Proposed, stage 2, after relaxation", hs2["s2_max_A"].to_numpy(float), "proposed", "--"),
    ("baseline_s1_post", "Baseline, stage 1, after relaxation", S1[("baseline", "post")]["gb_symfit_max_A"].to_numpy(float), "baseline", "-"),
    ("baseline_s2_post", "Baseline, stage 2, after relaxation", vs2["s2_max_A"].dropna().to_numpy(float), "baseline", "--"),
]
assert [len(s[2]) for s in series] == [350, 350, 350, 74, n_v_s2] and n_v_s2 == 73
assert all(np.isfinite(s[2]).all() and (s[2] > 0).all() for s in series)

# ============================================================ comparison with the table
vmin12 = float(vs2[["s1_max_A", "s2_max_A"]].min(axis=1).min())   # min(s1,s2) per structure (for the 1 row where s2 is empty, only s1; skipna)
vs1_min = float(series[3][2].min())
vs2_min = float(series[4][2].min())
proposed_s1_max = float(series[1][2].max())
proposed_all_max = max(float(s[2].max()) for s in series[:3])
baseline_all_min = min(float(s[2].min()) for s in series[3:])
P11, LO_T, HI_T = lv("P11"), lv("EQ99lo"), lv("EQ99hi")


def sig(x, n):
    return float("%.*g" % (n, x))


checks = [
    ("EQ93 proposed stage1 post max", proposed_s1_max, sig(proposed_s1_max, 4), lv("EQ93")),
    ("EQ91 baseline min(s1,s2) min", vmin12, round(vmin12, 4), lv("EQ91")),
    ("EQ91 (= stage-2 min of baseline)", vs2_min, round(vs2_min, 4), lv("EQ91")),
    ("EQ92 baseline stage1 min", vs1_min, round(vs1_min, 4), lv("EQ92")),
    ("B60 baseline stage1 n", len(series[3][2]), len(series[3][2]), lv("B60")),
]
lines = ["check\trecomputed\trounded\tledger\tstatus"]
bad = []
for nm, raw, rnd, ref in checks:
    ok = abs(rnd - ref) < 1e-12
    lines.append("%s\t%.6g\t%.6g\t%.6g\t%s" % (nm, raw, rnd, ref, "OK" if ok else "MISMATCH"))
    if not ok:
        bad.append((nm, raw, ref))
# positional relation of the threshold and the distributions (evidence for the claim written in the figure): maximum over all series of the proposed model < P11 < minimum over all series of the baseline
sep = proposed_all_max < P11 < baseline_all_min
lines.append("separation proposed_max(%.6g) < P11(%.3g) < baseline_min(%.6g)\t\t\t\t%s" % (proposed_all_max, P11, baseline_all_min, "OK" if sep else "FAIL"))
if not sep:
    bad.append(("separation", proposed_all_max, baseline_all_min))
lines.append("P11 / EQ99lo / EQ99hi read from ledger\t%g / %g / %g" % (P11, LO_T, HI_T))
print("\n".join(lines))
if bad:
    print("MISMATCH:", bad)
    raise SystemExit("disagreement with the table values. Stopped without writing the CSV or the figure.")

# ============================================================ statistics CSV
os.makedirs(DRAFT, exist_ok=True)
st = pd.DataFrame([dict(series=s[0], label=s[1], n=len(s[2]), min=float(s[2].min()), median=float(np.median(s[2])), max=float(s[2].max()))
                   for s in series])
st.to_csv(DRAFT + "/F9_stats.csv", index=False, float_format="%.6g")
with open(DRAFT + "/F9_stats.csv", "a", encoding="utf-8") as f:
    f.write("# reference values for the assertions (table values)\n")
    for ln in lines:
        f.write("# " + ln.replace("\t", " | ") + "\n")
print(st.to_string())
print("wrote F9_stats.csv")

# ============================================================ drawing
setup_style()
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
C = {"proposed": OKABE_ITO[5], "baseline": OKABE_ITO[1]}
# The plan has the horizontal axis 1e-6 to 10, but the data include the minimum 4.2e-7 of the proposed model (stage 2) and the maximum 15.4 of the baseline.
# The range is widened so that the empirical cumulative distribution is not cut (author's choice; stated in the report).
allmin = min(float(s[2].min()) for s in series)
allmax = max(float(s[2].max()) for s in series)
XL = (10.0 ** np.floor(np.log10(allmin)), 30.0)
assert allmax < XL[1] and allmin > XL[0]
print("xlim:", XL, " data min/max:", allmin, allmax)

fig, ax = plt.subplots(figsize=(COL_1, COL_1 * 1.22))
ax.axvspan(LO_T, HI_T, color="#e6e6e6", lw=0, zorder=0)
ax.axvline(P11, color="black", lw=0.9, zorder=2)
for key, lab, v, arm, ls in series:
    x = np.sort(v)
    n = len(x)
    xx = np.concatenate([[XL[0]], x, [XL[1]]])
    yy = np.concatenate([[0.0], np.arange(1, n + 1) / n, [1.0]])
    ax.step(xx, yy, where="post", color=C[arm], ls=ls, lw=1.1 if ls != ":" else 2.2, zorder=3 if ls != ":" else 2.5)
ax.set_xscale("log")
ax.set_xlim(*XL)
ax.set_ylim(0, 1.02)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("Maximum deviation $d_{max}$ [Å]")
ax.set_ylabel("Empirical cumulative distribution")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles, labels = [], []
for key, lab, v, arm, ls in series:
    handles.append(Line2D([], [], color=C[arm], ls=ls, lw=1.1))
    labels.append("%s (n = %d)" % (lab, len(v)))
handles.append(Line2D([], [], color="black", lw=0.9))
labels.append("Threshold %s" % LEDGER["P11"]["text"])
handles.append(Patch(facecolor="#e6e6e6", edgecolor="none"))
labels.append("Sensitivity range %s – %s" % (LEDGER["EQ99lo"]["text"].replace(" Å", ""), LEDGER["EQ99hi"]["text"]))
fig.legend(handles=handles, labels=labels, loc="outside lower center", ncol=1, frameon=False, handlelength=2.2, labelspacing=0.3, fontsize=6.3)

_, png = save_fig(fig, NAME, PNG_DIR)
fig.savefig(os.path.join(DRAFT, NAME + ".svg"), format="svg", metadata={"Date": None})
fig.savefig(os.path.join(DRAFT, NAME + ".pdf"), metadata={"Creator": "paperfig %s / fig_F9_si_dmax_ecdf.py" % paperfig.__version__})
plt.close(fig)
print("wrote", png)
print("wrote", os.path.join(DRAFT, NAME + ".svg"), "and .pdf")
print("DONE")
