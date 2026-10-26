#!/usr/bin/env python3
"""Manuscript version of the relaxation/novelty grid figure.

Copy of make_figure (aggregate_relax_novelty.py) fed by replot_relax_novelty_grid.load_sets, with only the
drawn labels replaced by reviewer-readable English. Data reading, computation, coordinates and colors are unchanged.
Self-contained: does not import the aggregation module. Refuses to overwrite.

Outputs (results/figs_manuscript/): relax_novelty_grid.png / relax_novelty_grid.pdf
Run: mace-mh env (matplotlib 3.10); cwd does not matter.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN.parent / "gen_eval" / "scripts"))  # paperfig.py (same as the original)
from paperfig import COL_2, OKABE_ITO, save_fig, setup_style  # noqa: E402

# ==== BEGIN DATA KEYS ====
# Keys of the input JSON. They are read, never drawn. Everything outside this block is label/figure code.
ARM_CON = "proposed"
ARM_UNC = "baseline"
SET_INNER = "inner5"
M_CONV = "survive"
M_NOVEL = "novel_post"
M_NUN = "nun_post"
M_NEAR = "near_post"
LAYER_KEY_1 = "main"
LAYER_KEY_2 = "additional"
# ==== END DATA KEYS ====

LAYER_TITLE = {LAYER_KEY_1: "Expt. 1", LAYER_KEY_2: "Expt. 2"}   # layer key -> drawn name

SUMMARY = RUN / "results/relax_novelty_summary.json"
FIGDIR = RUN / "results/figs_manuscript"
NAME = "relax_novelty_grid"
LAB_CON = "Constrained"
LAB_UNC = "Unconstrained"


def load_sets(summary_path):
    d = json.loads(Path(summary_path).read_text())
    sets = {}
    for key, v in d["sets"].items():
        layer, arm, w, s = key.split("|")
        assert w.startswith("w"), key
        sets[(layer, arm, w[1:], s)] = v
    return sets, d["layers"]


def make_figure(sets, figdir=None):
    figdir = Path(figdir) if figdir is not None else FIGDIR
    for ext in ("png", "pdf"):
        if (figdir / f"{NAME}.{ext}").exists():
            raise FileExistsError(str(figdir / f"{NAME}.{ext}"))
    setup_style()
    cols = []
    for (layer, arm, w, s) in sorted(sets, key=lambda k: (k[0], float(k[2]))):
        if s == SET_INNER and arm == ARM_CON:
            cols.append((layer, w))
    # Row labels. Definitions (from the aggregation code): Converged = relaxation converged / requested;
    # Novel = converged and no match among same-size training structures; Near target = converged and
    # |gamma_relaxed - gamma_target| <= 0.10 J/m^2; Novel, unique & near target = connected components of the novel,
    # unique, near-target set / requested.
    rows = [(M_CONV, "Converged"), (M_NOVEL, "Novel"),
            (M_NUN, "Novel, unique\n& near target"), (M_NEAR, "Near target\n(≤ 0.10 J/m$^2$)")]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(COL_2, COL_2 * 0.14 * len(rows) + 0.6), squeeze=False, sharey="row")
    for ci, (layer, w) in enumerate(cols):
        for ri, (mid, ylab) in enumerate(rows):
            ax = axes[ri][ci]
            for ai, (arm, color) in enumerate(((ARM_CON, OKABE_ITO[5]), (ARM_UNC, OKABE_ITO[6]))):
                m = sets.get((layer, arm, w, SET_INNER))
                if m is None:
                    continue
                x = m[mid]
                ax.bar(ai, x["p"], color=color, width=0.7)
                if x["lo"] is not None:
                    ax.errorbar(ai, x["p"], yerr=[[max(0.0, x["p"] - x["lo"])], [max(0.0, x["hi"] - x["p"])]], color="black", capsize=1.5, linewidth=0.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([LAB_CON, LAB_UNC] if ri == len(rows) - 1 else ["", ""], rotation=90)  # vertical: the longer names would collide side by side
            ax.set_ylim(0, 1.05)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=7)
            if ri == 0:
                ax.set_title(f"{LAYER_TITLE[layer]} ($w$ = {w})")
    fig.supylabel("Fraction", x=0.005, fontsize=9)
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    figdir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, NAME, str(figdir), pdf=True)
    plt.close(fig)
    return [str(figdir / f"{NAME}.png"), str(figdir / f"{NAME}.pdf")]


def main():
    sets, layers = load_sets(SUMMARY)
    print(make_figure(sets))


if __name__ == "__main__":
    main()
