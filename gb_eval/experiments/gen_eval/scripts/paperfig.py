#!/usr/bin/env python3
"""paperfig.py -- common style and saving utilities for publication-quality figures (depends only on matplotlib).

Usage:
    from paperfig import setup_style, save_fig, OKABE_ITO, COL_1, COL_2
    setup_style()
    fig, ax = plt.subplots(figsize=(COL_1, COL_1 * 0.75))
    ...
    save_fig(fig, "epa_hist", "figures")        # -> only figures/epa_hist.png
    save_fig(fig, "fig3", "figures", pdf=True)  # also PDF only when a vector file for submission is needed
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- column widths [inch] (1 inch = 25.4 mm)
COL_1 = 86.0 / 25.4    # single column 86 mm
COL_2 = 178.0 / 25.4   # double column 178 mm

# ---- Okabe-Ito color-blind-safe palette ----
# 8 colors that can be distinguished under color-vision diversity (safe if used in this order)
OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]


def setup_style():
    """Sets rcParams in bulk to publication-quality standards.

    - Fonts: equivalent to 7-9 pt for the body (axis labels 9 pt, ticks 7 pt)
    - Thinner lines and ticks, tighter margins
    - Replaces the palette with Okabe-Ito (the default tab10 is not color-vision safe)
    """
    mpl.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "font.family": "sans-serif",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # embed TrueType (avoid Type3 = required by many submission systems)
        "ps.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    })


def save_fig(fig, name, outdir=".", pdf=False):
    """Saves the figure as PNG (raster). PDF is not produced by default.

    Reason why PNG only is the default: PNG is enough for daily checks and pasting into documents,
    and producing PDF all the time would double the disk use (user instruction).
    Give pdf=True explicitly only for figures that need a vector file for submission.
    Return value: (pdf_path or None, png_path)
    """
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"{name}.png")
    fig.savefig(png, dpi=600)
    pdf_path = None
    if pdf:
        pdf_path = os.path.join(outdir, f"{name}.pdf")
        fig.savefig(pdf_path)
    return pdf_path, png
