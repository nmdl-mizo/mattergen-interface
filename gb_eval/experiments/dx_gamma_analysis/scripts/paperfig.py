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

__version__ = "1.1.0"   # 1.1.0: save_fig writes metadata to PNG/PDF. setup_style uses bbox=standard + constrained layout (checked by the figure check of the paper pipeline)

# ---- column widths [inch] (1 inch = 25.4 mm)
COL_1 = 86.0 / 25.4    # single column 86 mm
COL_2 = 178.0 / 25.4   # double column 178 mm

# ---- figure widths for posters [inch] (for 3 columns of A0 portrait; same values as the poster template) ----
POSTER_W1 = 240.0 / 25.4   # 1 column (column width of about 253.7 mm minus the left and right padding 2x7 mm)
POSTER_W2 = 505.0 / 25.4   # spanning 2 columns

PROFILES = ("paper", "poster")

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


def setup_style(profile="paper"):
    """Sets rcParams in bulk to publication-quality standards.

    profile="paper" (default): as before (paper figures). A call without arguments is exactly the same as before the modification.
    profile="poster": for conference posters (A0 read from 1.5 to 2 m; text in figures 22 pt or larger, thicker lines).
    An unknown profile raises ValueError (it does not silently fall back to paper).

    - Fonts: equivalent to 7-9 pt for the body (axis labels 9 pt, ticks 7 pt)
    - Thinner lines and ticks, tighter margins
    - Replaces the palette with Okabe-Ito (the default tab10 is not color-vision safe)
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
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
        # 1.1.0: tight trims margins and makes the physical width of the PNG smaller than figsize (a width check of the figure pipeline looks at this).
        # standard + constrained layout prevents axis labels from sticking out while keeping the physical width as figsize.
        "savefig.bbox": "standard",
        "figure.constrained_layout.use": True,
        "pdf.fonttype": 42,   # embed TrueType (avoid Type3 = required by many submission systems)
        "ps.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
    })
    if profile == "poster":
        mpl.rcParams.update({
            "font.size": 24,
            "axes.titlesize": 28,
            "axes.labelsize": 26,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "svg.fonttype": "none",   # keep text in figures as <text> of the SVG (so that the text size and prohibited words can be checked)
            "pdf.fonttype": 42,
            "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        })


def save_fig(fig, name, outdir=".", pdf=False, svg=False, dpi=None):
    """Saves the figure as PNG (raster). PDF is not produced by default.

    Reason why PNG only is the default: PNG is enough for daily checks and pasting into documents,
    and producing PDF all the time would double the disk use (user instruction).
    Give pdf=True explicitly only for figures that need a vector file for submission.
    With svg=True, outdir/name.svg is also written (for posters; not included in the return value, so that the existing
    `a, b = save_fig(...)` does not break; the path follows the convention outdir/<name>.svg).
    With dpi=None it is 600 as before (for posters the caller passes 300).
    Return value: (pdf_path or None, png_path)
    """
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"{name}.png")
    # Software tag: lets the figure pipeline (check G3) decide whether the figure went through paperfig.save_fig
    fig.savefig(png, dpi=600 if dpi is None else dpi, metadata={
        "Software": f"paperfig {__version__}",
        "paperfig_width_in": f"{fig.get_size_inches()[0]:.4f}",
    })
    pdf_path = None
    if pdf:
        pdf_path = os.path.join(outdir, f"{name}.pdf")
        fig.savefig(pdf_path, metadata={"Creator": f"paperfig {__version__}"})
    if svg:
        fig.savefig(os.path.join(outdir, f"{name}.svg"), metadata={"Creator": f"paperfig {__version__}"})
    return pdf_path, png
