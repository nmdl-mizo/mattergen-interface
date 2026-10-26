#!/usr/bin/env python3
"""fig_F1_schematic.py -- adopted version of the paper's Fig. 1 (F1): in the formula at the top of (a), the numerator E_excess is replaced by the notation of the text, E_cell - N E_bulk. Otherwise identical to the previous adopted version. Compared with the earlier labelfix version it reflects the typesetting of subscripts and fractions in formulas, the separate placement of Readout (GemNet-T -> Readout -> CFG, pos and species go straight), the term "grain boundary energy" in (a), and pos as the variable updated by the Corrector. The 120 atoms and the colors are unchanged. This is a wrapper that puts the adopted drawing into the paper PNG (via paperfig) without redrawing it.

Created as a copy of the earlier wrapper; only the input SVG and the output name were changed. The content of the image is not changed.
Procedure: rasterize the adopted SVG with rsvg-convert at the equivalent of 600 dpi (width = physical width of the figure x 600 px) -> matplotlib figure
      (width double = 178 mm, axes off, imshow at 1:1, zero margins) -> paperfig.save_fig (adds the Software tag and pHYs).
Input: figure_sources/fig1/vector/Fig1.svg
Output: figures/png/F1_schematic.png (earlier PNG files are not overwritten)
Run: with the evaluation python on a login node:  ulimit -v 8000000; export OMP_NUM_THREADS=1; timeout 900 ${MACE_PY} <this file>
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)            # figures/scripts/paperfig.py (copy of the master)
PA = os.path.abspath(os.path.join(HERE, "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import paperfig
from paperfig import COL_2, save_fig, setup_style

assert paperfig.__version__ == "1.1.0", paperfig.__version__
SVG = os.path.join(PA, "figure_sources/fig1/vector/Fig1.svg")
OUT_NAME = "F1_schematic"
OUT_DIR = os.path.join(PA, "figures", "png")
DPI = 600
assert os.path.exists(SVG), SVG
if os.path.exists(os.path.join(OUT_DIR, OUT_NAME + ".png")):
    sys.exit("existing files are not overwritten: " + OUT_NAME + ".png")

head = open(SVG, encoding="utf-8").read(3000)
vb = re.search(r'viewBox="([^"]+)"', head)
assert vb, "no viewBox"
vw, vh = [float(t) for t in vb.group(1).split()[2:4]]
W_IN = COL_2
H_IN = W_IN * vh / vw
W_PX = int(round(W_IN * DPI))
H_PX = int(round(H_IN * DPI))

with tempfile.TemporaryDirectory() as td:
    tmp = os.path.join(td, "raster.png")
    subprocess.run(["rsvg-convert", "-w", str(W_PX), "-h", str(H_PX), "--background-color", "white", "-o", tmp, SVG], check=True, timeout=600)
    img = np.asarray(Image.open(tmp).convert("RGB"))
print("raster:", img.shape, " figure: %.4f x %.4f in" % (W_IN, H_IN))

setup_style()
fig = plt.figure(figsize=(W_IN, H_IN))
fig.set_layout_engine("none")
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
ax.imshow(img, aspect="auto", interpolation="nearest", extent=(0, 1, 0, 1))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
_, png = save_fig(fig, OUT_NAME, OUT_DIR)
print("saved:", png)
