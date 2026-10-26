#!/usr/bin/env python3
"""Build the symlink view `$GBREL_WORK/<link>` -> `$GBREL_CODE/<target>` or `$GBREL_DATA/<target>` listed in compat_layout.tsv.

The evaluation scripts compose many paths from a "run root" variable (e.g. <root>/runs/<run>/results/...). Instead of rewriting every
composition, this script recreates the directory layout they expect under $GBREL_WORK/workspace as symlinks to the released files.
Existing links/files are never overwritten. Output directories that the scripts write to are created as real directories.
Usage:  GBREL_CODE=... GBREL_DATA=... GBREL_WORK=... python gb_eval/make_compat_tree.py
"""
import os
import sys
from pathlib import Path

code, data, work = (Path(os.environ[k]) for k in ("GBREL_CODE", "GBREL_DATA", "GBREL_WORK"))
base = {"CODE": code, "DATA": data}
rows = [l.rstrip("\n").split("\t") for l in (code / "gb_eval" / "compat_layout.tsv").read_text().split("\n")[1:] if l]
n_new = n_skip = 0
for link, b, tgt in rows:
    if link.startswith("workspace/paper/figures/png/"):
        continue   # published figure PNGs are outputs of the figure scripts (they refuse to overwrite): never linked
    target = base[b] / tgt
    if not target.exists():
        print("MISSING target:", target, file=sys.stderr)
        continue
    dst = work / link
    # skip if it can already be reached through a link (via an upper directory)
    cur = work
    covered = False
    for part in Path(link).parts[:-1]:
        cur = cur / part
        if cur.is_symlink():
            covered = True
            break
    if covered or dst.exists() or dst.is_symlink():
        n_skip += 1
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, dst)
    n_new += 1
for d in ("workspace/paper/figures/png",
          "workspace/paper/figure_sources/si_w_sweep",
          "workspace/paper/figure_sources/si_main_w0",
          "workspace/paper/figure_sources/si_f6_f9"):   # output directories of the figure scripts
    (work / d).mkdir(parents=True, exist_ok=True)
print("links created:", n_new, " skipped:", n_skip)
