"""Variable-length tilt-STGB package (internal notes).

**Scope note (2026-09-21, updated by the Step 32/32b/33/33b commit, "T5b")**: Step 31
(``layout.py``), Step 32 (``projection.py``), Step 32b (``adapter.py``), and Step 33
(``dataset.py``) are implemented. Step 33b (num_atoms distribution registration) lives in
``mattergen.common.data.num_atoms_distribution`` (not this package -- it is a single dict entry in
a shared, cross-package registry, not variable_length-specific code). Step 34 (loss), Step 35
(config), Step 36 (training sbatch), and Step 37 (dry-run/full-row precheck) are **out of scope**
for T5b and are intentionally absent.
"""

from __future__ import annotations
