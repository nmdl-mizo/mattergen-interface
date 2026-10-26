"""``gb_dx`` score readout shared by the symmetry-constrained adapter.

The generic diffusion score function reads a score for every key of ``corruption.sdes`` off the
model output BY NAME, so the ``gb_dx`` SDE needs a ``gb_dx`` field on the output ChemGraph.
``gb_dx_readout`` reads ``out.cell[:, 0, 0]`` (the stress(0,0)-derived lattice-score output that
``GemNetTCtrl.forward`` already provides; no new parameters) and is applied unconditionally by
the adapter's ``forward`` (harmless when no ``gb_dx`` SDE is configured).

The fixed-length readout head and the fixed-length denoiser/adapter classes that used to live here were
removed from the public release (not reachable from the paper path; see NOTICE).
"""

from __future__ import annotations

import torch


def gb_dx_readout(cell_output: torch.Tensor) -> torch.Tensor:
    """The ``gb_dx`` score is read straight off the existing stress(0,0)-derived lattice-score
    output (``cell_output`` == the model's own ``out.cell``, shape ``(n_graphs, 3, 3)``), NOT a new
    parameterised head. Returns shape ``(n_graphs, 1)`` (the graph-level scalar-field convention of
    ``gb_dx``)."""
    return cell_output[:, 0, 0].unsqueeze(-1)
