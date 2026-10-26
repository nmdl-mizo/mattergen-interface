"""Variable-length ``gb_t``/``gb_dx`` readout heads for the GemNetTDenoiser/adapter (internal notes).

``mattergen.symmetry_constrained.adapter``'s ``gb_t_readout`` does a fixed ``pos_output.reshape(n_graphs,
N_PAD, 3)`` (``adapter.py:33``), which is meaningless for a padding-free, variable-length batch --
there is no single ``N_PAD`` every graph shares. This module replaces it with a ``ptr``-based
segment mean over SLAVE nodes, using the SAME selection semantics (mean of the SLAVE-role node
positions, per graph) but grouping by graph index instead of by a fixed reshape.

``gb_dx_readout`` is UNCHANGED by the variable_length/padding-free distinction (it reads
``cell_output[:, 0, 0]``, shape ``(n_graphs, 3, 3)`` -> ``(n_graphs, 1)``, which has no per-node
dependency at all) -- it is imported directly from ``mattergen.symmetry_constrained.adapter``, not
re-implemented (the same function, not a duplicate).

``gb_role`` (needed to select SLAVE nodes) is assumed MATERIALIZED on the input batch
requirement ("``gb_role`` must be materialized at the dataset stage ... `loss.py:225` and
`adapter.py:53,78` read `gb_role` by name, so lazy derivation is too late") -- this module reads
``x.gb_role`` directly, exactly as ``mattergen.symmetry_constrained.adapter``'s two ``forward`` methods do
(``adapter.py:53,78``), never re-deriving it from ``gb_block``.
"""

from __future__ import annotations

import torch

from mattergen.adapter import GemNetTAdapter
from mattergen.symmetry_constrained.adapter import gb_dx_readout  # re-exported; same function, not duplicated
from mattergen.symmetry_constrained.constants import GBRole

__all__ = [
    "gb_dx_readout",
    "gb_t_readout_variable_length",
    "resolve_batch_vec",
    "SymmetryConstrainedAdapter",
]


def resolve_batch_vec(x) -> torch.Tensor:
    """Per-node "which graph do I belong to" index vector, ``(N_total,)`` long, values in
    ``[0, n_graphs)``. Uses PyG's auto-populated ``batch.batch`` attribute when present (the normal
    case for a collated ``Batch``); falls back to deriving it from ``ptr``/``num_atoms`` for a
    single, non-collated ``ChemGraph`` item (mirrors
    ``mattergen.symmetry_constrained.variable_length.projection.SymmetryProjector``'s own ``ptr`` fallback
    "reference only" shape borrowed from another projection module (lines 31-36), not its formula).
    """
    batch_vec = getattr(x, "batch", None)
    if batch_vec is not None:
        return batch_vec.to(dtype=torch.long)
    ptr = getattr(x, "ptr", None)
    if ptr is not None:
        ptr = ptr.to(dtype=torch.long)
    else:
        num_atoms = torch.as_tensor(x.num_atoms, dtype=torch.long).reshape(-1)
        ptr = torch.cat((num_atoms.new_zeros(1), num_atoms.cumsum(0)))
    n_graphs = ptr.numel() - 1
    counts = ptr[1:] - ptr[:-1]
    return torch.repeat_interleave(
        torch.arange(n_graphs, device=ptr.device, dtype=torch.long), counts
    )


def gb_t_readout_variable_length(
    pos_output: torch.Tensor, slave_mask: torch.Tensor, batch_vec: torch.Tensor, n_graphs: int
) -> torch.Tensor:
    """Per-graph mean of ``pos_output`` over nodes where ``slave_mask`` is True (:
    ``ptr``-based segment mean, replacing ``mattergen.symmetry_constrained.adapter.gb_t_readout``'s fixed
    ``reshape(n_graphs, N_PAD, 3)``). No padding/real-atom mask is needed here (unlike
    ``mattergen.symmetry_constrained.adapter.gb_t_readout``'s ``slave_real_mask``): every node in a
    ``variable_length`` batch is a real atom by construction.

    Args:
        pos_output: ``(N_total, 3)`` -- the denoiser's raw position-score output over the WHOLE
            (collated) batch.
        slave_mask: ``(N_total,)`` bool -- True where the node's role is SLAVE
            (``x.gb_role == GBRole.SLAVE``).
        batch_vec: ``(N_total,)`` long, values in ``[0, n_graphs)`` (see :func:`resolve_batch_vec`).
        n_graphs: number of graphs in the batch.

    Returns:
        ``(n_graphs, 3)``. A graph with zero SLAVE nodes would divide by zero; that is a
        Step-31-invariant violation (``m >= 1`` is required for every graph) and is therefore not
        silently guarded against here -- ``assert_graph_invariants`` (Step 31) is the place that
        catches it, before this function is ever reached in the intended call order (
        Step 32 -> Step 32b).
    """
    device, dtype = pos_output.device, pos_output.dtype
    idx = batch_vec[slave_mask]
    sums = torch.zeros(n_graphs, 3, device=device, dtype=dtype)
    sums.index_add_(0, idx, pos_output[slave_mask])
    counts = torch.zeros(n_graphs, 1, device=device, dtype=dtype)
    counts.index_add_(
        0, idx, torch.ones(idx.shape[0], 1, device=device, dtype=dtype)
    )
    return sums / counts.clamp(min=1)


def _gb_t_from_forward_output(x, out) -> "torch.Tensor | None":
    if "gb_role" not in x or "gb_t" not in x:
        return None
    slave_mask = x.gb_role == int(GBRole.SLAVE)
    batch_vec = resolve_batch_vec(x)
    n_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0
    return gb_t_readout_variable_length(out.pos, slave_mask, batch_vec, n_graphs)




class SymmetryConstrainedAdapter(GemNetTAdapter):
    """Variable-length counterpart of ``mattergen.symmetry_constrained.adapter.FixedSlotAdapter`` (
    Step 32b): composes :func:`gb_t_readout_variable_length` with ``GemNetTAdapter``'s property-conditioning
    machinery, mirroring the single-parent-chain shape of the fixed-length class."""

    def forward(self, x, t):
        out = super().forward(x, t)
        out = out.replace(gb_dx=gb_dx_readout(out.cell))
        gb_t = _gb_t_from_forward_output(x, out)
        if gb_t is None:
            return out
        return out.replace(gb_t=gb_t)
