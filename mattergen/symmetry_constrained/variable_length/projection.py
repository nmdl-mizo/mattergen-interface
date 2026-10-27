"""Variable-length PRIMARY<->REPLICA / FIXED_A<->FIXED_B deterministic projector for tilt-STGB
Sigma5(210)[001] (internal notes).

This is the ``variable_length`` counterpart of ``mattergen.symmetry_constrained.projection.FixedSlotProjector``: it
implements the SAME symmetry operation (``mattergen.symmetry_constrained.symmetry_ops`` --
imported, never re-derived,), but operates on a variable-length, padding-free batch
where per-node block membership (``gb_block``) and within-block correspondence (``gb_pair_rank``,
"K4" -- named to avoid the substring ``'index'``, see
``mattergen.symmetry_constrained.variable_length.layout`` module docstring) replace the fixed N_PAD=128 slot positions
that ``FixedSlotProjector`` relies on.

**Correctness invariant this module preserves **: unlike
a projector class of another module (reference only -- never imported for its formula, only
its ``ptr``-based graph-loop *shape*, per "the reason a similar module must not be copied as it is"), this projector DOES restore FIXED_A/FIXED_B every call (the deterministic
(see the above).
``reconstruct_fixed_b_cart``/cell-anchor formula), not just REPLICA.

**Lattice constants are never hardcoded here **: ``a_template``/``b_template``/
``c_template`` are read per-graph off ``batch.gb_a_template``/``gb_b_template``/``gb_c_template``
(populated by :mod:`mattergen.symmetry_constrained.variable_length.dataset`, Step 33, from each row's own manifest cell),
never a module constant. ``symmetry_ops.reconstruct_cell`` is called ONCE PER GRAPH (inside the loop)
with that graph's own template axes and stacked into the batch cell tensor afterward -- this is
DELIBERATELY not a single whole-batch call with one scalar template (unlike the
pseudocode's outer ``reconstruct_cell(batch.gb_dx, ...)` sketch), because per-graph values let this
module stay correct even if a future host system's template axes are not batch-uniform (for the
Al system this module currently targets has verified they ARE uniform across all 10
CNID templates, so the two call patterns are numerically identical here); this is the same
``reconstruct_cell`` function, called with the same argument semantics, just G times instead of
once (/C13: the operation sequence is unchanged, only the batching of the *call site*
differs).

The rotation matrix ``_R_REPLICA_MATRIX`` and additive-translation gauge ``_B0_CONST`` are
architectural constants of the symmetry operation itself, not host-system-specific
lattice values -- they are imported directly from ``mattergen.symmetry_constrained.projection`` (the single
source of truth names), not redefined here.
"""

from __future__ import annotations

import torch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.symmetry_constrained import symmetry_ops
from mattergen.symmetry_constrained.constants import SCM_HOST_Z as HOST_Z
from mattergen.symmetry_constrained.projection import _B0_CONST, _R_REPLICA_MATRIX
from mattergen.symmetry_constrained.symmetry_ops import _periodic_wrap
from mattergen.symmetry_constrained.variable_length.layout import (
    GB_BLOCK_FIELD,
    GB_BLOCK_FIXED_A,
    GB_BLOCK_FIXED_B,
    GB_BLOCK_PRIMARY,
    GB_BLOCK_REPLICA,
    GB_PAIR_RANK_FIELD,
    assert_graph_invariants,
)

# Graph-level field names this projector reads (populated by variable_length.dataset, Step 33).
GB_FIXED_A_TEMPLATE_KEY = "gb_fixed_a_template"  # (n_graphs, N_BULK_PER_GRAIN, 3) Cartesian, Å
# here -- FIXED_B is always DERIVED from FIXED_A_TEMPLATE, mirroring
# mattergen.symmetry_constrained.projection's own module docstring: "FIXED_B is therefore no longer copied
# verbatim ... instead it is derived from FIXED_A_template").
GB_DX_KEY = "gb_dx"  # (n_graphs, 1) or (n_graphs,) Å, per-graph scalar
GB_T_KEY = "gb_t"  # (n_graphs, 3) fractional; only axes [1,2] (in-plane) are read here
GB_A_TEMPLATE_KEY = "gb_a_template"  # (n_graphs,) or (n_graphs, 1) Å -- "K2"
GB_B_TEMPLATE_KEY = "gb_b_template"
GB_C_TEMPLATE_KEY = "gb_c_template"


def _sorted_block_indices(
    block: torch.Tensor, rank: torch.Tensor, block_id: int, start: int
) -> torch.Tensor:
    """Local (within-graph) node indices whose ``gb_block == block_id``, sorted by ``gb_pair_rank``
    ascending, offset by ``start`` to become batch-global indices. Depends only on ``(gb_block,
    gb_pair_rank)`` values, never on the atoms' raw array position (design note:
    "does not depend at all on the order of atoms in the batch")."""
    local = torch.nonzero(block == block_id, as_tuple=False).flatten()
    order = torch.argsort(rank[local])
    return local[order] + start




class SymmetryProjector:
    """Deterministic FIXED_A/FIXED_B restore + PRIMARY->REPLICA projection for a variable-length,
    padding-free batch. Correspondence is read from the per-node ``gb_block``/
    ``gb_pair_rank`` fields (Step 31), never inferred from array position.
    """

    def __call__(self, batch: ChemGraph) -> ChemGraph:
        pos = batch.pos
        device, dtype = pos.device, pos.dtype
        n_total = pos.shape[0]

        if GB_BLOCK_FIELD not in batch or GB_PAIR_RANK_FIELD not in batch:
            raise ValueError(
                f"SymmetryProjector: batch is missing '{GB_BLOCK_FIELD}'/'{GB_PAIR_RANK_FIELD}' "
                ". This projector requires them; it does not infer "
                "correspondence from array position."
            )
        gb_block = getattr(batch, GB_BLOCK_FIELD)
        gb_pair_rank = getattr(batch, GB_PAIR_RANK_FIELD)
        if gb_block.shape != (n_total,) or gb_pair_rank.shape != (n_total,):
            raise ValueError(
                f"SymmetryProjector: gb_block shape {tuple(gb_block.shape)} / gb_pair_rank "
                f"shape {tuple(gb_pair_rank.shape)} != expected ({n_total},)"
            )

        ptr = getattr(batch, "ptr", None)
        if ptr is not None:
            ptr = ptr.to(device=device, dtype=torch.long)
        else:
            # fallback shape of another projection module, lines 34-35 ("reference only" -- same shape,
            # not the same formula: this module's correspondence logic is entirely its own).
            num_atoms = torch.as_tensor(batch.num_atoms, device=device, dtype=torch.long).reshape(-1)
            ptr = torch.cat((num_atoms.new_zeros(1), num_atoms.cumsum(0)))
        n_graphs = ptr.numel() - 1
        if n_graphs < 1:
            raise ValueError(f"SymmetryProjector: ptr implies n_graphs={n_graphs} < 1")

        for missing_key in (GB_FIXED_A_TEMPLATE_KEY, GB_DX_KEY, GB_T_KEY, GB_A_TEMPLATE_KEY,
                            GB_B_TEMPLATE_KEY, GB_C_TEMPLATE_KEY):
            if missing_key not in batch:
                raise ValueError(f"SymmetryProjector: batch is missing '{missing_key}'")

        fixed_a_template = torch.as_tensor(
            getattr(batch, GB_FIXED_A_TEMPLATE_KEY), device=device, dtype=dtype
        ).reshape(n_graphs, -1, 3)
        n_bulk_per_grain = fixed_a_template.shape[1]

        gb_dx = torch.as_tensor(batch.gb_dx, device=device, dtype=dtype).reshape(n_graphs)
        gb_t = torch.as_tensor(batch.gb_t, device=device, dtype=dtype).reshape(n_graphs, 3)
        gb_t_ip = gb_t[:, 1:3]
        a_tmpl = torch.as_tensor(batch.gb_a_template, device=device, dtype=dtype).reshape(n_graphs)
        b_tmpl = torch.as_tensor(batch.gb_b_template, device=device, dtype=dtype).reshape(n_graphs)
        c_tmpl = torch.as_tensor(batch.gb_c_template, device=device, dtype=dtype).reshape(n_graphs)

        rotation = torch.tensor(_R_REPLICA_MATRIX, device=device, dtype=dtype)
        b0 = torch.tensor(_B0_CONST, device=device, dtype=dtype)

        out_pos = pos.clone()
        out_z = batch.atomic_numbers.clone()
        new_cells: list[torch.Tensor] = []

        for g in range(n_graphs):
            s, e = int(ptr[g]), int(ptr[g + 1])
            block = gb_block[s:e]
            rank = gb_pair_rank[s:e]

            idx_fa = _sorted_block_indices(block, rank, GB_BLOCK_FIXED_A, s)
            idx_fb = _sorted_block_indices(block, rank, GB_BLOCK_FIXED_B, s)
            idx_m = _sorted_block_indices(block, rank, GB_BLOCK_PRIMARY, s)
            idx_s = _sorted_block_indices(block, rank, GB_BLOCK_REPLICA, s)

            # per-graph invariants (n_fixed_a==n_fixed_b==N_BULK_PER_GRAIN,
            # n_primary==n_replica, n==2*N_BULK_PER_GRAIN+2*m, gb_pair_rank a within-block
            # permutation). Raises GbBlockInvariantViolation on any violation -- never caught here
            # (no silent continuation on a malformed graph).
            assert_graph_invariants(block, rank)

            if idx_fa.numel() != n_bulk_per_grain or idx_fb.numel() != n_bulk_per_grain:
                raise ValueError(
                    f"SymmetryProjector: graph {g}: FIXED_A/FIXED_B count "
                    f"{idx_fa.numel()}/{idx_fb.numel()} != gb_fixed_a_template's "
                    f"N_BULK_PER_GRAIN={n_bulk_per_grain} (graph-level field/per-node field "
                    "disagree)."
                )

            a_t = float(a_tmpl[g])
            b_t = float(b_tmpl[g])
            c_t = float(c_tmpl[g])

            fa_cart_g = fixed_a_template[g].unsqueeze(0)  # (1, N_BULK_PER_GRAIN, 3)
            gb_t_ip_g = gb_t_ip[g].unsqueeze(0)  # (1, 2)
            gb_dx_g = gb_dx[g].unsqueeze(0)  # (1,)

            new_cell_g = symmetry_ops.reconstruct_cell(
                gb_dx_g, a_template=a_t, b_template=b_t, c_template=c_t
            )[0]  # (3, 3)
            new_cells.append(new_cell_g)
            cell_inv_g = torch.linalg.inv(new_cell_g)

            # FIXED_A: Cartesian template anchor -> fractional in the just-reconstructed cell
            # (mirrors mattergen.symmetry_constrained.projection.py's gb_dx-present branch, D7 item c).
            frac_a_new = torch.einsum("kj,ji->ki", fa_cart_g[0], cell_inv_g)
            out_pos[idx_fa] = _periodic_wrap(frac_a_new)

            # FIXED_B: derived from FIXED_A_template via (R, b0_cart) + gb_dx + gb_t's in-plane
            # component (symmetry_ops.reconstruct_fixed_b_cart --, same formula as
            # projection.py:262-282).
            cart_b = symmetry_ops.reconstruct_fixed_b_cart(
                fa_cart_g, gb_t_ip_g, gb_dx_g, rotation=rotation,
                a_template=a_t, b_template=b_t, c_template=c_t,
            )[0]  # (N_BULK_PER_GRAIN, 3)
            frac_b_new = torch.einsum("kj,ji->ki", cart_b, cell_inv_g)
            out_pos[idx_fb] = _periodic_wrap(frac_b_new)

            # PRIMARY->REPLICA: PRIMARY's fractional coordinate is left UNCHANGED (it is already a
            # valid fractional coordinate in the current cell, diffused live --
            # pseudocode / mattergen.symmetry_constrained.projection.py module docstring
            # item 9). REPLICA is the deterministic function of PRIMARY via symmetry_ops
            #.
            primary_pos_g = out_pos[idx_m].unsqueeze(0)  # (1, m, 3)
            replica_pos_g = symmetry_ops.reconstruct_replica_pos(
                primary_pos_g, gb_t_ip_g, rotation=rotation, b0=b0
            )[0]  # (m, 3)
            out_pos[idx_s] = replica_pos_g

            # Species: REPLICA mirrors PRIMARY (solute decorates both mirror-image atoms together,
            # mattergen.symmetry_constrained.layout._validate_species_frac docstring item d); FIXED_A/FIXED_B
            # are always the pure host species (mattergen.symmetry_constrained.projection.py:322-323).
            out_z[idx_s] = out_z[idx_m]
            out_z[idx_fa] = HOST_Z
            out_z[idx_fb] = HOST_Z

        out_cell = torch.stack(new_cells, dim=0)  # (n_graphs, 3, 3)

        return batch.replace(pos=out_pos, cell=out_cell, atomic_numbers=out_z)
