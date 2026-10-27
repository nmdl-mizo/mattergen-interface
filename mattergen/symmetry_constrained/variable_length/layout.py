"""Per-node block/pair-rank field design for the variable-length tilt-STGB layout (internal notes).

**Why this module exists **: ``mattergen.symmetry_constrained.constants.GBRole`` (FIXED=0,
PRIMARY=1, REPLICA=2) cannot distinguish FIXED_A from FIXED_B. In the fixed N_PAD=128 slot layout
(``mattergen.symmetry_constrained.layout``), array *position* carried that distinction (``FIXED_A_SLICE`` vs
``FIXED_B_SLICE``); a variable-length layout has no fixed positions, so the distinction (and
the within-block PRIMARY<->REPLICA / FIXED_A<->FIXED_B correspondence order) must be carried as an
explicit per-node integer field instead. This module defines that field (``gb_block``), the
companion ordering field (``gb_pair_rank``), the derivation of the existing coarser ``gb_role``
field from ``gb_block``, and the per-graph invariant checks that a correctly-built graph must
satisfy.

**Naming ("K4")**: the ordering field is named
``gb_pair_rank``, **not** ``gb_pair_index``. PyTorch Geometric's ``torch_geometric.data.Data.__inc__``
offsets any attribute whose key contains the substring ``'index'`` by ``num_nodes`` at collate
time (to translate a within-graph node index into a within-batch node index, the same mechanism
used for e.g. ``edge_index``). ``mattergen.common.data.chemgraph.ChemGraph`` does not override
``__inc__`` (verified: it only overrides ``__cat_dim__``), so a field named ``gb_pair_index`` would
be silently corrupted by collate for every graph after the first in a batch. ``gb_pair_rank`` does
not contain the substring ``'index'`` and is therefore left untouched by collate (verified in
``tests/test_layout.py``, AC-T5a-5).

This module is intentionally standalone: it does not import anything from
``mattergen.symmetry_constrained.layout`` (the fixed N_PAD=128 slot layout) or build a ``ChemGraph``. It takes
and returns plain ``torch.Tensor`` so it can be exercised without an N_PAD-shaped batch and so a
future ``variable_length.dataset``/``variable_length.projection`` (, not implemented
here) can call it directly on a variable-length graph.
"""

from __future__ import annotations

import torch

from mattergen.symmetry_constrained.constants import N_BULK_PER_GRAIN, GBRole

# --- gb_block: the single source of truth for FIXED_A/FIXED_B/PRIMARY/REPLICA membership (
# SS3 Step 31 table). Values are plain ints (not an IntEnum instance) so they compare directly
# against a ``torch.long`` tensor without a dtype/enum boxing mismatch.
GB_BLOCK_FIXED_A: int = 0
GB_BLOCK_FIXED_B: int = 1
GB_BLOCK_PRIMARY: int = 2
GB_BLOCK_REPLICA: int = 3

_ALL_BLOCKS: tuple[int, ...] = (
    GB_BLOCK_FIXED_A,
    GB_BLOCK_FIXED_B,
    GB_BLOCK_PRIMARY,
    GB_BLOCK_REPLICA,
)

#: gb_role (mattergen.symmetry_constrained.constants.GBRole) is derived FROM gb_block, never
# stored independently at construction time by this module -- callers that already carry a
# gb_role field must pass it to derive_gb_role for a consistency check (silent
# fallback is forbidden -- a mismatch between an existing gb_role and the gb_block-derived one
# must raise, not be silently overwritten or silently ignored).
_BLOCK_TO_ROLE: dict[int, GBRole] = {
    GB_BLOCK_FIXED_A: GBRole.FIXED,
    GB_BLOCK_FIXED_B: GBRole.FIXED,
    GB_BLOCK_PRIMARY: GBRole.PRIMARY,
    GB_BLOCK_REPLICA: GBRole.REPLICA,
}

# Field name constants.
GB_BLOCK_FIELD: str = "gb_block"
GB_PAIR_RANK_FIELD: str = "gb_pair_rank"


class GbBlockInvariantViolation(RuntimeError):
    """Raised by ``assert_graph_invariants`` when a graph's ``gb_block``/``gb_pair_rank`` fields
    violate one of per-graph invariants. Never caught and silently downgraded by
    this module ."""


def derive_gb_role(
    gb_block: torch.Tensor, existing_gb_role: torch.Tensor | None = None
) -> torch.Tensor:
    """Derive the ``gb_role`` (``GBRole`` values: FIXED=0, PRIMARY=1, REPLICA=2) field from
    ``gb_block``.

    Args:
        gb_block: ``(N,)`` long tensor, values in ``{0,1,2,3}`` (``GB_BLOCK_*``).
        existing_gb_role: optional ``(N,)`` long tensor. If provided, the derived role is checked
            against it and a mismatch **raises** ``ValueError`` (silent fallback is
            forbidden when a pre-existing value conflicts with the value this function derives).

    Returns:
        ``(N,)`` long tensor of ``GBRole`` values, same device/shape as ``gb_block``.

    Raises:
        ValueError: if ``gb_block`` contains a value outside ``{0,1,2,3}``, or if
            ``existing_gb_role`` is provided and disagrees with the derived value at any node.
    """
    if gb_block.ndim != 1:
        raise ValueError(f"derive_gb_role: gb_block must be 1-D, got ndim={gb_block.ndim}")
    unknown = ~torch.isin(
        gb_block, torch.tensor(_ALL_BLOCKS, device=gb_block.device, dtype=gb_block.dtype)
    )
    if bool(torch.any(unknown).item()):
        bad = sorted(set(gb_block[unknown].tolist()))
        raise ValueError(f"derive_gb_role: gb_block contains value(s) outside {_ALL_BLOCKS}: {bad}")

    derived = torch.empty_like(gb_block)
    for block_id, role in _BLOCK_TO_ROLE.items():
        derived[gb_block == block_id] = int(role)

    if existing_gb_role is not None:
        if existing_gb_role.shape != gb_block.shape:
            raise ValueError(
                f"derive_gb_role: existing_gb_role shape {tuple(existing_gb_role.shape)} != "
                f"gb_block shape {tuple(gb_block.shape)}"
            )
        if not torch.equal(derived, existing_gb_role.to(dtype=derived.dtype)):
            n_mismatch = int((derived != existing_gb_role).sum().item())
            raise ValueError(
                f"derive_gb_role: existing_gb_role disagrees with the gb_block-derived value at "
                f"{n_mismatch} node(s). gb_role must equal derive_gb_role(gb_block) exactly "
                "(silent fallback forbidden)."
            )
    return derived


def _block_counts(gb_block: torch.Tensor) -> dict[int, int]:
    return {block_id: int((gb_block == block_id).sum().item()) for block_id in _ALL_BLOCKS}


def _check_block_is_permutation(gb_block: torch.Tensor, gb_pair_rank: torch.Tensor, block_id: int, size: int) -> None:
    ranks = gb_pair_rank[gb_block == block_id]
    if ranks.numel() != size:
        raise GbBlockInvariantViolation(
            f"gb_pair_rank: block {block_id} has {ranks.numel()} node(s) but size={size} was "
            "expected (internal inconsistency in caller)."
        )
    if size == 0:
        return
    sorted_ranks, _ = torch.sort(ranks)
    expected = torch.arange(size, device=ranks.device, dtype=ranks.dtype)
    if not torch.equal(sorted_ranks, expected):
        counts = torch.bincount(ranks.clamp(min=0), minlength=size) if bool((ranks >= 0).all()) else None
        dup_or_missing = "duplicate and/or out-of-range" if counts is None else (
            "duplicate" if bool((counts[:size] > 1).any()) else "missing/out-of-range"
        )
        raise GbBlockInvariantViolation(
            f"gb_pair_rank: block {block_id} (size={size}) values are not a permutation of "
            f"0..{size - 1} ({dup_or_missing} rank(s) found). Got sorted values "
            f"{sorted_ranks.tolist()}."
        )


def assert_graph_invariants(gb_block: torch.Tensor, gb_pair_rank: torch.Tensor) -> None:
    """Check per-graph invariants for a single graph's ``gb_block``/
    ``gb_pair_rank`` fields. Raises ``GbBlockInvariantViolation`` on any violation (no
    silent continuation).

    Invariants checked:
        1. ``n_fixed_a == n_fixed_b == N_BULK_PER_GRAIN``
        2. ``n_primary == n_replica`` (call the common value ``m``)
        3. ``n == 2*N_BULK_PER_GRAIN + 2*m``
        4. ``n % 2 == 0`` and ``m >= 1``
        5. within each block, ``gb_pair_rank`` is a permutation of ``0..(size-1)`` (no
           duplicates, no gaps)

    Args:
        gb_block: ``(N,)`` long tensor for ONE graph (not a batch), values in ``{0,1,2,3}``.
        gb_pair_rank: ``(N,)`` long tensor for the same graph.

    Raises:
        GbBlockInvariantViolation: on any invariant violation.
        ValueError: on malformed input (shape mismatch, unknown block id).
    """
    if gb_block.ndim != 1:
        raise ValueError(f"assert_graph_invariants: gb_block must be 1-D, got ndim={gb_block.ndim}")
    if gb_pair_rank.shape != gb_block.shape:
        raise ValueError(
            f"assert_graph_invariants: gb_pair_rank shape {tuple(gb_pair_rank.shape)} != "
            f"gb_block shape {tuple(gb_block.shape)}"
        )
    unknown = ~torch.isin(
        gb_block, torch.tensor(_ALL_BLOCKS, device=gb_block.device, dtype=gb_block.dtype)
    )
    if bool(torch.any(unknown).item()):
        bad = sorted(set(gb_block[unknown].tolist()))
        raise ValueError(
            f"assert_graph_invariants: gb_block contains value(s) outside {_ALL_BLOCKS}: {bad}"
        )

    n = int(gb_block.shape[0])
    counts = _block_counts(gb_block)
    n_fixed_a = counts[GB_BLOCK_FIXED_A]
    n_fixed_b = counts[GB_BLOCK_FIXED_B]
    n_primary = counts[GB_BLOCK_PRIMARY]
    n_replica = counts[GB_BLOCK_REPLICA]

    # Invariant 1
    if n_fixed_a != N_BULK_PER_GRAIN or n_fixed_b != N_BULK_PER_GRAIN:
        raise GbBlockInvariantViolation(
            f"n_fixed_a={n_fixed_a} n_fixed_b={n_fixed_b} != N_BULK_PER_GRAIN={N_BULK_PER_GRAIN} "
            "(both required)."
        )
    # Invariant 2
    if n_primary != n_replica:
        raise GbBlockInvariantViolation(f"n_primary={n_primary} != n_replica={n_replica}.")
    m = n_primary
    # Invariant 3
    expected_n = 2 * N_BULK_PER_GRAIN + 2 * m
    if n != expected_n:
        raise GbBlockInvariantViolation(
            f"n={n} != 2*N_BULK_PER_GRAIN + 2*m = 2*{N_BULK_PER_GRAIN} + 2*{m} = {expected_n}."
        )
    # Invariant 4
    if n % 2 != 0:
        raise GbBlockInvariantViolation(f"n={n} is odd (n % 2 == 0 required).")
    if m < 1:
        raise GbBlockInvariantViolation(f"m={m} < 1 (m >= 1 required).")

    # Invariant 5: gb_pair_rank is a within-block permutation of 0..(size-1).
    _check_block_is_permutation(gb_block, gb_pair_rank, GB_BLOCK_FIXED_A, n_fixed_a)
    _check_block_is_permutation(gb_block, gb_pair_rank, GB_BLOCK_FIXED_B, n_fixed_b)
    _check_block_is_permutation(gb_block, gb_pair_rank, GB_BLOCK_PRIMARY, m)
    _check_block_is_permutation(gb_block, gb_pair_rank, GB_BLOCK_REPLICA, m)


