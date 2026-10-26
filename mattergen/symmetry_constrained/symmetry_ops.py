"""Pure symmetry-operation functions for the tilt-STGB Sigma5(210)[001] MASTER<->SLAVE /
FIXED_A<->FIXED_B reconstruction (internal notes).

**Purpose **: the same formula was at risk of being duplicated across
``mattergen.symmetry_constrained`` (fixed N_PAD=128 slot layout) and a future variable-length package
if the arithmetic stayed embedded inside
``mattergen.symmetry_constrained.projection.FixedSlotProjector.__call__``. This module extracts that arithmetic
into functions that know nothing about array layout (no ``N_PAD``, no slot slicing) and nothing
about which host system (Al or another host) is being processed (no module-level lattice constants --
"K2": those are passed in as keyword-only arguments so a future host system does not
require a new hardcoded constant here).

**This is a pure move, not a rewrite.** Every function body below is copied verbatim (same
operand order, same einsum subscripts, same intermediate tensor construction) from
``mattergen.symmetry_constrained.projection`` as it stood at commit ``4f14c2f``, specifically:

- ``reconstruct_slave_pos``          <- ``projection.py:316-322`` (MASTER->SLAVE, both branches)
- ``reconstruct_fixed_b_cart``       <- ``projection.py:262-282`` (FIXED_B, gb_dx-present branch)
- ``reconstruct_fixed_b_cart_legacy``<- ``projection.py:293-300`` (FIXED_B, gb_dx-absent branch)
- ``reconstruct_cell``               <- ``projection.py:249-252`` (cell, gb_dx-present branch)
- ``_periodic_wrap``                 <- ``projection.py:124-131`` (shared wrap helper)

The MASTER->SLAVE in-plane coefficient ``T_coeff`` is **not** parameterized here:
records it as part of the symmetry operation's own definition (``T_COEFF_INPLANE = np.eye(2)``,
``mattergen.symmetry_constrained.manifest.py:49``), not a system-dependent quantity, so it is kept as a
module-level identity constant exactly as ``projection.py`` had it (``_T_COEFF_INPLANE_MATRIX``),
preserving the original operation sequence bit-for-bit (/C13: do not silently change
which operations run just to make a signature shorter).
"""

from __future__ import annotations

import torch

# T_coeff = I_2.
# Part of the symmetry operation's definition, not a system-specific parameter -- kept fixed here
# (not exposed as an argument) so this module's public functions reproduce the exact operation
# sequence of projection.py:316-322 bit-for-bit.
_T_COEFF_INPLANE_MATRIX = [[1.0, 0.0], [0.0, 1.0]]


def _periodic_wrap(x: torch.Tensor) -> torch.Tensor:
    """Fractional-coordinate wrap into [0, 1) (projection.py:124-131 mod-1 / SS6 edge
    case 1). Plain ``torch.remainder`` (not ``x - floor(x)`` by hand) for the float32
    1.0-eps -> 1.0 rounding edge case."""
    return torch.remainder(x, 1.0)


def reconstruct_slave_pos(
    master_frac: torch.Tensor,
    gb_t_ip: torch.Tensor,
    *,
    rotation: torch.Tensor,
    b0: torch.Tensor,
) -> torch.Tensor:
    """MASTER -> SLAVE fractional-coordinate reconstruction (``projection.py:316-322``).

    Args:
        master_frac: ``(G, M, 3)`` fractional coordinates.
        gb_t_ip: ``(G, 2)`` in-plane component of ``gb_t`` (y, z).
        rotation: ``(3, 3)`` -- ``_R_SLAVE_MATRIX`` (``diag(+1,-1,-1)``).
        b0: ``(3,)`` -- ``_B0_CONST`` (``(0.5, 0.5, 0.5)``).

    Returns:
        ``(G, M, 3)`` fractional coordinates, periodic-wrapped into ``[0, 1)``. Identical in both
        the legacy (gb_dx-absent) and gb_dx-present branches of ``FixedSlotProjector.__call__``
        ("do not touch the fractional coordinates of MASTER/SLAVE").
    """
    device, dtype = master_frac.device, master_frac.dtype
    n_graphs = master_frac.shape[0]
    t_coeff = torch.tensor(_T_COEFF_INPLANE_MATRIX, device=device, dtype=dtype)
    rotated = torch.einsum("ij,gkj->gki", rotation, master_frac)
    inplane_term = torch.einsum("ij,gj->gi", t_coeff, gb_t_ip)
    add = torch.zeros(n_graphs, 3, device=device, dtype=dtype)
    add[:, 1:3] = inplane_term
    return _periodic_wrap(rotated + b0.view(1, 1, 3) + add.unsqueeze(1))


def reconstruct_fixed_b_cart(
    fixed_a_cart: torch.Tensor,
    gb_t_ip: torch.Tensor,
    gb_dx: torch.Tensor,
    *,
    rotation: torch.Tensor,
    a_template: float,
    b_template: float,
    c_template: float,
) -> torch.Tensor:
    """FIXED_B Cartesian-coordinate reconstruction, gb_dx-present branch
    (``projection.py:262-282``). Caller converts the returned Cartesian tensor to fractional
    coordinates via the current cell's inverse (``projection.py:281`` -- not duplicated here since
    that step is a plain ``einsum`` against a caller-owned ``cell_inv``, not part of the symmetry
    operation itself).

    Args:
        fixed_a_cart: ``(G, N_FIXED, 3)`` Cartesian FIXED_A template (Angstrom).
        gb_t_ip: ``(G, 2)`` in-plane component of ``gb_t`` (y, z), fractional.
        gb_dx: ``(G,)`` per-graph dx shift (Angstrom), already flattened to 1-D.
        rotation: ``(3, 3)`` -- ``_R_SLAVE_MATRIX``.
        a_template / b_template / c_template: template cell axis lengths (Angstrom). **System
        specific -- always pass the value appropriate to the host system being processed
        . Never hardcode Al's value here.**

    Returns:
        ``(G, N_FIXED, 3)`` Cartesian coordinates (not yet wrapped/converted to fractional).
    """
    device, dtype = fixed_a_cart.device, fixed_a_cart.dtype
    n_graphs = fixed_a_cart.shape[0]
    b0_cart = torch.tensor(
        [0.5 * a_template, 0.5 * b_template, 0.5 * c_template], device=device, dtype=dtype
    )
    rotated_a_cart = torch.einsum("ij,gkj->gki", rotation, fixed_a_cart)
    gb_t_ip_cart = torch.zeros(n_graphs, 3, device=device, dtype=dtype)
    gb_t_ip_cart[:, 1] = gb_t_ip[:, 0] * b_template
    gb_t_ip_cart[:, 2] = gb_t_ip[:, 1] * c_template
    dx_shift_cart = torch.zeros(n_graphs, 3, device=device, dtype=dtype)
    dx_shift_cart[:, 0] = gb_dx
    return (
        rotated_a_cart
        + b0_cart.view(1, 1, 3)
        + dx_shift_cart.unsqueeze(1)
        + gb_t_ip_cart.unsqueeze(1)
    )




def reconstruct_cell(
    gb_dx: torch.Tensor,
    *,
    a_template: float,
    b_template: float,
    c_template: float,
) -> torch.Tensor:
    """Cell reconstruction as a pure function of ``gb_dx`` (``projection.py:249-252``,
    gb_dx-present branch). ``new_cell[:, 1, 1]`` / ``new_cell[:, 2, 2]`` are the **system-specific**
    in-plane axis lengths ("K2" -- the direct cause of the r0 KILL finding: these were
    Al's hardcoded values in ``projection.py``, which are wrong by ~11% for another host). Always pass the
    value appropriate to the host system being processed; never hardcode here.

    Args:
        gb_dx: ``(G,)`` per-graph dx shift (Angstrom), already flattened to 1-D.
        a_template / b_template / c_template: template cell axis lengths (Angstrom).

    Returns:
        ``(G, 3, 3)`` cell tensor. Off-diagonal entries are zero (orthorhombic cell, unchanged from
        the original -- this function never populates them).
    """
    device, dtype = gb_dx.device, gb_dx.dtype
    n_graphs = gb_dx.shape[0]
    new_cell = torch.zeros(n_graphs, 3, 3, device=device, dtype=dtype)
    new_cell[:, 0, 0] = a_template + 2.0 * gb_dx
    new_cell[:, 1, 1] = b_template
    new_cell[:, 2, 2] = c_template
    return new_cell
