#!/usr/bin/env python3
"""
sym_common.py -- common to P1/P2/P3: utility for the audit of the symmetry operation GB1(x=0) <-> GB2(x=0.5).

Specification: see the description of the data production.

Symmetry operation: x' = R x + t in fractional coordinates
  [v1] R candidates: diag(-1,1,1) (mirror on the GB normal axis), diag(1,1,1) (pure translation). Assumes GB normal axis = index 0 (a axis).
  [v2 follow-up] The R candidates were extended to 8 after an acceptance remark (_r_candidates_v2).
    Physical background: the standard operation that maps GB1(x=0) <-> GB2(x=0.5) of a tilt bicrystal is the
    C2 rotation about the tilt axis. For a [001] tilt, GB normal = a axis, tilt axis = c axis, R=diag(-1,-1,1), which was not included in the v1 candidates
    (pure translation diag(1,1,1) / mirror on the normal axis only diag(-1,1,1)).
    The direct product of R_nn (normal-axis component) in {+1,-1} and (R_others1, R_others2) (two in-plane axes) in
    {(+1,+1) translation type, (-1,-1) C2 rotation type, (-1,+1), (+1,-1)} = 8 kinds, all fractional diagonal matrices.
  For P2/P3 (mirror system, GB normal = c axis = index 2) the normal axis is rearranged to the corresponding component of the diagonal
  (switched by the normal_axis argument of audit_symmetry. Physically the meaning "mirror/translate the GB1 plane along the normal direction
  onto the GB2 plane" is the same; only the position of the sign-reversed component of the matrix follows the normal axis).
  t = (0.5, ty, tz) as the initial coarse search (0.5 is put at the position of the normal-axis component) ->
  refinement by a least-squares fit from the atom-pair correspondence (Hungarian).

Residual = maximum of the nearest-atom distances (MIC) between the structure after applying the operation and the original structure (max over all atom pairs).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment


def _r_candidates(normal_axis: int):
    """[v1 backward compatibility] R candidates for the normal axis normal_axis (0/1/2) (a mirror that reverses the sign of the normal-axis component only, and a pure translation)."""
    mirror = np.eye(3)
    mirror[normal_axis, normal_axis] = -1.0
    return {
        "mirror_normal": mirror,
        "translate": np.eye(3),
    }


def _r_candidates_v2(normal_axis: int):
    """[v2 follow-up] 8 R candidates: normal-axis component R_nn in {+1,-1} x in-plane two axes (R_o1,R_o2) in
    {(+1,+1),(-1,-1),(-1,+1),(+1,-1)}. All fractional diagonal matrices (diag).
    R_nn=-1 with in-plane (-1,-1) corresponds to a C2 rotation about the tilt axis (one of the in-plane axes) (the physical operation pointed out in the acceptance remark)."""
    other_axes = [i for i in range(3) if i != normal_axis]
    o1, o2 = other_axes
    cands = {}
    for nn in (1.0, -1.0):
        for (s1, s2) in itertools.product((1.0, -1.0), repeat=2):
            diag = [1.0, 1.0, 1.0]
            diag[normal_axis] = nn
            diag[o1] = s1
            diag[o2] = s2
            nn_tag = "refl" if nn < 0 else "id"
            name = f"Rnn{nn_tag}_o{'m' if s1 < 0 else 'p'}{'m' if s2 < 0 else 'p'}"
            cands[name] = np.diag(diag)
    return cands


# backward compatibility (default assuming GB normal = a axis = index 0)
R_CANDIDATES = _r_candidates(0)


def _apply_op(frac: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return frac @ R.T + t


def _mic_frac_diff(df: np.ndarray) -> np.ndarray:
    """Folds a fractional difference into the minimum image (MIC) (each component [-0.5,0.5))."""
    return df - np.round(df)


def _pairwise_cost_matrix(cell: np.ndarray, frac_a: np.ndarray, frac_b: np.ndarray) -> np.ndarray:
    """Matrix (N,N) of all-pair MIC Cartesian distances between frac_a (N,3) and frac_b (N,3)."""
    n = frac_a.shape[0]
    diff = frac_a[:, None, :] - frac_b[None, :, :]           # (N,N,3)
    diff = _mic_frac_diff(diff)
    cart = diff @ cell                                        # (N,N,3)
    dist = np.linalg.norm(cart, axis=2)
    return dist


def _match_and_residual(cell: np.ndarray, frac_ref: np.ndarray, frac_moved: np.ndarray):
    """Matches frac_moved to frac_ref by the Hungarian method and returns (residual array after matching, matching indices)."""
    n_ref = frac_ref.shape[0]
    n_mov = frac_moved.shape[0]
    if n_ref != n_mov:
        return None, None
    D = _pairwise_cost_matrix(cell, frac_ref, frac_moved)
    row_ind, col_ind = linear_sum_assignment(D)
    residuals = D[row_ind, col_ind]
    return residuals, col_ind


def _refine_t(cell: np.ndarray, frac_ref: np.ndarray, frac0: np.ndarray, R: np.ndarray,
              t_init: np.ndarray, n_iter: int = 5):
    """Iteratively refines t from the centroid difference of the matched atom pairs.
    Each iteration: apply R with the current t -> Hungarian matching -> add the mean Cartesian displacement of the matched pairs to t (converted to fractional).
    """
    t = t_init.copy()
    cell_inv = np.linalg.inv(cell)
    residuals = None
    col_ind = None
    for _ in range(n_iter):
        moved = _apply_op(frac0, R, t)
        residuals, col_ind = _match_and_residual(cell, frac_ref, moved)
        if residuals is None:
            break
        # feed back the mean displacement of the matched pairs (fractional, including MIC) to t
        diff = frac_ref - moved[col_ind]
        diff = _mic_frac_diff(diff)
        # convert diff to Cartesian, average, and return to fractional again (for numerical stability)
        cart_diff = diff @ cell
        mean_cart = cart_diff.mean(axis=0)
        mean_frac = mean_cart @ cell_inv
        t = t + mean_frac
    return t, residuals, col_ind


def audit_symmetry(atoms, ty_grid=None, tz_grid=None, n_refine=5, normal_axis=0, use_v2_candidates=False,
                   force_r_name=None):
    """For one structure, searches the GB1<->GB2 symmetry operation (half-cell translation 0.5 along the normal axis; search ty, tz over the two in-plane axes).

    Parameters
    ----------
    atoms : ase.Atoms
    ty_grid, tz_grid : coarse search grids (fractional coordinates, default 0..1 in steps of 0.05; correspond to the two axes other than the normal axis)
    n_refine : number of iterations of the refinement of t
    normal_axis : cell index of the GB normal axis (0=a axis/1=b axis/2=c axis). 0 for P1 (im system), 2 for P2/P3 (mirror system).
    use_v2_candidates : if True, the [v2 follow-up] 8 R candidates (_r_candidates_v2) are used. False (default) is
        the 2 candidates of v1 (backward compatible; does not change the behavior of the existing p1_audit.py/p2_build.py).
    force_r_name : [follow-up of the degeneracy check] if given, t is optimized for only the one of the 8 v2 candidates that corresponds
        (a forced search, not an argmin. Example: "Rnnid_omm"). Used together with use_v2_candidates=True.

    Returns
    -------
    dict: best_R(str), best_t(tuple), max_residual_A(float), mean_residual_A(float)
    """
    if ty_grid is None:
        ty_grid = np.arange(0.0, 1.0, 0.05)
    if tz_grid is None:
        tz_grid = np.arange(0.0, 1.0, 0.05)

    cell = np.array(atoms.get_cell())
    frac = np.array(atoms.get_scaled_positions(wrap=True))
    n = len(atoms)
    other_axes = [i for i in range(3) if i != normal_axis]

    best = {
        "best_R": None, "best_t": None,
        "max_residual_A": float("inf"), "mean_residual_A": float("inf"),
        "n_matched": 0, "n_atoms": n,
    }

    r_candidates = _r_candidates_v2(normal_axis) if use_v2_candidates else _r_candidates(normal_axis)
    if force_r_name is not None:
        if force_r_name not in r_candidates:
            raise ValueError(f"force_r_name={force_r_name!r} is not in the candidate set: {list(r_candidates)}")
        r_candidates = {force_r_name: r_candidates[force_r_name]}
    for r_name, R in r_candidates.items():
        # --- coarse search: normal axis fixed at 0.5; choose the minimum sum of residuals after matching on the grid of the two in-plane axes (ty,tz) ---
        best_coarse = None
        best_coarse_cost = np.inf
        for ty in ty_grid:
            for tz in tz_grid:
                t0 = np.zeros(3)
                t0[normal_axis] = 0.5
                t0[other_axes[0]] = ty
                t0[other_axes[1]] = tz
                moved = _apply_op(frac, R, t0)
                residuals, col_ind = _match_and_residual(cell, frac, moved)
                if residuals is None:
                    continue
                cost = float(residuals.sum())
                if cost < best_coarse_cost:
                    best_coarse_cost = cost
                    best_coarse = t0
        if best_coarse is None:
            # a mismatch of the number of atoms etc.: continue as a fail (residual inf)
            continue

        # --- refinement ---
        t_ref, residuals, col_ind = _refine_t(cell, frac, frac, R, best_coarse, n_iter=n_refine)
        if residuals is None:
            continue
        max_res = float(residuals.max())
        mean_res = float(residuals.mean())

        if max_res < best["max_residual_A"]:
            best.update({
                "best_R": r_name,
                "best_t": tuple(round(float(x), 6) for x in t_ref),
                "max_residual_A": max_res,
                "mean_residual_A": mean_res,
                "n_matched": n,
            })

    return best
