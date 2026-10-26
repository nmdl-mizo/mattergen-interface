#!/usr/bin/env python3
"""
removal_builder.py -- subtask B: builder of the removal of symmetric atom pairs.

Plan: see the experiment description.

Function remove_symmetric_pairs(atoms, k, seed, op=None):
1. Refit the symmetry operation op to the template (reusing sym_common, force_r_name="Rnnid_omm"
   = screw R=diag(+1,-1,-1) of the plan). An exception if the fit residual > 1e-6 A.
2. Identification of the GB2 core atoms: fractional x in [0.5-delta, 0.5+delta], delta=0.05. The list of the atoms concerned = C2.
3. Draw k atoms from C2 without replacement with a seeded RNG.
4. For each selected atom, find the nearest atom at the position of the image under op (MIC distance < 0.3 A) and
   delete the selected atom and the image atom, 2 atoms in total.
   - If no image is found -> redo that selection (at most 10 times; when exhausted, record a skip).
   - A case in which the image returns to the GB2 core side (self pair, x in [0.5-delta, 0.5+delta]) is also allowed as a pair.
     However, if image = the selected atom itself, choose again (to keep the number of atoms in steps of 2).
5. After the removal, confirm with the symmetry audit that the residual is < 1e-3 A (a failure is recorded as a skip).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sym_common import audit_symmetry, _r_candidates_v2, _apply_op  # noqa: E402

DELTA = 0.05           # width of the GB2 core decision (fractional)
IMAGE_TOL_A = 0.3       # distance tolerance for the image atom search (A)
FIT_RESIDUAL_TOL_A = 1e-6   # upper limit of the tolerance of the op fit residual
ASBUILT_RESIDUAL_TOL_A = 1e-3   # upper limit of the tolerance of the as-built audit after the removal
FORCE_R_NAME = "Rnnid_omm"   # candidate name that corresponds to the screw R=diag(+1,-1,-1)
MAX_RETRY_PER_SITE = 10


class RemovalSkip(Exception):
    """Exception that holds the reason why this sample should be skipped (caught by the caller and recorded in the manifest)."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def fit_symmetry_op(atoms, normal_axis: int = 0, force_r_name: str = FORCE_R_NAME):
    """Refits the symmetry operation to the template. An exception if the residual > 1e-6 A.

    Returns
    -------
    R : (3,3) ndarray (fractional diagonal matrix)
    t : (3,) ndarray
    residual_A : float
    """
    res = audit_symmetry(atoms, normal_axis=normal_axis, use_v2_candidates=True,
                          force_r_name=force_r_name)
    residual = res["max_residual_A"]
    if not (residual < FIT_RESIDUAL_TOL_A):
        raise RemovalSkip(f"symmetry fit residual too large: {residual:.3e}A "
                           f"(tol={FIT_RESIDUAL_TOL_A:.1e}A)")
    R = _r_candidates_v2(normal_axis)[force_r_name]
    t = np.array(res["best_t"], dtype=float)
    return R, t, residual


def _mic_cart_distance(cell, frac_a, frac_b):
    """MIC Cartesian distance vector (N,) of a point-to-point set."""
    diff = frac_a[None, :] - frac_b
    diff = diff - np.round(diff)
    cart = diff @ cell
    return np.linalg.norm(cart, axis=1)


def remove_symmetric_pairs(atoms, k: int, seed: int, R=None, t=None, normal_axis: int = 0):
    """Returns new Atoms from which k symmetric atom pairs were removed.

    Parameters
    ----------
    atoms : ase.Atoms (template, not changed)
    k : number of atom pairs to remove (total number of deleted atoms = 2k)
    seed : random seed
    R, t : symmetry operation (refitted with fit_symmetry_op if omitted)
    normal_axis : GB normal axis (0 for Sigma5_210)

    Returns
    -------
    new_atoms : ase.Atoms (after deleting 2k atoms)
    info : dict {removed_sites, natoms, op_residual_A}

    Raises
    ------
    RemovalSkip : when the fit fails, no image is found (retries exhausted), the retries for a self mapping are exhausted, or
                  the audit residual after the removal is >= 1e-3 A.
    """
    if R is None or t is None:
        R, t, op_residual = fit_symmetry_op(atoms, normal_axis=normal_axis)
    else:
        op_residual = None

    if k == 0:
        info = {"removed_sites": 0, "natoms": len(atoms), "op_residual_A": op_residual}
        return atoms.copy(), info

    cell = np.array(atoms.get_cell())
    frac_all = np.array(atoms.get_scaled_positions(wrap=True))
    n = len(atoms)

    core_mask = (frac_all[:, 0] > 0.5 - DELTA) & (frac_all[:, 0] < 0.5 + DELTA)
    C2 = np.where(core_mask)[0]
    if len(C2) < k:
        raise RemovalSkip(f"GB2 core has only {len(C2)} atoms, need k={k}")

    rng = np.random.default_rng(seed)
    # draw from C2 without replacement. A larger candidate pool is prepared and consumed sequentially to allow for re-selection after a self mapping.
    pool = list(C2)
    rng.shuffle(pool)
    pool_iter = iter(pool)

    remove_ids: set[int] = set()
    n_selected = 0
    max_pool_exhaust_retries = len(C2) * 3 + 10  # safety valve against re-consuming the pool at a self mapping

    def next_candidate():
        nonlocal pool_iter
        try:
            return next(pool_iter)
        except StopIteration:
            return None

    exhaust_guard = 0
    while n_selected < k:
        exhaust_guard += 1
        if exhaust_guard > max_pool_exhaust_retries:
            raise RemovalSkip("exhausted candidate pool while avoiding self-mapping")

        sel = next_candidate()
        if sel is None:
            raise RemovalSkip(f"GB2 core pool exhausted before selecting k={k} pairs "
                               f"(selected {n_selected})")
        if sel in remove_ids:
            continue

        found_image = None
        for _attempt in range(MAX_RETRY_PER_SITE):
            moved_frac = _apply_op(frac_all[sel:sel + 1], R, t)[0]
            dists = _mic_cart_distance(cell, moved_frac, frac_all)
            # atoms that are already deletion targets are excluded from the image candidates (prevents double assignment)
            dists_masked = dists.copy()
            for rid in remove_ids:
                dists_masked[rid] = np.inf
            img = int(np.argmin(dists_masked))
            dmin = float(dists_masked[img])

            if dmin >= IMAGE_TOL_A:
                # if no image is found -> redo this selection
                sel_retry = next_candidate()
                if sel_retry is None:
                    raise RemovalSkip("no image found and candidate pool exhausted")
                sel = sel_retry
                if sel in remove_ids:
                    continue
                continue

            if img == sel:
                # image = the selected atom itself (self mapping on the symmetry plane) -> choose again
                sel_retry = next_candidate()
                if sel_retry is None:
                    raise RemovalSkip("self-mapping and candidate pool exhausted")
                sel = sel_retry
                if sel in remove_ids:
                    continue
                continue

            found_image = img
            break

        if found_image is None:
            # MAX_RETRY_PER_SITE exhausted -> skip this sample
            raise RemovalSkip(f"exceeded {MAX_RETRY_PER_SITE} retries finding valid image")

        remove_ids.add(sel)
        remove_ids.add(found_image)
        n_selected += 1

    new_atoms = atoms.copy()
    keep_mask = np.ones(n, dtype=bool)
    for rid in remove_ids:
        keep_mask[rid] = False
    del new_atoms[[i for i in range(n) if not keep_mask[i]]]

    # as-built symmetry audit after the removal (confirm a residual < 1e-3 A with the symmetry audit).
    # Speed-up [author's choice]: first check quickly with force_r_name (one of the 8 candidates),
    # and if the criterion is met it is OK (a sufficient condition for "some of the candidates meets the criterion").
    # Only if it is not met, fall back to the full search over the 8 candidates. The decision is identical to the full search over the 8 candidates.
    res_fast = audit_symmetry(new_atoms, normal_axis=normal_axis, use_v2_candidates=True,
                               force_r_name=FORCE_R_NAME)
    if res_fast["max_residual_A"] < ASBUILT_RESIDUAL_TOL_A:
        residual_after = res_fast["max_residual_A"]
    else:
        res_after = audit_symmetry(new_atoms, normal_axis=normal_axis, use_v2_candidates=True)
        residual_after = res_after["max_residual_A"]
    if not (residual_after < ASBUILT_RESIDUAL_TOL_A):
        raise RemovalSkip(f"post-removal symmetry residual too large: {residual_after:.3e}A "
                           f"(tol={ASBUILT_RESIDUAL_TOL_A:.1e}A)")

    info = {
        "removed_sites": len(remove_ids),
        "natoms": len(new_atoms),
        "op_residual_A": op_residual,
        "asbuilt_residual_A": residual_after,
    }
    return new_atoms, info
