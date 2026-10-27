"""Variable-length, padding-free ``torch.utils.data.Dataset`` for the proposed tilt-STGB
Sigma5(210)[001] data (internal notes).

**Geometry/correspondence layer is REUSED, not re-derived **: role assignment
(``assign_roles``), the PRIMARY<->REPLICA correspondence fit (``fit_projection``), and the canonical
sub_lo->sub_hi PRIMARY order (``_primary_replica_order``) are exactly
``mattergen.symmetry_constrained.layout.build_canonical_layout``'s own machinery -- this module calls that
function directly (its ``manifest`` argument is unused by the function body, verified by grep at
implementation time; this module passes ``None``, see :func:`_UNUSED_MANIFEST_ARG` docstring
below) and only reads off its ``source_*_local_idx``/``effective_gb_t_yz``/``n_real_primary``
outputs, discarding the N_PAD=128/padding-padded ``pos``/``atomic_numbers``/``gb_role`` fields that
function also builds (this module builds its OWN padding-free, variable-length equivalents of
those three from the same ``source_*_local_idx`` arrays).

**REPLICA correspondence is RE-DERIVED from the applied gauge, not read off
``layout.source_replica_local_idx`` directly**: the fitted
per-sub-block correspondence table (``mattergen.symmetry_constrained.manifest.fit_projection``'s
``primary_to_replica``, which underlies ``layout.source_replica_local_idx``) is confirmed correct for
sub-block A but is fit against sub-block B's OWN ``b0`` -- for 103/2,136 (4.82%) rows where
sub-block A's and B's fitted in-plane gauge differ by exactly the (0.5, 0.5) fractional
degeneracy, applying the single graph-level ``gb_t`` (= sub-block A's gauge, ``layout.
effective_gb_t_yz``) to ALL PRIMARY atoms and comparing against that stale sub-block-B-gauge
correspondence produces a 4.4498 A per-atom residual on the sub-block-B half of REPLICA (
"what was happening"), even though the reconstructed ATOM SET is still exactly correct
(same 1.7e-14 A precision) -- only the atom-to-atom PAIRING for that half is wrong.
:func:`_derive_replica_correspondence` (below) re-derives the pairing by applying the single
graph-level gauge and finding each reconstructed REPLICA position's periodic nearest neighbor
within ``layout.source_replica_local_idx``'s ATOM SET (which is correct as a set regardless of the
gauge mismatch), then verifies **(main judgment, integer)** that mapping is a
bijection and that it agrees with the independent Hungarian-optimal assignment on the same
distance matrix -- these two integer checks together make the derived pairing unambiguous (
"the correspondence is uniquely determined"), not merely "close enough". The residual itself is checked only
against a loose **sanity-only, floating-point** upper bound (``GAUGE_CORRESPONDENCE_MAX_RESIDUAL_A``
= 1e-6 A r2 -- NOT the decision factor, a hard 1e-12 A threshold in
the r1 version of this function had raised on 9/2,136 rows that were in fact perfectly bijective
and Hungarian-agreeing, just with ordinary ``fit_projection`` refinement noise up to 2.6e-10 A).
This shared bijection/Hungarian/sanity logic lives in
:func:`_derive_correspondence_from_reconstruction` and is also used by
:func:`_derive_fixed_b_correspondence` for the independently
discovered FIXED_A<->FIXED_B analogue of this same gauge/correspondence mismatch. It reuses
``mattergen.symmetry_constrained.symmetry_ops.reconstruct_replica_pos`` (the SAME formula the production
projector runs, -- not re-derived) and ``mattergen.symmetry_constrained.manifest._pairwise_mic_dist``/
``_hungarian_match`` (the SAME nearest-neighbor/Hungarian primitives already used by
``fit_projection`` itself and by this project's own diagnosis scripts, ``experiments/earlier_experiment
variable_length_t5b/scripts/d2_per_atom_vs_setwise_residual.py``).

**PRIMARY<->REPLICA species check is done on the op_A correspondence **:
``build_canonical_layout``'s own check d (``layout._validate_species_frac``) pairs PRIMARY and REPLICA with
the per-sub-block-gauge ``replica_order``, whereas this loader orders its output with the single-gauge op_A
correspondence (:func:`_derive_replica_correspondence`). For a gauge-degenerate ("degenerate") parent -- sub-blocks A and B
differ by the (0, 1/2, 1/2) gauge -- the two correspondences pair the same atom SET differently, so a solute
decoration placed correctly on the op_A correspondence failed layout's check d (4,050 as-built solute decorations in
), and, conversely, a structure that passes check d but not op_A would be emitted with a broken species
pairing. This loader therefore calls ``build_canonical_layout(..., check_primary_replica_species=False)`` (checks a-c
still run) and checks ``species[m_idx] == species[s_idx]`` on the SAME ``s_idx`` that builds the output order
(``LayoutError`` naming "op_A" on mismatch). The output order and the correspondence functions are unchanged,
so tensors of rows that were readable before are bit-identical.

**Split is READ, never recomputed here ("do not cut the split yourself")**: the
train/val/test assignment comes from the shared assignment file
(:mod:`mattergen.symmetry_constrained.variable_length.dataset`'s ``split_assignment_csv_path`` constructor argument,
concretely a ``split_assignment.csv`` file) -- this module never shuffles or re-splits. A ``material_id`` present in the
split assignment file but absent from ``gamma_csv_path`` is an error (raised immediately -- never silently skipped).

**Lattice template axes are read from data, never hardcoded **: each row's
``a_template``/``b_template``/``c_template`` come from its CNID's dx=0 manifest JSON's ``cell``
field, cached per-cnid. This module never imports ``mattergen.symmetry_constrained.projection.A_TEMPLATE``/
``B_TEMPLATE``/``C_TEMPLATE`` (those are Al's values). ** (2026-10-01)**: the
manifest file is ``{manifest_dir}/{cnid}_dx_p0.000.json`` (dx-tagged name) OR ``{manifest_dir}/{cnid}.json``
(Al); exactly one of the two must exist (neither -> FileNotFoundError, both -> ValueError,).

**Two split forms **:
  * ``split_assignment_csv_path`` (assignment-CSV form, default): ``split_assignment.csv`` form, rows are looked
    up by ``(cnid, k, seed)`` (one row per parent). ``split`` in {train, val, test}.
  * ``split_indices_json_path`` (Al): a JSON with ``train_indices``/``val_indices``/
    ``val_dx_indices`` = integer positions into the ``converged``-filtered row list of
    ``gamma_csv_path`` (the same space ``mattergen.symmetry_constrained.dataset.TiltGBDataset(indices=...)`` uses).
    ``split`` in {train, val, val_dx, test }. Rows are looked up by CSV row POSITION, because Al parents
    ``(cnid, k, seed)`` carry several dx-sweep rows each (the key is not unique). The bulk membership
    and the FIXED_B/REPLICA correspondence are still keyed by the PARENT ``(cnid, k, seed)``.
Exactly one of the two must be given.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.symmetry_constrained import symmetry_ops
from mattergen.symmetry_constrained.layout import (
    LayoutError,
    build_canonical_layout,
    load_bulk_membership,
    load_gamma_csv,
    read_poscar_frac_species,
)
from mattergen.symmetry_constrained.manifest import _hungarian_match, _pairwise_mic_dist
from mattergen.symmetry_constrained.projection import _B0_CONST, _R_REPLICA_MATRIX
from mattergen.symmetry_constrained.variable_length.layout import (
    GB_BLOCK_FIXED_A,
    GB_BLOCK_FIXED_B,
    GB_BLOCK_PRIMARY,
    GB_BLOCK_REPLICA,
    derive_gb_role,
)

# Implementation convention 5 (revision r2): sanity-only upper limit (A) on the maximum of the nearest-neighbour distances.
# **This is not the deciding factor of the verdict** (where floating-point rounding can decide the direction of the verdict, a strict comparison
# is not used as the deciding factor). The main verdicts are convention 3 (bijection) and convention 4 (nearest = Hungarian, both integer invariants).
# 1e-6 A is placed 6 orders of magnitude below the nearest-neighbour interatomic distance of about 2.55 A of the other system, and 4 orders above the worst measured value of r1, 2.60e-10 A (
# errors in part1_full_population_regulation_check of the report json of an earlier run,
# an ordinary numerical error from the N_REFINE_ITERS=8 refinement iterations of fit_projection, not a (0.5,0.5) degeneracy);
# it is a loose upper limit for detecting that the order of magnitude has broken down.
# The absolute threshold 1e-12 A of the old r1 was a number set without measurement, and wrongly raised an exception for 9 of 2,136 rows
# (those rows satisfied both the bijection and the Hungarian agreement and were not degenerate).
GAUGE_CORRESPONDENCE_MAX_RESIDUAL_A = 1e-6


def _derive_correspondence_from_reconstruction(
    recon_frac: np.ndarray,
    true_idx: np.ndarray,
    frac: np.ndarray,
    cell: np.ndarray,
    *,
    max_residual_A: float,
    context: str,
    label: str,
) -> tuple[np.ndarray, dict]:
    """Common logic that applies implementation conventions 3-5 (revision r2) to the already reconstructed positions ``recon_frac`` (``(m,3)``
    fractional, wrapped into ``[0,1)``) and the true atom set ``true_idx`` (POSCAR-local index, ``(m,)``).
    **Called from both the REPLICA and the FIXED_B paths** (the bijection, the Hungarian agreement and the sanity check are not copied to two places.
    "apply conventions 1 to 6 in the same form to the correspondence FIXED_A -> FIXED_B").
    (see the above).

    Returns ``(matched_idx, diag)`` where ``matched_idx[i]`` is ``true_idx``'s entry paired with
    the ``i``-th row of ``recon_frac`` (i.e. with whichever source atom produced
    ``recon_frac[i]``), and ``diag`` carries the validation quantities (``max_nn_residual_A``,
    ``n_atoms``, ``n_distinct_images``, ``hungarian_agrees``) for reporting.

    Raises ``ValueError`` (silent fallback forbidden,) if any of:
      3. [main verdict, integer] the nearest-neighbor mapping is not a bijection (distinct image count
         != ``m``)
      4. [main verdict, integer] the nearest-neighbor mapping disagrees with the Hungarian-optimal
         assignment on the same distance matrix
      5. [sanity only, floating point] the max nearest-neighbor residual exceeds ``max_residual_A``
         (not a decision factor -- a loose upper bound to detect a broken-digit regression,
         )
    """
    m = int(recon_frac.shape[0])
    if int(true_idx.shape[0]) != m:
        raise ValueError(
            f"_derive_correspondence_from_reconstruction{context} [{label}]: reconstructed count "
            f"{m} != true_idx count {true_idx.shape[0]}"
        )

    true_frac = np.asarray(frac, dtype=np.float64)[true_idx]  # (m, 3)
    cell64 = np.asarray(cell, dtype=np.float64)
    dist_mat = _pairwise_mic_dist(cell64, recon_frac, true_frac)

    nn_col = dist_mat.argmin(axis=1)
    nn_dist = dist_mat[np.arange(m), nn_col]

    # Convention 3 [main verdict, integer]: bijection check.
    n_distinct = len(set(nn_col.tolist()))
    if n_distinct != m:
        raise ValueError(
            f"_derive_correspondence_from_reconstruction{context} [{label}]: nearest-neighbor "
            f"mapping is not a bijection ({n_distinct}/{m} distinct images)."
        )

    # Convention 4 [main verdict, integer]: agreement with the independent Hungarian-optimal assignment.
    row_h, col_h, _residuals_h = _hungarian_match(cell64, recon_frac, true_frac)
    if row_h is None:
        raise ValueError(
            f"_derive_correspondence_from_reconstruction{context} [{label}]: Hungarian match "
            "failed (atom count mismatch)."
        )
    hungarian_col = np.empty(m, dtype=np.int64)
    hungarian_col[row_h] = col_h
    hungarian_agrees = bool(np.array_equal(nn_col, hungarian_col))
    if not hungarian_agrees:
        mismatch = np.nonzero(nn_col != hungarian_col)[0]
        raise ValueError(
            f"_derive_correspondence_from_reconstruction{context} [{label}]: nearest-neighbor "
            f"mapping disagrees with the Hungarian-optimal assignment at row(s) "
            f"{mismatch.tolist()}."
        )

    # Convention 5 [sanity only, floating point]: residual upper bound (not the decision factor).
    max_dist = float(nn_dist.max())
    if max_dist > max_residual_A:
        raise ValueError(
            f"_derive_correspondence_from_reconstruction{context} [{label}]: max nearest-neighbor "
            f"residual {max_dist:.6e} A exceeds sanity threshold {max_residual_A:.1e} A "
            "(digits likely broken -- this is a sanity bound, not the decision factor)."
        )

    diag = {
        "max_nn_residual_A": max_dist,
        "n_atoms": m,
        "n_distinct_images": n_distinct,
        "hungarian_agrees": hungarian_agrees,
    }
    return np.asarray(true_idx)[nn_col], diag


def _derive_replica_correspondence(
    frac: np.ndarray,
    cell: np.ndarray,
    primary_order: np.ndarray,
    true_replica_idx: np.ndarray,
    gb_t_yz: np.ndarray,
    *,
    max_residual_A: float = GAUGE_CORRESPONDENCE_MAX_RESIDUAL_A,
    context: str = "",
) -> tuple[np.ndarray, dict]:
    """Implementation conventions 1-6 (option A): apply the single graph-level ``gb_t`` (``gb_t_yz`` -- the
    caller passes ``layout.effective_gb_t_yz``, sub-block A's gauge point 1) to
    every PRIMARY atom in ``primary_order`` via ``symmetry_ops.reconstruct_replica_pos`` (the SAME
    formula the production projector runs -- not re-derived here), then hands the
    reconstructed positions to :func:`_derive_correspondence_from_reconstruction` to find each
    one's periodic (MIC) nearest neighbor within the TRUE REPLICA atom SET ``true_replica_idx`` (the
    caller passes ``layout.source_replica_local_idx`` -- correct as a SET even for the 103/2,136
    sub-block-gauge-degenerate rows where its per-sub-block ORDER is wrong, "the atom set
    is exactly reproduced in both gauges"). Returns ``(replica_order, diag)`` where ``replica_order[i]``
    is ``true_replica_idx``'s entry paired with ``primary_order[i]``, and ``diag`` carries the validation
    quantities (``max_nn_residual_A``, ``n_primary`` (alias of ``n_atoms``), ``n_distinct_images``,
    ``hungarian_agrees``) for reporting.
    """
    m = int(primary_order.shape[0])

    primary_frac_t = torch.as_tensor(frac[primary_order], dtype=torch.float64).unsqueeze(0)
    gb_t_ip_t = torch.as_tensor(np.asarray(gb_t_yz, dtype=np.float64)).unsqueeze(0)
    rotation_t = torch.tensor(_R_REPLICA_MATRIX, dtype=torch.float64)
    b0_t = torch.tensor(_B0_CONST, dtype=torch.float64)
    recon_replica_frac = (
        symmetry_ops.reconstruct_replica_pos(primary_frac_t, gb_t_ip_t, rotation=rotation_t, b0=b0_t)[0]
        .numpy()
    )  # (m, 3) fractional, wrapped into [0, 1)

    matched_idx, diag = _derive_correspondence_from_reconstruction(
        recon_replica_frac, true_replica_idx, frac, cell,
        max_residual_A=max_residual_A,
        context=context,
        label="REPLICA",
    )
    diag["n_primary"] = diag["n_atoms"]  # back-compat alias (diagnosis scripts key on n_primary)
    assert diag["n_atoms"] == m
    return matched_idx, diag


def _derive_fixed_b_correspondence(
    frac: np.ndarray,
    cell: np.ndarray,
    fixed_a_order: np.ndarray,
    true_fixed_b_idx: np.ndarray,
    gb_t_yz: np.ndarray,
    gb_dx: float,
    a_template: float,
    b_template: float,
    c_template: float,
    *,
    max_residual_A: float = GAUGE_CORRESPONDENCE_MAX_RESIDUAL_A,
    context: str = "",
) -> tuple[np.ndarray, dict]:
    """ r2 【K6】: apply the SAME single graph-level ``gb_t``/``gb_dx`` gauge to every
    FIXED_A atom in ``fixed_a_order`` via ``symmetry_ops.reconstruct_fixed_b_cart`` (the SAME
    formula the production projector runs -- not re-derived here), convert to
    fractional coordinates via this row's own cell, then hand the result to
    :func:`_derive_correspondence_from_reconstruction` to re-derive the FIXED_A<->FIXED_B
    correspondence the same way :func:`_derive_replica_correspondence` re-derives PRIMARY<->REPLICA
    ("apply conventions 1 to 6 in the same form to the correspondence FIXED_A -> FIXED_B").

    ``true_fixed_b_idx`` (the caller passes ``layout.source_fixed_b_local_idx``, i.e.
    ``bulk_membership.csv``'s ``bulkB_indices``) is used only as a SET here -- its per-row
    ORDER (template-slot order r2 【K6】 root cause: fit via ``audit_symmetry``'s
    ``best_t``, a different fit than the ``gb_t`` applied here) is exactly what this function
    re-derives, not what it trusts.

    Returns ``(fixed_b_order, diag)`` where ``fixed_b_order[i]`` is ``true_fixed_b_idx``'s entry
    paired with ``fixed_a_order[i]``, and ``diag`` carries the validation quantities
    (``max_nn_residual_A``, ``n_atoms``, ``n_distinct_images``, ``hungarian_agrees``).
    """
    fixed_a_cart = frac[fixed_a_order] @ np.asarray(cell, dtype=np.float64)  # (n_bulk, 3) Cartesian
    fixed_a_cart_t = torch.as_tensor(fixed_a_cart, dtype=torch.float64).unsqueeze(0)
    gb_t_ip_t = torch.as_tensor(np.asarray(gb_t_yz, dtype=np.float64)).unsqueeze(0)
    gb_dx_t = torch.tensor([float(gb_dx)], dtype=torch.float64)
    rotation_t = torch.tensor(_R_REPLICA_MATRIX, dtype=torch.float64)

    recon_fixed_b_cart = symmetry_ops.reconstruct_fixed_b_cart(
        fixed_a_cart_t, gb_t_ip_t, gb_dx_t,
        rotation=rotation_t,
        a_template=float(a_template), b_template=float(b_template), c_template=float(c_template),
    )[0].numpy()  # (n_bulk, 3) Cartesian, NOT yet wrapped

    cell64 = np.asarray(cell, dtype=np.float64)
    recon_fixed_b_frac = np.mod(recon_fixed_b_cart @ np.linalg.inv(cell64), 1.0)

    return _derive_correspondence_from_reconstruction(
        recon_fixed_b_frac, true_fixed_b_idx, frac, cell,
        max_residual_A=max_residual_A,
        context=context,
        label="FIXED_B",
    )


GB_GAMMA_PROPERTY_KEY = "gb_gamma_J_m2"  # same constant name as mattergen.symmetry_constrained.layout's
# GB_GAMMA_PROPERTY_KEY ("use gb_gamma_J_m2 registered in PROPERTY_SOURCE_IDS as it is. Do not invent a new property name"). Not imported from symmetry_constrained.layout to avoid pulling
# in the whole module.
# in that module's own GB_GAMMA_PROPERTY_KEY-adjacent PROPERTY_SOURCE_IDS assertion at this
# module's import time (this module has no need for that assertion; symmetry_constrained.dataset already runs
# it whenever symmetry_constrained is imported, which it always is here via symmetry_constrained.layout above).

GB_T_KEY = "gb_t"
GB_DX_KEY = "gb_dx"
GB_FIXED_A_TEMPLATE_KEY = "gb_fixed_a_template"
GB_FIXED_B_TEMPLATE_KEY = "gb_fixed_b_template"
GB_A_TEMPLATE_KEY = "gb_a_template"
GB_B_TEMPLATE_KEY = "gb_b_template"
GB_C_TEMPLATE_KEY = "gb_c_template"

_DX0_TAG = "dx_p0.000"  #: template axes are the dx=0 base value (gb_dx is the
# per-row OFFSET from this base, mirroring mattergen.symmetry_constrained.projection.py's
# gb_dx = (cell[0,0,0] - A_TEMPLATE) / 2.0, just data-sourced instead of a module constant).


def resolve_view(path: str) -> str:
    """Absorb the login-node-vs-compute-node mount-prefix discrepancy (``<CLUSTER_MOUNT>`` vs
    ``<CLUSTER_MOUNT>``) for an EXISTING path. Same idiom as
    a data-preparation script's ``resolve_view`` (and its several precedents in this codebase) -- not imported (that script lives
    under ``runs/``, not an importable package) but the same ~10-line idiom, not a re-derivation of
    any non-trivial logic (targets duplicated FORMULAS/algorithms, not this kind of
    environment-glue snippet, which is reproduced verbatim across many scripts in this repo by
    established convention).
    """
    candidates = [path]
    if path.startswith('/mnt/CLUSTER_ROOT_A'):
        candidates.append(path.replace('/mnt/CLUSTER_ROOT_A', '/CLUSTER_ROOT_B', 1))
    elif path.startswith('/CLUSTER_ROOT_B'):
        candidates.append("/mnt" + path)
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"path not found in any view: {path} (tried {candidates})")


@dataclass
class SplitAssignmentRow:
    material_id: str
    cnid: str
    k: int
    seed: int
    dx_A: float
    split: str


def load_split_assignment(csv_path: str | Path, split: str) -> list[SplitAssignmentRow]:
    """Read the shared split-assignment CSV (
    ``split_assignment.csv``, columns ``material_id,cnid,k,seed,dx_A,split``) and return only
    the rows for ``split``. Raises if a row's ``material_id`` does not match
    ``f"{cnid}_k{k}_s{seed}"`` (point 3 format; a format
    inconsistency in the shared assignment file must halt loudly, not be silently trusted)."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"load_split_assignment: split={split!r} not one of train/val/test")
    out: list[SplitAssignmentRow] = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            cnid, k, seed = r["cnid"], int(r["k"]), int(r["seed"])
            material_id = r["material_id"]
            expected_id = f"{cnid}_k{k}_s{seed}"
            if material_id != expected_id:
                raise ValueError(
                    f"load_split_assignment: material_id={material_id!r} != expected "
                    f"{expected_id!r} (cnid/k/seed-derived) -- split assignment file format "
                    "inconsistency."
                )
            out.append(
                SplitAssignmentRow(
                    material_id=material_id, cnid=cnid, k=k, seed=seed,
                    dx_A=float(r["dx_A"]), split=r["split"],
                )
            )
    return out


#: "test" added (``split.json`` carries ``test_indices``). Append-only: a JSON that
# lacks ``test_indices`` still loads, because only the
# REQUESTED split's key must be present; the other keys are checked only if they exist.
SPLIT_INDEX_KEYS = {
    "train": "train_indices", "val": "val_indices", "val_dx": "val_dx_indices", "test": "test_indices",
}


def load_split_indices(json_path: str | Path, split: str) -> list[int]:
    """: read the row-position split JSON (``indices_audit_dedup.json`` form:
    ``train_indices``/``val_indices``/``val_dx_indices`` and, since, ``test_indices``; optional
    ``n_train``/``n_val``/``n_val_dx``/``n_test``)
    and return the list of integer positions for ``split`` (order preserved). Only the requested split's key must
    exist; the keys that do exist are checked against each other. Raises  on:
    unknown split / missing key / non-integer or negative entry / duplicates within the split /
    overlap between the splits present in the file / a declared ``n_*`` that disagrees with the list."""
    if split not in SPLIT_INDEX_KEYS:
        raise ValueError(f"load_split_indices: split={split!r} not one of {sorted(SPLIT_INDEX_KEYS)}")
    with open(json_path) as f:
        payload = json.load(f)
    seen: dict[str, set[int]] = {}
    for sp, key in SPLIT_INDEX_KEYS.items():
        if key not in payload:
            if sp == split:
                raise ValueError(f"load_split_indices: key {key!r} missing in {json_path}")
            continue
        vals = payload[key]
        if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in vals):
            raise ValueError(f"load_split_indices: {key!r} in {json_path} has a non-integer or negative entry")
        if len(set(vals)) != len(vals):
            raise ValueError(f"load_split_indices: {key!r} in {json_path} has duplicate positions")
        declared = payload.get("n_" + sp)
        if declared is not None and int(declared) != len(vals):
            raise ValueError(f"load_split_indices: n_{sp}={declared} != len({key})={len(vals)} in {json_path}")
        seen[sp] = set(vals)
    names = sorted(seen)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if seen[a] & seen[b]:
                raise ValueError(f"load_split_indices: splits {a!r} and {b!r} overlap in {json_path}")
    return list(payload[SPLIT_INDEX_KEYS[split]])


class SymmetricTiltGBDataset(Dataset):
    """One padding-free, variable-length ChemGraph per (cnid,k,seed) proposed training structure
    . Node layout is ``[FIXED_A(38) | FIXED_B(38) | PRIMARY(m) | REPLICA(m)]``
    (``gb_block``/``gb_pair_rank`` carry block membership and within-block correspondence order,
    -- NOT array position; the concatenation order here is a convenience, not a
    correctness requirement, mirroring the dataset of another package, whose own
    ``order = np.concatenate((fixed, primary, replica))`` convention, "reference only" shape).
    """

    def __init__(
        self,
        gamma_csv_path: str | Path,
        bulk_membership_csv_path: str | Path,
        manifest_dir: str | Path,
        split_assignment_csv_path: str | Path | None = None,
        split: str | None = None,
        expected_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        split_indices_json_path: str | Path | None = None,
    ):
        """``dtype`` (default ``torch.float32``, matching the training-time convention of
        ``mattergen.symmetry_constrained.dataset.TiltGBDataset``): floating-point dtype for every tensor field
        this dataset constructs (``pos``/``cell``/``gb_t``/``gb_dx``/``gb_fixed_a_template``/
        ``gb_fixed_b_template``/``gb_a_template``/``gb_b_template``/``gb_c_template``/the gamma
        property). Long/int tensors (``atomic_numbers``, ``gb_block``, ``gb_pair_rank``,
        ``gb_role``, ``num_atoms``) are unaffected.

        Exists so that the blanket rule "run the validation with device=cpu and dtype=float64 ..."
        requirement (check-2/check-4/check-5) can be honored WITHOUT first truncating the
        underlying ``numpy`` (float64-by-default) geometry to float32 and only THEN upcasting a
        ``ChemGraph`` built from it (which would silently bake in float32 quantization noise
        ~1e-6..1e-7 before the cast ever runs -- a validation methodology bug this project hit and
        fixed at implementation time, not a change to the training-time default: the default
        stays ``torch.float32``).

        ``split_assignment_csv_path`` / ``split_indices_json_path``: exactly one
        must be given. The former (default) is the ``split_assignment.csv`` form
        (``split`` in train/val/test, rows looked up by ``(cnid,k,seed)``); the latter (Al) is a
        JSON of row positions (``split`` in train/val/val_dx/test, rows looked up by position in the
        ``converged``-filtered ``gamma_csv_path`` rows -- see the module docstring). In the JSON form
        ``self.row_positions[i]`` is the position of ``self.rows[i]`` (``None`` in the CSV form)."""
        if split is None:
            raise ValueError("SymmetricTiltGBDataset: 'split' is required")
        if (split_assignment_csv_path is None) == (split_indices_json_path is None):
            raise ValueError(
                "SymmetricTiltGBDataset: exactly one of split_assignment_csv_path (assignment-CSV form) and "
                "split_indices_json_path (Al form) must be given "
                f"(got csv={split_assignment_csv_path!r}, json={split_indices_json_path!r})"
            )
        self.dtype = dtype
        self.row_positions: list[int] | None = None

        if split_indices_json_path is not None:
            # Al form: look rows up by position in the converged-filtered CSV row list.
            positions = load_split_indices(split_indices_json_path, split)
            all_rows = [r for r in load_gamma_csv(gamma_csv_path) if r.converged]
            for p in positions:
                if p >= len(all_rows):
                    raise ValueError(
                        f"SymmetricTiltGBDataset: split index {p} out of range for {len(all_rows)} "
                        f"converged rows in {gamma_csv_path} (split={split!r}, "
                        f"{split_indices_json_path}) --."
                    )
            self.rows = [all_rows[p] for p in positions]
            self.row_positions = list(positions)
            self.split_form = "indices_json"
        else:
            split_rows = load_split_assignment(split_assignment_csv_path, split)

            gamma_by_key = {
                (r.cnid, r.k, r.seed): r for r in load_gamma_csv(gamma_csv_path) if r.converged
            }
            self.rows = []
            for sr in split_rows:
                key = (sr.cnid, sr.k, sr.seed)
                if key not in gamma_by_key:
                    raise ValueError(
                        f"SymmetricTiltGBDataset: split assignment row material_id={sr.material_id!r} "
                        f"(key={key}) has no corresponding converged row in {gamma_csv_path} "
                        "(silent fallback forbidden --)."
                    )
                self.rows.append(gamma_by_key[key])
            self.split_form = "assignment_csv"

        if expected_size is not None and len(self.rows) != expected_size:
            raise ValueError(
                f"SymmetricTiltGBDataset: expected {expected_size} structures for split={split!r}, "
                f"found {len(self.rows)}"
            )

        self.membership = load_bulk_membership(bulk_membership_csv_path)
        self.manifest_dir = Path(manifest_dir)
        self.split = split
        self._template_axes_cache: dict[str, tuple[float, float, float]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _template_axes(self, cnid: str) -> tuple[float, float, float]:
        """ "K2": a/b/c_template read from THIS cnid's dx=0 manifest cell, never a
        module constant. Cached per-cnid (the value is identical across all 10 CNID templates,
        , but this module never assumes that -- it looks each cnid up independently)."""
        if cnid not in self._template_axes_cache:
            #: The dx=0 manifest is named "{cnid}_dx_p0.000.json" or "{cnid}.json".
            # Exactly one must exist: neither -> FileNotFoundError, both -> ValueError (ambiguous
            # which one to read; never a silent preference).
            cand_tagged = self.manifest_dir / f"{cnid}_{_DX0_TAG}.json"
            cand_al = self.manifest_dir / f"{cnid}.json"
            exists = [p for p in (cand_tagged, cand_al) if p.is_file()]
            if len(exists) == 0:
                raise FileNotFoundError(
                    f"SymmetricTiltGBDataset: no template manifest for cnid={cnid!r} in {self.manifest_dir} "
                    f"(tried {cand_tagged.name} and {cand_al.name})"
                )
            if len(exists) == 2:
                raise ValueError(
                    f"SymmetricTiltGBDataset: both {cand_tagged.name} and {cand_al.name} exist in "
                    f"{self.manifest_dir} -- ambiguous which holds the dx=0 template cell"
                )
            path = exists[0]
            with open(path) as f:
                payload = json.load(f)
            cell = payload["cell"]
            self._template_axes_cache[cnid] = (
                float(cell[0][0]), float(cell[1][1]), float(cell[2][2]),
            )
        return self._template_axes_cache[cnid]

    def __getitem__(self, index: int) -> ChemGraph:
        row = self.rows[index]
        key = (row.cnid, row.k, row.seed)
        if key not in self.membership:
            raise ValueError(f"SymmetricTiltGBDataset: no bulk_membership row for {key}")
        membership = self.membership[key]

        poscar_path = Path(resolve_view(str(Path(row.dir) / "POSCAR_relaxed")))
        frac, cell, species = read_poscar_frac_species(poscar_path)

        # a_t/b_t/c_t/gb_dx_val computed here (moved up from their original later position in this
        # method) because the FIXED_B re-derivation below needs them --
        # NOT recomputed a second time later in this method.
        a_t, b_t, c_t = self._template_axes(row.cnid)
        gb_dx_val = (float(cell[0, 0]) - a_t) / 2.0

        # Role assignment + PRIMARY<->REPLICA correspondence: REUSE build_canonical_layout entirely
        # (module docstring above). `manifest` is unused by the function body (verified by grep at
        # implementation time -- "a case whose manifest lacks gb_t_effective_frac_yz also raises an exception without filling in a default value": passing None here never triggers a silent default
        # (see the above).
        # for that field because the field is never read for this call path at all).
        rng = np.random.default_rng()
        # L-b: layout's own PRIMARY<->REPLICA species check (d, per-sub-block gauge) is
        # skipped here; the species are checked below on the SAME op_A correspondence (s_idx)
        # that orders the output. Checks a-c (length / allowed Z / FIXED pure host) still run.
        layout = build_canonical_layout(
            frac=frac, cell=cell, membership=membership, manifest=None, rng=rng,
            species_frac=species, check_primary_replica_species=False,
        )

        fa_idx = layout.source_fixed_a_local_idx  # (N_BULK,) POSCAR-local index
        # fb_idx: RE-DERIVED from the single graph-level gauge, not
        # read off layout.source_fixed_b_local_idx directly -- see module docstring and
        # _derive_fixed_b_correspondence. layout.source_fixed_b_local_idx (bulk_membership.csv's
        # bulkB_indices, template-slot order from a DIFFERENT fit -- build_bulk_membership.py's
        # audit_symmetry best_t) is still the correct TRUE FIXED_B atom SET (passed in as
        # true_fixed_b_idx); only its per-row ORDER is stale for the 100/2,136 rows where this
        # gauge-degenerate mismatch occurs (r2 K6 census: experiments/earlier_experiment
        # k6_fixedb/results/k6_fixedb_census.json, set-wise residual machine-precision (max
        # 2.30e-14 A) and fully bijective/Hungarian-agreeing across all 2,136 rows -- confirms this
        # is the SAME ordering-only bug as REPLICA, not a set-reproduction failure).
        fb_idx, _fixed_b_correspondence_diag = _derive_fixed_b_correspondence(
            frac=frac,
            cell=cell,
            fixed_a_order=fa_idx,
            true_fixed_b_idx=layout.source_fixed_b_local_idx,
            gb_t_yz=layout.effective_gb_t_yz,
            gb_dx=gb_dx_val,
            a_template=a_t, b_template=b_t, c_template=c_t,
            context=f" ({row.cnid} k={row.k} seed={row.seed})",
        )
        m_idx = layout.source_primary_local_idx  # (m,) sub_lo->sub_hi order
        # s_idx: RE-DERIVED from the single graph-level gauge ("option A"), not read off
        # layout.source_replica_local_idx directly -- see module docstring and
        # _derive_replica_correspondence. layout.source_replica_local_idx is still the correct TRUE
        # REPLICA atom SET (passed in as true_replica_idx); only its per-sub-block PAIRING is stale
        # for the 103/2,136 gauge-degenerate rows.
        s_idx, _replica_correspondence_diag = _derive_replica_correspondence(
            frac=frac,
            cell=cell,
            primary_order=m_idx,
            true_replica_idx=layout.source_replica_local_idx,
            gb_t_yz=layout.effective_gb_t_yz,
            context=f" ({row.cnid} k={row.k} seed={row.seed})",
        )
        # L-b: PRIMARY<->REPLICA species check on the op_A correspondence (m_idx, s_idx),
        # i.e. the SAME pairing the output order below uses (not layout's per-sub-block replica_order).
        context = f" ({row.cnid} k={row.k} seed={row.seed})"
        ms = species[m_idx]
        ss = species[s_idx]
        bad = np.nonzero(ms != ss)[0]
        if bad.size:
            raise LayoutError(
                f"op_A PRIMARY<->REPLICA species mismatch at slot(s) {bad.tolist()}: "
                f"PRIMARY rows {m_idx[bad].tolist()} species {ms[bad].tolist()} != "
                f"REPLICA rows {s_idx[bad].tolist()} species {ss[bad].tolist()}{context}"
                " ")
        n_bulk = len(fa_idx)
        m = len(m_idx)

        order = np.concatenate([fa_idx, fb_idx, m_idx, s_idx])
        n = int(order.shape[0])

        gb_block = torch.cat(
            [
                torch.full((n_bulk,), GB_BLOCK_FIXED_A, dtype=torch.long),
                torch.full((n_bulk,), GB_BLOCK_FIXED_B, dtype=torch.long),
                torch.full((m,), GB_BLOCK_PRIMARY, dtype=torch.long),
                torch.full((m,), GB_BLOCK_REPLICA, dtype=torch.long),
            ]
        )
        gb_pair_rank = torch.cat(
            [
                torch.arange(n_bulk, dtype=torch.long),
                torch.arange(n_bulk, dtype=torch.long),
                torch.arange(m, dtype=torch.long),
                torch.arange(m, dtype=torch.long),
            ]
        )
        gb_role = derive_gb_role(gb_block)

        dtype = self.dtype
        pos = torch.as_tensor(frac[order], dtype=dtype)
        atomic_numbers = torch.as_tensor(species[order], dtype=torch.long)
        cell_t = torch.as_tensor(cell, dtype=dtype).reshape(1, 3, 3)

        gb_t_yz = torch.as_tensor(layout.effective_gb_t_yz, dtype=dtype)
        gb_t = torch.zeros(1, 3, dtype=dtype)
        gb_t[0, 1:3] = gb_t_yz

        # a_t/b_t/c_t/gb_dx_val: already computed above (before the FIXED_B re-derivation) --
        # NOT recomputed here.
        gb_dx = torch.tensor([[gb_dx_val]], dtype=dtype)  # (1, 1)

        fixed_a_cart = frac[fa_idx] @ cell  # (N_BULK, 3), Cartesian, Angstrom
        fixed_b_cart = frac[fb_idx] @ cell
        gb_fixed_a_template = torch.as_tensor(fixed_a_cart, dtype=dtype).unsqueeze(0)
        gb_fixed_b_template = torch.as_tensor(fixed_b_cart, dtype=dtype).unsqueeze(0)

        gb_a_template = torch.tensor([a_t], dtype=dtype)
        gb_b_template = torch.tensor([b_t], dtype=dtype)
        gb_c_template = torch.tensor([c_t], dtype=dtype)

        gamma = torch.tensor(row.gamma_Jm2, dtype=dtype)

        graph = ChemGraph(
            pos=pos,
            atomic_numbers=atomic_numbers,
            cell=cell_t,
            num_atoms=torch.tensor(n, dtype=torch.long),
            num_nodes=n,
        )
        graph = graph.replace(
            **{
                "gb_block": gb_block,
                "gb_pair_rank": gb_pair_rank,
                "gb_role": gb_role,
                GB_T_KEY: gb_t,
                GB_DX_KEY: gb_dx,
                GB_FIXED_A_TEMPLATE_KEY: gb_fixed_a_template,
                GB_FIXED_B_TEMPLATE_KEY: gb_fixed_b_template,
                GB_A_TEMPLATE_KEY: gb_a_template,
                GB_B_TEMPLATE_KEY: gb_b_template,
                GB_C_TEMPLATE_KEY: gb_c_template,
                GB_GAMMA_PROPERTY_KEY: gamma,
                # assignment-CSV form: unchanged. JSON form: (cnid,k,seed) is
                # not unique across dx-sweep rows, so the CSV row position is prefixed.
                "structure_id": (
                    f"{row.cnid}_k{row.k}_s{row.seed}"
                    if self.row_positions is None
                    else f"r{self.row_positions[index]}_{row.cnid}_k{row.k}_s{row.seed}"
                ),
            }
        )
        return graph
