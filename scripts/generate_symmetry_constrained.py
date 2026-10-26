# Explicit requirement of SCM_HOST_Z/SCM_SOLUTE_ZS (silent fallback is prohibited):
# `mattergen.symmetry_constrained.constants` resolves these two variables at import time (inside the import chain of other modules
# that runs before the `import mattergen...` of this file), and if they are unset it silently falls back to its defaults
# (HOST_Z=13, SOLUTE_ZS={12}). Therefore
# the check must be placed "before any mattergen submodule" -- even if it is placed in main() instead of here,
# `mattergen.symmetry_constrained.*` would by then already be contaminated by the default values that were imported.
from __future__ import annotations

import os
import sys


if "SCM_HOST_Z" not in os.environ:
    print(
        "generate_symmetry_constrained.py: SCM_HOST_Z is not set. "
        "mattergen.symmetry_constrained.constants would silently fall back to its Al default (13) the moment "
        "any mattergen.symmetry_constrained* module is imported below -- refusing rather than silently "
        "generating an Al-configured batch . Set e.g. 'export SCM_HOST_Z=13' "
        "for Al before invoking this script.",
        file=sys.stderr,
    )
    sys.exit(1)
if "SCM_SOLUTE_ZS" not in os.environ:
    print(
        "generate_symmetry_constrained.py: SCM_SOLUTE_ZS is not set. "
        "mattergen.symmetry_constrained.constants would silently fall back to its default solute set ('12') the moment "
        "any mattergen.symmetry_constrained* module is imported below -- refusing rather than silently mixing "
        "in a solute species this run does not intend to use . Set "
        "'export SCM_SOLUTE_ZS=' (empty, for no solute) before invoking this script.",
        file=sys.stderr,
    )
    sys.exit(1)

"""Generation driver of the proposed symmetry-constrained model: variable-length, padding-free,
gamma-conditioned sampling of the Al Sigma5(210)[001] tilt grain boundary.

It reuses the sampler wiring of ``sampling_conf/scm_cond.yaml`` (pos / atomic_numbers / gb_t predictors and the
classifier-free-guidance hooks) and replaces every piece that assumed a fixed 128-slot layout with the
``variable_length`` equivalents. The symmetry operation itself (``mattergen.symmetry_constrained.symmetry_ops``) is
imported, never re-derived, via ``mattergen.symmetry_constrained.variable_length.projection.SymmetryProjector``.

**Generation-time contract.** For a freshly generated graph there is no relaxed POSCAR to read ``gb_block``,
``gb_pair_rank``, the FIXED_A template, ``gb_t`` and ``gb_dx`` from. This module builds them as follows:

  - **FIXED_A** (38 atoms/grain, Cartesian anchor) is READ from a single REFERENCE structure (``REFERENCE_CNID`` =
    ``cnid_0_0``, ``k=0``, the unique converged row without a ``/dx_`` path component) for EVERY generated graph,
    whatever its target atom count ``n``. This is valid because ``N_BULK_PER_GRAIN`` (76 atoms in total, 38/38) is
    constant across all ``k``: the bulk region does not depend on how many MASTER/SLAVE atoms a level has.
  - **MASTER/SLAVE** (``m`` atoms per grain each, ``m = (n - 2*N_BULK_PER_GRAIN) / 2``) are FULLY DIFFUSED (``pos``
    mask = 0). They are seeded with finite fractional coordinates in [0, 1) taken from the reference structure's own
    MASTER/SLAVE atoms (truncated to the first ``m`` in canonical sub_lo -> sub_hi order). These values are
    placeholders only: ``_sample_prior`` overwrites them from the SDE prior before the first denoiser call. Finite
    values (not NaN) are used to avoid a ``0 * NaN`` hazard in mask-multiply code paths.
  - **``gb_pair_rank``** of MASTER/SLAVE is ``arange(m)`` for both blocks (the k-th MASTER <-> k-th SLAVE
    convention); at generation time there is no real SLAVE atom set against which a correspondence could be fitted.
  - **``gb_a_template`` / ``gb_b_template`` / ``gb_c_template``** are read from the ``cell`` of the reference
    structure's dx=0 manifest (``{manifest_dir}/{REFERENCE_CNID}.json``, or the dx-tagged name
    ``{REFERENCE_CNID}_dx_p0.000.json``; exactly one of the two must exist), the same source
    ``SymmetricTiltGBDataset._template_axes`` uses. They are never module constants.
  - **``gb_dx``** is FULLY DIFFUSED: the corruption config of the proposed model
    (``mattergen/conf/lightning_module/diffusion_module/corruption/scm_proposed.yaml``) always includes a
    ``gb_dx: GBTranslationVPSDE`` entry, so the checkpoint expects ``gb_dx`` to be reverse-diffused at generation
    time. A ``gb_dx`` predictor is therefore always injected into the sampler; there is no fixed-dx mode. The
    initial ``gb_dx`` / cell used only to seed the FIXED_A / FIXED_B placeholder coordinates is ``gb_dx = 0.0``;
    the Cartesian anchors of ``symmetry_ops.reconstruct_fixed_b_cart`` do not depend on dx.

**Not done here.** There is no vector / per-structure-target mode (only a scalar ``--gamma_target`` that applies to
the whole batch), no ``--dx`` override, and no padding-reduction step (this architecture has no padding tokens).

**Imports refused on principle.** The fixed-length slot-count constant, the SLAVE array-position slice constant, the
padding sentinel-species constant (all three named in ``mattergen.symmetry_constrained.layout``) and the removed
fixed-length projector are not imported. Only ``mattergen.symmetry_constrained.symmetry_ops`` (layout-agnostic pure
functions), ``mattergen.symmetry_constrained.variable_length.*``, the host-agnostic I/O helpers
``mattergen.symmetry_constrained.layout.{GB_GAMMA_PROPERTY_KEY, build_canonical_layout, load_bulk_membership,
load_gamma_csv, read_poscar_frac_species}`` and the symmetry operation's own constants
``mattergen.symmetry_constrained.projection.{_R_SLAVE_MATRIX, _B0_CONST}`` are used.

Usage (on a GPU node; SCM_HOST_Z and SCM_SOLUTE_ZS must be exported before the call, see the check at the top of the file):
    export SCM_HOST_Z=13
    export SCM_SOLUTE_ZS=
    python scripts/generate_symmetry_constrained.py \\
        --checkpoint <path/to/*.ckpt> --config <path/to/resolved_config.yaml> \\
        --guidance_scale 2.0 --seed 42 --gamma_target 1.0 --num_samples 8 \\
        --output_path <dir> --host-z 13 \\
        --gamma_csv_path <.../gamma_dedup.csv> --bulk_membership_csv_path <.../bulk_membership.csv> \\
        --manifest_dir <.../manifests>
``--host-z`` must equal the exported SCM_HOST_Z (otherwise RuntimeError). The three path arguments have no default
(they depend on where the training data were unpacked). The ``dir`` column of the gamma CSV must contain real paths
(expand ``${GBREL_DATA}`` first, see docs/reproduce.md).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch
from mattergen.common.utils.eval_utils import save_structures
from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.generator import structure_from_model_output
from mattergen.symmetry_constrained import symmetry_ops
from mattergen.symmetry_constrained.constants import SCM_HOST_Z as HOST_Z
from mattergen.symmetry_constrained.constants import N_BULK_PER_GRAIN
from mattergen.symmetry_constrained.layout import (
    GB_GAMMA_PROPERTY_KEY,
    build_canonical_layout,
    load_bulk_membership,
    load_gamma_csv,
    read_poscar_frac_species,
)
from mattergen.symmetry_constrained.projection import _B0_CONST, _R_SLAVE_MATRIX
from mattergen.symmetry_constrained.variable_length.dataset import resolve_view
from mattergen.symmetry_constrained.variable_length.layout import (
    GB_BLOCK_FIELD,
    GB_BLOCK_FIXED_A,
    GB_BLOCK_FIXED_B,
    GB_BLOCK_MASTER,
    GB_BLOCK_SLAVE,
    GB_PAIR_RANK_FIELD,
    assert_graph_invariants,
    derive_gb_role,
)
from mattergen.symmetry_constrained.variable_length.projection import SymmetryProjector
from mattergen.property_embeddings import SetConditionalEmbeddingType, set_conditional_property_values

SAMPLING_CONF_DIR = str(Path(__file__).resolve().parent.parent / "sampling_conf")
# -adjacent task: reused UNCHANGED (not copied/modified) -- the predictor/corrector wiring
# (pos/atomic_numbers/gb_t + CFG remove/keep-conditioning hooks) is architecture-agnostic; it knows
# nothing about the fixed-length slot count or gb_block/gb_pair_rank. Only the sampler's state_projector (set in main(),
# below) and the injected gb_dx predictor (also below) are proposed-specific.
SAMPLING_CONFIG_NAME = "scm_cond"

NUM_ATOMS_DISTRIBUTION_KEY = "AL_S5_210_TRAIN"
REFERENCE_CNID = "cnid_0_0"
REFERENCE_K = 0
_DX0_TAG = "dx_p0.000"

GB_T_KEY = "gb_t"
GB_DX_KEY = "gb_dx"
GB_FIXED_A_TEMPLATE_KEY = "gb_fixed_a_template"
GB_A_TEMPLATE_KEY = "gb_a_template"
GB_B_TEMPLATE_KEY = "gb_b_template"
GB_C_TEMPLATE_KEY = "gb_c_template"



class ReferenceTemplate:
    """Everything :func:`build_generation_graph` needs, computed ONCE per process from the single
    reference structure (module docstring "Generation-time contract" above)."""

    def __init__(
        self,
        fixed_a_cart: np.ndarray,
        a_template: float,
        b_template: float,
        c_template: float,
        master_frac_ref: np.ndarray,
        slave_frac_ref: np.ndarray,
        reference_dir: str,
    ):
        self.fixed_a_cart = fixed_a_cart  # (N_BULK_PER_GRAIN, 3) Cartesian, Angstrom
        self.a_template = a_template
        self.b_template = b_template
        self.c_template = c_template
        self.master_frac_ref = master_frac_ref  # (m_ref, 3) fractional, [0,1)
        self.slave_frac_ref = slave_frac_ref  # (m_ref, 3) fractional, [0,1)
        self.m_ref = master_frac_ref.shape[0]
        self.reference_dir = reference_dir


def build_reference_template(
    gamma_csv_path: str, bulk_membership_csv_path: str, manifest_dir: str
) -> ReferenceTemplate:
    """Locate the unique converged (``REFERENCE_CNID``, ``k=REFERENCE_K``, ``dx_p0.000``) row and
    build the FIXED_A Cartesian template + a/b/c template axes + MASTER/SLAVE placeholder-source
    fractional coordinates from it (module docstring above)."""
    rows = load_gamma_csv(gamma_csv_path)
    dx0_marker = f"/{_DX0_TAG}/k{REFERENCE_K}_s"
    candidates = [
        r
        for r in rows
        if r.cnid == REFERENCE_CNID and r.k == REFERENCE_K and r.converged and dx0_marker in r.dir
    ]
    if len(candidates) == 0:
        # (2026-10-01): the Al CSV (integrated_gamma_dedup.csv) names no dx tag in the dir of
        # its dx=0 original rows (".../structures/cnid_0_0/k0_s0"); only its dx!=0 augmentation rows
        # carry ".../dx_m0.750"-style dirs. A row with a dx-tagged directory name is handled by the rule above. Fallback: the converged (REFERENCE_CNID, REFERENCE_K) rows whose dir has no
        # "/dx_" path component. Still exactly one or an exception .
        candidates = [
            r
            for r in rows
            if r.cnid == REFERENCE_CNID
            and r.k == REFERENCE_K
            and r.converged
            and "/dx_" not in r.dir
        ]
    if len(candidates) != 1:
        raise ValueError(
            f"build_reference_template: expected exactly one converged {_DX0_TAG!r} row for "
            f"cnid={REFERENCE_CNID!r} k={REFERENCE_K}, found {len(candidates)}: "
            f"{[c.dir for c in candidates]} (refusing to guess which one is the "
            "reference)."
        )
    row = candidates[0]

    membership_map = load_bulk_membership(bulk_membership_csv_path)
    key = (row.cnid, row.k, row.seed)
    if key not in membership_map:
        raise ValueError(f"build_reference_template: no bulk_membership row for {key}")
    membership = membership_map[key]

    poscar_path = Path(resolve_view(str(Path(row.dir) / "POSCAR_relaxed")))
    frac, cell, species = read_poscar_frac_species(poscar_path)

    #: The dx=0 template manifest is named "{cnid}_dx_p0.000.json" or "{cnid}.json" (same
    # either-or rule as SymmetricTiltGBDataset._template_axes: exactly one must exist).
    _cand_tagged = Path(manifest_dir) / f"{REFERENCE_CNID}_{_DX0_TAG}.json"
    _cand_al = Path(manifest_dir) / f"{REFERENCE_CNID}.json"
    _exists = [p for p in (_cand_tagged, _cand_al) if p.is_file()]
    if len(_exists) != 1:
        raise FileNotFoundError(
            f"build_reference_template: expected exactly one template manifest in {manifest_dir} "
            f"({_cand_tagged.name} or {_cand_al.name}), found {[p.name for p in _exists]} "
            "(neither -> missing, both -> ambiguous)."
        )
    manifest_path = _exists[0]
    with open(manifest_path) as f:
        manifest_payload = json.load(f)
    manifest_cell = manifest_payload["cell"]
    a_t = float(manifest_cell[0][0])
    b_t = float(manifest_cell[1][1])
    c_t = float(manifest_cell[2][2])

    rng = np.random.default_rng(0)
    layout = build_canonical_layout(
        frac=frac, cell=cell, membership=membership, manifest=None, rng=rng, species_frac=species
    )
    fa_idx = layout.source_fixed_a_local_idx  # (N_BULK_PER_GRAIN,)
    if fa_idx.shape[0] != N_BULK_PER_GRAIN:
        raise ValueError(
            f"build_reference_template: reference FIXED_A count {fa_idx.shape[0]} != "
            f"N_BULK_PER_GRAIN={N_BULK_PER_GRAIN}"
        )
    m_idx = layout.source_master_local_idx  # (m_ref,) canonical sub_lo->sub_hi order
    s_idx = layout.source_slave_local_idx  # (m_ref,)
    if m_idx.shape[0] != s_idx.shape[0]:
        raise ValueError(
            f"build_reference_template: reference MASTER count {m_idx.shape[0]} != SLAVE count "
            f"{s_idx.shape[0]}"
        )

    fixed_a_cart = frac[fa_idx] @ cell  # (N_BULK_PER_GRAIN, 3) Cartesian, Angstrom

    return ReferenceTemplate(
        fixed_a_cart=fixed_a_cart,
        a_template=a_t,
        b_template=b_t,
        c_template=c_t,
        master_frac_ref=frac[m_idx],
        slave_frac_ref=frac[s_idx],
        reference_dir=row.dir,
    )


def sample_num_atoms(
    num_samples: int, rng: np.random.Generator, distribution_key: str | None = None
) -> list[int]:
    """Draw ``num_samples`` values of ``n`` from ``NUM_ATOMS_DISTRIBUTIONS[NUM_ATOMS_DISTRIBUTION_KEY]``
    , the SAME algorithm
    ``mattergen.common.data.dataset.NumAtomsCrystalDataset.from_num_atoms_distribution`` uses
    (``np.random.choice`` weighted by ``p``) -- not re-derived, just inlined because
    that classmethod builds a ``NumAtomsCrystalDataset``/``ChemGraph`` this module does not use
    (this module needs ``n`` alone, before the rest of the proposed-specific graph is built).

    ``distribution_key``: ``None`` (default, every pre-existing caller) -> the
    module constant ``NUM_ATOMS_DISTRIBUTION_KEY`` (resolved at call time); otherwise that key.
    An unknown key raises ``KeyError`` ."""
    key = NUM_ATOMS_DISTRIBUTION_KEY if distribution_key is None else distribution_key
    if key not in NUM_ATOMS_DISTRIBUTIONS:
        raise KeyError(f"sample_num_atoms: unknown num_atoms distribution key {key!r}")
    dist = NUM_ATOMS_DISTRIBUTIONS[key]
    p = np.array([v["p"] for v in dist], dtype=np.float64)
    p = p / p.sum()
    idx = rng.choice(len(dist), size=num_samples, p=p)
    return [int(dist[i]["n"]) for i in idx]


def build_generation_graph(
    template: ReferenceTemplate, n_target: int, dtype: torch.dtype = torch.float32
) -> ChemGraph:
    """Build one variable-length, padding-free ``ChemGraph`` for generation with exactly
    ``n_target`` atoms (module docstring "Generation-time contract" above). Raises ``ValueError``
    if ``n_target`` is not realizable under ``N_BULK_PER_GRAIN`` (
    ``m = (n - 2*N_BULK_PER_GRAIN) / 2`` invariant) -- never silently rounds/clamps ."""
    if (n_target - 2 * N_BULK_PER_GRAIN) % 2 != 0:
        raise ValueError(
            f"build_generation_graph: n_target={n_target} is not realizable "
            f"(n - 2*N_BULK_PER_GRAIN must be even; N_BULK_PER_GRAIN={N_BULK_PER_GRAIN})."
        )
    m = (n_target - 2 * N_BULK_PER_GRAIN) // 2
    if n_target != 2 * N_BULK_PER_GRAIN + 2 * m:
        raise ValueError(
            f"build_generation_graph: internal inconsistency, n_target={n_target} != "
            f"2*{N_BULK_PER_GRAIN} + 2*{m}"
        )
    if m < 1:
        raise ValueError(f"build_generation_graph: n_target={n_target} implies m={m} < 1")
    if m > template.m_ref:
        raise ValueError(
            f"build_generation_graph: n_target={n_target} implies m={m}, which exceeds the "
            f"reference template's own MASTER/SLAVE count m_ref={template.m_ref} (cannot source "
            "placeholder MASTER/SLAVE fractional coordinates)."
        )

    n_bulk = N_BULK_PER_GRAIN
    gb_block = torch.cat(
        [
            torch.full((n_bulk,), GB_BLOCK_FIXED_A, dtype=torch.long),
            torch.full((n_bulk,), GB_BLOCK_FIXED_B, dtype=torch.long),
            torch.full((m,), GB_BLOCK_MASTER, dtype=torch.long),
            torch.full((m,), GB_BLOCK_SLAVE, dtype=torch.long),
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
    assert_graph_invariants(gb_block, gb_pair_rank)
    gb_role = derive_gb_role(gb_block)

    a_t, b_t, c_t = template.a_template, template.b_template, template.c_template
    gb_dx0 = torch.zeros(1, dtype=dtype)
    cell0 = symmetry_ops.reconstruct_cell(gb_dx0, a_template=a_t, b_template=b_t, c_template=c_t)[
        0
    ]  # (3,3)
    cell_inv0 = torch.linalg.inv(cell0)

    fixed_a_cart_t = torch.as_tensor(template.fixed_a_cart, dtype=dtype)  # (N_BULK_PER_GRAIN, 3)
    frac_a0 = torch.remainder(torch.einsum("kj,ji->ki", fixed_a_cart_t, cell_inv0), 1.0)

    rotation = torch.tensor(_R_SLAVE_MATRIX, dtype=dtype)
    gb_t0 = torch.zeros(1, 3, dtype=dtype)
    cart_b0 = symmetry_ops.reconstruct_fixed_b_cart(
        fixed_a_cart_t.unsqueeze(0),
        gb_t0[:, 1:3],
        gb_dx0.reshape(1),
        rotation=rotation,
        a_template=a_t,
        b_template=b_t,
        c_template=c_t,
    )[0]
    frac_b0 = torch.remainder(torch.einsum("kj,ji->ki", cart_b0, cell_inv0), 1.0)

    master_frac0 = torch.as_tensor(np.asarray(template.master_frac_ref[:m]), dtype=dtype)
    slave_frac0 = torch.as_tensor(np.asarray(template.slave_frac_ref[:m]), dtype=dtype)

    pos = torch.cat([frac_a0, frac_b0, master_frac0, slave_frac0], dim=0)
    n = pos.shape[0]
    if n != n_target:
        raise RuntimeError(
            f"build_generation_graph: internal, built pos with n={n} != n_target={n_target}"
        )

    atomic_numbers = torch.full((n,), HOST_Z, dtype=torch.long)

    graph = ChemGraph(
        pos=pos,
        atomic_numbers=atomic_numbers,
        cell=cell0.unsqueeze(0).to(dtype),
        num_atoms=torch.tensor(n, dtype=torch.long),
        num_nodes=n,
    )
    graph = graph.replace(
        **{
            GB_BLOCK_FIELD: gb_block,
            GB_PAIR_RANK_FIELD: gb_pair_rank,
            "gb_role": gb_role,
            GB_T_KEY: gb_t0,
            GB_DX_KEY: gb_dx0.reshape(1, 1),
            GB_FIXED_A_TEMPLATE_KEY: fixed_a_cart_t.unsqueeze(0),
            GB_A_TEMPLATE_KEY: torch.tensor([a_t], dtype=dtype),
            GB_B_TEMPLATE_KEY: torch.tensor([b_t], dtype=dtype),
            GB_C_TEMPLATE_KEY: torch.tensor([c_t], dtype=dtype),
        }
    )
    return graph


def build_masks_variable_length(batch: ChemGraph) -> dict[str, torch.Tensor]:
    """``pos``: FIXED_A fixed (1, reference anchor), FIXED_B/MASTER/SLAVE all free (0) -- same
    convention as ``mattergen.scripts.generate_symmetry_constrained.build_masks`` (:
    ``SymmetryProjector`` reconstructs FIXED_B/SLAVE from FIXED_A_template/MASTER every step,
    so masking them here would only let stale prior values leak into the reconstruction). ``gb_t``:
    fully free (0). ``gb_dx``/``cell``: no mask entry at all (both are fully diffused by the
    injected ``gb_dx`` predictor / left untouched respectively, module docstring above --
    ``atomic_numbers`` likewise has no mask entry, same reasoning as the fixed-length
    ``build_masks``: it is a long-dtype field that cannot go through ``_sample_prior``'s ``lerp_``,
    and the projector re-unifies SLAVE=MASTER/FIXED=HOST_Z every step regardless)."""
    gb_block = getattr(batch, GB_BLOCK_FIELD)
    pos_mask = (gb_block == GB_BLOCK_FIXED_A).to(batch.pos.dtype).unsqueeze(-1)
    return {"pos": pos_mask, GB_T_KEY: torch.zeros_like(batch.gb_t)}


def extract_gamma_scaler_stats(lightning_module: DiffusionLightningModule) -> dict:
    """Same purpose/attribute chain as ``generate_scm_cond.py``'s function of the same name
    -- reproduced here (not imported: that function lives
    in a top-level ``scripts/`` file, not a package) so a later checkpoint mismatch is directly
    detectable from ``generation_metrics.json`` alone."""
    try:
        prop_emb = lightning_module.diffusion_module.model.property_embeddings_adapt[
            GB_GAMMA_PROPERTY_KEY
        ]
        scaler = prop_emb.scaler
        means = scaler.means.detach().cpu().tolist()
        stds = scaler.stds.detach().cpu().tolist()
    except (AttributeError, KeyError) as e:
        raise RuntimeError(
            "extract_gamma_scaler_stats: could not resolve "
            "lightning_module.diffusion_module.model.property_embeddings_adapt"
            f"[{GB_GAMMA_PROPERTY_KEY!r}].scaler.{{means,stds}}: {e!r}"
        ) from e
    return {"means": means, "stds": stds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a .ckpt file")
    parser.add_argument("--config", required=True, help="path to the run's saved (resolved) config.yaml")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        required=True,
        help="classifier-free guidance scale, passed straight to "
        "GuidedPredictorCorrector.from_pl_module(...). No config default (same sealing clause as "
        "generate_scm_cond.py).",
    )
    parser.add_argument("--seed", type=int, required=True, help="torch.manual_seed value for this invocation.")
    parser.add_argument(
        "--gamma_target",
        type=float,
        required=True,
        help="scalar grain-boundary energy target in J/m^2 (raw, unscaled), stamped uniformly "
        "across the whole batch via mattergen.property_embeddings.set_conditional_property_values "
        "(module docstring: vector/per-structure-target mode is out of this script's scope).",
    )
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--gamma_csv_path", type=str, required=True)
    parser.add_argument("--bulk_membership_csv_path", type=str, required=True)
    parser.add_argument("--manifest_dir", type=str, required=True)
    parser.add_argument(
        "--num-atoms-distribution-key",
        dest="num_atoms_distribution_key",
        type=str,
        default=NUM_ATOMS_DISTRIBUTION_KEY,
        help="key into NUM_ATOMS_DISTRIBUTIONS the atom counts n are drawn from (default: the distribution of the training split of the paper).",
    )
    parser.add_argument(
        "--host-z",
        dest="host_z",
        type=int,
        default=13,
        help="expected host atomic number (default 13 = Al). It must equal the SCM_HOST_Z environment "
        "variable that mattergen.symmetry_constrained.constants resolved at import; a mismatch raises "
        "(pass --host-z 13 together with SCM_HOST_Z=13).",
    )
    parser.add_argument(
        "--record_trajectories",
        action="store_true",
        default=False,
        help="save the state of each diffusion step (after projection) in generated_trajectories.zip",
    )
    args = parser.parse_args()

    #: --host-z vs the environment-resolved HOST_Z (/C23: explicit and consistent, no silent pick).
    if int(args.host_z) != int(HOST_Z):
        raise RuntimeError(
            f"--host-z={args.host_z} disagrees with SCM_HOST_Z={HOST_Z} (environment, resolved by "
            "mattergen.symmetry_constrained.constants at import). Pass --host-z matching the exported SCM_HOST_Z."
        )
    if args.num_atoms_distribution_key not in NUM_ATOMS_DISTRIBUTIONS:
        raise KeyError(
            f"--num-atoms-distribution-key={args.num_atoms_distribution_key!r} is not in NUM_ATOMS_DISTRIBUTIONS"
        )
    # Resolved values, printed for the check-8 / 5b(v) check (training side prints the same two).
    print(f"resolved SCM_HOST_Z={HOST_Z} (--host-z {args.host_z})")
    print(f"resolved num_atoms_distribution_key={args.num_atoms_distribution_key}")

    output_path = Path(args.output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError(
            f"output_path {output_path} already exists and is not empty "
            f"(contains: {sorted(p.name for p in output_path.iterdir())}). Refusing to write into "
            "it."
        )
    output_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    template = build_reference_template(args.gamma_csv_path, args.bulk_membership_csv_path, args.manifest_dir)

    rng = np.random.default_rng(args.seed)
    n_per_sample = sample_num_atoms(args.num_samples, rng, args.num_atoms_distribution_key)

    saved_cfg = OmegaConf.load(args.config)
    lightning_module: DiffusionLightningModule = instantiate(saved_cfg.lightning_module)
    checkpoint_path = Path(args.checkpoint).resolve()
    ckpt = torch.load(checkpoint_path, map_location=device)
    lightning_module.load_state_dict(ckpt["state_dict"], strict=True)
    lightning_module = lightning_module.to(device)
    lightning_module.eval()

    scaler_stats = extract_gamma_scaler_stats(lightning_module)

    with initialize_config_dir(config_dir=SAMPLING_CONF_DIR, version_base="1.1"):
        sampling_cfg = compose(config_name=SAMPLING_CONFIG_NAME)
    # gb_dx is ALWAYS diffused for the proposed architecture (module docstring "gb_dx is FULLY
    # DIFFUSED" above) -- scm_cond.yaml has no gb_dx predictor entry (it was written for the
    # fixed-length --dx-mode fixed default), so inject one, mirroring
    # generate_scm_cond.py's --dx-mode diffuse branch exactly (same _target_/_partial_).
    OmegaConf.set_struct(sampling_cfg, False)
    OmegaConf.update(
        sampling_cfg,
        "sampler_partial.predictor_partials.gb_dx",
        {
            "_target_": "mattergen.symmetry_constrained.corruption_dx.GBTranslationPredictor",
            "_partial_": True,
        },
        merge=True,
    )
    OmegaConf.set_struct(sampling_cfg, True)
    sampler_partial = instantiate(sampling_cfg.sampler_partial)
    sampler = sampler_partial(pl_module=lightning_module, guidance_scale=args.guidance_scale)
    sampler.state_projector = SymmetryProjector()

    graphs = [build_generation_graph(template, n) for n in n_per_sample]
    batch = collate(graphs).to(device)
    batch = set_conditional_property_values(batch, {GB_GAMMA_PROPERTY_KEY: args.gamma_target})
    batch = SetConditionalEmbeddingType()(batch)
    mask = build_masks_variable_length(batch)
    mask = {k: v.to(device) for k, v in mask.items()}

    if args.record_trajectories:
        sample, mean, recorded_samples = sampler.sample_with_record(batch, mask=mask)
    else:
        sample, mean = sampler.sample(batch, mask=mask)

    # Gate c2-style symmetry residual: compare the RAW sampler output against the projector's own
    # prediction from MASTER, index-preserving (do not use a
    # threshold-based classification here -- gb_pair_rank already fixes the correspondence, no
    # nearest-neighbor/Hungarian step is needed at generation time since there is no independent
    # "true" SLAVE to match against).
    projected = sampler.state_projector(sample)
    d = projected.pos.to(torch.float64) - sample.pos.to(torch.float64)
    dmod = ((d + 0.5) % 1.0) - 0.5
    slave_mask = (getattr(sample, GB_BLOCK_FIELD) == GB_BLOCK_SLAVE).cpu()
    sym_resid_frac_per_node = dmod.abs().max(dim=-1).values.cpu()
    sym_resid_frac_slave = float(sym_resid_frac_per_node[slave_mask].max().item()) if slave_mask.any() else 0.0
    sym_resid_frac_all = float(sym_resid_frac_per_node.max().item())

    n_atoms_per_sample = sample.num_atoms.detach().cpu().tolist()

    with open(output_path / "generation_metrics.json", "w") as f:
        json.dump(
            {
                "property_name": GB_GAMMA_PROPERTY_KEY,
                "mode": "scalar",
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(Path(args.config).resolve()),
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "gamma_target_J_m2": args.gamma_target,
                "gamma_scaler_means": scaler_stats["means"],
                "gamma_scaler_stds": scaler_stats["stds"],
                "num_samples_requested": args.num_samples,
                "n_atoms_per_sample": n_atoms_per_sample,
                "reference_dir": template.reference_dir,
                "resolved_host_z": int(HOST_Z),
                "num_atoms_distribution_key": args.num_atoms_distribution_key,
                "symmetry_residual_frac_max_slave": sym_resid_frac_slave,
                "symmetry_residual_frac_max_all": sym_resid_frac_all,
            },
            f,
            indent=2,
        )
    print("mode: scalar")
    print("checkpoint_path:", checkpoint_path)
    print("guidance_scale:", args.guidance_scale)
    print("seed:", args.seed)
    print("gamma_target_J_m2:", args.gamma_target)
    print("gamma_scaler_means:", scaler_stats["means"], "gamma_scaler_stds:", scaler_stats["stds"])
    print("n_atoms_per_sample:", n_atoms_per_sample)
    print("symmetry_residual_frac_max_slave:", sym_resid_frac_slave)

    lengths, angles = lattice_matrix_to_params_torch(sample.cell)
    structures = structure_from_model_output(
        frac_coords=sample.pos,
        atom_types=sample.atomic_numbers,
        lengths=lengths,
        angles=angles,
        num_atoms=sample.num_atoms,
        site_properties={},
    )
    save_structures(output_path, structures)
    print(f"saved {len(structures)} structures to {output_path}")
    if args.record_trajectories:
        from mattergen.generator import dump_trajectories, list_of_time_steps_to_list_of_trajectories

        all_trajs_list = list_of_time_steps_to_list_of_trajectories(recorded_samples)
        dump_trajectories(output_path=output_path, all_trajs_list=all_trajs_list)
        print(f"saved {len(all_trajs_list)} trajectories to {output_path}")


if __name__ == "__main__":
    main()
