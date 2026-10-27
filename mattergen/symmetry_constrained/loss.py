"""Loss for the tilt-STGB Sigma5(210)[001] diffusion loop (M-t3c).

internal notes (masking policy) / SS3.4
(gb_t) / SS4 M-t3c. Deliberately does NOT import mattergen.gb: the
_gb_t_inplane_loss and _masked_wrapped_pos_loss logic mirrored here from mattergen/gb/diffusion.py
is a straight port, not an import.

Masking policy:
    pos              -> PRIMARY block real (non-padding) slots only
    atomic_numbers   -> PRIMARY block ALL 26 slots (padding included)
    gb_t             -> in-plane axes [1, 2] only (axis 0 pinned at 0.0)
    cell             -> NOT SCORED (Step1 cell-fixed)
    FIXED / REPLICA    -> never scored

E2 fixed-N baseline (internal notes W-1, 2026-08-10):
``SymmetryConstrainedLoss`` gained an optional ``freeze_atomic_numbers`` kwarg, default False
(existing behaviour unchanged when omitted/False). When True, the ATOMIC_NUMBERS_FIELD entry is
excluded from the per-call ``per_field`` dict entirely (not computed, not aggregated into
``agg_loss``, and therefore also absent from the ``metrics`` dict returned to
``DiffusionLightningModule._calc_loss`` -- see mattergen/diffusion/lightning_module.py:162-171,
which logs every key of that dict as ``f"{k}_{step_type}"``; dropping the key is what makes
"atomic_numbers_train"/"atomic_numbers_val" absent from metrics.csv G-E2W-3). pos
and gb_t losses are unaffected. ``self.model_targets`` (set in ``__init__``, consumed by
mattergen/diffusion/diffusion_module.py:40 to build the score_fn's D3PM predictor head) is NOT
touched by this flag -- ``include_atomic_numbers=True`` is still passed to ``super().__init__``
unconditionally, so the generation-time score_fn structure is identical to the unfrozen case
("do not drop model_targets").
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import torch
from torch_scatter import scatter

from mattergen.common.loss import MaterialsLoss
from mattergen.diffusion.corruption.sde_lib import maybe_expand
from mattergen.diffusion.d3pm.d3pm import compute_kl_reverse_process
from mattergen.diffusion.discrete_time import to_discrete_time
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.training.field_loss import aggregate_per_sample, denoising_score_matching
from mattergen.diffusion.wrapped.wrapped_normal_loss import wrapped_normal_score
from mattergen.symmetry_constrained.constants import GBRole

logger = logging.getLogger(__name__)

_GB_T_INPLANE_AXIS_MASK = (0.0, 1.0, 1.0)

# (internal notes) SS3-3 Step6 -- NOT one of
# the literally listed output file, but a load-bearing gap discovered while preparing the
# Stage 8 training launch (same class of gap as adapter.py's gb_dx_readout, approved 2026-08-25):
# without a per_field entry, DiffusionLightningModule.score_fn (diffusion_module.py:130) would
# still compute a score for "gb_dx" (any key in self.corruption.sdes.keys --
# scm_dxvar.yaml adds one), via self.model_targets["gb_dx"] -- a KeyError today, since
# SymmetryConstrainedLoss never adds "gb_dx" to model_targets (only GB_T_FIELD is, __init__
# below). Even if that dict lookup were patched, __call__'s per_field dict below only ever
# builds POS_FIELD/GB_T_FIELD/ATOMIC_NUMBERS_FIELD entries -- gb_dx would simply never be
# trained against any target (loss=0 contribution), silently defeating entire purpose
# (a model that never sees a gb_dx loss term cannot learn a gamma->dx relationship). Both gaps
# are closed together below, gated on GB_DX_FIELD's presence in noisy_batch (backward-compatible
# with the existing scm_cond configs, which never diffuse gb_dx and therefore
# never populate this field -- mirrors the field-presence-gate pattern already used by
# projection.py's FixedSlotProjector and adapter.py's "gb_role" not in x checks).
GB_DX_FIELD = "gb_dx"

GB_T_FIELD: Literal["gb_t"] = "gb_t"
POS_FIELD: Literal["pos"] = "pos"
ATOMIC_NUMBERS_FIELD: Literal["atomic_numbers"] = "atomic_numbers"


def _masked_wrapped_pos_loss(
    *,
    corruption,
    score_model_output,
    t,
    batch_idx,
    batch_size,
    x,
    noisy_x,
    batch,
    node_is_unmasked,
    reduce,
):
    _, std = corruption.marginal_prob(
        x=torch.zeros((x.shape[0], 1), device=t.device), t=t, batch_idx=batch_idx, batch=batch
    )
    wb = corruption.wrapping_boundary * torch.eye(x.shape[-1], device=t.device)[None].expand(
        batch_size, -1, -1
    )
    target = (
        wrapped_normal_score(
            x=noisy_x, mean=x, wrapping_boundary=wb, variance_diag=std.squeeze() ** 2, batch=batch_idx
        )
        * std
    )
    losses = node_is_unmasked.unsqueeze(-1) * (target - score_model_output).square()
    loss_per_sample = aggregate_per_sample(losses, batch_idx, reduce="sum", batch_size=batch_size)
    if reduce == "mean":
        loss_per_sample = loss_per_sample / scatter(
            node_is_unmasked, batch_idx, dim=0, dim_size=batch_size, reduce="sum"
        ).clamp(min=1)
    return loss_per_sample


def _masked_d3pm_loss(
    *,
    corruption,
    score_model_output,
    t,
    batch_idx,
    batch_size,
    x,
    noisy_x,
    node_is_unmasked,
    reduce,
    d3pm_hybrid_lambda=0.0,
):
    assert hasattr(corruption, "N")
    assert hasattr(corruption, "_to_zero_based")
    assert hasattr(corruption, "d3pm")
    keep = node_is_unmasked.bool()
    x_kept = x[keep]
    noisy_x_kept = noisy_x[keep]
    score_kept = score_model_output[keep]
    batch_idx_kept = batch_idx[keep]

    t_expanded = maybe_expand(to_discrete_time(t, N=corruption.N, T=corruption.T), batch_idx_kept)
    metrics_dict = compute_kl_reverse_process(
        corruption._to_zero_based(x_kept.long()),
        t_expanded,
        diffusion=corruption.d3pm,
        log_space=True,
        denoise_fn=lambda targets, timestep: score_kept,
        hybrid_lambda=d3pm_hybrid_lambda,
        x_t_plus_1=corruption._to_zero_based(noisy_x_kept.long()),
    )
    loss = metrics_dict.pop("loss")
    loss_per_structure = aggregate_per_sample(
        loss, batch_idx=batch_idx_kept, reduce=reduce, batch_size=batch_size
    )
    return loss_per_structure


def _gb_t_inplane_loss(*, corruption, score_model_output, t, x, noisy_x, batch_size, reduce):
    device = t.device
    batch_idx = torch.arange(batch_size, device=device)
    _, std = corruption.marginal_prob(
        x=torch.zeros((batch_size, 1), device=device), t=t, batch_idx=None, batch=None
    )
    wb = corruption.wrapping_boundary * torch.eye(x.shape[-1], device=device)[None].expand(
        batch_size, -1, -1
    )
    target = (
        wrapped_normal_score(
            x=noisy_x, mean=x, wrapping_boundary=wb, variance_diag=std.squeeze(-1) ** 2, batch=batch_idx
        )
        * std
    )
    axis_mask = torch.tensor(_GB_T_INPLANE_AXIS_MASK, device=device, dtype=target.dtype)
    losses = axis_mask * (target - score_model_output).square()
    loss_per_sample = losses.sum(-1)
    if reduce == "mean":
        loss_per_sample = loss_per_sample / axis_mask.sum().clamp(min=1)
    return loss_per_sample


class SymmetryConstrainedLoss(MaterialsLoss):
    def __init__(
        self,
        reduce: Literal["sum", "mean"] = "mean",
        d3pm_hybrid_lambda: float = 0.0,
        weights=None,
        gb_t_loss_weight: float = 1.0,
        gb_dx_loss_weight: float = 1.0,
        freeze_atomic_numbers: bool = False,
    ):
        super().__init__(
            reduce=reduce,
            d3pm_hybrid_lambda=d3pm_hybrid_lambda,
            include_pos=True,
            include_cell=False,
            include_atomic_numbers=True,
            weights=weights,
        )
        self.gb_t_loss_weight = gb_t_loss_weight
        # (2026-08-25, necessary unlisted addition -- module docstring above):
        # gb_dx_loss_weight defaults to 1.0, matching gb_t_loss_weight's default -- both are
        # auxiliary rigid-translation latents (in-plane for gb_t, GB-normal for gb_dx) of the
        # same physical kind; this is a conservative, non-tuned choice, not a value pins
        # down explicitly. Inert (never read) for configs where gb_dx is absent (see __call__).
        self.gb_dx_loss_weight = gb_dx_loss_weight
        # W-1: does NOT change include_atomic_numbers above (model_targets
        # must keep atomic_numbers, see module docstring) -- only gates the per-call __call__
        # aggregation/logging below.
        self.freeze_atomic_numbers = freeze_atomic_numbers
        self.model_targets = {
            **self.model_targets,
            GB_T_FIELD: ModelTarget.score_times_std,
            #: added unconditionally (harmless when the corruption config does
            # not diffuse gb_dx, mirroring GB_T_FIELD's own unconditional registration above --
            # DiffusionLightningModule.score_fn only ever reads keys present in
            # self.corruption.sdes.keys(), so an unused model_targets entry is never referenced).
            GB_DX_FIELD: ModelTarget.score_times_std,
        }

    def __call__(
        self,
        *,
        multi_corruption,
        batch,
        noisy_batch,
        score_model_output,
        t,
        node_is_unmasked=None,
    ):
        if "gb_role" not in batch:
            raise ValueError("SymmetryConstrainedLoss requires the 'gb_role' node property")
        if GB_T_FIELD not in noisy_batch:
            raise ValueError(
                "SymmetryConstrainedLoss requires a gb_t corruption field in the training batch"
            )
        batch_size = batch.get_batch_size()
        role = noisy_batch.gb_role
        primary_mask = (role == int(GBRole.PRIMARY)).long()
        from mattergen.symmetry_constrained.layout import PAD_Z_TILT

        is_real = (batch.atomic_numbers != PAD_Z_TILT).long()
        pos_mask = primary_mask

        pos_idx = batch.get_batch_idx("pos")
        atomic_numbers_idx = batch.get_batch_idx("atomic_numbers")

        per_field = {
            POS_FIELD: _masked_wrapped_pos_loss(
                corruption=multi_corruption.corruptions[POS_FIELD],
                score_model_output=score_model_output[POS_FIELD],
                t=t,
                batch_idx=pos_idx,
                batch_size=batch_size,
                x=batch[POS_FIELD],
                noisy_x=noisy_batch[POS_FIELD],
                batch=batch,
                node_is_unmasked=pos_mask,
                reduce=self.reduce,
            ),
            GB_T_FIELD: _gb_t_inplane_loss(
                corruption=multi_corruption.corruptions[GB_T_FIELD],
                score_model_output=score_model_output[GB_T_FIELD],
                t=t,
                x=batch[GB_T_FIELD],
                noisy_x=noisy_batch[GB_T_FIELD],
                batch_size=batch_size,
                reduce=self.reduce,
            ),
        }
        if GB_DX_FIELD in noisy_batch:
            # (necessary unlisted addition, module docstring above): gb_dx is a
            # plain (non-periodic, non-masked) graph-level scalar VPSDE field -- standard
            # denoising score matching (mattergen.diffusion.training.field_loss, the SAME generic
            # loss used elsewhere in mattergen for non-wrapped continuous fields), not the
            # wrapped-normal/masked machinery pos/gb_t need. batch_idx=None: gb_dx is already
            # one-row-per-graph (aggregate_per_sample's documented "assume first dim is the batch
            # dim" convention when batch_idx is omitted).
            per_field[GB_DX_FIELD] = denoising_score_matching(
                corruption=multi_corruption.corruptions[GB_DX_FIELD],
                score_model_output=score_model_output[GB_DX_FIELD],
                t=t,
                batch_idx=None,
                batch_size=batch_size,
                x=batch[GB_DX_FIELD],
                noisy_x=noisy_batch[GB_DX_FIELD],
                reduce=self.reduce,
                batch=batch,
                model_target=ModelTarget.score_times_std,
            )
        if self.freeze_atomic_numbers:
            # W-1 / G-E2W-3: excluded from aggregation AND from the returned
            # metrics dict (module docstring) -- explicit log line so the exclusion is auditable
            # from the smoke training's stdout/log, not just inferable from metrics.csv's absent
            # columns.
            logger.info(
                "SymmetryConstrainedLoss.freeze_atomic_numbers=True: excluding %r from loss "
                "aggregation and per-field logging (workspace docs/design/28 E2 fixed-N "
                "baseline).",
                ATOMIC_NUMBERS_FIELD,
            )
        else:
            per_field[ATOMIC_NUMBERS_FIELD] = _masked_d3pm_loss(
                corruption=multi_corruption.corruptions[ATOMIC_NUMBERS_FIELD],
                score_model_output=score_model_output[ATOMIC_NUMBERS_FIELD],
                t=t,
                batch_idx=atomic_numbers_idx,
                batch_size=batch_size,
                x=batch[ATOMIC_NUMBERS_FIELD],
                noisy_x=noisy_batch[ATOMIC_NUMBERS_FIELD],
                node_is_unmasked=primary_mask,
                reduce=self.reduce,
                d3pm_hybrid_lambda=self.d3pm_hybrid_lambda,
            )
        weights = {
            **self.loss_weights,
            GB_T_FIELD: self.gb_t_loss_weight,
            #: only referenced when GB_DX_FIELD in per_field (gated above), so
            # this key's presence/absence in `weights` never matters for scm_cond
            # configs (no KeyError risk either way -- per_field never has the key there).
            GB_DX_FIELD: self.gb_dx_loss_weight,
        }
        agg_loss = torch.stack([weights[k] * v for k, v in per_field.items()], dim=0).sum(0)
        return agg_loss.mean(), {k: v.mean() for k, v in per_field.items()}
