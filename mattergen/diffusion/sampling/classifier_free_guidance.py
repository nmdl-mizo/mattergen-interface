# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable

import torch

from mattergen.diffusion.sampling.pc_sampler import Diffusable, PredictorCorrector
from mattergen.common.data.collate import collate

BatchTransform = Callable[[Diffusable], Diffusable]


def identity(x: Diffusable) -> Diffusable:
    """
    Default function that transforms data to its conditional state
    """
    return x


class GuidedPredictorCorrector(PredictorCorrector):
    """
    Sampler for classifier-free guidance.
    """

    def __init__(
        self,
        *,
        guidance_scale: float,
        remove_conditioning_fn: BatchTransform,
        keep_conditioning_fn: BatchTransform | None = None,
        guidance_scale_per_field: dict[str, float] | None = None,
        **kwargs,
    ):
        """
        guidance_scale: gamma in p_gamma(x|y)=p(x)p(y|x)**gamma for classifier-free guidance.
            When ``guidance_scale_per_field`` is *not* provided this single scalar is applied
            uniformly to every corrupted field (original behaviour, bit-exact).
        remove_conditioning_fn: function that removes conditioning from the data
        keep_conditioning_fn: function that will be applied to the data before evaluating the
            conditional score. For example, this function might drop some fields that you never
            want to condition on or add fields that indicate which conditions should be respected.
        guidance_scale_per_field: optional dict mapping corrupted-field names (e.g. ``pos``,
            ``cell``, ``atomic_numbers``) to per-field guidance weights.  A ``"default"`` key
            serves as fallback for fields not explicitly listed; if absent the scalar
            ``guidance_scale`` is used as the fallback.  When this parameter is ``None`` the
            sampler falls back to the uniform-scalar path (fully backward-compatible).
        **kwargs: passed on to parent class constructor.
        """

        super().__init__(**kwargs)
        self._remove_conditioning_fn = remove_conditioning_fn
        self._keep_conditioning_fn = keep_conditioning_fn or identity
        self._guidance_scale = guidance_scale
        self._guidance_scale_per_field = guidance_scale_per_field

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _field_guidance(self, field_key: str) -> float:
        """Return the guidance weight for *field_key*.

        Resolution order: ``guidance_scale_per_field[field_key]`` >
        ``guidance_scale_per_field["default"]`` > ``guidance_scale`` (scalar).
        """
        if self._guidance_scale_per_field is not None:
            return self._guidance_scale_per_field.get(
                field_key,
                self._guidance_scale_per_field.get("default", self._guidance_scale),
            )
        return self._guidance_scale

    def _score_fn(
        self,
        x: Diffusable,
        t: torch.Tensor,
    ) -> Diffusable:
        """For each field, regardless of whether the corruption process is SDE or D3PM, we guide the score in the same way here,
        by taking a linear combination of the conditional and unconditional score model output.

        For discrete fields, the score model outputs are interpreted as logits, so the linear combination here means we compute logits for
        p_\gamma(x|y)=p(x)^(1-\gamma) p(x|y)^\gamma

        """

        def get_unconditional_score():
            return super(GuidedPredictorCorrector, self)._score_fn(
                x=self._remove_conditioning_fn(x), t=t
            )

        def get_conditional_score():
            return super(GuidedPredictorCorrector, self)._score_fn(
                x=self._keep_conditioning_fn(x), t=t
            )

        # ----- per-field guidance path (new) -----
        if self._guidance_scale_per_field is not None:
            batch_no_condition = self._remove_conditioning_fn(x)
            batch_with_condition = self._keep_conditioning_fn(x)
            joint_batch = collate([batch_no_condition, batch_with_condition])

            for attr, value in batch_no_condition.items():
                if isinstance(value, list):
                    joint_batch[attr] = batch_no_condition[attr] + batch_with_condition[attr]

            combined_score = super(GuidedPredictorCorrector, self)._score_fn(
                x=joint_batch, t=torch.cat([t, t], dim=0),
            )
            unconditional_score = combined_score[0]
            conditional_score = combined_score[1]

            return unconditional_score.replace(
                **{
                    k: torch.lerp(
                        unconditional_score[k],
                        conditional_score[k],
                        self._field_guidance(k),
                    )
                    for k in self._multi_corruption.corrupted_fields
                }
            )

        # ----- original scalar path (unchanged, bit-exact) -----
        if abs(self._guidance_scale - 1) < 1e-15:
            return get_conditional_score()
        elif abs(self._guidance_scale) < 1e-15:
            return get_unconditional_score()
        else:
            # guided_score = guidance_factor * conditional_score + (1-guidance_factor) * unconditional_score
            batch_no_condition = self._remove_conditioning_fn(x)
            batch_with_condition = self._keep_conditioning_fn(x)
            joint_batch = collate([batch_no_condition, batch_with_condition])

            for attr,value in batch_no_condition.items():
                if isinstance(value, list):
                    joint_batch[attr] = batch_no_condition[attr]+batch_with_condition[attr]


            combined_score = super(GuidedPredictorCorrector, self)._score_fn(
                x=joint_batch, t=torch.cat([t, t], dim=0),
            )
            # Split the combined score back into unconditional and conditional parts.
            # Any batch.attr: list fields will be wrong here because of the manual concatenation above
            # this should be ok as self._multi_corruption.corrupted_fields are always torch.Tensor
            unconditional_score = combined_score[0]
            conditional_score = combined_score[1]

            return unconditional_score.replace(
                **{
                    k: torch.lerp(
                        unconditional_score[k], conditional_score[k], self._guidance_scale
                    )
                    for k in self._multi_corruption.corrupted_fields
                }
            )
