"""``gb_dx`` scalar-latent corruption for the tilt-STGB diffusion loop (Step 2, internal notes).

``gb_dx`` is a genuine per-graph diffused scalar latent (like ``gb_t``), representing the
GB-normal rigid-translation half-gap. This
module is an INTENTIONAL, DELIBERATE COMPLEMENT of ``<ORIGINAL_ROOT>/mattergen/mattergen/
gb/corruption.py``'s ``DeltaAVPSDE``/``DeltaAAncestralSamplingPredictor`` (read-only reference,
) -- NOT an import, a from-scratch reimplementation with the SAME non-symmetrised-noise
structure, for two reasons pins down explicitly:

  1. ``LatticeVPSDE``'s ``sample_marginal``/``prior_sampling`` route sampled noise through
     ``make_noise_symmetric_preserve_variance``, which hard-asserts a square ``(batch, 3, 3)``
     matrix -- incompatible with a ``(n_graphs, 1)`` scalar field. Subclassing ``LatticeVPSDE``
     directly is therefore not viable; this subclasses ``VPSDE`` instead (mirrors the earlier
     reference's own reasoning, ``corruption.py`` module docstring).
  2. (design decision, not to be changed by the author): the terminal distribution
     for workspace's ``gb_dx`` is **gamma-INDEPENDENT** (``mu_uncond``/``sigma2`` constants only,
     P-1's measured train-split statistics). The earlier reference's ``DeltaAVPSDE`` primarily
     supports a gamma-CONDITIONED terminal mean (``mu(gamma)``, a frozen linear/quadratic fit) with
     the unconditional case as a fallback/opt-in (``gamma_independent_prior``); reusing that class
     here would import a gamma-conditioning code surface explicitly forbids exercising
     (learning the gamma->dx relationship would be pre-empted by baking the confound into the
     prior stated rationale). This module implements ONLY the constant-mean
     branch directly (no ``mu_gamma_json``, no ``gamma_field`` branch at all) -- a strict subset of
     the reference's surface, not a superset, so there is no unused-but-reachable gamma-dependent
     code path to accidentally trigger.

Coefficient of 2 (the definition of gb_delta_a ... has a relation of coefficient 2 with dx of the project): the
earlier reference defines ``gb_delta_a = cell[0,0] - A_TEMPLATE`` (a single-grain-boundary
half-cell convention). workspace's bicrystal has TWO grain boundaries, so opening each gap by
``gb_dx`` stretches the cell by ``2*gb_dx``: ``cell[0,0] = A_TEMPLATE + 2*gb_dx``,
i.e. ``gb_dx = (cell[0,0] - A_TEMPLATE) / 2 = gb_delta_a / 2`` in the reference's units. This
module's constant/coefficient names (``mu_uncond``, ``sigma2``) are therefore ``gb_dx``-scaled
values (P-1's ``prior_stats.json``), NOT directly interchangeable with the reference's
``mu_uncond``/``sigma2_delta_a`` (which are ``gb_delta_a``-scaled, i.e. 2x workspace's values).
"""
from __future__ import annotations

import torch

from mattergen.common.diffusion.corruption import expand
from mattergen.diffusion.corruption.corruption import B, BatchedData, maybe_expand
from mattergen.diffusion.corruption.sde_lib import VPSDE
from mattergen.diffusion.sampling.predictors import AncestralSamplingPredictor
from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean



class GBTranslationVPSDE(VPSDE):
    """Scalar VPSDE for the per-graph latent ``gb_dx``. Terminal mean/variance are
    FIXED CONSTANTS (``mu_uncond``/``sigma2``: gamma-independent -- no gamma-conditioned
    branch exists in this class at all, unlike the earlier reference's ``DeltaAVPSDE``).

    ``marginal_prob``/``sde``/``prior_sampling``/``sample_marginal`` are plain-scalar
    reimplementations of the ``LatticeVPSDE`` recipe minus ``make_noise_symmetric_preserve_variance``
    (module docstring reason 1) -- mirrors the earlier reference's own structure, not its content.
    """

    @staticmethod
    def from_vpsde_config(vpsde_config):
        return GBTranslationVPSDE(**vpsde_config)

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20,
        mu_uncond: float | None = None,
        sigma2: float | None = None,
        sigma2_floor: float = 0.01,
    ):
        """``mu_uncond``/``sigma2`` (required, no default -- silent fallback is prohibited):
        the measured train-split (3,735 structures) statistics, wired verbatim via
        ``conf/.../corruption/scm_dxvar.yaml``. ``sigma2_floor`` mirrors the
        earlier reference's clipping floor (protects against a degenerate near-zero variance
        collapsing the forward/reverse process); P-1's measured ``sigma2=0.1503`` is far above this
        floor so it is inert for the values actually uses, kept only as a defensive guard.
        """
        super().__init__(beta_min=beta_min, beta_max=beta_max)
        if mu_uncond is None or sigma2 is None:
            raise ValueError(
                "GBTranslationVPSDE requires both mu_uncond and sigma2 explicitly (: gamma-independent "
                "constant terminal distribution; no gamma-conditioned fallback exists to silently "
                "substitute --)."
            )
        self.mu_uncond = float(mu_uncond)
        self.sigma2_floor = float(sigma2_floor)
        self.sigma2 = max(float(sigma2), self.sigma2_floor)

    def get_limit_mean(self, x: torch.Tensor, batch: BatchedData) -> torch.Tensor:
        n = x.shape[0]
        return torch.full((n, 1), self.mu_uncond, device=x.device, dtype=x.dtype)

    def get_limit_var(self, x: torch.Tensor, batch: BatchedData) -> torch.Tensor:
        n = x.shape[0]
        return torch.full((n, 1), self.sigma2, device=x.device, dtype=x.dtype)

    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch is not None
        mean_coeff = self._marginal_mean_coeff(t)
        limit_mean = self.get_limit_mean(x=x, batch=batch)
        limit_var = self.get_limit_var(x=x, batch=batch)
        mean_coeff_expanded = maybe_expand(mean_coeff, batch_idx, x)
        mean = mean_coeff_expanded * x + (1 - mean_coeff_expanded) * limit_mean
        std = torch.sqrt((1.0 - mean_coeff_expanded**2) * limit_var)
        return mean, std

    def mean_coeff_and_std(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_coeff = self._marginal_mean_coeff(t)
        std = self.marginal_prob(x, t, batch_idx, batch)[1]
        return maybe_expand(mean_coeff, batch=None, like=x), std

    def sample_marginal(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> torch.Tensor:
        mean, std = self.marginal_prob(x=x, t=t, batch=batch)
        z = torch.randn_like(x)
        return mean + expand(std, z.shape) * z

    def prior_sampling(
        self,
        shape: torch.Size | tuple,
        conditioning_data: BatchedData | None = None,
        batch_idx: B = None,
    ) -> torch.Tensor:
        assert conditioning_data is not None
        x_sample = torch.randn(*shape)
        device = conditioning_data["num_atoms"].device
        x_sample = x_sample.to(device)
        limit_mean = self.get_limit_mean(x=x_sample, batch=conditioning_data)
        limit_var = self.get_limit_var(x=x_sample, batch=conditioning_data)
        return x_sample * limit_var.sqrt() + limit_mean

    def sde(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch is not None
        limit_mean = self.get_limit_mean(x=x, batch=batch)
        limit_var = self.get_limit_var(x=x, batch=batch)
        beta_t = self.beta(t)
        drift = -0.5 * expand(beta_t, x.shape) * (x - limit_mean)
        diffusion = torch.sqrt(expand(beta_t, limit_var.shape) * limit_var)
        return maybe_expand(drift, batch_idx), maybe_expand(diffusion, batch_idx)


class GBTranslationPredictor(AncestralSamplingPredictor):
    """Ancestral-sampling predictor for ``gb_dx`` that honours ``GBTranslationVPSDE``'s non-zero terminal
    (limit) mean ``mu_uncond`` (, earlier reference
    ``DeltaAAncestralSamplingPredictor`` counterpart, corruption.py:245-ish).

    Bug this mirrors avoiding (earlier reference's own docstring, reviewer red-team review
    ``docs/gb/GB_STEP2_FAIL_REVIEW_20260712.md`` SS1): the generic ``AncestralSamplingPredictor``
    assumes a zero-limit-mean linear SDE; ``GBTranslationVPSDE``'s forward process has a non-zero
    ``(1-alpha_t)*mu_uncond`` term, so the reverse update needs the matching
    ``mean_coeff * get_limit_mean(...)`` correction (``prior_term`` below) -- omitting it injects a
    spurious drift toward zero instead of toward ``mu_uncond`` every reverse step (
    / AC-9a's explicit purpose: catching exactly this class of wiring bug).
    """

    @classmethod
    def is_compatible(cls, corruption) -> bool:
        _super = super()
        assert hasattr(_super, "is_compatible")
        return _super.is_compatible(corruption) or isinstance(corruption, GBTranslationVPSDE)

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
        batch: BatchedData | None,
    ) -> SampleAndMean:
        x_coeff, score_coeff, std = self._get_coeffs(
            x=x,
            t=t,
            dt=dt,
            batch_idx=batch_idx,
            batch=batch,
        )
        # mean = x_coeff * x + score_coeff * score + mean_coeff * limit_mean, mean_coeff = 1 - x_coeff
        # (same algebra as the earlier reference's DeltaAAncestralSamplingPredictor /
        # LatticeAncestralSamplingPredictor).
        mean_coeff = 1 - x_coeff
        z = torch.randn_like(x_coeff)  # plain (unsymmetrised) noise -- gb_dx is a scalar field.
        assert hasattr(self.corruption, "get_limit_mean")  # mypy
        x_drift = x_coeff * x
        score_term = score_coeff * score
        prior_term = mean_coeff * self.corruption.get_limit_mean(x=x, batch=batch)
        mean = x_drift + score_term + prior_term
        sample = mean + std * z
        return sample, mean
