# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from typing import Any, TypeVar

import torch

from mattergen.diffusion.corruption.sde_lib import SDE
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.model_target import ModelTarget

T = TypeVar("T", bound=BatchedData)


def convert_model_out_to_score(
    *,
    model_target: ModelTarget,
    sde: SDE,
    model_out: torch.Tensor,
    batch_idx: torch.LongTensor,
    t: torch.Tensor,
    batch: Any
) -> torch.Tensor:
    """
    Convert a model output to a score, according to the specified model_target.

    model_target: says what the model predicts.
        For example, in RFDiffusion the model predicts clean coordinates;
        in EDM the model predicts the raw noise.
    sde: corruption process
    model_out: model output
    batch_idx: indicates which sample each row of model_out belongs to
    noisy_x: noisy data
    t: diffusion timestep
    batch: noisy batch, ignored except by strange SDEs
    """
    _, std = sde.marginal_prob(
        x=torch.ones_like(model_out),
        t=t,
        batch_idx=batch_idx,
        batch=batch,
    )
    # Note the slack tolerances in test_model_utils.py: the choice of ModelTarget does make a difference.
    if model_target == ModelTarget.score_times_std:
        return model_out / std
    elif model_target == ModelTarget.logits:
        # Not really a score, but logits will be handled downstream.
        return model_out
    else:
        raise NotImplementedError


class NoiseLevelEncoding(torch.nn.Module):
    """
    From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor, shape [batch_size]
        """
        x = torch.zeros((t.shape[0], self.d_model), device=self.div_term.device)
        x[:, 0::2] = torch.sin(t[:, None] * self.div_term[None])
        x[:, 1::2] = torch.cos(t[:, None] * self.div_term[None])
        return self.dropout(x)

class CategoryValueEncoding(torch.nn.Module):
    """
    Learnable embedding for categorical (discrete) properties.

    Args:
    ----
    n_classes: int
        Total number of categories (e.g., 2 for binary).
    d_model: int
        Output vector dimension (align with other node/edge features).
    padding_idx: int | None, default None
        Index to use for missing values. None means no padding.
    dropout: float, default 0.0
        Dropout rate for embedding output.
    """

    def __init__(
        self,
        n_classes: int,
        d_model: int,
        padding_idx: int | None = None,
        dropout: float = 0.0,
        atol: float = 1e-8,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=n_classes,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        # Xavier initialization (the PyTorch default would also do, but it is stated explicitly)
        torch.nn.init.xavier_uniform_(self.embedding.weight)

        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0.0)

        self.dropout = torch.nn.Dropout(p=dropout)
        self.atol = atol

    def _to_long_idx(self, idx: torch.Tensor) -> torch.LongTensor:
        if idx.dtype == torch.bool:
            return idx.long()

        if torch.is_floating_point(idx):
            assert torch.allclose(idx, idx.round(), atol=self.atol), f"Unable to convert FloatTensor to LongTensor: {idx}"
            return idx.round().long()

        if idx.dtype in (torch.int32, torch.int64):
            return idx.long()

        assert False, f"Unsupported tensor dtype: {idx.dtype}"

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        ----------
        idx : Tensor[*,]  (Bool / Long / Float)

        Returns:
        -------
        Tensor[* , d_model]
        """
        idx_long = self._to_long_idx(idx)
        x = self.embedding(idx_long)
        return self.dropout(x)


class ScalarMLPEncoding(torch.nn.Module):
    """Learnable MLP encoding for scalar properties.

    Replaces sinusoidal NoiseLevelEncoding for narrow-range continuous values
    (e.g. interface_energy) where the sinusoidal basis cannot distinguish
    closely spaced inputs (cos_sim > 0.99).

    Args:
        n_layers: number of linear layers (default 2 for v2b compat, 3 for v3).
        normalize: if True, L2-normalize the output and scale to ||out||=sqrt(d_model),
            so the embedding norm is constant regardless of the input value.
    """

    def __init__(self, d_model: int, dropout: float = 0.0,
                 normalize: bool = False, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.normalize = normalize
        layers: list[torch.nn.Module] = [torch.nn.Linear(1, d_model), torch.nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [torch.nn.Linear(d_model, d_model), torch.nn.SiLU()]
        layers.pop()
        self.net = torch.nn.Sequential(*layers)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        out = self.net(t)
        if self.normalize:
            out = torch.nn.functional.normalize(out, dim=-1) * math.sqrt(self.d_model)
        return self.dropout(out)
