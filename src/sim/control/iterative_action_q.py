"""State-only structured action-value model for iterative policy training."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class EntityStateEncoder(nn.Module):
    """Encode shared system features and equal-sized per-vessel feature blocks."""

    def __init__(self, state_feature_names: list[str], output_features: int = 64):
        super().__init__()
        vessel_ids = sorted(
            {
                name.split(".")[1]
                for name in state_feature_names
                if name.startswith("greedy_proposal.")
            }
        )
        vessel_indices = []
        used = set()
        for vessel_id in vessel_ids:
            indices = [
                index
                for index, name in enumerate(state_feature_names)
                if name.startswith(f"{vessel_id}.")
                or name.startswith(f"greedy_proposal.{vessel_id}.")
            ]
            vessel_indices.append(indices)
            used.update(indices)
        if not vessel_indices or len({len(indices) for indices in vessel_indices}) != 1:
            raise ValueError("entity encoder requires equal per-vessel feature blocks")
        global_indices = [
            index for index in range(len(state_feature_names)) if index not in used
        ]
        self.register_buffer(
            "vessel_indices", torch.as_tensor(vessel_indices, dtype=torch.long)
        )
        self.register_buffer(
            "global_indices", torch.as_tensor(global_indices, dtype=torch.long)
        )
        self.vessel_encoder = nn.Sequential(
            nn.Linear(len(vessel_indices[0]), 32), nn.SiLU()
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(len(global_indices), 32), nn.SiLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(32 * (len(vessel_ids) + 1), output_features), nn.SiLU()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        vessel_state = state[:, self.vessel_indices].reshape(
            batch * self.vessel_indices.shape[0], self.vessel_indices.shape[1]
        )
        vessels = self.vessel_encoder(vessel_state).reshape(batch, -1)
        global_features = self.global_encoder(
            state.index_select(1, self.global_indices)
        )
        return self.fusion(torch.cat((global_features, vessels), dim=1))


class StatelessGRUGate(nn.Module):
    """One GRU transform with a fixed zero incoming hidden state."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_r, input_z, input_n = F.linear(
            inputs, self.weight_ih, self.bias_ih
        ).chunk(3, dim=-1)
        hidden_r, hidden_z, hidden_n = self.bias_hh.chunk(3)
        reset = torch.sigmoid(input_r + hidden_r)
        update = torch.sigmoid(input_z + hidden_z)
        candidate = torch.tanh(input_n + reset * hidden_n)
        return (1.0 - update) * candidate


class IterativeActionQuantileQ(nn.Module):
    """State-only ensemble quantile Q-network over structured vessel actions."""

    is_stateless = True

    def __init__(
        self,
        state_feature_names: list[str],
        joint_actions: np.ndarray | list[list[int]],
        *,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        return_scale: float,
        heads: int = 5,
        quantiles: int = 51,
        hidden_size: int = 128,
        prior_scale: float = 0.25,
        action_embedding_size: int = 16,
        action_feature_size: int = 64,
    ) -> None:
        super().__init__()
        joint_array = np.asarray(joint_actions, dtype=np.int64)
        if joint_array.ndim != 2 or len(joint_array) == 0:
            raise ValueError("joint_actions must be a non-empty action-by-vessel matrix")
        if (joint_array < 0).any():
            raise ValueError("joint action indices must be non-negative")

        self.action_count = int(len(joint_array))
        self.heads = int(heads)
        self.quantiles = int(quantiles)
        self.return_scale = float(return_scale)
        self.prior_scale = float(prior_scale)
        self.action_feature_size = int(action_feature_size)
        self.register_buffer(
            "state_mean", torch.as_tensor(state_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "state_std", torch.as_tensor(state_std, dtype=torch.float32)
        )
        self.register_buffer(
            "joint_action_indices", torch.as_tensor(joint_array, dtype=torch.long)
        )

        self.state_encoder = EntityStateEncoder(state_feature_names, 64)
        self.state_projection = nn.Sequential(nn.Linear(64, hidden_size), nn.SiLU())
        self.stateless_gate = StatelessGRUGate(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, self.heads * self.quantiles)

        cardinalities = [
            int(joint_array[:, index].max()) + 1
            for index in range(joint_array.shape[1])
        ]
        self.structured_action_embeddings = nn.ModuleList(
            [nn.Embedding(size, action_embedding_size) for size in cardinalities]
        )
        self.structured_action_fusion = nn.Sequential(
            nn.Linear(
                len(cardinalities) * action_embedding_size,
                action_feature_size,
            ),
            nn.SiLU(),
            nn.LayerNorm(action_feature_size, elementwise_affine=False),
        )
        self.structured_query = nn.Linear(
            hidden_size, self.heads * self.quantiles * action_feature_size
        )

        self.structured_prior_embeddings = nn.ModuleList(
            [nn.Embedding(size, action_embedding_size) for size in cardinalities]
        )
        self.structured_prior_fusion = nn.Sequential(
            nn.Linear(
                len(cardinalities) * action_embedding_size,
                action_feature_size,
            ),
            nn.SiLU(),
            nn.LayerNorm(action_feature_size, elementwise_affine=False),
        )
        self.structured_prior_query = nn.Linear(
            hidden_size, self.heads * self.quantiles * action_feature_size
        )
        for module in (
            self.structured_prior_embeddings,
            self.structured_prior_fusion,
            self.structured_prior_query,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _action_features(self, embeddings, fusion) -> torch.Tensor:
        pieces = [
            embedding(self.joint_action_indices[:, vessel_index])
            for vessel_index, embedding in enumerate(embeddings)
        ]
        return fusion(torch.cat(pieces, dim=-1))

    def quantiles_from_features(self, features: torch.Tensor) -> torch.Tensor:
        batch, sequence = features.shape[:2]
        query = self.structured_query(features).reshape(
            batch,
            sequence,
            self.heads,
            self.quantiles,
            self.action_feature_size,
        )
        action_features = self._action_features(
            self.structured_action_embeddings, self.structured_action_fusion
        )
        advantage = torch.einsum("bshqk,ak->bshaq", query, action_features)
        advantage = advantage / np.sqrt(self.action_feature_size)

        prior_query = self.structured_prior_query(features.detach()).reshape_as(query)
        prior_features = self._action_features(
            self.structured_prior_embeddings, self.structured_prior_fusion
        )
        prior = torch.einsum("bshqk,ak->bshaq", prior_query, prior_features)
        prior = prior / np.sqrt(self.action_feature_size)

        value = self.value(features).reshape(
            batch, sequence, self.heads, 1, self.quantiles
        )
        q = value + advantage - advantage.mean(dim=3, keepdim=True)
        return q + self.prior_scale * prior

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        batch, sequence = states.shape[:2]
        normalized_state = (states - self.state_mean) / self.state_std
        encoded = self.state_encoder(normalized_state.reshape(batch * sequence, -1))
        projected = self.state_projection(encoded)
        features = self.stateless_gate(projected).reshape(batch, sequence, -1)
        return self.quantiles_from_features(features)


def quantile_huber_loss(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    *,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Pairwise QR-DQN loss; final axes are predicted and target quantiles."""

    if predicted.shape[-1] <= 0 or targets.shape[-1] <= 0:
        raise ValueError("quantile tensors must have non-empty final axes")
    error = targets.unsqueeze(-2) - predicted.unsqueeze(-1)
    absolute_error = error.abs()
    huber = torch.where(
        absolute_error <= kappa,
        0.5 * error.square(),
        kappa * (absolute_error - 0.5 * kappa),
    )
    count = predicted.shape[-1]
    tau = (
        torch.arange(count, device=predicted.device, dtype=predicted.dtype) + 0.5
    ) / count
    view_shape = [1] * (error.ndim - 2) + [count, 1]
    weight = (tau.reshape(view_shape) - (error.detach() < 0).to(predicted.dtype)).abs()
    return (weight * huber / kappa).mean(dim=(-2, -1))
