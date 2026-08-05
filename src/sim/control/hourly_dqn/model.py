"""Joint-action utilities and network for masked Double DQN."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def joint_action_table(action_dims: Sequence[int]) -> np.ndarray:
    """Enumerate a MultiDiscrete action space in row-major order."""

    dims = tuple(int(value) for value in action_dims)
    if not dims or any(value <= 0 for value in dims):
        raise ValueError("action_dims must contain positive integers")
    return np.stack(
        np.meshgrid(*(np.arange(value) for value in dims), indexing="ij"),
        axis=-1,
    ).reshape(-1, len(dims))


def split_flat_action_mask(
    flat_mask: np.ndarray | Sequence[bool],
    action_dims: Sequence[int],
) -> tuple[np.ndarray, ...]:
    """Split the MaskablePPO-style concatenated per-dimension mask."""

    dims = tuple(int(value) for value in action_dims)
    mask = np.asarray(flat_mask, dtype=bool).reshape(-1)
    if mask.size != sum(dims):
        raise ValueError(
            f"flat action mask has {mask.size} entries; expected {sum(dims)}"
        )
    boundaries = np.cumsum(dims)[:-1]
    return tuple(np.split(mask, boundaries))


def joint_action_mask(
    per_dimension_masks: Sequence[np.ndarray | Sequence[bool]],
    action_table: np.ndarray,
) -> np.ndarray:
    """Return legal joint actions from independent per-vessel masks."""

    table = np.asarray(action_table, dtype=np.int64)
    masks = tuple(np.asarray(mask, dtype=bool).reshape(-1) for mask in per_dimension_masks)
    if table.ndim != 2 or table.shape[1] != len(masks):
        raise ValueError("action table and per-dimension masks do not align")
    legal = np.ones(len(table), dtype=bool)
    for dimension, mask in enumerate(masks):
        if table[:, dimension].max(initial=-1) >= len(mask):
            raise ValueError("action table index exceeds mask cardinality")
        legal &= mask[table[:, dimension]]
    if not legal.any():
        raise RuntimeError("physical action mask contains no legal joint action")
    return legal


class QNetwork(nn.Module):
    """Small MLP over the shared hourly state and future summary."""

    def __init__(
        self,
        observation_dim: int,
        action_count: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_count <= 0:
            raise ValueError("observation_dim and action_count must be positive")
        layers: list[nn.Module] = []
        width = int(observation_dim)
        for hidden_size in hidden_sizes:
            hidden_size = int(hidden_size)
            if hidden_size <= 0:
                raise ValueError("hidden_sizes must contain positive integers")
            layers.extend((nn.Linear(width, hidden_size), nn.ReLU()))
            width = hidden_size
        layers.append(nn.Linear(width, int(action_count)))
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class MaskedDoubleDQNPolicy:
    """Evaluation adapter exposing the same ``predict`` API as MaskablePPO."""

    def __init__(
        self,
        network: QNetwork,
        action_dims: Sequence[int],
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.action_dims = tuple(int(value) for value in action_dims)
        self.action_table = joint_action_table(self.action_dims)

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
        action_masks: np.ndarray | Sequence[bool] | None = None,
    ) -> tuple[np.ndarray, None]:
        if not deterministic:
            raise ValueError("formal DQN evaluation is deterministic")
        state = np.asarray(observation, dtype=np.float32)
        if state.ndim != 1:
            raise ValueError("predict expects one unbatched observation")
        if action_masks is None:
            legal = np.ones(len(self.action_table), dtype=bool)
        else:
            legal = joint_action_mask(
                split_flat_action_mask(action_masks, self.action_dims),
                self.action_table,
            )
        self.network.eval()
        with torch.no_grad():
            q_values = self.network(
                torch.as_tensor(state, device=self.device).unsqueeze(0)
            )[0]
            legal_tensor = torch.as_tensor(legal, device=self.device)
            action_index = int(
                q_values.masked_fill(~legal_tensor, -torch.inf).argmax().item()
            )
        return self.action_table[action_index].copy(), None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "MaskedDoubleDQNPolicy":
        checkpoint: dict[str, Any] = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("kind") != "hourly_masked_double_dqn":
            raise ValueError(f"{path} is not a masked Double-DQN checkpoint")
        network = QNetwork(
            int(checkpoint["observation_dim"]),
            int(checkpoint["action_count"]),
            tuple(int(value) for value in checkpoint["hidden_sizes"]),
        )
        network.load_state_dict(checkpoint["network_state_dict"])
        return cls(network, checkpoint["action_dims"], device=device)
