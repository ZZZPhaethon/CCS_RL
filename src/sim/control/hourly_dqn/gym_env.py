"""Discrete joint-action wrapper for the formal hourly control interface."""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional training dependency.
    raise ImportError(
        "HourlyJointActionDQNEnv requires gymnasium."
    ) from exc

from sim.control.hourly_ppo.gym_env import HourlyCentralizedPPOEnv

from .model import joint_action_mask, joint_action_table, split_flat_action_mask


class HourlyJointActionDQNEnv(gym.Wrapper):
    """Enumerate the hourly MultiDiscrete vessel action as one Discrete action."""

    def __init__(self, env: HourlyCentralizedPPOEnv) -> None:
        super().__init__(env)
        self.action_dims = tuple(int(value) for value in env.action_space.nvec)
        self.action_table = joint_action_table(self.action_dims)
        self.action_space = spaces.Discrete(len(self.action_table))
        self.observation_space = spaces.Dict(
            {
                "state": env.observation_space,
                "action_mask": spaces.MultiBinary(len(self.action_table)),
            }
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._packed_observation(observation), info

    def step(self, action):
        action_index = int(np.asarray(action).item())
        legal = self.action_masks()
        if action_index < 0 or action_index >= len(self.action_table):
            raise ValueError(f"joint action index out of range: {action_index}")
        if not legal[action_index]:
            raise ValueError(f"joint action {action_index} is physically illegal")
        observation, reward, terminated, truncated, info = self.env.step(
            self.action_table[action_index]
        )
        return (
            self._packed_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def action_masks(self) -> np.ndarray:
        return joint_action_mask(
            split_flat_action_mask(self.env.action_masks(), self.action_dims),
            self.action_table,
        )

    def training_simulator_usage(self) -> dict[str, int | float]:
        return self.env.training_simulator_usage()

    def _packed_observation(self, state: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "state": np.asarray(state, dtype=np.float32),
            "action_mask": self.action_masks(),
        }
