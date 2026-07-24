"""Gymnasium adapter for the sparse vessel-dispatch environment.

Use this adapter with ordinary MaskablePPO only with ``gamma=1.0``.  Each Gym
step aggregates a variable number of hourly rewards, while standard SB3 PPO has
one fixed discount per Gym transition.  A time-aware PPO is needed to support
the exact hourly ``gamma ** elapsed_hours`` objective.
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("EventDrivenGymEnv requires gymnasium. Install the RL extra.") from exc

from .event_driven import EventDrivenCCSEnv
from .gym_adapter import flat_action_mask, native_action_from_flat


class EventDrivenGymEnv(gym.Env):
    """MaskablePPO view of :class:`EventDrivenCCSEnv`."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: EventDrivenCCSEnv,
        *,
        observation_provider=None,
        observation_size: int | None = None,
    ) -> None:
        super().__init__()
        self.env = env
        self.observation_provider = observation_provider
        native = env.env
        self.action_space = spaces.MultiDiscrete(native.vessel_action_dims + native.well_rate_action_dims)
        size = native.observation_size if observation_provider is None else observation_size
        if size is None or int(size) <= 0:
            raise ValueError("custom event observations require a positive observation_size")
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(int(size),),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        transition = self.env.reset(seed=episode_seed)
        return self._observation(transition.observation), transition.info

    def step(self, action):
        transition = self.env.step(native_action_from_flat(self.env.env, action))
        return (
            self._observation(transition.observation),
            float(transition.reward),
            transition.terminated,
            transition.truncated,
            transition.info,
        )

    def action_masks(self) -> np.ndarray:
        native = self.env.env
        event = self.env.current_event
        if event is None:
            raise RuntimeError("No active DecisionEvent; reset or step the environment first.")
        vessel_masks = [
            list(event.vessel_action_masks[vessel_id])
            if vessel_id in event.vessel_action_masks
            else [True] + [False] * (native.vessel_action_count - 1)
            for vessel_id in native.vessel_ids
        ]
        return flat_action_mask(vessel_masks, native.well_rate_action_mask())

    @staticmethod
    def _to_array(observation: list[float]) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32)

    def _observation(self, fallback: list[float]) -> np.ndarray:
        if self.observation_provider is None:
            result = self._to_array(fallback)
        else:
            result = np.asarray(self.observation_provider(self.env), dtype=np.float32)
        if result.shape != self.observation_space.shape:
            raise ValueError(
                f"event observation has shape {result.shape}, expected {self.observation_space.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("event observation must contain only finite values")
        return result
