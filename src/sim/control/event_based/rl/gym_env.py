"""Gymnasium adapter for the sparse high-level dispatch environment.

稀疏高层调度环境的 Gymnasium 适配器。
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional training dependency.
    raise ImportError(
        "HighLevelDispatchGymEnv requires gymnasium. Install with `pip install gymnasium`."
    ) from exc

from .high_level_env import HighLevelDispatchEnv


class HighLevelDispatchGymEnv(gym.Env):
    """Expose :class:`HighLevelDispatchEnv` as a Gymnasium ``Discrete`` task.

    将 :class:`HighLevelDispatchEnv` 暴露为 Gymnasium ``Discrete`` 任务。
    """

    metadata = {"render_modes": []}

    def __init__(self, env: HighLevelDispatchEnv) -> None:
        """Initialise action and observation spaces from the wrapped environment.

        根据被包装环境初始化动作与观测空间。
        """
        super().__init__()
        self.env = env
        self.action_space = spaces.Discrete(env.action_count)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(env.observation_size,),
            dtype=np.float32,
        )

    def action_masks(self) -> np.ndarray:
        """Return the legal-action mask consumed by MaskablePPO."""

        return self.env.action_masks()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Sample a reproducible per-episode scenario seed and reset.

        采样可复现的每回合场景种子并重置环境。
        """
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        observation = self.env.reset(seed=episode_seed)
        return observation.astype(np.float32, copy=False), {}

    def step(self, action):
        """Execute one sparse high-level action.

        执行一个稀疏高层动作。
        """
        observation, reward, terminated, truncated, info = self.env.step(int(action))
        return observation.astype(np.float32, copy=False), reward, terminated, truncated, info
