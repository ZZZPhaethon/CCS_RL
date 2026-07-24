"""Gymnasium adapter for the residual CCS semi-MDP.

残差 CCS 半 MDP 的 Gymnasium 适配器。
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional dependency.
    raise ImportError(
        "ResidualDispatchGymEnv requires gymnasium."
    ) from exc

from .env import ResidualDispatchEnv


class ResidualDispatchGymEnv(gym.Env):
    """Expose residual interventions as a compact ``Discrete`` task.

    将残差干预暴露为紧凑的 ``Discrete`` 任务。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: ResidualDispatchEnv,
        *,
        episode_seed_min: int = 100_000,
        episode_seed_max: int = 999_999,
    ) -> None:
        """Configure disjoint training-seed sampling.

        配置与验证集互不重叠的训练 seed 采样范围。
        """
        super().__init__()
        if episode_seed_min > episode_seed_max:
            raise ValueError("episode_seed_min must not exceed episode_seed_max.")
        self.env = env
        self.episode_seed_min = int(episode_seed_min)
        self.episode_seed_max = int(episode_seed_max)
        self.action_space = spaces.Discrete(env.action_count)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(env.observation_size,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset with a reproducible seed drawn only from the training range.

        使用仅来自训练范围的可复现 seed 重置环境。
        """
        super().reset(seed=seed)
        episode_seed = int(
            self.np_random.integers(
                self.episode_seed_min,
                self.episode_seed_max + 1,
            )
        )
        observation = self.env.reset(seed=episode_seed)
        return observation.astype(np.float32, copy=False), {
            "episode_seed": episode_seed,
        }

    def step(self, action):
        """Execute one residual action.

        执行一个残差动作。
        """
        observation, reward, terminated, truncated, info = self.env.step(
            int(action)
        )
        return (
            observation.astype(np.float32, copy=False),
            reward,
            terminated,
            truncated,
            info,
        )
