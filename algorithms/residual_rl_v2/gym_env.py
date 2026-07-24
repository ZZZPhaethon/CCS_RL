"""Gymnasium adapter exposing dynamic masks for MaskablePPO.

向 MaskablePPO 暴露动态掩码的 Gymnasium 适配器。
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional dependency.
    raise ImportError(
        "MaskedResidualGymEnv requires gymnasium."
    ) from exc

from .env import MaskedResidualDispatchEnv


class MaskedResidualGymEnv(gym.Env):
    """Expose the masked residual semi-MDP as a discrete Gym task.

    将掩码残差半 MDP 暴露为离散 Gym 任务。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: MaskedResidualDispatchEnv,
        *,
        episode_seed_min: int = 100_000,
        episode_seed_max: int = 999_999,
    ) -> None:
        """Configure spaces and a training-only scenario seed range.

        配置空间和仅供训练使用的场景 seed 范围。
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

    def action_masks(self) -> np.ndarray:
        """Return the current boolean mask required by MaskablePPO.

        返回 MaskablePPO 所需的当前布尔动作掩码。
        """
        return self.env.action_masks()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset with a reproducible seed from the training range.

        使用训练范围内的可复现 seed 重置环境。
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
        """Execute one action already filtered by MaskablePPO.

        执行一个已被 MaskablePPO 掩码过滤的动作。
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
