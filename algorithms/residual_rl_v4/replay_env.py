"""Curriculum Gym environment with top-failure episode replay.

带高放空失败场景重放的课程 Gym 环境。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from algorithms.residual_rl_v2.curriculum import (
    CurriculumMaskedResidualGymEnv,
)

from .scenario import ReplayableDifficultyScenarioGenerator


@dataclass(frozen=True)
class FailureEpisode:
    """Store one reproducible training failure.

    保存一个可复现的训练失败场景。
    """

    seed: int
    difficulty: str
    vented_t: float


class TailFailureReplayGymEnv(CurriculumMaskedResidualGymEnv):
    """Replay the highest-vent training episodes with fixed probability.

    按固定概率重放放空量最高的训练 episode。

    Each vector worker owns an independent bounded replay pool. This keeps
    ``SubprocVecEnv`` simple and avoids cross-process mutable state.

    每个向量环境 worker 拥有独立且有界的回放池，从而保持
    ``SubprocVecEnv`` 简单，并避免跨进程可变状态。
    """

    def __init__(
        self,
        env,
        *,
        episode_seed_min: int = 100_000,
        episode_seed_max: int = 999_999,
        replay_probability: float = 0.30,
        replay_capacity: int = 20,
        minimum_replay_pool: int = 4,
    ) -> None:
        """Configure the seed range and top-failure replay pool.

        配置训练 seed 范围和高放空失败回放池。
        """
        super().__init__(
            env,
            episode_seed_min=episode_seed_min,
            episode_seed_max=episode_seed_max,
        )
        if not 0.0 <= replay_probability <= 1.0:
            raise ValueError("replay_probability must be inside [0, 1].")
        if replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive.")
        if minimum_replay_pool <= 0:
            raise ValueError("minimum_replay_pool must be positive.")
        if minimum_replay_pool > replay_capacity:
            raise ValueError(
                "minimum_replay_pool must not exceed replay_capacity."
            )
        self.replay_probability = float(replay_probability)
        self.replay_capacity = int(replay_capacity)
        self.minimum_replay_pool = int(minimum_replay_pool)
        self._failures: dict[tuple[int, str], FailureEpisode] = {}
        self._episode_seed: int | None = None
        self._episode_difficulty = "normal"
        self._episode_from_replay = False
        self._episodes_completed = 0
        self._replay_episodes_completed = 0

    @property
    def scenario_generator(
        self,
    ) -> ReplayableDifficultyScenarioGenerator:
        """Return and type-check the replay-aware generator.

        返回并检查支持重放的场景生成器。
        """
        generator = self.env.env.scenario_generator
        if not isinstance(
            generator,
            ReplayableDifficultyScenarioGenerator,
        ):
            raise TypeError(
                "TailFailureReplayGymEnv requires "
                "ReplayableDifficultyScenarioGenerator."
            )
        return generator

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """Reset from a fresh training seed or the local failure pool.

        使用新训练 seed 或本地失败池重置环境。
        """
        del options
        gym.Env.reset(self, seed=seed)
        failure = self._sample_replay_failure()
        if failure is None:
            episode_seed = int(
                self.np_random.integers(
                    self.episode_seed_min,
                    self.episode_seed_max + 1,
                )
            )
            forced_difficulty = None
            from_replay = False
        else:
            episode_seed = failure.seed
            forced_difficulty = failure.difficulty
            from_replay = True

        generator = self.scenario_generator
        generator.set_forced_difficulty(forced_difficulty)
        try:
            observation = self.env.reset(seed=episode_seed)
        finally:
            generator.set_forced_difficulty(None)

        self._episode_seed = episode_seed
        self._episode_difficulty = generator.last_difficulty
        self._episode_from_replay = from_replay
        info = {
            "episode_seed": episode_seed,
            "scenario_difficulty": self._episode_difficulty,
            "hard_probability": float(generator.hard_probability),
            "failure_replay": from_replay,
            "replay_buffer_size": len(self._failures),
        }
        return observation.astype(np.float32, copy=False), info

    def step(self, action):
        """Advance the episode and record terminal failures.

        推进 episode，并在结束时记录失败场景。
        """
        observation, reward, terminated, truncated, info = self.env.step(
            int(action)
        )
        done = bool(terminated or truncated)
        if done:
            self._record_completed_episode()
        info.update(
            {
                "scenario_difficulty": self._episode_difficulty,
                "hard_probability": float(
                    self.scenario_generator.hard_probability
                ),
                "episode_seed": self._episode_seed,
                "failure_replay": self._episode_from_replay,
                "replay_buffer_size": len(self._failures),
            }
        )
        return (
            observation.astype(np.float32, copy=False),
            reward,
            terminated,
            truncated,
            info,
        )

    def get_replay_snapshot(self) -> dict[str, Any]:
        """Return JSON-serializable replay diagnostics.

        返回可写入 JSON 的回放诊断信息。
        """
        ranked = self._ranked_failures()
        return {
            "replay_probability": self.replay_probability,
            "replay_capacity": self.replay_capacity,
            "minimum_replay_pool": self.minimum_replay_pool,
            "buffer_size": len(ranked),
            "episodes_completed": self._episodes_completed,
            "replay_episodes_completed": (
                self._replay_episodes_completed
            ),
            "replay_episode_rate": (
                self._replay_episodes_completed
                / max(1, self._episodes_completed)
            ),
            "failures": [asdict(record) for record in ranked],
        }

    def _sample_replay_failure(self) -> FailureEpisode | None:
        """Sample a high-vent failure using vent-weighted probabilities.

        根据放空量权重采样高放空失败场景。
        """
        failures = self._ranked_failures()
        if len(failures) < self.minimum_replay_pool:
            return None
        if float(self.np_random.random()) >= self.replay_probability:
            return None
        weights = np.asarray(
            [max(1.0, record.vented_t) for record in failures],
            dtype=np.float64,
        )
        probabilities = weights / weights.sum()
        index = int(
            self.np_random.choice(
                len(failures),
                p=probabilities,
            )
        )
        return failures[index]

    def _record_completed_episode(self) -> None:
        """Insert or update the current seed and keep only the worst cases.

        插入或更新当前 seed，并只保留最严重的失败场景。
        """
        if self._episode_seed is None:
            return
        vented_t = float(self.env.env.ledger.vented_t)
        key = (self._episode_seed, self._episode_difficulty)
        self._failures[key] = FailureEpisode(
            seed=self._episode_seed,
            difficulty=self._episode_difficulty,
            vented_t=vented_t,
        )
        keep = self._ranked_failures()[: self.replay_capacity]
        self._failures = {
            (record.seed, record.difficulty): record
            for record in keep
        }
        self._episodes_completed += 1
        self._replay_episodes_completed += int(
            self._episode_from_replay
        )

    def _ranked_failures(self) -> list[FailureEpisode]:
        """Return failures from highest to lowest venting.

        按放空量从高到低返回失败场景。
        """
        return sorted(
            self._failures.values(),
            key=lambda record: (
                -record.vented_t,
                record.seed,
                record.difficulty,
            ),
        )
