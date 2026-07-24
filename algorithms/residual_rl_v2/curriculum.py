"""Curriculum utilities for masked residual PPO training.

用于掩码残差 PPO 训练的课程学习工具。
"""

from __future__ import annotations

from dataclasses import dataclass

from algorithms.rl.reward import HighLevelRewardConfig

from .factory import make_masked_residual_native_env
from .gym_env import MaskedResidualGymEnv


@dataclass(frozen=True)
class CurriculumStage:
    """Describe one difficulty stage starting at a training fraction.

    描述一个从指定训练进度开始的难度阶段。
    """

    start_fraction: float
    hard_probability: float


DEFAULT_CURRICULUM = (
    CurriculumStage(0.00, 0.00),
    CurriculumStage(0.20, 0.15),
    CurriculumStage(0.40, 0.30),
    CurriculumStage(0.70, 0.50),
)


def validate_curriculum(
    stages: tuple[CurriculumStage, ...],
) -> tuple[CurriculumStage, ...]:
    """Validate and return an ordered curriculum schedule.

    校验并返回按顺序排列的课程表。
    """
    if not stages:
        raise ValueError("At least one curriculum stage is required.")
    ordered = tuple(sorted(stages, key=lambda stage: stage.start_fraction))
    if ordered[0].start_fraction != 0.0:
        raise ValueError("The first curriculum stage must start at 0.")
    previous = -1.0
    for stage in ordered:
        if not 0.0 <= stage.start_fraction < 1.0:
            raise ValueError("Stage fractions must be inside [0, 1).")
        if stage.start_fraction <= previous:
            raise ValueError("Stage fractions must be strictly increasing.")
        if not 0.0 <= stage.hard_probability <= 1.0:
            raise ValueError("Hard probabilities must be inside [0, 1].")
        previous = stage.start_fraction
    return ordered


def parse_curriculum_specs(
    values: list[str],
) -> tuple[CurriculumStage, ...]:
    """Parse CLI values formatted as ``fraction:hard_probability``.

    解析格式为 ``训练比例:困难概率`` 的命令行参数。
    """
    stages: list[CurriculumStage] = []
    for value in values:
        try:
            fraction_text, probability_text = value.split(":", maxsplit=1)
            stage = CurriculumStage(
                float(fraction_text),
                float(probability_text),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Curriculum stages must use 'fraction:probability'."
            ) from exc
        stages.append(stage)
    return validate_curriculum(tuple(stages))


def curriculum_stage_index(
    stages: tuple[CurriculumStage, ...],
    progress_fraction: float,
) -> int:
    """Return the active stage index for a training fraction.

    返回给定训练进度对应的活动阶段索引。
    """
    index = 0
    for position, stage in enumerate(stages):
        if progress_fraction + 1e-12 < stage.start_fraction:
            break
        index = position
    return index


class CurriculumMaskedResidualGymEnv(MaskedResidualGymEnv):
    """Expose a runtime-adjustable hard-scenario probability.

    暴露可在训练过程中调整的困难场景概率。
    """

    def set_hard_probability(self, probability: float) -> float:
        """Update the mixture used by future episode resets.

        更新后续 episode 重置时使用的场景混合比例。
        """
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("probability must be inside [0, 1].")
        generator = self.env.env.scenario_generator
        if not hasattr(generator, "hard_probability"):
            raise TypeError(
                "Curriculum requires MixedDifficultyScenarioGenerator."
            )
        generator.hard_probability = float(probability)
        return float(generator.hard_probability)

    def get_hard_probability(self) -> float:
        """Return the probability used for the next reset.

        返回下一次重置时使用的困难场景概率。
        """
        return float(self.env.env.scenario_generator.hard_probability)

    def reset(self, *, seed=None, options=None):
        """Reset and expose the sampled difficulty in ``info``.

        重置环境，并在 ``info`` 中暴露采样难度。
        """
        observation, info = super().reset(seed=seed, options=options)
        generator = self.env.env.scenario_generator
        info["scenario_difficulty"] = generator.last_difficulty
        info["hard_probability"] = float(generator.hard_probability)
        return observation, info

    def step(self, action):
        """Step the environment and attach curriculum diagnostics.

        推进环境，并附加课程学习诊断信息。
        """
        observation, reward, terminated, truncated, info = super().step(
            action
        )
        generator = self.env.env.scenario_generator
        info["scenario_difficulty"] = generator.last_difficulty
        info["hard_probability"] = float(generator.hard_probability)
        return observation, reward, terminated, truncated, info


def make_curriculum_masked_residual_gym_env(
    *,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    decision_interval_h: float = 24.0,
    event_triggered: bool = True,
    weather_mode: str = "window",
    initial_hard_probability: float = 0.0,
    reward: HighLevelRewardConfig | None = None,
    episode_seed_min: int = 100_000,
    episode_seed_max: int = 999_999,
) -> CurriculumMaskedResidualGymEnv:
    """Build one curriculum-aware Gym training environment.

    构建一个支持课程学习的 Gym 训练环境。
    """
    native = make_masked_residual_native_env(
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        decision_interval_h=decision_interval_h,
        event_triggered=event_triggered,
        weather_mode=weather_mode,
        hard_scenario_probability=initial_hard_probability,
        reward=reward,
    )
    return CurriculumMaskedResidualGymEnv(
        native,
        episode_seed_min=episode_seed_min,
        episode_seed_max=episode_seed_max,
    )
