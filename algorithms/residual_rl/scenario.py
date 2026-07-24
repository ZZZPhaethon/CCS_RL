"""Generate a reproducible normal/hard mixture for residual-RL training.

为残差 RL 训练生成可复现的普通/困难混合场景。
"""

from __future__ import annotations

import random

from Simulation.scenario_generation import Scenario, ScenarioConfig, ScenarioGenerator


class MixedDifficultyScenarioGenerator:
    """Sample normal scenarios and a configurable fraction of hard scenarios.

    按可配置比例采样普通场景和困难场景。

    The difficulty choice is a deterministic function of the episode seed.
    Validation and training can therefore use disjoint seed ranges without
    silently changing scenario difficulty between repeated runs.

    难度选择是 episode seed 的确定性函数，因此训练与验证可以使用互不重叠的
    seed 范围，并在重复实验中保持难度不变。
    """

    def __init__(
        self,
        *,
        episode_hours: int,
        weather_process: str = "window",
        hard_probability: float = 0.30,
    ) -> None:
        """Build normal and tail-risk generators.

        创建普通场景与尾部风险场景生成器。
        """
        if not 0.0 <= hard_probability <= 1.0:
            raise ValueError("hard_probability must be inside [0, 1].")
        self.hard_probability = float(hard_probability)
        self.normal = ScenarioGenerator(
            ScenarioConfig(
                episode_hours=episode_hours,
                time_step_hours=1.0,
                weather_process=weather_process,
            )
        )
        self.hard = ScenarioGenerator(
            ScenarioConfig(
                episode_hours=episode_hours,
                time_step_hours=1.0,
                weather_process=weather_process,
                capture_noise_std=0.40,
                capture_high_output_rate_per_week=0.9,
                capture_high_output_mean_hours=60.0,
                capture_high_output_multiplier_range=(1.35, 1.85),
                weather_window_rate_per_week=0.7,
                weather_window_mean_hours=60.0,
                weather_window_speed_factor_range=(0.45, 0.72),
                well_maintenance_rate_per_week=0.5,
                well_maintenance_mean_hours=30.0,
                emitter_initial_fill_range=(0.25, 0.75),
                terminal_initial_fill_range=(0.10, 0.65),
            )
        )
        self.last_difficulty = "normal"

    def sample(self, network, seed: int | None = None) -> Scenario:
        """Sample one reproducible normal or hard scenario.

        采样一个可复现的普通或困难场景。
        """
        chooser = random.Random(seed)
        is_hard = chooser.random() < self.hard_probability
        self.last_difficulty = "hard" if is_hard else "normal"
        generator = self.hard if is_hard else self.normal
        return generator.sample(network, seed=seed)
