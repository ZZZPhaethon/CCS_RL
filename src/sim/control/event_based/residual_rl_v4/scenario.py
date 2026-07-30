"""Scenario generator with an explicit replay difficulty override.

支持显式重放难度覆盖的场景生成器。
"""

from __future__ import annotations

from sim.scenario_generation import (
    Scenario,
    ScenarioConfig,
    ScenarioGenerator,
)

from sim.control.event_based.residual_rl.scenario import (
    MixedDifficultyScenarioGenerator,
)


UNIFIED_WINDOW_STRESS_PROFILES = {
    "low": {
        "capture_high_output_mean_hours": 24.0,
        "capture_high_output_multiplier_range": (1.10, 1.40),
        "weather_window_mean_hours": 24.0,
        "weather_window_speed_factor_range": (0.70, 0.90),
        "well_maintenance_mean_hours": 6.0,
    },
    "medium": {
        "capture_high_output_mean_hours": 48.0,
        "capture_high_output_multiplier_range": (1.25, 1.75),
        "weather_window_mean_hours": 48.0,
        "weather_window_speed_factor_range": (0.50, 0.80),
        "well_maintenance_mean_hours": 12.0,
    },
    "high": {
        "capture_high_output_mean_hours": 96.0,
        "capture_high_output_multiplier_range": (1.50, 2.00),
        "weather_window_mean_hours": 96.0,
        "weather_window_speed_factor_range": (0.40, 0.70),
        "well_maintenance_mean_hours": 24.0,
    },
}


def unified_window_scenario_config(
    episode_hours: int,
    stress_level: str = "medium",
) -> ScenarioConfig:
    """Build the locked E4 stress profile without changing other factors."""

    if stress_level not in UNIFIED_WINDOW_STRESS_PROFILES:
        raise ValueError(f"unknown unified-window stress level: {stress_level}")
    stress = UNIFIED_WINDOW_STRESS_PROFILES[stress_level]
    return ScenarioConfig(
        episode_hours=int(episode_hours),
        time_step_hours=1.0,
        weather_process="window",
        capture_noise_std=0.30,
        capture_outage_rate_per_week=0.5,
        capture_outage_mean_hours=12.0,
        capture_high_output_rate_per_week=0.5,
        capture_high_output_mean_hours=stress[
            "capture_high_output_mean_hours"
        ],
        capture_high_output_multiplier_range=stress[
            "capture_high_output_multiplier_range"
        ],
        weather_window_rate_per_week=0.5,
        weather_window_mean_hours=stress["weather_window_mean_hours"],
        weather_window_speed_factor_range=stress[
            "weather_window_speed_factor_range"
        ],
        well_maintenance_rate_per_week=0.3,
        well_maintenance_mean_hours=stress[
            "well_maintenance_mean_hours"
        ],
        emitter_initial_fill_range=(0.0, 0.50),
        terminal_initial_fill_range=(0.0, 0.50),
        warm_start=True,
        reservoir_initial_pressure_fill_range=(0.0, 0.50),
    )


class ReplayableDifficultyScenarioGenerator(
    MixedDifficultyScenarioGenerator
):
    """Recreate a saved seed with its original sampled difficulty.

    使用训练时的原始难度重新生成已保存 seed。

    A curriculum changes ``hard_probability`` over time. Reusing only a seed
    could consequently turn a former hard episode into a normal episode.
    ``forced_difficulty`` prevents that drift during failure replay.

    课程会随时间改变 ``hard_probability``。如果只复用 seed，之前的困难
    episode 可能在之后变成普通 episode。失败重放期间通过
    ``forced_difficulty`` 避免这种难度漂移。
    """

    def __init__(
        self,
        *,
        scenario_protocol: str = "v4_mixed_window",
        stress_level: str = "medium",
        **kwargs,
    ) -> None:
        """Initialize the normal/hard generators.

        初始化普通与困难场景生成器。
        """
        super().__init__(**kwargs)
        if scenario_protocol not in {
            "v4_mixed_window",
            "unified_window_v1",
        }:
            raise ValueError(
                f"unknown scenario protocol: {scenario_protocol}"
            )
        self.scenario_protocol = scenario_protocol
        self.stress_level = stress_level
        if scenario_protocol == "unified_window_v1":
            episode_hours = int(kwargs["episode_hours"])
            weather_process = str(
                kwargs.get("weather_process", "window")
            )
            if weather_process != "window":
                raise ValueError(
                    "unified_window_v1 requires window weather"
                )
            config = unified_window_scenario_config(
                episode_hours,
                stress_level=stress_level,
            )
            self.normal = ScenarioGenerator(config)
            self.hard = ScenarioGenerator(config)
        self.forced_difficulty: str | None = None

    def set_forced_difficulty(
        self,
        difficulty: str | None,
    ) -> None:
        """Force the next sample to normal/hard, or restore mixture sampling.

        强制下一次采样为普通/困难，或恢复按混合概率采样。
        """
        if difficulty not in {None, "normal", "hard"}:
            raise ValueError(
                "difficulty must be None, 'normal', or 'hard'."
            )
        self.forced_difficulty = difficulty

    def sample(self, network, seed: int | None = None) -> Scenario:
        """Sample normally or reproduce the explicitly requested difficulty.

        正常采样，或重建显式指定难度的场景。
        """
        difficulty = self.forced_difficulty
        if difficulty is None:
            return super().sample(network, seed=seed)
        self.last_difficulty = difficulty
        generator = self.hard if difficulty == "hard" else self.normal
        return generator.sample(network, seed=seed)
