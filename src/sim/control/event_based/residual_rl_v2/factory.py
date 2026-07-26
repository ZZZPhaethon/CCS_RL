"""Factories for masked residual environments.

掩码残差环境的工厂函数。
"""

from __future__ import annotations

from sim.environment import CCSEnvConfig, build_phase1_env

from sim.control.event_based.residual_rl.scenario import MixedDifficultyScenarioGenerator
from sim.control.event_based.rl.reward import HighLevelRewardConfig

from .env import MaskedResidualDispatchEnv, MaskedResidualEnvConfig
from .gym_env import MaskedResidualGymEnv


def make_masked_residual_native_env(
    *,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    decision_interval_h: float = 24.0,
    event_triggered: bool = True,
    weather_mode: str = "window",
    hard_scenario_probability: float = 0.30,
    reward: HighLevelRewardConfig | None = None,
    scenario_generator=None,
) -> MaskedResidualDispatchEnv:
    """Build a native masked residual environment.

    构建原生掩码残差环境。
    """
    if weather_mode not in {"window", "block"}:
        raise ValueError("weather_mode must be 'window' or 'block'.")
    generator = scenario_generator or MixedDifficultyScenarioGenerator(
        episode_hours=episode_hours + forecast_context_hours,
        weather_process=weather_mode,
        hard_probability=hard_scenario_probability,
    )
    physical_env = build_phase1_env(
        scenario=scenario,
        scenario_generator=generator,
        weather_mode=weather_mode,
        config=CCSEnvConfig(
            episode_hours=episode_hours,
            include_goal_obs=False,
            reward_mode="vent_first",
        ),
    )
    return MaskedResidualDispatchEnv(
        physical_env,
        config=MaskedResidualEnvConfig(
            decision_interval_h=decision_interval_h,
            event_triggered=event_triggered,
            reward=reward or HighLevelRewardConfig(),
        ),
    )


def make_masked_residual_gym_env(
    *,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    decision_interval_h: float = 24.0,
    event_triggered: bool = True,
    weather_mode: str = "window",
    hard_scenario_probability: float = 0.30,
    reward: HighLevelRewardConfig | None = None,
    episode_seed_min: int = 100_000,
    episode_seed_max: int = 999_999,
) -> MaskedResidualGymEnv:
    """Build one Gym environment for vectorised MaskablePPO.

    构建一个用于向量化 MaskablePPO 的 Gym 环境。
    """
    native = make_masked_residual_native_env(
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        decision_interval_h=decision_interval_h,
        event_triggered=event_triggered,
        weather_mode=weather_mode,
        hard_scenario_probability=hard_scenario_probability,
        reward=reward,
    )
    return MaskedResidualGymEnv(
        native,
        episode_seed_min=episode_seed_min,
        episode_seed_max=episode_seed_max,
    )

