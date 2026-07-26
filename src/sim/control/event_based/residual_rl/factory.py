"""Factories for residual native and Gymnasium environments.

残差原生环境与 Gymnasium 环境的工厂函数。
"""

from __future__ import annotations

from sim.environment import CCSEnvConfig, build_phase1_env

from sim.control.event_based.rl.reward import HighLevelRewardConfig

from .env import ResidualDispatchEnv, ResidualEnvConfig
from .gym_env import ResidualDispatchGymEnv
from .scenario import MixedDifficultyScenarioGenerator


def make_residual_native_env(
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
) -> ResidualDispatchEnv:
    """Build the residual environment with 168 h disturbance context.

    构建带 168 小时扰动上下文的残差环境。
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
    return ResidualDispatchEnv(
        physical_env,
        config=ResidualEnvConfig(
            decision_interval_h=decision_interval_h,
            event_triggered=event_triggered,
            reward=reward or HighLevelRewardConfig(),
        ),
    )


def make_residual_gym_env(
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
) -> ResidualDispatchGymEnv:
    """Build one Gymnasium residual environment for vector training.

    构建一个用于向量化训练的 Gymnasium 残差环境。
    """
    native = make_residual_native_env(
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        decision_interval_h=decision_interval_h,
        event_triggered=event_triggered,
        weather_mode=weather_mode,
        hard_scenario_probability=hard_scenario_probability,
        reward=reward,
    )
    return ResidualDispatchGymEnv(
        native,
        episode_seed_min=episode_seed_min,
        episode_seed_max=episode_seed_max,
    )

