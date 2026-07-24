"""Factories for risk-gated residual RL v3 environments.

风险门控残差强化学习 v3 的环境工厂。
"""

from __future__ import annotations

from algorithms.rl.reward import HighLevelRewardConfig
from Simulation.environment import CCSEnvConfig, build_phase1_env

from algorithms.residual_rl.scenario import MixedDifficultyScenarioGenerator
from algorithms.residual_rl_v2.curriculum import (
    CurriculumMaskedResidualGymEnv,
)
from algorithms.residual_rl_v2.env import MaskedResidualEnvConfig
from algorithms.residual_rl_v2.gym_env import MaskedResidualGymEnv

from .env import (
    RiskGatedResidualDispatchEnv,
    RiskGatedResidualEnvConfig,
)
from .risk_gate import AdaptiveRiskGateConfig


def make_risk_gated_native_env(
    *,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    decision_interval_h: float = 24.0,
    event_triggered: bool = True,
    weather_mode: str = "window",
    hard_scenario_probability: float = 0.30,
    reward: HighLevelRewardConfig | None = None,
    gate: AdaptiveRiskGateConfig | None = None,
    gate_mode: str = "hard",
    outside_risk_intervention_penalty: float = 0.0,
    scenario_generator=None,
) -> RiskGatedResidualDispatchEnv:
    """Build one native v3 environment.

    构建一个原生 v3 环境。
    """
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
    return RiskGatedResidualDispatchEnv(
        physical_env,
        config=RiskGatedResidualEnvConfig(
            residual=MaskedResidualEnvConfig(
                decision_interval_h=decision_interval_h,
                event_triggered=event_triggered,
                reward=reward or HighLevelRewardConfig(),
            ),
            adaptive_gate=gate or AdaptiveRiskGateConfig(),
            gate_mode=gate_mode,
            outside_risk_intervention_penalty=(
                outside_risk_intervention_penalty
            ),
        ),
    )


def make_risk_gated_gym_env(
    *,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    decision_interval_h: float = 24.0,
    event_triggered: bool = True,
    weather_mode: str = "window",
    hard_scenario_probability: float = 0.30,
    reward: HighLevelRewardConfig | None = None,
    gate: AdaptiveRiskGateConfig | None = None,
    gate_mode: str = "hard",
    outside_risk_intervention_penalty: float = 0.0,
    episode_seed_min: int = 100_000,
    episode_seed_max: int = 999_999,
) -> MaskedResidualGymEnv:
    """Build one Gym-compatible v3 environment.

    构建一个兼容 Gym 的 v3 环境。
    """
    native = make_risk_gated_native_env(
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        decision_interval_h=decision_interval_h,
        event_triggered=event_triggered,
        weather_mode=weather_mode,
        hard_scenario_probability=hard_scenario_probability,
        reward=reward,
        gate=gate,
        gate_mode=gate_mode,
        outside_risk_intervention_penalty=(
            outside_risk_intervention_penalty
        ),
    )
    return MaskedResidualGymEnv(
        native,
        episode_seed_min=episode_seed_min,
        episode_seed_max=episode_seed_max,
    )


def make_curriculum_risk_gated_gym_env(
    *,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    decision_interval_h: float = 24.0,
    event_triggered: bool = True,
    weather_mode: str = "window",
    initial_hard_probability: float = 0.0,
    reward: HighLevelRewardConfig | None = None,
    gate: AdaptiveRiskGateConfig | None = None,
    gate_mode: str = "soft",
    outside_risk_intervention_penalty: float = 0.02,
    episode_seed_min: int = 100_000,
    episode_seed_max: int = 999_999,
) -> CurriculumMaskedResidualGymEnv:
    """Build one curriculum-aware v3 Gym environment.

    构建一个支持课程学习的 v3 Gym 环境。
    """
    native = make_risk_gated_native_env(
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        decision_interval_h=decision_interval_h,
        event_triggered=event_triggered,
        weather_mode=weather_mode,
        hard_scenario_probability=initial_hard_probability,
        reward=reward,
        gate=gate,
        gate_mode=gate_mode,
        outside_risk_intervention_penalty=(
            outside_risk_intervention_penalty
        ),
    )
    return CurriculumMaskedResidualGymEnv(
        native,
        episode_seed_min=episode_seed_min,
        episode_seed_max=episode_seed_max,
    )
