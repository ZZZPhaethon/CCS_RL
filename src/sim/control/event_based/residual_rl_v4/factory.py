"""Factories for tail-robust residual RL v4 environments.

面向尾部风险的残差强化学习 v4 环境工厂。
"""

from __future__ import annotations

from sim.control.event_based.rl.reward import HighLevelRewardConfig

from sim.control.event_based.residual_rl_v3.factory import (
    make_risk_gated_native_env,
)
from sim.control.event_based.residual_rl_v3.risk_gate import (
    AdaptiveRiskGateConfig,
)

from .replay_env import TailFailureReplayGymEnv
from .scenario import ReplayableDifficultyScenarioGenerator


def make_tail_robust_native_env(
    *,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    decision_interval_h: float = 24.0,
    event_triggered: bool = True,
    weather_mode: str = "window",
    scenario_protocol: str = "v4_mixed_window",
    hard_scenario_probability: float = 0.40,
    reward: HighLevelRewardConfig | None = None,
    gate: AdaptiveRiskGateConfig | None = None,
    gate_mode: str = "soft",
    outside_risk_intervention_penalty: float = 0.02,
    override_windows_h: tuple[tuple[float, float], ...] = (),
):
    """Build a native v4 environment without a replay wrapper.

    构建不包含回放包装器的原生 v4 环境。
    """
    generator = ReplayableDifficultyScenarioGenerator(
        episode_hours=episode_hours + forecast_context_hours,
        weather_process=weather_mode,
        hard_probability=hard_scenario_probability,
        scenario_protocol=scenario_protocol,
    )
    return make_risk_gated_native_env(
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
        override_windows_h=override_windows_h,
        scenario_generator=generator,
        well_control_mode=(
            "automatic_max"
            if scenario_protocol == "unified_window_v1"
            else "agent_selected"
        ),
    )


def make_tail_replay_gym_env(
    *,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    decision_interval_h: float = 24.0,
    event_triggered: bool = True,
    weather_mode: str = "window",
    scenario_protocol: str = "v4_mixed_window",
    initial_hard_probability: float = 0.10,
    reward: HighLevelRewardConfig | None = None,
    gate: AdaptiveRiskGateConfig | None = None,
    gate_mode: str = "soft",
    outside_risk_intervention_penalty: float = 0.02,
    episode_seed_min: int = 100_000,
    episode_seed_max: int = 999_999,
    replay_probability: float = 0.30,
    replay_capacity: int = 20,
    minimum_replay_pool: int = 4,
    override_windows_h: tuple[tuple[float, float], ...] = (),
) -> TailFailureReplayGymEnv:
    """Build one curriculum and failure-replay training environment.

    构建一个支持课程学习和失败重放的训练环境。
    """
    native = make_tail_robust_native_env(
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        decision_interval_h=decision_interval_h,
        event_triggered=event_triggered,
        weather_mode=weather_mode,
        scenario_protocol=scenario_protocol,
        hard_scenario_probability=initial_hard_probability,
        reward=reward,
        gate=gate,
        gate_mode=gate_mode,
        outside_risk_intervention_penalty=(
            outside_risk_intervention_penalty
        ),
        override_windows_h=override_windows_h,
    )
    return TailFailureReplayGymEnv(
        native,
        episode_seed_min=episode_seed_min,
        episode_seed_max=episode_seed_max,
        replay_probability=replay_probability,
        replay_capacity=replay_capacity,
        minimum_replay_pool=minimum_replay_pool,
    )
