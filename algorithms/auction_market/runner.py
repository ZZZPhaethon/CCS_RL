"""Build a physical episode and roll out a dispatch policy on it.

构建一个物理回合并在其上回放一个调度策略。

This reuses the exact scenario machinery of the residual-RL factory so the
auction is evaluated on the same disturbance model (capture, weather, well
availability, ship speed) as the rest of the project. It never resets the
environment itself, so callers can deep-copy one reset environment and give an
identical scenario to several controllers.

它复用残差 RL 工厂完全相同的场景机制,使拍卖在与项目其余部分相同的扰动模型
(捕集、天气、井可用性、船速)上评估。它自身从不重置环境,因此调用者可以深拷贝
一个已重置的环境,并把完全相同的场景交给多个控制器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from Simulation.environment import CCSEnv, CCSEnvConfig, build_phase1_env

from algorithms.residual_rl.scenario import MixedDifficultyScenarioGenerator
from algorithms.rl.reward import HARD_VIOLATION_CODES


PolicyFn = Callable[[CCSEnv], dict]


@dataclass
class EpisodeResult:
    """Physical, economic, and per-emitter outcomes of one rollout.

    一次回放的物理、经济与逐排放源结果。
    """

    captured_t: float
    stored_t: float
    vented_t: float
    operating_cost_eur: float
    total_cost_eur: float
    hard_violations: int
    steps: int
    ledger: object = None
    per_emitter_captured_t: dict[str, float] = field(default_factory=dict)
    per_emitter_vented_t: dict[str, float] = field(default_factory=dict)

    @property
    def storage_rate(self) -> float:
        """Realised stored / captured fraction. / 实际封存/捕集比例。"""
        return self.stored_t / self.captured_t if self.captured_t > 1e-9 else 0.0

    @property
    def unit_total_cost_eur_per_t(self) -> float:
        """Total cost per stored tonne. / 单位封存总成本。"""
        return (
            self.total_cost_eur / self.stored_t
            if self.stored_t > 1e-9
            else float("nan")
        )


def build_env(
    *,
    scenario: str = "northern_lights_phase1_milkrun_imbalanced",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    hard_scenario_probability: float = 0.0,
    weather_mode: str = "window",
) -> CCSEnv:
    """Build a physical CCS environment matching the residual-RL setup.

    构建一个与残差 RL 设置一致的物理 CCS 环境。
    """
    generator = MixedDifficultyScenarioGenerator(
        episode_hours=episode_hours + forecast_context_hours,
        weather_process=weather_mode,
        hard_probability=hard_scenario_probability,
    )
    return build_phase1_env(
        scenario=scenario,
        scenario_generator=generator,
        weather_mode=weather_mode,
        config=CCSEnvConfig(
            episode_hours=episode_hours,
            include_goal_obs=False,
            reward_mode="vent_first",
        ),
    )


def rollout(env: CCSEnv, policy: PolicyFn) -> EpisodeResult:
    """Roll out one policy on an already-reset environment.

    在一个已重置的环境上回放一个策略。
    """
    if env.simulator is None:
        raise RuntimeError("Reset the environment before calling rollout().")
    dt = float(env.network.time_step_hours)
    per_captured = {emitter_id: 0.0 for emitter_id in env.emitter_ids}
    per_vented = {emitter_id: 0.0 for emitter_id in env.emitter_ids}
    hard_violations = 0
    steps = 0
    max_steps = int(env.config.episode_hours / dt) + 8
    done = False
    while not done and steps < max_steps:
        action = policy(env)
        _obs, _reward, terminated, truncated, info = env.step(action)
        state = env.simulator.state
        for emitter_id in env.emitter_ids:
            stored_tph = float(state.last_capture_tph.get(emitter_id, 0.0))
            vent_tph = float(state.last_vent_tph.get(emitter_id, 0.0))
            per_captured[emitter_id] += (stored_tph + vent_tph) * dt
            per_vented[emitter_id] += vent_tph * dt
        for code in info.get("violations", []):
            if str(code) in HARD_VIOLATION_CODES:
                hard_violations += 1
        steps += 1
        done = bool(terminated or truncated)

    return EpisodeResult(
        captured_t=float(env.cumulative_captured_t),
        stored_t=float(env.cumulative_stored_t),
        vented_t=float(env.ledger.vented_t),
        operating_cost_eur=float(env.ledger.operating_cost),
        total_cost_eur=float(env.ledger.total_cost),
        hard_violations=hard_violations,
        steps=steps,
        ledger=env.ledger,
        per_emitter_captured_t=per_captured,
        per_emitter_vented_t=per_vented,
    )
