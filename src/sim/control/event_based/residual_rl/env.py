"""Event-triggered residual semi-MDP for CCS vessel interventions.

用于 CCS 船舶干预的事件触发残差半 MDP。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from sim.environment import CCSEnv

from sim.control.event_based.rl.high_level_env import (
    _decision_event_snapshot,
    _replan_event_reason,
)
from sim.control.event_based.rl.reward import HighLevelRewardConfig, high_level_reward

from .action_codec import ResidualActionCodec
from .executor import ResidualRuleExecutor
from .observation import residual_observation, residual_observation_size


@dataclass(frozen=True)
class ResidualEnvConfig:
    """Configure residual decision timing, event thresholds, and reward.

    配置残差决策时序、事件阈值与奖励。
    """

    decision_interval_h: float = 24.0
    event_triggered: bool = True
    emitter_fill_event_thresholds: tuple[float, ...] = (0.8, 0.95)
    minimum_operable_speed_factor: float = 0.75
    reward: HighLevelRewardConfig = HighLevelRewardConfig()

    def __post_init__(self) -> None:
        """Validate the residual semi-MDP configuration.

        校验残差半 MDP 配置。
        """
        if self.decision_interval_h <= 0.0:
            raise ValueError("decision_interval_h must be positive.")
        thresholds = self.emitter_fill_event_thresholds
        if tuple(sorted(set(thresholds))) != thresholds:
            raise ValueError(
                "emitter_fill_event_thresholds must be unique and increasing."
            )
        if any(not 0.0 < value < 1.0 for value in thresholds):
            raise ValueError("Emitter fill thresholds must be inside (0, 1).")
        if self.minimum_operable_speed_factor < 0.0:
            raise ValueError(
                "minimum_operable_speed_factor must be non-negative."
            )


class ResidualDispatchEnv:
    """Let PPO keep or minimally correct a safe hourly rule controller.

    让 PPO 保持或小幅修正规则控制器的逐小时安全动作。
    """

    def __init__(
        self,
        env: CCSEnv,
        *,
        config: ResidualEnvConfig | None = None,
    ) -> None:
        """Wrap a physical environment without changing its constraints.

        包装物理环境，同时不改变其中的任何约束。
        """
        self.env = env
        self.config = config or ResidualEnvConfig()
        steps = self.config.decision_interval_h / env.network.time_step_hours
        if steps <= 0.0 or abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                "decision_interval_h must be a positive multiple of "
                "the physical time step."
            )
        self._decision_steps = int(round(steps))
        self.codec = ResidualActionCodec(env.emitter_ids)
        self.executor = ResidualRuleExecutor()
        self.last_decision_trigger = "initial"
        self.last_decision_elapsed_h = 0.0

    @property
    def action_count(self) -> int:
        """Return ``1 + 2 * n_emitters`` residual actions.

        返回 ``1 + 2 * 排放源数量`` 个残差动作。
        """
        return self.codec.action_count

    @property
    def observation_size(self) -> int:
        """Return the augmented event/risk observation size.

        返回扩展后的事件/风险观测维度。
        """
        return residual_observation_size(self.env)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset one physical episode and expose the ``initial`` event.

        重置一个物理回合，并暴露 ``initial`` 事件。
        """
        self.env.reset(seed=seed)
        self.executor = ResidualRuleExecutor()
        self.last_decision_trigger = "initial"
        self.last_decision_elapsed_h = 0.0
        return self._observation()

    def step(
        self,
        action_index: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one residual intervention until an event or time limit.

        应用一次残差干预，直至事件发生或达到最大决策间隔。
        """
        if self.env.simulator is None:
            raise RuntimeError("Call reset() before step().")
        intervention = self.codec.decode(int(action_index))
        self.executor.begin_decision(intervention)

        started_at_h = self.env.simulator.state.time_h
        start_stored_t = self.env.cumulative_stored_t
        start_vented_t = self.env.ledger.vented_t
        start_operating_cost = self.env.ledger.operating_cost
        start_total_cost = self.env.ledger.total_cost
        start_captured_t = self.env.cumulative_captured_t
        overflow_risk_t_hours = 0.0
        native_reward = 0.0
        violations: Counter[str] = Counter()
        overridden_vessels: set[str] = set()
        terminated = False
        truncated = False
        native_steps = 0
        event_snapshot = _decision_event_snapshot(self.env, self.config)
        decision_trigger = "maximum_interval"

        for _ in range(self._decision_steps):
            action = self.executor.propose_action(self.env)
            overridden_vessels.update(self.executor.last_overridden_vessels)
            _obs, reward, terminated, truncated, info = self.env.step(action)
            native_steps += 1
            native_reward += float(reward)
            overflow_risk_t_hours += (
                float(info.get("overflow_risk_t", 0.0))
                * self.env.network.time_step_hours
            )
            violations.update(str(code) for code in info.get("violations", []))
            if terminated or truncated:
                decision_trigger = "episode_end"
                break
            next_snapshot = _decision_event_snapshot(self.env, self.config)
            event_reason = _replan_event_reason(
                event_snapshot,
                next_snapshot,
                self.env.vessel_ids,
                self.env.emitter_ids,
                self.env.well_ids,
            )
            event_snapshot = next_snapshot
            if self.config.event_triggered and event_reason is not None:
                decision_trigger = event_reason
                break

        elapsed_hours = self.env.simulator.state.time_h - started_at_h
        stored_t = self.env.cumulative_stored_t - start_stored_t
        vented_t = self.env.ledger.vented_t - start_vented_t
        operating_cost = self.env.ledger.operating_cost - start_operating_cost
        total_cost = self.env.ledger.total_cost - start_total_cost
        captured_t = self.env.cumulative_captured_t - start_captured_t
        reward, reward_breakdown = high_level_reward(
            stored_t=stored_t,
            captured_t=captured_t,
            vented_t=vented_t,
            operating_cost=operating_cost,
            overflow_risk_t_hours=overflow_risk_t_hours,
            violation_counts=violations,
            config=self.config.reward,
        )
        self.last_decision_trigger = decision_trigger
        self.last_decision_elapsed_h = elapsed_hours
        difficulty = getattr(
            self.env.scenario_generator,
            "last_difficulty",
            "fixed_or_standard",
        )
        info = {
            "action_label": self.codec.label(int(action_index)),
            "intervention_kind": intervention.kind,
            "intervention_emitter": intervention.emitter_id,
            "intervention_applied": bool(overridden_vessels),
            "overridden_vessels": sorted(overridden_vessels),
            "elapsed_hours": elapsed_hours,
            "native_steps": native_steps,
            "decision_trigger": decision_trigger,
            "scenario_difficulty": difficulty,
            "stored_t": stored_t,
            "vented_t": vented_t,
            "captured_t": captured_t,
            "operating_cost": operating_cost,
            "total_cost": total_cost,
            "overflow_risk_t_hours": overflow_risk_t_hours,
            "violation_counts": dict(sorted(violations.items())),
            "high_level_reward": reward_breakdown,
            "native_hourly_reward": native_reward,
            "cumulative_stored_t": self.env.cumulative_stored_t,
            "cumulative_vented_t": self.env.ledger.vented_t,
            "cumulative_total_cost": self.env.ledger.total_cost,
        }
        return self._observation(), reward, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        """Encode the state visible at the current decision point.

        编码当前决策点可见的状态。
        """
        return residual_observation(
            self.env,
            decision_trigger=self.last_decision_trigger,
            hours_since_decision=self.last_decision_elapsed_h,
            maximum_interval_h=self.config.decision_interval_h,
        )

