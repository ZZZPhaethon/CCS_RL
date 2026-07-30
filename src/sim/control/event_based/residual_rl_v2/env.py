"""Masked residual semi-MDP with a rule-counterfactual reward.

带规则反事实奖励的掩码残差半 MDP。
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.control.cplex_milp import _terminal_cleanup_cost_for_state
from sim.environment import CCSEnv

from sim.control.event_based.residual_rl.observation import (
    residual_observation,
    residual_observation_size,
)
from sim.control.event_based.rl.high_level_env import (
    _decision_event_snapshot,
    _replan_event_reason,
)
from sim.control.event_based.rl.observation_encoder import (
    FORECAST_WINDOWS_H,
    validated_future_summary_windows,
)
from sim.control.event_based.rl.reward import HighLevelRewardConfig, high_level_reward

from .executor import (
    MaskedResidualRuleExecutor,
    adaptive_greedy_action,
    default_rule_action,
    eligible_override_vessels,
    residual_action_mask,
)
from .action_codec import MaskedResidualActionCodec


@dataclass(frozen=True)
class MaskedResidualEnvConfig:
    """Configure timing, event thresholds, and counterfactual reward.

    配置时序、事件阈值与反事实奖励。
    """

    decision_interval_h: float = 24.0
    event_triggered: bool = True
    emitter_fill_event_thresholds: tuple[float, ...] = (0.8, 0.95)
    minimum_operable_speed_factor: float = 0.75
    reward: HighLevelRewardConfig = HighLevelRewardConfig()
    future_summary_windows_h: tuple[int, ...] = FORECAST_WINDOWS_H
    max_simulator_hour_steps: int | None = None

    def __post_init__(self) -> None:
        """Validate semi-MDP settings.

        校验半 MDP 设置。
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
        validated_future_summary_windows(
            self.future_summary_windows_h
        )
        budget = self.max_simulator_hour_steps
        if budget is not None and (
            int(budget) != budget or budget <= 0
        ):
            raise ValueError(
                "max_simulator_hour_steps must be a positive integer."
            )


class MaskedResidualDispatchEnv:
    """Learn only the incremental value of changing the rule decision.

    仅学习改变规则决策所产生的增量价值。
    """

    def __init__(
        self,
        env: CCSEnv,
        *,
        config: MaskedResidualEnvConfig | None = None,
    ) -> None:
        """Wrap a physical environment and preserve all native constraints.

        包装物理环境，并保留所有原生约束。
        """
        self.env = env
        self.config = config or MaskedResidualEnvConfig()
        steps = self.config.decision_interval_h / env.network.time_step_hours
        if steps <= 0.0 or abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                "decision_interval_h must be a positive multiple of "
                "the physical time step."
            )
        self._decision_steps = int(round(steps))
        self.codec = MaskedResidualActionCodec(env.emitter_ids)
        self.executor = MaskedResidualRuleExecutor()
        self._simulator_usage_at_start = (
            self.env.simulator_step_usage()
        )
        self.last_decision_trigger = "initial"
        self.last_decision_elapsed_h = 0.0
        self.counterfactual_env: CCSEnv | None = None

    @property
    def action_count(self) -> int:
        """Return the compact residual action count.

        返回紧凑残差动作数量。
        """
        return self.codec.action_count

    @property
    def observation_size(self) -> int:
        """Return the event/risk observation size.

        返回事件/风险观测维度。
        """
        return residual_observation_size(
            self.env,
            self.config.future_summary_windows_h,
        )

    def action_masks(self) -> np.ndarray:
        """Return valid actions at the current decision point.

        返回当前决策点的合法动作。
        """
        if self.env.simulator is None:
            raise RuntimeError("Call reset() before requesting action masks.")
        return residual_action_mask(self.env, self.codec)

    def training_simulator_usage(self):
        """Return physical-simulator use since this wrapper was created."""

        return (
            self.env.simulator_step_usage()
            - self._simulator_usage_at_start
        )

    def _remaining_paired_native_steps(self) -> int | None:
        """Return actual steps affordable with matching counterfactual steps."""

        budget = self.config.max_simulator_hour_steps
        if budget is None:
            return None
        remaining_h = max(
            0.0,
            float(budget)
            - self.training_simulator_usage().hour_steps,
        )
        paired_step_h = 2.0 * self.env.network.time_step_hours
        return int((remaining_h + 1e-9) // paired_step_h)

    def simulator_budget_exhausted(self) -> bool:
        """Return whether another actual/counterfactual pair is affordable."""

        remaining_steps = self._remaining_paired_native_steps()
        return remaining_steps is not None and remaining_steps <= 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset an episode and expose the initial event.

        重置回合并暴露初始事件。
        """
        self.env.reset(seed=seed)
        self.counterfactual_env = deepcopy(self.env)
        self.executor = MaskedResidualRuleExecutor()
        self.last_decision_trigger = "initial"
        self.last_decision_elapsed_h = 0.0
        return self._observation()

    def step(
        self,
        action_index: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute a feasible intervention and compare it with the rule.

        执行可行干预，并与规则反事实进行比较。
        """
        if self.env.simulator is None or self.counterfactual_env is None:
            raise RuntimeError("Call reset() before step().")
        index = int(action_index)
        mask = self.action_masks()
        if not 0 <= index < self.action_count:
            raise ValueError(f"Residual action index out of range: {index}.")
        if not bool(mask[index]):
            raise ValueError(
                f"Residual action {index} is masked at this decision point."
            )

        intervention = self.codec.decode(index)
        baseline_action_at_decision = default_rule_action(self.env)
        if intervention.kind == "adaptive_greedy":
            adaptive_action = adaptive_greedy_action(self.env)
            eligible_at_decision = tuple(
                vessel_id
                for position, vessel_id in enumerate(self.env.vessel_ids)
                if adaptive_action["vessels"][position]
                != baseline_action_at_decision["vessels"][position]
            )
        elif intervention.emitter_id is not None:
            eligible_at_decision = eligible_override_vessels(
                self.env,
                intervention.emitter_id,
            )
        else:
            eligible_at_decision = ()
        self.executor.begin_decision(intervention)

        start = _metric_snapshot(self.env)
        started_at_h = self.env.simulator.state.time_h
        overflow_risk_t_hours = 0.0
        native_reward = 0.0
        violations: Counter[str] = Counter()
        eligible_seen: set[str] = set(eligible_at_decision)
        overridden_vessels: set[str] = set()
        changed_native_steps = 0
        terminated = False
        truncated = False
        native_steps = 0
        event_snapshot = _decision_event_snapshot(self.env, self.config)
        decision_trigger = "maximum_interval"
        budget_exhausted = False
        remaining_paired_steps = self._remaining_paired_native_steps()
        allowed_native_steps = (
            self._decision_steps
            if remaining_paired_steps is None
            else min(self._decision_steps, remaining_paired_steps)
        )

        for _ in range(allowed_native_steps):
            action = self.executor.propose_action(self.env)
            eligible_seen.update(self.executor.last_eligible_vessels)
            overridden_vessels.update(self.executor.last_overridden_vessels)
            changed_native_steps += int(
                self.executor.last_native_action_changed
            )
            _obs, native_step_reward, terminated, truncated, info = (
                self.env.step(action)
            )
            native_steps += 1
            native_reward += float(native_step_reward)
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

        actual = _metric_delta(self.env, start)
        actual["terminal_cleanup_operating_cost_eur"] = 0.0
        if (
            truncated
            and self.env.t >= self.env.n_steps
            and self.config.reward.objective == "realised_total_cost"
        ):
            actual_cleanup_cost = float(
                _terminal_cleanup_cost_for_state(
                    self.env,
                    self.env.cost_model.parameters,
                )
            )
            actual["terminal_cleanup_operating_cost_eur"] = (
                actual_cleanup_cost
            )
            actual["operating_cost_eur"] += actual_cleanup_cost
            actual["total_cost_eur"] += actual_cleanup_cost
        actual_reward, actual_breakdown = high_level_reward(
            stored_t=actual["stored_t"],
            captured_t=actual["captured_t"],
            vented_t=actual["vented_t"],
            operating_cost=actual["operating_cost_eur"],
            overflow_risk_t_hours=overflow_risk_t_hours,
            violation_counts=violations,
            config=self.config.reward,
            total_cost_eur=actual["total_cost_eur"],
        )
        counterfactual = _advance_rule_counterfactual(
            self.counterfactual_env,
            native_steps=native_steps,
            config=self.config,
        )
        if (
            not terminated
            and not truncated
            and self.simulator_budget_exhausted()
        ):
            budget_exhausted = True
            truncated = True
            decision_trigger = "simulator_budget_exhausted"
        incremental_reward = actual_reward - counterfactual["reward"]
        elapsed_hours = self.env.simulator.state.time_h - started_at_h
        self.last_decision_trigger = decision_trigger
        self.last_decision_elapsed_h = elapsed_hours

        avoided_vent_t = (
            counterfactual["vented_t"] - actual["vented_t"]
        )
        incremental_stored_t = (
            actual["stored_t"] - counterfactual["stored_t"]
        )
        operating_cost_saving = (
            counterfactual["operating_cost_eur"]
            - actual["operating_cost_eur"]
        )
        total_cost_saving = (
            counterfactual["total_cost_eur"]
            - actual["total_cost_eur"]
        )
        simulator_usage = self.training_simulator_usage()
        simulator_budget = self.config.max_simulator_hour_steps
        info = {
            "action_label": self.codec.label(index),
            "intervention_selected": index != 0,
            "intervention_kind": intervention.kind,
            "intervention_emitter": intervention.emitter_id,
            "action_mask": mask.tolist(),
            "feasible_intervention_count": int(mask.sum()) - 1,
            "eligible_vessels_at_decision": list(eligible_at_decision),
            "eligible_vessels_seen": sorted(eligible_seen),
            "intervention_feasible_at_decision": bool(
                eligible_at_decision
            ),
            "overridden_vessels": sorted(overridden_vessels),
            "native_action_changed": changed_native_steps > 0,
            "changed_native_steps": changed_native_steps,
            "elapsed_hours": elapsed_hours,
            "native_steps": native_steps,
            "decision_trigger": decision_trigger,
            "stored_t": actual["stored_t"],
            "vented_t": actual["vented_t"],
            "captured_t": actual["captured_t"],
            "operating_cost": actual["operating_cost_eur"],
            "total_cost": actual["total_cost_eur"],
            "terminal_cleanup_operating_cost_eur": actual[
                "terminal_cleanup_operating_cost_eur"
            ],
            "counterfactual_stored_t": counterfactual["stored_t"],
            "counterfactual_vented_t": counterfactual["vented_t"],
            "counterfactual_operating_cost_eur": counterfactual[
                "operating_cost_eur"
            ],
            "counterfactual_total_cost_eur": counterfactual[
                "total_cost_eur"
            ],
            "counterfactual_terminal_cleanup_operating_cost_eur": (
                counterfactual[
                    "terminal_cleanup_operating_cost_eur"
                ]
            ),
            "incremental_stored_t": incremental_stored_t,
            "avoided_vent_t": avoided_vent_t,
            "operating_cost_saving_eur": operating_cost_saving,
            "total_cost_saving_eur": total_cost_saving,
            "overflow_risk_t_hours": overflow_risk_t_hours,
            "violation_counts": dict(sorted(violations.items())),
            "actual_reward": actual_reward,
            "counterfactual_reward": counterfactual["reward"],
            "incremental_reward": incremental_reward,
            "actual_reward_breakdown": actual_breakdown,
            "counterfactual_reward_breakdown": counterfactual[
                "reward_breakdown"
            ],
            "native_hourly_reward": native_reward,
            "baseline_action_at_decision": baseline_action_at_decision,
            "cumulative_stored_t": self.env.cumulative_stored_t,
            "cumulative_vented_t": self.env.ledger.vented_t,
            "cumulative_total_cost": self.env.ledger.total_cost,
            "cumulative_reported_total_cost": (
                self.env.ledger.total_cost
                + actual["terminal_cleanup_operating_cost_eur"]
            ),
            "counterfactual_cumulative_stored_t": (
                self.counterfactual_env.cumulative_stored_t
            ),
            "counterfactual_cumulative_vented_t": (
                self.counterfactual_env.ledger.vented_t
            ),
            "counterfactual_cumulative_total_cost": (
                self.counterfactual_env.ledger.total_cost
            ),
            **simulator_usage.as_dict(),
            "max_simulator_hour_steps": simulator_budget,
            "simulator_budget_exhausted": budget_exhausted,
            "simulator_budget_fraction": (
                simulator_usage.hour_steps / simulator_budget
                if simulator_budget is not None
                else None
            ),
        }
        return (
            self._observation(),
            incremental_reward,
            terminated,
            truncated,
            info,
        )

    def _observation(self) -> np.ndarray:
        """Encode the current event-triggered decision state.

        编码当前事件触发决策状态。
        """
        return residual_observation(
            self.env,
            decision_trigger=self.last_decision_trigger,
            hours_since_decision=self.last_decision_elapsed_h,
            maximum_interval_h=self.config.decision_interval_h,
            future_summary_windows_h=(
                self.config.future_summary_windows_h
            ),
        )


def _metric_snapshot(env: CCSEnv) -> dict[str, float]:
    """Capture cumulative physical/economic metrics.

    捕获累计物理与经济指标。
    """
    return {
        "captured_t": float(env.cumulative_captured_t),
        "stored_t": float(env.cumulative_stored_t),
        "vented_t": float(env.ledger.vented_t),
        "operating_cost_eur": float(env.ledger.operating_cost),
        "total_cost_eur": float(env.ledger.total_cost),
    }


def _metric_delta(
    env: CCSEnv,
    start: dict[str, float],
) -> dict[str, float]:
    """Return realised metric increments since a snapshot.

    返回相对快照的实际指标增量。
    """
    current = _metric_snapshot(env)
    return {
        key: current[key] - start[key]
        for key in start
    }


def _advance_rule_counterfactual(
    env: CCSEnv,
    *,
    native_steps: int,
    config: MaskedResidualEnvConfig,
) -> dict[str, Any]:
    """Advance the persistent rule baseline for the same physical duration.

    在相同物理时长内推进持续存在的纯规则基线。
    """
    start = _metric_snapshot(env)
    overflow_risk_t_hours = 0.0
    violations: Counter[str] = Counter()
    for _ in range(native_steps):
        action = default_rule_action(env)
        _obs, _reward, terminated, truncated, info = env.step(action)
        overflow_risk_t_hours += (
            float(info.get("overflow_risk_t", 0.0))
            * env.network.time_step_hours
        )
        violations.update(str(code) for code in info.get("violations", []))
        if terminated or truncated:
            break
    delta = _metric_delta(env, start)
    delta["terminal_cleanup_operating_cost_eur"] = 0.0
    if (
        env.t >= env.n_steps
        and config.reward.objective == "realised_total_cost"
    ):
        cleanup_cost = float(
            _terminal_cleanup_cost_for_state(
                env,
                env.cost_model.parameters,
            )
        )
        delta["terminal_cleanup_operating_cost_eur"] = cleanup_cost
        delta["operating_cost_eur"] += cleanup_cost
        delta["total_cost_eur"] += cleanup_cost
    reward, breakdown = high_level_reward(
        stored_t=delta["stored_t"],
        captured_t=delta["captured_t"],
        vented_t=delta["vented_t"],
        operating_cost=delta["operating_cost_eur"],
        overflow_risk_t_hours=overflow_risk_t_hours,
        violation_counts=violations,
        config=config.reward,
        total_cost_eur=delta["total_cost_eur"],
    )
    return {
        **delta,
        "overflow_risk_t_hours": overflow_risk_t_hours,
        "violation_counts": dict(sorted(violations.items())),
        "reward": reward,
        "reward_breakdown": breakdown,
    }
