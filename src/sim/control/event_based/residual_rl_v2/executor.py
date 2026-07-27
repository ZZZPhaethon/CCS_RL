"""Execute only residual interventions that change the rule action.

仅执行能够真实改变规则动作的残差干预。
"""

from __future__ import annotations

from math import inf

import numpy as np

from sim.control.baselines import (
    balanced_capture_assignment,
    greedy_shuttle_policy,
    make_cluster_shuttle_policy,
)
from sim.environment import CCSEnv

from .action_codec import (
    MaskedResidualActionCodec,
    MaskedResidualIntervention,
)


_EPS = 1e-9


def default_rule_action(env: CCSEnv) -> dict[str, list[int]]:
    """Return the same balanced rule action used by the benchmark.

    返回与正式基准相同的平衡规则动作。
    """
    assignment = balanced_capture_assignment(env)
    env.set_goal_assignment(assignment)
    action = make_cluster_shuttle_policy(env, assignment)(env)
    action["wells"] = [
        env.highest_feasible_well_rate_index(well_id)
        for well_id in env.well_ids
    ]
    return {
        "vessels": list(action["vessels"]),
        "wells": list(action["wells"]),
    }


def eligible_override_vessels(
    env: CCSEnv,
    emitter_id: str,
    baseline_action: dict[str, list[int]] | None = None,
) -> tuple[str, ...]:
    """Return empty vessels whose destination can genuinely be changed.

    返回目的地能够被真实改变的空载船舶。

    A vessel is excluded when the rule already sends it to the requested
    emitter. This prevents a nominal intervention from being logged as an
    applied intervention when its native action equals the baseline.

    如果规则已经把船派往指定排放源，则该船不属于候选。这样可以避免
    原生动作与基线相同时仍被错误记录为“已干预”。
    """
    if env.simulator is None:
        raise RuntimeError("Call env.reset() before checking residual actions.")
    baseline = baseline_action or default_rule_action(env)
    state = env.simulator.state
    masks = env.vessel_action_mask()
    target_action = env.vessel_go_emitter_action(emitter_id)
    candidates: list[str] = []
    for position, vessel_id in enumerate(env.vessel_ids):
        vessel_state = env.simulator.vessel_states[vessel_id]
        cargo_t = state.entity_inventory_t.get(vessel_id, 0.0)
        if (
            vessel_state["mode"] == "berthed"
            and cargo_t <= _EPS
            and masks[position][target_action]
            and baseline["vessels"][position] != target_action
        ):
            candidates.append(vessel_id)
    return tuple(candidates)


def residual_action_mask(
    env: CCSEnv,
    codec: MaskedResidualActionCodec,
) -> np.ndarray:
    """Return a dynamic mask that always permits the rule default.

    返回始终允许规则默认动作的动态动作掩码。
    """
    baseline = default_rule_action(env)
    adaptive = adaptive_greedy_action(env)
    candidates = {
        emitter_id: eligible_override_vessels(
            env,
            emitter_id,
            baseline,
        )
        for emitter_id in codec.emitter_ids
    }
    mask = [True, adaptive != baseline]
    # ``prioritise`` only differs from ``add_one`` when at least two vessels
    # can be redirected. Mask it otherwise to remove duplicate policy actions.
    # 只有至少两艘船可重定向时，``prioritise`` 才不同于 ``add_one``；
    # 否则将其掩码，避免策略看到重复动作。
    mask.extend(
        len(candidates[emitter]) >= 2
        for emitter in codec.emitter_ids
    )
    mask.extend(
        bool(candidates[emitter])
        for emitter in codec.emitter_ids
    )
    return np.asarray(mask, dtype=bool)


class MaskedResidualRuleExecutor:
    """Overlay a feasible residual intervention on the balanced rule action.

    在平衡规则动作上叠加一个可执行的残差干预。
    """

    def __init__(self) -> None:
        """Initialise per-decision diagnostic state.

        初始化每次决策的诊断状态。
        """
        self.intervention = MaskedResidualIntervention("keep_default")
        self.add_one_consumed = False
        self.last_eligible_vessels: tuple[str, ...] = ()
        self.last_overridden_vessels: tuple[str, ...] = ()
        self.last_native_action_changed = False

    def begin_decision(
        self,
        intervention: MaskedResidualIntervention,
    ) -> None:
        """Start one sparse residual decision interval.

        开始一个稀疏残差决策区间。
        """
        intervention.validate()
        self.intervention = intervention
        self.add_one_consumed = False
        self.last_eligible_vessels = ()
        self.last_overridden_vessels = ()
        self.last_native_action_changed = False

    def propose_action(self, env: CCSEnv) -> dict[str, list[int]]:
        """Return the rule action or one genuinely different native action.

        返回规则动作或一个真正不同的原生动作。
        """
        baseline = default_rule_action(env)
        if self.intervention.kind == "adaptive_greedy":
            action = adaptive_greedy_action(env)
            changed = [
                vessel_id
                for position, vessel_id in enumerate(env.vessel_ids)
                if action["vessels"][position]
                != baseline["vessels"][position]
            ]
            self.last_eligible_vessels = tuple(changed)
            self.last_overridden_vessels = tuple(changed)
            self.last_native_action_changed = action != baseline
            return action
        target = self.intervention.emitter_id
        if self.intervention.kind == "keep_default" or target is None:
            self._clear_hourly_diagnostics()
            return baseline

        candidates = eligible_override_vessels(env, target, baseline)
        self.last_eligible_vessels = candidates
        if self.intervention.kind == "add_one":
            if self.add_one_consumed or not candidates:
                self.last_overridden_vessels = ()
                self.last_native_action_changed = False
                return baseline
            candidates = (
                min(
                    candidates,
                    key=lambda vessel: _travel_hours(env, vessel, target),
                ),
            )
            self.add_one_consumed = True

        action = {
            "vessels": list(baseline["vessels"]),
            "wells": list(baseline["wells"]),
        }
        target_action = env.vessel_go_emitter_action(target)
        for vessel_id in candidates:
            position = env.vessel_ids.index(vessel_id)
            action["vessels"][position] = target_action
        self.last_overridden_vessels = tuple(candidates)
        self.last_native_action_changed = action != baseline
        return action

    def _clear_hourly_diagnostics(self) -> None:
        """Clear diagnostics when the default action is selected.

        在选择默认动作时清空逐小时诊断信息。
        """
        self.last_eligible_vessels = ()
        self.last_overridden_vessels = ()
        self.last_native_action_changed = False


def adaptive_greedy_action(env: CCSEnv) -> dict[str, list[int]]:
    """Return the global adaptive greedy action with safe well rates.

    返回带安全注入率的全局 adaptive greedy 动作。
    """
    action = greedy_shuttle_policy(env)
    action["wells"] = [
        env.highest_feasible_well_rate_index(well_id)
        for well_id in env.well_ids
    ]
    return {
        "vessels": list(action["vessels"]),
        "wells": list(action["wells"]),
    }


def _travel_hours(
    env: CCSEnv,
    vessel_id: str,
    destination_id: str,
) -> float:
    """Estimate current sailing time for nearest-vessel selection.

    估计当前航行时间，用于选择最近船舶。
    """
    assert env.simulator is not None
    route = env._routes[vessel_id]
    origin_id = env._weather_reference_origin(vessel_id)
    if origin_id == destination_id:
        return 0.0
    leg_id = f"{origin_id}->{destination_id}"
    factor = env._weather_speed_at(leg_id, vessel_id, 0)
    speed_knots = float(
        route.get("speed_knots") or env.config.default_speed_knots
    )
    effective_speed = speed_knots * max(0.0, float(factor))
    if effective_speed <= _EPS:
        return inf
    distance_km = env._leg_distance_km(
        origin_id,
        destination_id,
        route,
    )
    return max(0.0, distance_km / (effective_speed * 1.852))

