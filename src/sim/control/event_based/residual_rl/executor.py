"""Apply residual vessel interventions over the balanced physical rule.

在平衡物理规则之上应用船舶残差干预。
"""

from __future__ import annotations

from math import inf

from sim.control.baselines import (
    balanced_capture_assignment,
    make_cluster_shuttle_policy,
)
from sim.environment import CCSEnv

from .action_codec import ResidualIntervention


_EPS = 1e-9


class ResidualRuleExecutor:
    """Execute a safe rule action and optionally alter vessel destinations.

    执行安全规则动作，并可选择性修改船舶目的地。

    Well-rate choices are never exposed to RL. Every physical hour uses the
    highest currently feasible rate reported by the environment, so reservoir
    pressure, availability, and injectivity masks remain authoritative.

    注入档位不再暴露给 RL。每个物理小时均使用环境给出的当前最高可行档位，
    因此储层压力、井可用性与注入能力掩码仍具有最终约束权。
    """

    def __init__(self) -> None:
        """Initialise per-decision intervention state.

        初始化每次决策的干预状态。
        """
        self.intervention = ResidualIntervention("keep_default")
        self.add_one_consumed = False
        self.last_overridden_vessels: tuple[str, ...] = ()

    def begin_decision(self, intervention: ResidualIntervention) -> None:
        """Start a new sparse decision interval.

        开始新的稀疏决策区间。
        """
        intervention.validate()
        self.intervention = intervention
        self.add_one_consumed = False
        self.last_overridden_vessels = ()

    def propose_action(self, env: CCSEnv) -> dict[str, list[int]]:
        """Return one masked native action for the current physical hour.

        返回当前物理小时的一个掩码合法原生动作。
        """
        if env.simulator is None:
            raise RuntimeError("Call env.reset() before requesting an action.")
        assignment = balanced_capture_assignment(env)
        env.set_goal_assignment(assignment)
        action = make_cluster_shuttle_policy(env, assignment)(env)
        if not env.automatic_well_control:
            action["wells"] = env.automatic_well_rate_indices()
        target = self.intervention.emitter_id
        if self.intervention.kind == "keep_default" or target is None:
            self.last_overridden_vessels = ()
            return action

        candidates = _dispatchable_empty_vessels(env, target)
        if self.intervention.kind == "add_one":
            if self.add_one_consumed or not candidates:
                self.last_overridden_vessels = ()
                return action
            candidates = [
                vessel_id
                for vessel_id in candidates
                if action["vessels"][env.vessel_ids.index(vessel_id)]
                != env.vessel_go_emitter_action(target)
            ] or candidates
            candidates = [
                min(
                    candidates,
                    key=lambda vessel: _travel_hours(env, vessel, target),
                )
            ]
            self.add_one_consumed = True

        target_action = env.vessel_go_emitter_action(target)
        for vessel_id in candidates:
            action["vessels"][env.vessel_ids.index(vessel_id)] = target_action
        self.last_overridden_vessels = tuple(candidates)
        return action


def _dispatchable_empty_vessels(env: CCSEnv, emitter_id: str) -> list[str]:
    """Return empty, berthed vessels that may legally sail to an emitter.

    返回可合法驶向指定排放源的空载靠泊船舶。
    """
    assert env.simulator is not None
    state = env.simulator.state
    target_action = env.vessel_go_emitter_action(emitter_id)
    masks = env.vessel_action_mask()
    candidates: list[str] = []
    for position, vessel_id in enumerate(env.vessel_ids):
        vessel_state = env.simulator.vessel_states[vessel_id]
        cargo_t = state.entity_inventory_t.get(vessel_id, 0.0)
        if (
            vessel_state["mode"] == "berthed"
            and cargo_t <= _EPS
            and masks[position][target_action]
        ):
            candidates.append(vessel_id)
    return candidates


def _travel_hours(env: CCSEnv, vessel_id: str, destination_id: str) -> float:
    """Estimate current sailing hours for nearest-vessel selection.

    估计当前航行小时数，用于选择最近船舶。
    """
    assert env.simulator is not None
    route = env._routes[vessel_id]
    origin_id = env._weather_reference_origin(vessel_id)
    if origin_id == destination_id:
        return 0.0
    leg_id = f"{origin_id}->{destination_id}"
    speed_factor = env._weather_speed_at(leg_id, vessel_id, 0)
    speed_knots = float(
        route.get("speed_knots") or env.config.default_speed_knots
    )
    effective_speed = speed_knots * max(0.0, float(speed_factor))
    if effective_speed <= _EPS:
        return inf
    distance_km = env._leg_distance_km(origin_id, destination_id, route)
    return max(0.0, distance_km / (effective_speed * 1.852))
