"""Replay-validated MPC over the environment's native action space."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import permutations
import math
from typing import Callable

from ..environment import CCSEnv, VESSEL_GO_TERMINAL, VESSEL_WAIT
from .baselines import greedy_shuttle_policy
from .replay import ReplayExpectation, replay_native_actions
from .rolling_milp import _capture_tonnes, _sail_hours_between

Policy = Callable[[CCSEnv], dict[str, list]]


@dataclass(frozen=True)
class _NativeMpcCandidate:
    name: str
    native_actions_by_hour: list[dict[str, list[int]]]
    vented_t: float
    end_unstored_t: float
    operating_cost: float
    total_cost: float
    is_valid: bool
    is_exact: bool = False
    mismatches: tuple[str, ...] = ()


class RollingNativeMpcController:
    """Environment-grounded rolling controller over only native actions.

    Candidate schedules are rolled out in a copied ``CCSEnv`` so automatic
    loading, terminal unload priority, and discrete well actions are identical
    to execution. Greedy is an evaluated candidate, never a fallback path.
    """

    def __init__(
        self,
        env: CCSEnv,
        replan_every: int = 24,
        planning_horizon_h: int = 168,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.replan_every = max(1, int(replan_every))
        self.planning_horizon_h = max(1, int(planning_horizon_h))
        self.progress = progress
        self._native_actions_by_hour: list[dict[str, list[int]]] = []
        self._plan_origin_h = -1e9
        self.last_candidate_name = ""
        self.last_trace_replay_is_valid = False
        self.last_trace_replay_is_exact = False
        self.last_trace_replay_mismatches: tuple[str, ...] = ()
        self.candidate_evaluations = 0

    def __call__(self, env: CCSEnv) -> dict[str, list]:
        return self.policy(env)

    def policy(self, env: CCSEnv) -> dict[str, list]:
        now = env.simulator.state.time_h
        if now < self._plan_origin_h or now - self._plan_origin_h >= self.replan_every:
            self._replan(env, now)
        elapsed = int(max(0.0, math.floor(now - self._plan_origin_h)))
        if elapsed >= len(self._native_actions_by_hour):
            raise RuntimeError("rolling_native_mpc trace ended before the next replan")
        action = self._native_actions_by_hour[elapsed]
        self._validate_native_action(env, action)
        return {
            "vessels": [int(choice) for choice in action["vessels"]],
            "wells": [int(choice) for choice in action["wells"]],
        }

    def _replan(self, env: CCSEnv, now: float) -> None:
        remaining_h = max(1, min(self.planning_horizon_h, env.n_steps - env.t))
        candidates = [
            _rollout_native_candidate(env, greedy_shuttle_policy, remaining_h, "greedy"),
            _rollout_native_candidate(env, _forecast_urgency_policy, remaining_h, "forecast_urgency"),
        ]
        for assignment in _dedicated_assignments(env):
            name = "dedicated:" + ",".join(assignment[vessel_id] for vessel_id in env.vessel_ids)
            candidates.append(
                _rollout_native_candidate(
                    env,
                    _make_dedicated_policy(assignment),
                    remaining_h,
                    name,
                )
            )
        self.candidate_evaluations += len(candidates)
        valid = [candidate for candidate in candidates if candidate.is_valid]
        if not valid:
            raise RuntimeError("rolling_native_mpc produced no replay-valid native candidate")
        best = min(valid, key=self._candidate_key)
        self._native_actions_by_hour = best.native_actions_by_hour
        self._plan_origin_h = now
        self.last_candidate_name = best.name
        self.last_trace_replay_is_valid = best.is_valid
        self.last_trace_replay_is_exact = best.is_exact
        self.last_trace_replay_mismatches = best.mismatches
        if self.progress is not None:
            self.progress(
                f"  rolling_native_mpc replan at t={now:.0f} h; candidate={best.name}; "
                f"forecast_vent={best.vented_t:,.1f} t; end_unstored={best.end_unstored_t:,.1f} t"
            )
    @staticmethod
    def _candidate_key(candidate: _NativeMpcCandidate) -> tuple[float, float, float]:
        return (
            candidate.vented_t,
            candidate.end_unstored_t,
            candidate.operating_cost,
        )

    @staticmethod
    def _validate_native_action(env: CCSEnv, action: dict[str, list[int]]) -> None:
        vessel_actions = action.get("vessels", [])
        well_actions = action.get("wells", [])
        if len(vessel_actions) != len(env.vessel_ids) or len(well_actions) != len(env.well_ids):
            raise RuntimeError("rolling_native_mpc trace has the wrong action dimension")
        for vessel_id, choice, mask in zip(env.vessel_ids, vessel_actions, env.vessel_action_mask()):
            if not (0 <= int(choice) < len(mask) and mask[int(choice)]):
                raise RuntimeError(f"rolling_native_mpc action is infeasible for {vessel_id}: {choice}")
        for well_id, choice, mask in zip(env.well_ids, well_actions, env.well_rate_action_mask()):
            if not (0 <= int(choice) < len(mask) and mask[int(choice)]):
                raise RuntimeError(f"rolling_native_mpc action is infeasible for {well_id}: {choice}")


def native_mpc_candidate_names(env: CCSEnv) -> tuple[str, ...]:
    """Return candidate names in the stable order used by the native MPC."""
    names = ["greedy", "forecast_urgency"]
    names.extend(
        "dedicated:" + ",".join(assignment[vessel_id] for vessel_id in env.vessel_ids)
        for assignment in _dedicated_assignments(env)
    )
    return tuple(names)


def _rollout_native_candidate(env: CCSEnv, policy: Policy, horizon_h: int, name: str) -> _NativeMpcCandidate:
    replay_env = copy.deepcopy(env)
    start_stored_t = float(replay_env.cumulative_stored_t)
    start_vented_t = float(replay_env.ledger.vented_t)
    start_captured_t = float(replay_env.cumulative_captured_t)
    start_ledger = copy.deepcopy(replay_env.ledger)
    start_operating_cost = float(replay_env.ledger.operating_cost)
    start_total_cost = float(replay_env.ledger.total_cost)
    actions: list[dict[str, list[int]]] = []
    violations: list[str] = []
    injection_tph: list[float] = []
    total_reward = 0.0
    overflow_risk_t = 0.0
    for _ in range(horizon_h):
        action = policy(replay_env)
        native_action = {
            "vessels": [int(choice) for choice in action["vessels"]],
            "wells": [int(choice) for choice in action["wells"]],
        }
        actions.append(native_action)
        before_stored_t = float(replay_env.cumulative_stored_t)
        _obs, reward, terminated, truncated, info = replay_env.step(native_action)
        injection_tph.append(float(replay_env.cumulative_stored_t) - before_stored_t)
        total_reward += float(reward)
        overflow_risk_t += float(info.get("overflow_risk_t", 0.0))
        violations.extend(str(value) for value in info.get("violations", []))
        if terminated or truncated:
            break
    invalid = {"berth_required", "bottomhole_pressure_clipped"}
    state = replay_env.simulator.state
    expected = ReplayExpectation(
        required_fields=frozenset(
            {
                "elapsed_hours",
                "stored_t",
                "vented_t",
                "captured_t",
                "in_transit_t",
                "vessel_fuel",
                "conditioning",
                "reconditioning",
                "loading",
                "unloading",
                "operating_cost",
                "total_cost",
                "total_reward",
                "objective_value",
                "overflow_risk_t",
                "injection_tph",
                "entity_inventory_t",
                "vessel_berths",
            }
        ),
        elapsed_hours=len(actions),
        stored_t=float(replay_env.cumulative_stored_t) - start_stored_t,
        vented_t=float(replay_env.ledger.vented_t) - start_vented_t,
        captured_t=float(replay_env.cumulative_captured_t) - start_captured_t,
        in_transit_t=replay_env._in_transit_inventory(),
        vessel_fuel=float(replay_env.ledger.vessel_fuel) - float(start_ledger.vessel_fuel),
        conditioning=float(replay_env.ledger.conditioning) - float(start_ledger.conditioning),
        reconditioning=float(replay_env.ledger.reconditioning) - float(start_ledger.reconditioning),
        loading=float(replay_env.ledger.loading) - float(start_ledger.loading),
        unloading=float(replay_env.ledger.unloading) - float(start_ledger.unloading),
        operating_cost=float(replay_env.ledger.operating_cost) - start_operating_cost,
        total_cost=float(replay_env.ledger.total_cost) - start_total_cost,
        total_reward=total_reward,
        objective_value=-total_reward / float(replay_env.config.reward_scale),
        overflow_risk_t=overflow_risk_t,
        injection_tph=tuple(injection_tph),
        entity_inventory_t={
            entity_id: float(state.entity_inventory_t.get(entity_id, 0.0))
            for entity_id in replay_env.network.entities
        },
        vessel_berths={
            vessel_id: state.vessel_berths.get(vessel_id)
            for vessel_id in replay_env.vessel_ids
        },
    )
    replay = replay_native_actions(
        env,
        actions,
        horizon_h=horizon_h,
        expected=expected,
    )
    actual = replay.actual
    legacy_is_valid = not (set(violations) & invalid)
    return _NativeMpcCandidate(
        name=name,
        native_actions_by_hour=actions,
        vented_t=actual.vented_t,
        end_unstored_t=actual.in_transit_t,
        operating_cost=actual.operating_cost,
        total_cost=actual.total_cost,
        is_valid=legacy_is_valid and replay.is_executable and replay.is_exact,
        is_exact=replay.is_exact,
        mismatches=replay.mismatches,
    )


def _dedicated_assignments(env: CCSEnv) -> list[dict[str, str]]:
    vessels = list(env.vessel_ids)
    emitters = list(env.emitter_ids)
    if not vessels or not emitters:
        return []
    if len(vessels) == len(emitters):
        return [
            dict(zip(vessels, emitter_order))
            for emitter_order in permutations(emitters)
        ]
    return [
        {
            vessel_id: emitters[(vessel_index + offset) % len(emitters)]
            for vessel_index, vessel_id in enumerate(vessels)
        }
        for offset in range(len(emitters))
    ]


def _make_dedicated_policy(assignment: dict[str, str]) -> Policy:
    def policy(env: CCSEnv) -> dict[str, list]:
        state = env.simulator.state
        vessel_actions: list[int] = []
        masks = env.vessel_action_mask()
        for vessel_id, mask in zip(env.vessel_ids, masks):
            vessel = env.network.entities[vessel_id]
            cargo_t = float(state.entity_inventory_t.get(vessel_id, 0.0))
            berth = state.vessel_berths.get(vessel_id)
            if berth in env.terminal_ids and cargo_t > 1e-9:
                vessel_actions.append(VESSEL_WAIT)
            elif cargo_t >= vessel.capacity_t - 1e-9 and mask[VESSEL_GO_TERMINAL]:
                vessel_actions.append(VESSEL_GO_TERMINAL)
            elif berth == assignment[vessel_id] and cargo_t < vessel.capacity_t - 1e-9:
                vessel_actions.append(VESSEL_WAIT)
            else:
                action = env.vessel_go_emitter_action(assignment[vessel_id])
                vessel_actions.append(action if mask[action] else VESSEL_WAIT)
        return {
            "vessels": vessel_actions,
            "wells": [env.highest_feasible_well_rate_index(well_id) for well_id in env.well_ids],
        }

    return policy


def _forecast_urgency_policy(env: CCSEnv) -> dict[str, list]:
    state = env.simulator.state
    masks = env.vessel_action_mask()
    vessel_actions: list[int] = []
    for vessel_id, mask in zip(env.vessel_ids, masks):
        vessel = env.network.entities[vessel_id]
        cargo_t = float(state.entity_inventory_t.get(vessel_id, 0.0))
        berth = state.vessel_berths.get(vessel_id)
        if berth in env.terminal_ids and cargo_t > 1e-9:
            vessel_actions.append(VESSEL_WAIT)
        elif cargo_t >= vessel.capacity_t - 1e-9 and mask[VESSEL_GO_TERMINAL]:
            vessel_actions.append(VESSEL_GO_TERMINAL)
        elif berth in env.emitter_ids and cargo_t < vessel.capacity_t - 1e-9:
            vessel_actions.append(VESSEL_WAIT)
        else:
            action = _most_urgent_emitter_action(env, vessel_id, mask)
            vessel_actions.append(action if action is not None else VESSEL_WAIT)
    return {
        "vessels": vessel_actions,
        "wells": [env.highest_feasible_well_rate_index(well_id) for well_id in env.well_ids],
    }


def _most_urgent_emitter_action(env: CCSEnv, vessel_id: str, mask: list[bool]) -> int | None:
    state = env.simulator.state
    berth = state.vessel_berths.get(vessel_id)
    if berth is None:
        return None
    best: tuple[tuple[float, float], int] | None = None
    for emitter_id in env.emitter_ids:
        action = env.vessel_go_emitter_action(emitter_id)
        if not mask[action]:
            continue
        eta_h = _sail_hours_between(env, str(berth), emitter_id, vessel_id, max_horizon_h=168)
        emitter = env.network.entities[emitter_id]
        inventory_t = float(state.entity_inventory_t.get(emitter_id, 0.0))
        overflow_h = 168.0
        projected_t = inventory_t
        for t in range(168):
            projected_t += _capture_tonnes(env, emitter_id, t)
            if projected_t > emitter.buffer_capacity_t + 1e-9:
                overflow_h = float(t + 1)
                break
        score = (overflow_h - eta_h, -inventory_t)
        if best is None or score < best[0]:
            best = (score, action)
    return None if best is None else best[1]
