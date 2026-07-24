"""Encode event reasons and operational risk for residual dispatch RL.

为残差调度 RL 编码事件原因与运行风险。
"""

from __future__ import annotations

from math import inf

import numpy as np

from Simulation.entities.emitter import Emitter
from Simulation.environment import CCSEnv
from Simulation.operations.unloading import terminal_unload_queue_snapshot

from algorithms.rl.observation_encoder import (
    high_level_observation,
    high_level_observation_size,
)


EVENT_TYPES = (
    "initial",
    "maximum_interval",
    "vessel_arrival",
    "loading_completed",
    "unloading_completed",
    "emitter_fill_threshold",
    "overflow_risk_started",
    "weather_threshold",
    "well_availability_changed",
)
"""Stable event one-hot order. / 稳定的事件 one-hot 顺序。"""

RISK_HORIZON_H = 168.0
_EPS = 1e-9


def residual_observation(
    env: CCSEnv,
    *,
    decision_trigger: str,
    hours_since_decision: float,
    maximum_interval_h: float,
) -> np.ndarray:
    """Return base state, forecasts, event context, and risk features.

    返回基础状态、预测、事件上下文与风险特征。
    """
    if env.simulator is None:
        raise RuntimeError("Call env.reset() before requesting an observation.")
    values = [
        *high_level_observation(env),
        *_event_one_hot(decision_trigger),
        *_hours_to_overflow(env),
        *_travel_hours_matrix(env),
        _dispatchable_empty_fraction(env),
        _terminal_queue_fraction(env),
        min(1.0, max(0.0, hours_since_decision / max(_EPS, maximum_interval_h))),
    ]
    return np.asarray(values, dtype=np.float32)


def residual_observation_size(env: CCSEnv) -> int:
    """Return the fixed residual-observation length.

    返回固定的残差观测长度。
    """
    return (
        high_level_observation_size(env)
        + len(EVENT_TYPES)
        + len(env.emitter_ids)
        + len(env.vessel_ids) * len(env.emitter_ids)
        + 3
    )


def residual_feature_names(env: CCSEnv) -> tuple[str, ...]:
    """Return feature names in encoder order for experiment metadata.

    按编码顺序返回特征名称，用于实验元数据。
    """
    names = list(env.feature_names)
    from Simulation.environment.vessel_mode import (
        vessel_operation_mode_feature_names,
        vessel_sailing_destination_feature_names,
    )

    names.extend(vessel_operation_mode_feature_names(env))
    names.extend(vessel_sailing_destination_feature_names(env))
    for window_h in (24, 72):
        names.extend(
            f"{emitter}.availability_mean_{window_h}h"
            for emitter in env.emitter_ids
        )
        names.extend(f"{well}.available_mean_{window_h}h" for well in env.well_ids)
        names.extend(f"{well}.injectivity_min_{window_h}h" for well in env.well_ids)
        names.extend(
            (
                f"fleet.speed_mean_{window_h}h",
                f"fleet.speed_min_{window_h}h",
            )
        )
    names.extend(f"event.{event}" for event in EVENT_TYPES)
    names.extend(f"{emitter}.hours_to_overflow_norm" for emitter in env.emitter_ids)
    for vessel_id in env.vessel_ids:
        names.extend(
            f"{vessel_id}.to_{emitter}.travel_hours_norm"
            for emitter in env.emitter_ids
        )
    names.extend(
        (
            "fleet.dispatchable_empty_fraction",
            "terminal.unload_queue_fraction",
            "decision.hours_since_previous_norm",
        )
    )
    return tuple(names)


def _event_one_hot(decision_trigger: str) -> list[float]:
    """Map a detailed trigger such as ``vessel_arrival:vessel_1`` to one-hot.

    将带实体 ID 的详细触发原因映射为 one-hot。
    """
    trigger_type = str(decision_trigger).split(":", 1)[0]
    return [1.0 if trigger_type == event else 0.0 for event in EVENT_TYPES]


def _hours_to_overflow(env: CCSEnv) -> list[float]:
    """Estimate buffer headroom in hours, normalized to 168 h.

    估计排放源缓冲区剩余小时数，并归一化至 168 小时。
    """
    assert env.simulator is not None
    state = env.simulator.state
    values: list[float] = []
    for emitter_id in env.emitter_ids:
        emitter = env.network.entities[emitter_id]
        assert isinstance(emitter, Emitter)
        inventory_t = state.entity_inventory_t.get(emitter_id, 0.0)
        headroom_t = max(0.0, emitter.buffer_capacity_t - inventory_t)
        availability = state.emitter_availability.get(
            emitter_id,
            emitter.availability,
        )
        capture_tph = emitter.nominal_capture_tph * max(0.0, availability)
        hours = inf if capture_tph <= _EPS else headroom_t / capture_tph
        values.append(min(1.0, max(0.0, hours / RISK_HORIZON_H)))
    return values


def _travel_hours_matrix(env: CCSEnv) -> list[float]:
    """Return current vessel-to-emitter travel times normalized to 168 h.

    返回当前各船至各排放源的航行时间，并归一化至 168 小时。
    """
    assert env.simulator is not None
    values: list[float] = []
    for vessel_id in env.vessel_ids:
        route = env._routes[vessel_id]
        origin_id = env._weather_reference_origin(vessel_id)
        speed_knots = float(
            route.get("speed_knots") or env.config.default_speed_knots
        )
        for emitter_id in env.emitter_ids:
            if origin_id == emitter_id:
                values.append(0.0)
                continue
            leg_id = f"{origin_id}->{emitter_id}"
            factor = env._weather_speed_at(leg_id, vessel_id, 0)
            effective_speed = speed_knots * max(0.0, float(factor))
            if effective_speed <= _EPS:
                hours = RISK_HORIZON_H
            else:
                distance_km = env._leg_distance_km(
                    origin_id,
                    emitter_id,
                    route,
                )
                hours = distance_km / (effective_speed * 1.852)
            values.append(min(1.0, max(0.0, hours / RISK_HORIZON_H)))
    return values


def _dispatchable_empty_fraction(env: CCSEnv) -> float:
    """Return the fraction of empty vessels with at least one legal departure.

    返回至少存在一个合法出发动作的空载船舶比例。
    """
    assert env.simulator is not None
    state = env.simulator.state
    masks = env.vessel_action_mask()
    count = 0
    for position, vessel_id in enumerate(env.vessel_ids):
        vessel_state = env.simulator.vessel_states[vessel_id]
        cargo_t = state.entity_inventory_t.get(vessel_id, 0.0)
        emitter_actions = [
            env.vessel_go_emitter_action(emitter_id)
            for emitter_id in env.emitter_ids
        ]
        if (
            vessel_state["mode"] == "berthed"
            and cargo_t <= _EPS
            and any(masks[position][action] for action in emitter_actions)
        ):
            count += 1
    return count / max(1, len(env.vessel_ids))


def _terminal_queue_fraction(env: CCSEnv) -> float:
    """Return total unload-queue length normalized by fleet size.

    返回卸载队列总长度，并按船队规模归一化。
    """
    assert env.simulator is not None
    total = 0
    for terminal_id in env.terminal_ids:
        terminal = env.network.entities[terminal_id]
        total += len(
            terminal_unload_queue_snapshot(
                env.network,
                terminal,
                env.simulator.state,
            )
        )
    return total / max(1, len(env.vessel_ids))
