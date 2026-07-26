"""Per-emitter state features for a learned bidding policy.

用于学习型竞价策略的逐排放源状态特征。

Each emitter (agent) observes a fixed-length, O(1)-scaled local feature vector.
The features are forward-looking on purpose: the myopic bid only sees the
projected vent over a fixed window, whereas a learned policy can weigh weather
exposure, vessel reachability, and competitor congestion to reorder who is
served -- which is where the gap to the centralized controller can be closed.

每个排放源(智能体)观测一个定长、量级约为 O(1) 的局部特征向量。特征刻意包含前瞻信息:
近视出价只看固定窗口内的预计放空,而学习型策略可以权衡天气暴露、船舶可达性与对手拥塞
来重排服务顺序——这正是可以缩小与中心化控制器差距的地方。
"""

from __future__ import annotations

from math import inf

import numpy as np

from Simulation.entities.emitter import Emitter
from Simulation.environment import CCSEnv

from .policies import _travel_hours


FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "fill_ratio",
    "hours_to_overflow_norm",
    "projected_vent_frac",
    "capture_frac",
    "nearest_vessel_travel_norm",
    "forecast_speed_min_72h",
    "max_other_fill",
    "dispatchable_empty_frac",
)
N_FEATURES = len(FEATURE_NAMES)

_EPS = 1e-9
_BID_HORIZON_H = 48.0
_OVERFLOW_CLIP_H = 168.0
_TRAVEL_CLIP_H = 72.0
_FORECAST_WINDOW_H = 72.0


def emitter_features(env: CCSEnv, emitter_id: str) -> np.ndarray:
    """Return the fixed-length local feature vector for one emitter.

    返回单个排放源的定长局部特征向量。
    """
    if env.simulator is None or env.scenario is None:
        raise RuntimeError("Call env.reset() before extracting features.")
    state = env.simulator.state
    emitter = env.network.entities[emitter_id]
    if not isinstance(emitter, Emitter):
        raise TypeError(f"{emitter_id} is not an Emitter.")

    capacity_t = max(_EPS, float(emitter.buffer_capacity_t))
    inventory_t = float(state.entity_inventory_t.get(emitter_id, 0.0))
    fill_ratio = inventory_t / capacity_t
    headroom_t = max(0.0, capacity_t - inventory_t)
    availability = float(
        state.emitter_availability.get(emitter_id, emitter.availability)
    )
    capture_tph = float(emitter.nominal_capture_tph) * max(0.0, availability)
    hours_to_overflow = (
        inf if capture_tph <= _EPS else headroom_t / capture_tph
    )
    hours_to_overflow_norm = min(1.0, hours_to_overflow / _OVERFLOW_CLIP_H)
    projected_vent_frac = (
        max(0.0, capture_tph * _BID_HORIZON_H - headroom_t) / capacity_t
    )

    max_capture = max(
        (
            float(env.network.entities[other].nominal_capture_tph)
            for other in env.emitter_ids
        ),
        default=1.0,
    )
    capture_frac = capture_tph / max(_EPS, max_capture)

    nearest_travel_h = _nearest_empty_vessel_travel_h(env, emitter_id)
    nearest_vessel_travel_norm = min(1.0, nearest_travel_h / _TRAVEL_CLIP_H)

    forecast_speed_min = _forecast_fleet_speed_min(env, _FORECAST_WINDOW_H)

    other_fills = [
        float(state.entity_inventory_t.get(other, 0.0))
        / max(_EPS, float(env.network.entities[other].buffer_capacity_t))
        for other in env.emitter_ids
        if other != emitter_id
    ]
    max_other_fill = max(other_fills, default=0.0)

    dispatchable_empty_frac = _dispatchable_empty_count(env) / max(
        1, len(env.vessel_ids)
    )

    return np.array(
        [
            1.0,
            fill_ratio,
            hours_to_overflow_norm,
            projected_vent_frac,
            capture_frac,
            nearest_vessel_travel_norm,
            forecast_speed_min,
            max_other_fill,
            dispatchable_empty_frac,
        ],
        dtype=float,
    )


def _dispatchable_empty_count(env: CCSEnv) -> int:
    """Count empty berthed vessels with a legal emitter departure.

    统计可合法驶向排放源的空载靠泊船舶。
    """
    assert env.simulator is not None
    state = env.simulator.state
    masks = env.vessel_action_mask()
    emitter_actions = [
        env.vessel_go_emitter_action(emitter_id)
        for emitter_id in env.emitter_ids
    ]
    count = 0
    for position, vessel_id in enumerate(env.vessel_ids):
        vessel_state = env.simulator.vessel_states[vessel_id]
        cargo_t = float(state.entity_inventory_t.get(vessel_id, 0.0))
        if (
            vessel_state["mode"] == "berthed"
            and cargo_t <= _EPS
            and any(masks[position][action] for action in emitter_actions)
        ):
            count += 1
    return count


def _nearest_empty_vessel_travel_h(env: CCSEnv, emitter_id: str) -> float:
    """Return travel hours of the nearest empty vessel that can reach an emitter.

    返回能够到达该排放源的最近空船的航行小时数。
    """
    assert env.simulator is not None
    state = env.simulator.state
    masks = env.vessel_action_mask()
    target_action = env.vessel_go_emitter_action(emitter_id)
    best = inf
    for position, vessel_id in enumerate(env.vessel_ids):
        vessel_state = env.simulator.vessel_states[vessel_id]
        cargo_t = float(state.entity_inventory_t.get(vessel_id, 0.0))
        if (
            vessel_state["mode"] == "berthed"
            and cargo_t <= _EPS
            and masks[position][target_action]
        ):
            best = min(best, _travel_hours(env, vessel_id, emitter_id))
    return best if best != inf else _TRAVEL_CLIP_H


def _forecast_fleet_speed_min(env: CCSEnv, window_h: float) -> float:
    """Return the minimum fleet speed factor over the forecast window.

    返回预测窗口内的最小船队速度系数。
    """
    assert env.simulator is not None and env.scenario is not None
    scenario = env.scenario
    start = scenario.step_index(env.simulator.state.time_h)
    steps = max(1, int(round(window_h / scenario.time_step_hours)))
    end = min(scenario.n_steps, start + steps)
    values: list[float] = []
    for vessel_id in env.vessel_ids:
        series = scenario.vessel_speed_factor.get(vessel_id, ())
        for index in range(start, max(start + 1, end)):
            values.append(
                float(series[index]) if 0 <= index < len(series) else 1.0
            )
    return min(values, default=1.0)
