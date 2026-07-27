"""Encode a compact state and forecast summary for high-level RL.

为高层 RL 编码紧凑的状态与预测摘要。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from sim.environment import CCSEnv
from sim.environment.vessel_mode import (
    vessel_operation_mode_observation,
    vessel_operation_mode_feature_names,
    vessel_sailing_destination_feature_names,
    vessel_sailing_destination_observation,
)


FORECAST_WINDOWS_H = (24, 72)


def future_summary_observation(env: CCSEnv) -> np.ndarray:
    """Return the exact 24 h/72 h future summaries used by high-level PPO."""

    if env.simulator is None or env.scenario is None:
        raise RuntimeError("Call env.reset() before requesting future summaries.")
    return np.asarray(
        [
            value
            for window in FORECAST_WINDOWS_H
            for value in _window_summary(env, window)
        ],
        dtype=np.float32,
    )


def future_summary_feature_names(env: CCSEnv) -> tuple[str, ...]:
    """Return feature names for :func:`future_summary_observation`."""

    names: list[str] = []
    for window_h in FORECAST_WINDOWS_H:
        names.extend(
            f"{emitter}.availability_mean_{window_h}h"
            for emitter in env.emitter_ids
        )
        names.extend(
            f"{well}.available_mean_{window_h}h" for well in env.well_ids
        )
        names.extend(
            f"{well}.injectivity_min_{window_h}h" for well in env.well_ids
        )
        names.extend(
            (
                f"fleet.speed_mean_{window_h}h",
                f"fleet.speed_min_{window_h}h",
            )
        )
    return tuple(names)


def high_level_observation(env: CCSEnv) -> np.ndarray:
    """Return native state features plus fixed-size 24 h/72 h forecasts.

    返回原生状态特征与固定长度的 24 小时/72 小时预测。

    The native observation includes inventories, vessel cargo/location,
    terminal condition, injection availability, and reservoir pressure margin.
    Explicit vessel-operation modes and sailing destinations identify event
    context. Each forecast window adds average emitter/well availability,
    minimum injectivity, and mean/minimum vessel speed factor.

    原生观测包含库存、船载重/位置、终端状态、注入可用性和储层压力裕度；显式船舶作业
    模式与航行目的地用于标识事件上下文。每个预测窗口还加入 emitter/井可用率均值、
    最低注入能力以及船速系数的均值/最小值。
    """
    if env.simulator is None or env.scenario is None:
        raise RuntimeError("Call env.reset() before requesting a high-level observation.")
    base = np.asarray(env._observation(), dtype=np.float32)
    vessel_context = np.asarray(
        [
            *vessel_operation_mode_observation(env),
            *vessel_sailing_destination_observation(env),
        ],
        dtype=np.float32,
    )
    forecast = future_summary_observation(env)
    return np.concatenate((base, vessel_context, forecast)).astype(
        np.float32,
        copy=False,
    )


def high_level_observation_size(env: CCSEnv) -> int:
    """Return the fixed observation length for a configured environment.

    返回已配置环境的固定观测长度。
    """
    per_window = len(env.emitter_ids) + 2 * len(env.well_ids) + 2
    vessel_context = len(vessel_operation_mode_feature_names(env)) + len(
        vessel_sailing_destination_feature_names(env)
    )
    return (
        env.observation_size
        + vessel_context
        + len(FORECAST_WINDOWS_H) * per_window
    )


def _window_summary(env: CCSEnv, window_h: int) -> list[float]:
    """Summarise the available part of one future disturbance window.

    汇总一个未来扰动窗口中已可获得的部分。
    """
    assert env.scenario is not None
    assert env.simulator is not None
    scenario = env.scenario
    start = scenario.step_index(env.simulator.state.time_h)
    window_steps = max(1, int(round(window_h / scenario.time_step_hours)))
    end = min(scenario.n_steps, start + window_steps)
    indices = range(start, max(start + 1, end))
    values: list[float] = []
    for emitter_id in env.emitter_ids:
        values.append(_mean(scenario.emitter_availability.get(emitter_id, ()), indices, 1.0))
    for well_id in env.well_ids:
        values.append(_mean_bool(scenario.well_available.get(well_id, ()), indices, True))
    for well_id in env.well_ids:
        values.append(_minimum(scenario.injectivity_factor.get(well_id, ()), indices, 1.0))
    speed_series = [
        scenario.vessel_speed_factor.get(vessel_id, ())
        for vessel_id in env.vessel_ids
    ]
    flattened_speed = [
        _series_value(series, index, 1.0)
        for series in speed_series
        for index in indices
    ]
    values.extend((float(np.mean(flattened_speed)), float(np.min(flattened_speed))))
    return values


def _mean(values: Iterable[float], indices: Iterable[int], default: float) -> float:
    """Return a default-padded average for a disturbance series.

    返回带默认补值的扰动序列平均值。
    """
    series = tuple(float(value) for value in values)
    sampled = [_series_value(series, index, default) for index in indices]
    return float(np.mean(sampled))


def _mean_bool(values: Iterable[bool], indices: Iterable[int], default: bool) -> float:
    """Return an availability fraction for a boolean disturbance series.

    返回布尔扰动序列的可用比例。
    """
    return _mean((1.0 if value else 0.0 for value in values), indices, float(default))


def _minimum(values: Iterable[float], indices: Iterable[int], default: float) -> float:
    """Return a default-padded minimum for a disturbance series.

    返回带默认补值的扰动序列最小值。
    """
    series = tuple(float(value) for value in values)
    sampled = [_series_value(series, index, default) for index in indices]
    return float(np.min(sampled))


def _series_value(values: tuple[float, ...], index: int, default: float) -> float:
    """Read one series value or use a safe default beyond its horizon.

    读取一个序列值，在超出时域时使用安全默认值。
    """
    return values[index] if 0 <= index < len(values) else default
