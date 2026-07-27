from __future__ import annotations

import numpy as np

from .env import CCSEnv

FORECAST_HORIZON_H = 168
REPLAN_EVERY_H = 24
REPLAN_PHASE_FEATURE_NAMES = ("hours_since_replan", "is_replan")


def replan_phase_observation(
    time_h: float,
    replan_every_h: int = REPLAN_EVERY_H,
) -> tuple[float, float]:
    """Return normalized plan phase and an explicit replan indicator."""
    period = int(replan_every_h)
    if period <= 1:
        raise ValueError("replan_every_h must be greater than one")
    phase = int(time_h) % period
    return phase / float(period - 1), float(phase == 0)


def forecast_channel_names(env: CCSEnv) -> tuple[str, ...]:
    names = [f"capture.{emitter_id}" for emitter_id in env.emitter_ids]
    names += [f"emitter_available.{emitter_id}" for emitter_id in env.emitter_ids]
    names += [f"well_available.{well_id}" for well_id in env.well_ids]
    names += [f"injectivity.{well_id}" for well_id in env.well_ids]
    names += ["weather.global_speed_factor"]
    return tuple(names)


def current_state_feature_names(env: CCSEnv) -> tuple[str, ...]:
    if env.config.include_weather_obs:
        raise ValueError("forecast experiment requires include_weather_obs=False")
    return tuple([*env.feature_names, *env._global_current_weather_feature_names()])


def current_state_observation(env: CCSEnv) -> list[float]:
    if env.simulator is None or env.scenario is None:
        raise RuntimeError("Call env.reset() before requesting forecast observations.")
    if env.config.include_weather_obs:
        raise ValueError("forecast experiment requires include_weather_obs=False")
    return [*env._observation(), *env._global_current_weather_observation()]


def future_forecast_observation(
    env: CCSEnv,
    horizon_h: int = FORECAST_HORIZON_H,
) -> list[list[float]]:
    if env.simulator is None or env.scenario is None:
        raise RuntimeError("Call env.reset() before requesting forecast observations.")
    if len(env.emitter_ids) != 3 or len(env.well_ids) != 1:
        raise ValueError("the comparison forecast schema requires 3 emitters and 1 well")
    now_index = env.scenario.step_index(env.simulator.state.time_h)
    final_index = now_index + int(horizon_h)
    if final_index > env.scenario.n_steps:
        raise RuntimeError(
            f"forecast requires scenario index {final_index - 1}, "
            f"but trajectory ends at {env.scenario.n_steps - 1}"
        )
    vessel_id = env.vessel_ids[0]
    rows: list[list[float]] = []
    for index in range(now_index, final_index):
        capture = []
        emitter_online = []
        for emitter_id in env.emitter_ids:
            emitter = env.network.entities[emitter_id]
            multiplier = float(env.scenario.emitter_availability[emitter_id][index])
            capture_tph = emitter.capture_rate_tph_at(
                index * env.scenario.time_step_hours
            )
            capture.append(
                capture_tph * multiplier / max(1e-9, emitter.max_production_tph)
            )
            emitter_online.append(1.0 if multiplier > 0.0 else 0.0)
        well_available = [
            1.0 if env.scenario.well_available[well_id][index] else 0.0
            for well_id in env.well_ids
        ]
        injectivity = [
            float(env.scenario.injectivity_factor[well_id][index])
            for well_id in env.well_ids
        ]
        weather = [float(env.scenario.vessel_speed_factor[vessel_id][index])]
        rows.append([*capture, *emitter_online, *well_available, *injectivity, *weather])
    return rows


def masked_future_forecast_observation(
    env: CCSEnv,
    horizon_h: int = FORECAST_HORIZON_H,
) -> list[list[float]]:
    """Return finite-episode forecasts with a binary valid-horizon channel."""

    if env.simulator is None or env.scenario is None:
        raise RuntimeError("Call env.reset() before requesting forecast observations.")
    horizon = int(horizon_h)
    if horizon <= 0:
        raise ValueError("horizon_h must be positive")
    now_index = env.scenario.step_index(env.simulator.state.time_h)
    valid_steps = min(horizon, max(0, int(env.n_steps) - now_index))
    values = future_forecast_observation(env, valid_steps) if valid_steps else []
    channel_count = len(forecast_channel_names(env))
    rows = [[*row, 1.0] for row in values]
    rows.extend([[0.0] * (channel_count + 1) for _ in range(horizon - valid_steps)])
    return rows


def masked_forecast_summary(
    forecast: np.ndarray | list[list[float]],
    windows_h: tuple[int, ...],
) -> np.ndarray:
    """Summarise masked hourly physics over one or more look-ahead windows."""

    values = np.asarray(forecast, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] != 10:
        raise ValueError("masked forecast must end with shape [hours, 10]")
    summaries = []
    for window_h in windows_h:
        window = int(window_h)
        if window <= 0 or window > values.shape[-2]:
            raise ValueError("summary windows must be within the forecast horizon")
        part = values[..., :window, :]
        mask = part[..., -1] > 0.5
        has_valid = mask.any(axis=-1)
        count = np.maximum(mask.sum(axis=-1, keepdims=True), 1)
        masked = np.where(mask[..., None], part[..., :-1], 0.0)
        mean = masked.sum(axis=-2) / count
        injectivity = np.where(
            has_valid,
            np.where(mask, part[..., 7], np.inf).min(axis=-1),
            0.0,
        )
        weather_min = np.where(
            has_valid,
            np.where(mask, part[..., 8], np.inf).min(axis=-1),
            0.0,
        )
        valid_fraction = mask.mean(axis=-1)
        summaries.extend(
            (
                mean[..., 0],
                mean[..., 1],
                mean[..., 2],
                mean[..., 6],
                injectivity,
                mean[..., 8],
                weather_min,
                valid_fraction,
            )
        )
    return np.stack(summaries, axis=-1).astype(np.float32, copy=False)


def masked_forecast_band_summary(
    forecast: np.ndarray | list[list[float]],
    bands_h: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Summarise non-overlapping forecast bands without cumulative duplication."""

    values = np.asarray(forecast, dtype=np.float32)
    summaries = []
    for start_h, end_h in bands_h:
        start = int(start_h)
        end = int(end_h)
        if start < 0 or end <= start or end > values.shape[-2]:
            raise ValueError("summary bands must be ordered inside the forecast horizon")
        summaries.append(
            masked_forecast_summary(values[..., start:end, :], (end - start,))
        )
    return np.concatenate(summaries, axis=-1).astype(np.float32, copy=False)


def masked_forecast_summary_observation(
    env: CCSEnv,
    windows_h: tuple[int, ...],
    horizon_h: int = FORECAST_HORIZON_H,
) -> np.ndarray:
    return masked_forecast_summary(
        masked_future_forecast_observation(env, horizon_h=horizon_h),
        windows_h,
    )


def masked_forecast_band_summary_observation(
    env: CCSEnv,
    bands_h: tuple[tuple[int, int], ...],
    horizon_h: int = FORECAST_HORIZON_H,
) -> np.ndarray:
    return masked_forecast_band_summary(
        masked_future_forecast_observation(env, horizon_h=horizon_h),
        bands_h,
    )


def masked_forecast_summary_feature_names(
    env: CCSEnv,
    windows_h: tuple[int, ...],
) -> tuple[str, ...]:
    names = []
    for window_h in windows_h:
        names.extend(
            f"{emitter_id}.effective_capture_mean_{window_h}h"
            for emitter_id in env.emitter_ids
        )
        names.extend(
            (
                f"{env.well_ids[0]}.available_mean_{window_h}h",
                f"{env.well_ids[0]}.injectivity_min_{window_h}h",
                f"fleet.speed_mean_{window_h}h",
                f"fleet.speed_min_{window_h}h",
                f"valid_fraction_{window_h}h",
            )
        )
    return tuple(names)


def masked_forecast_band_summary_feature_names(
    env: CCSEnv,
    bands_h: tuple[tuple[int, int], ...],
) -> tuple[str, ...]:
    names = []
    for start_h, end_h in bands_h:
        label = f"{start_h}_{end_h}h"
        names.extend(
            f"{emitter_id}.effective_capture_mean_{label}"
            for emitter_id in env.emitter_ids
        )
        names.extend(
            (
                f"{env.well_ids[0]}.available_mean_{label}",
                f"{env.well_ids[0]}.injectivity_min_{label}",
                f"fleet.speed_mean_{label}",
                f"fleet.speed_min_{label}",
                f"valid_fraction_{label}",
            )
        )
    return tuple(names)
