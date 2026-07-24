"""Build deterministic future forecasts and replanning features for RL agents.

为强化学习智能体构建确定性未来预测和重规划特征。
"""

from __future__ import annotations

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


def unload_queue_feature_names(env: CCSEnv) -> tuple[str, ...]:
    """FIFO unload-queue features that disambiguate otherwise aliased states."""
    names = []
    for vessel_id in env.vessel_ids:
        names += [f"{vessel_id}.queue_position", f"{vessel_id}.queue_head"]
    for terminal_id in env.terminal_ids:
        names.append(f"{terminal_id}.queue_length")
    return tuple(names)


def unload_queue_observation(env: CCSEnv) -> list[float]:
    if env.simulator is None:
        raise RuntimeError("Call env.reset() before requesting forecast observations.")
    queues = env.simulator.state.terminal_unload_queues
    vessel_count = max(1, len(env.vessel_ids))
    position_by_vessel: dict[str, float] = {}
    head_by_vessel: dict[str, float] = {}
    for queue in queues.values():
        for index, vessel_id in enumerate(queue):
            position_by_vessel[vessel_id] = (index + 1) / vessel_count
            head_by_vessel[vessel_id] = 1.0 if index == 0 else 0.0
    values: list[float] = []
    for vessel_id in env.vessel_ids:
        values.append(position_by_vessel.get(vessel_id, 0.0))
        values.append(head_by_vessel.get(vessel_id, 0.0))
    for terminal_id in env.terminal_ids:
        values.append(len(queues.get(terminal_id, [])) / vessel_count)
    return values


def current_state_feature_names(env: CCSEnv) -> tuple[str, ...]:
    if env.config.include_weather_obs:
        raise ValueError("forecast experiment requires include_weather_obs=False")
    return tuple(
        [
            *env.feature_names,
            *env._global_current_weather_feature_names(),
            *unload_queue_feature_names(env),
        ]
    )


def current_state_observation(env: CCSEnv) -> list[float]:
    if env.simulator is None or env.scenario is None:
        raise RuntimeError("Call env.reset() before requesting forecast observations.")
    if env.config.include_weather_obs:
        raise ValueError("forecast experiment requires include_weather_obs=False")
    return [
        *env._observation(),
        *env._global_current_weather_observation(),
        *unload_queue_observation(env),
    ]


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
