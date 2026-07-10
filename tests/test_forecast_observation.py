import numpy as np

from sim.environment.forecast import (
    FORECAST_HORIZON_H,
    current_state_feature_names,
    current_state_observation,
    forecast_channel_names,
    future_forecast_observation,
)
from sim.train import make_native_env


def _env(hours=2):
    return make_native_env(
        episode_hours=hours,
        scenario="northern_lights_phase1_3vessels",
        weather_mode="block",
        include_weather_obs=False,
    )


def test_three_vessel_forecast_is_168_by_9_and_starts_next_hour():
    env = _env()
    env.reset(seed=7)
    forecast = np.asarray(future_forecast_observation(env), dtype=np.float32)
    assert FORECAST_HORIZON_H == 168
    assert forecast.shape == (168, 9)
    assert forecast_channel_names(env) == (
        "capture.brevik",
        "capture.celsio",
        "capture.yara_sluiskil",
        "emitter_available.brevik",
        "emitter_available.celsio",
        "emitter_available.yara_sluiskil",
        "well_available.aurora_well_a7_ah",
        "injectivity.aurora_well_a7_ah",
        "weather.global_speed_factor",
    )
    vessel_id = env.vessel_ids[0]
    assert forecast[0, 8] == env.scenario.vessel_speed_factor[vessel_id][1]


def test_current_state_has_current_weather_but_no_future_summaries():
    env = _env()
    env.reset(seed=7)
    names = current_state_feature_names(env)
    state = current_state_observation(env)
    assert len(names) == len(state) == 51
    assert "weather.speed_now" in names
    assert "weather.speed_24h_mean" not in names
    assert "weather.speed_168h_mean" not in names
    assert all(np.isfinite(state))


def test_last_rl_step_still_has_full_forecast_context():
    env = _env(hours=720)
    env.reset(seed=7)
    idle = {"vessels": [0] * len(env.vessel_ids), "wells": [0] * len(env.well_ids)}
    for _ in range(719):
        _observation, _reward, terminated, truncated, _info = env.step(idle)
        assert not terminated
        assert not truncated

    forecast = np.asarray(future_forecast_observation(env))
    vessel_id = env.vessel_ids[0]
    assert forecast.shape == (168, 9)
    assert forecast[0, 8] == env.scenario.vessel_speed_factor[vessel_id][720]
    assert forecast[-1, 8] == env.scenario.vessel_speed_factor[vessel_id][887]
