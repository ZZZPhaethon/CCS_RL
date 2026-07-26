import numpy as np

from sim.control.rolling_milp import _capture_tonnes
from sim.environment.forecast import (
    FORECAST_HORIZON_H,
    current_state_feature_names,
    current_state_observation,
    forecast_channel_names,
    future_forecast_observation,
    masked_future_forecast_observation,
)
from sim.train import make_native_env


def _env(hours=2):
    return make_native_env(
        episode_hours=hours,
        scenario="northern_lights_phase1_3vessels",
        weather_mode="block",
        include_weather_obs=False,
    )


def test_three_vessel_forecast_is_168_by_9_and_starts_current_hour():
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
    assert forecast[0, 8] == env.scenario.vessel_speed_factor[vessel_id][0]


def test_future_capture_uses_hourly_emission_profile():
    env = _env()
    env.reset(seed=7)
    emitter_id = env.emitter_ids[0]
    emitter = env.network.entities[emitter_id]
    env.scenario.emitter_availability[emitter_id][0] = 2.0

    forecast = np.asarray(future_forecast_observation(env), dtype=np.float32)
    expected = (
        emitter.capture_rate_tph_at(0.0)
        * env.scenario.emitter_availability[emitter_id][0]
        / emitter.max_production_tph
    )

    assert np.isclose(forecast[0, 0], expected)
    assert expected > 1.0
    assert not np.isclose(
        emitter.capture_rate_tph_at(0.0),
        emitter.nominal_capture_tph,
    )


def test_future_capture_matches_mpc_rollout_offsets_zero_through_167():
    env = _env(hours=24)
    env.reset(seed=7)
    forecast = np.asarray(future_forecast_observation(env), dtype=np.float64)

    for channel, emitter_id in enumerate(env.emitter_ids):
        emitter = env.network.entities[emitter_id]
        expected = np.asarray(
            [
                _capture_tonnes(env, emitter_id, offset_h)
                / emitter.max_production_tph
                for offset_h in range(168)
            ]
        )
        np.testing.assert_allclose(forecast[:, channel], expected)


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
    assert forecast[0, 8] == env.scenario.vessel_speed_factor[vessel_id][719]
    assert forecast[-1, 8] == env.scenario.vessel_speed_factor[vessel_id][886]


def test_masked_forecast_hides_post_episode_context():
    env = _env(hours=720)
    env.reset(seed=7)
    env.simulator.state.time_h = 700.0
    forecast = np.asarray(masked_future_forecast_observation(env))
    assert forecast.shape == (168, 10)
    assert np.all(forecast[:20, -1] == 1.0)
    assert np.all(forecast[20:, -1] == 0.0)
    assert np.all(forecast[20:, :-1] == 0.0)
