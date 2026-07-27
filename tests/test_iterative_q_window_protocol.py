from types import SimpleNamespace

import pytest

from experiments import iterative_q_data_common as common
from sim.control.event_based.rl.observation_encoder import (
    future_summary_observation,
    high_level_observation,
)


def _args(hard_probability: float):
    return SimpleNamespace(
        episode_hours=24,
        reward_scale=1e-5,
        variant=common.DEFAULT_VARIANT,
        scenario_protocol="v4_mixed_window",
        hard_scenario_probability=hard_probability,
        forecast_context_hours=168,
        seeds=[123],
    )


@pytest.mark.parametrize(
    ("hard_probability", "expected_difficulty"),
    ((0.0, "normal"), (1.0, "hard")),
)
def test_iterative_q_window_protocol_uses_v4_difficulty_generator(
    hard_probability,
    expected_difficulty,
):
    args = _args(hard_probability)
    env = common.make_native_env(args)
    env.reset(seed=123)

    assert env.scenario_generator.last_difficulty == expected_difficulty
    assert env.config.reward_mode == "economic"
    assert env.config.reward_scale == pytest.approx(1e-5)
    assert env.config.weather_observation_layout == "global"
    assert all(
        env.network.entities[emitter_id].hourly_capture_profile_tph
        for emitter_id in env.emitter_ids
    )


def test_iterative_q_window_policy_state_excludes_v4_future_aggregates():
    args = _args(0.5)
    wrapper = common.make_event_env(args)
    observation, _info = wrapper.reset_native_seed(123)
    names = common.state_feature_names(wrapper)

    assert observation["state"].shape == (len(names),)
    assert not any("mean_24h" in name or "mean_72h" in name for name in names)


def test_iterative_q_future_summary_exactly_matches_v4():
    wrapper = common.make_event_env(_args(0.5))
    wrapper.reset_native_seed(123)

    summary = common.v4_future_summary(wrapper)
    names = common.v4_future_feature_names(wrapper)

    assert summary.shape == (14,)
    assert len(names) == 14
    assert summary == pytest.approx(high_level_observation(wrapper.env)[-14:])


def test_iterative_q_uses_shared_configurable_future_summary():
    args = _args(0.5)
    args.future_summary_windows_h = (168,)
    wrapper = common.make_event_env(args)
    wrapper.reset_native_seed(123)

    summary = common.v4_future_summary(wrapper)
    names = common.v4_future_feature_names(wrapper)

    assert summary.shape == (7,)
    assert len(names) == 7
    assert summary == pytest.approx(
        future_summary_observation(wrapper.env, (168,))
    )


def test_unified_window_protocol_uses_one_fixed_configuration():
    args = _args(0.0)
    args.scenario_protocol = "unified_window_v1"
    first = common.make_native_env(args)
    first.reset(seed=123)
    config = first.scenario_generator.normal.config

    assert first.automatic_well_control
    assert first.well_rate_action_dims == []
    assert "wells" not in common.greedy_shuttle_policy(first)
    assert config.capture_noise_std == pytest.approx(0.30)
    assert config.capture_high_output_rate_per_week == pytest.approx(0.5)
    assert config.capture_high_output_mean_hours == pytest.approx(48.0)
    assert config.capture_high_output_multiplier_range == (1.25, 1.75)
    assert config.weather_window_rate_per_week == pytest.approx(0.5)
    assert config.weather_window_mean_hours == pytest.approx(48.0)
    assert config.weather_window_speed_factor_range == (0.50, 0.80)
    assert config.well_maintenance_rate_per_week == pytest.approx(0.3)
    assert config.well_maintenance_mean_hours == pytest.approx(12.0)
    assert config.emitter_initial_fill_range == (0.0, 0.50)
    assert config.terminal_initial_fill_range == (0.0, 0.50)
    assert config.warm_start
    assert config.reservoir_initial_pressure_fill_range == (0.0, 0.50)

    args.hard_scenario_probability = 1.0
    second = common.make_native_env(args)
    second.reset(seed=123)

    assert (
        first.scenario.initial_inventory_t
        == second.scenario.initial_inventory_t
    )
    assert (
        first.scenario.emitter_availability
        == second.scenario.emitter_availability
    )
    assert (
        first.scenario.vessel_speed_factor
        == second.scenario.vessel_speed_factor
    )
    assert first.scenario.well_available == second.scenario.well_available
