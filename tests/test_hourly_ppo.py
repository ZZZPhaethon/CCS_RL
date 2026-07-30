from __future__ import annotations

import numpy as np
import pytest

from sim.control.event_based.rl.observation_encoder import (
    future_summary_observation,
    high_level_observation_size,
)
from sim.control.hourly_ppo.gym_env import HourlyCentralizedPPOEnv
from sim.control.hourly_ppo.train_hourly_ppo import make_hourly_native_env


def test_hourly_ppo_uses_direct_masked_action_and_one_hour_transition() -> None:
    native = make_hourly_native_env(
        episode_hours=4,
        forecast_context_hours=168,
        scenario_protocol="local_formal",
    )
    gym_env = HourlyCentralizedPPOEnv(
        native,
        future_summary_windows_h=(168,),
        episode_seed_min=7,
        episode_seed_max=7,
        max_simulator_hour_steps=1,
        include_terminal_cleanup_reward=False,
    )

    observation, info = gym_env.reset(seed=0)
    assert info["episode_seed"] == 7
    assert observation.shape == (high_level_observation_size(native, (168,)),)
    assert future_summary_observation(native, (168,)).size > 0
    assert gym_env.action_space.nvec.tolist() == native.vessel_action_dims
    assert gym_env.action_masks().dtype == np.bool_

    next_observation, _reward, terminated, truncated, step_info = gym_env.step(
        np.zeros(len(native.vessel_ids), dtype=np.int64)
    )

    assert next_observation.shape == observation.shape
    assert not terminated
    assert truncated
    assert step_info["decision_interval_h"] == pytest.approx(1.0)
    assert step_info["elapsed_hours"] == pytest.approx(1.0)
    assert step_info["native_steps"] == 1
    assert step_info["simulator_budget_exhausted"]
    assert native.simulator_step_usage().hour_steps == pytest.approx(1.0)
    assert gym_env.training_simulator_usage()["simulator_hour_steps"] == (
        pytest.approx(1.0)
    )


def test_hourly_ppo_terminal_cleanup_closes_fixed_horizon(
    monkeypatch,
) -> None:
    native = make_hourly_native_env(
        episode_hours=1,
        forecast_context_hours=168,
        scenario_protocol="local_formal",
        reward_scale=1e-6,
    )
    gym_env = HourlyCentralizedPPOEnv(
        native,
        episode_seed_min=11,
        episode_seed_max=11,
    )
    gym_env.reset(seed=0)
    monkeypatch.setattr(
        "sim.control.hourly_ppo.gym_env._terminal_cleanup_cost_for_state",
        lambda *_args: 123.0,
    )

    _observation, reward, terminated, truncated, info = gym_env.step(
        np.zeros(len(native.vessel_ids), dtype=np.int64)
    )

    realised_cost = float(info["economics"]["total_cost"])
    assert reward == pytest.approx(-(realised_cost + 123.0) * 1e-6)
    assert terminated
    assert not truncated
    assert info["terminal_cleanup_operating_cost_eur"] == pytest.approx(123.0)
    assert info["terminal_cleanup_included_in_reward"]
