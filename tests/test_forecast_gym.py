import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sim.environment.forecast_gym import ForecastGymEnv, forecast_policy_observation
from sim.environment.forecast import current_state_feature_names
from sim.environment.gym_adapter import flat_action_mask
from sim.train import make_native_env


def _native():
    return make_native_env(
        episode_hours=2,
        scenario_context_hours=169,
        scenario="northern_lights_phase1_3vessels",
        weather_mode="block",
    )


def test_state_flat_and_tcn_observation_shapes():
    state_env = ForecastGymEnv(_native(), "state")
    flat_env = ForecastGymEnv(_native(), "flat")
    tcn_env = ForecastGymEnv(_native(), "tcn")

    state, _ = state_env.reset(seed=4)
    flat, _ = flat_env.reset(seed=4)
    structured, _ = tcn_env.reset(seed=4)

    assert state.shape == (51,)
    assert flat.shape == (51 + 168 * 9,)
    assert structured["state"].shape == (51,)
    assert structured["forecast"].shape == (168, 9)
    assert np.allclose(flat[:51], state)
    assert np.allclose(flat[51:].reshape(168, 9), structured["forecast"])


def test_operation_mode_variants_append_modes_to_current_state_only():
    native = _native()
    native.reset(seed=4)
    base_state = forecast_policy_observation(native, "state")
    state_mode = forecast_policy_observation(native, "state_mode")
    tcn_mode = forecast_policy_observation(native, "tcn_mode")
    mode_size = len(native.vessel_ids) * 5

    assert state_mode.shape == (len(base_state) + mode_size,)
    np.testing.assert_array_equal(state_mode[: len(base_state)], base_state)
    modes = state_mode[-mode_size:].reshape(len(native.vessel_ids), 5)
    np.testing.assert_array_equal(modes.sum(axis=1), np.ones(len(native.vessel_ids)))
    np.testing.assert_array_equal(tcn_mode["state"], state_mode)
    assert tcn_mode["forecast"].shape == (168, 9)


def test_tcn_mode_destination_appends_sailing_target_without_changing_legacy_variant():
    native = _native()
    native.reset(seed=4)
    vessel_id = native.vessel_ids[0]
    origin_id = native.emitter_ids[0]
    terminal_id = native.terminal_ids[0]
    other_emitter_id = native.emitter_ids[1]
    native.simulator.state.vessel_berths.pop(vessel_id, None)
    native.simulator.vessel_states[vessel_id].update(
        {
            "mode": "sailing",
            "berth": None,
            "origin": origin_id,
            "destination": terminal_id,
            "progress": 0.5,
        }
    )

    legacy_before = forecast_policy_observation(native, "tcn_mode")
    destination_before = forecast_policy_observation(native, "tcn_mode_destination")
    native.simulator.vessel_states[vessel_id]["destination"] = other_emitter_id
    legacy_after = forecast_policy_observation(native, "tcn_mode")
    destination_after = forecast_policy_observation(native, "tcn_mode_destination")
    destination_size = len(native.vessel_ids) * (
        len(native.terminal_ids) + len(native.emitter_ids)
    )

    np.testing.assert_array_equal(legacy_before["state"], legacy_after["state"])
    np.testing.assert_array_equal(
        destination_before["state"][:-destination_size],
        legacy_before["state"],
    )
    assert not np.array_equal(
        destination_before["state"][-destination_size:],
        destination_after["state"][-destination_size:],
    )
    wrapped = ForecastGymEnv(_native(), "tcn_mode_destination")
    observation, _ = wrapped.reset(seed=4)
    assert wrapped.observation_space["state"].shape == observation["state"].shape

    state_env = ForecastGymEnv(_native(), "state_mode")
    tcn_env = ForecastGymEnv(_native(), "tcn_mode")
    state_obs, _ = state_env.reset(seed=4)
    tcn_obs, _ = tcn_env.reset(seed=4)
    assert state_env.observation_space.contains(state_obs)
    assert tcn_env.observation_space.contains(tcn_obs)


@pytest.mark.parametrize("variant", ["state", "flat", "tcn", "state_mode", "tcn_mode"])
def test_terminal_observation_retains_declared_shape(variant):
    env = ForecastGymEnv(_native(), variant)
    observation, _ = env.reset(seed=4)
    action = np.zeros(len(env.action_space.nvec), dtype=np.int64)

    observation, _, terminated, truncated, _ = env.step(action)
    assert not terminated
    assert not truncated
    if variant == "flat":
        assert np.any(observation[51:] != 0.0)
    elif variant in {"tcn", "tcn_mode"}:
        assert np.any(observation["forecast"] != 0.0)

    observation, _, terminated, truncated, _ = env.step(action)
    assert not terminated
    assert truncated
    assert env.observation_space.contains(observation)
    if variant == "flat":
        assert np.any(observation[51:] != 0.0)
    elif variant in {"tcn", "tcn_mode"}:
        assert np.any(observation["forecast"] != 0.0)


def test_dummy_vec_env_bootstraps_from_real_terminal_forecast():
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: ForecastGymEnv(_native(), "tcn")])
    _observation = vec_env.reset()
    action = np.zeros((1, 4), dtype=np.int64)

    pre_terminal, _, dones, _ = vec_env.step(action)
    assert not dones[0]

    _reset_observation, _, dones, infos = vec_env.step(action)
    assert dones[0]
    assert infos[0]["TimeLimit.truncated"] is True
    terminal = infos[0]["terminal_observation"]
    assert terminal["forecast"].shape == (168, 9)
    assert np.allclose(terminal["forecast"][:-1], pre_terminal["forecast"][0, 1:])
    assert np.any(terminal["forecast"][-1] != 0.0)


def test_terminal_state_uses_timeout_hour_disturbances():
    from stable_baselines3.common.vec_env import DummyVecEnv

    wrapped = ForecastGymEnv(_native(), "tcn")
    vec_env = DummyVecEnv([lambda: wrapped])
    _observation = vec_env.reset()
    emitter_id = wrapped.env.emitter_ids[0]
    well_id = wrapped.env.well_ids[0]
    wrapped.env.scenario.emitter_availability[emitter_id][2] = 0.123
    wrapped.env.scenario.injectivity_factor[well_id][2] = 0.456
    wrapped.env.scenario.well_available[well_id][2] = False
    action = np.zeros((1, 4), dtype=np.int64)

    _observation, _, dones, _ = vec_env.step(action)
    assert not dones[0]
    _observation, _, dones, infos = vec_env.step(action)

    assert dones[0]
    terminal_state = infos[0]["terminal_observation"]["state"]
    names = current_state_feature_names(wrapped.env)
    assert terminal_state[names.index(f"{emitter_id}.availability")] == pytest.approx(
        0.123
    )
    assert terminal_state[names.index(f"{well_id}.injectivity")] == pytest.approx(
        0.456
    )
    assert terminal_state[names.index(f"{well_id}.available")] == 0.0


def test_action_masks_preserve_native_multidiscrete_order():
    env = ForecastGymEnv(_native(), "state")
    env.reset(seed=4)

    expected = flat_action_mask(
        env.env.vessel_action_mask(), env.env.well_rate_action_mask()
    )

    assert np.array_equal(env.action_masks(), expected)


@pytest.mark.parametrize("variant", ["state", "flat", "tcn", "state_mode", "tcn_mode"])
def test_policy_wrapper_forwards_observation_mask_and_native_action(variant):
    from sim.environment.forecast_gym import make_forecast_ppo_policy

    captured = {}

    class FakeModel:
        def predict(self, observation, deterministic=True, action_masks=None):
            captured["observation"] = observation
            captured["deterministic"] = deterministic
            captured["action_masks"] = action_masks
            return np.array([1, 2, 3, 4], dtype=np.int64), None

    env = _native()
    env.reset(seed=4)
    action = make_forecast_ppo_policy(FakeModel(), variant)(env)

    assert action == {"vessels": [1, 2, 3], "wells": [4]}
    assert not captured["deterministic"]
    assert np.array_equal(
        captured["action_masks"],
        flat_action_mask(env.vessel_action_mask(), env.well_rate_action_mask()),
    )
    if variant == "state":
        assert captured["observation"].shape == (51,)
    elif variant == "state_mode":
        assert captured["observation"].shape == (66,)
    elif variant == "flat":
        assert captured["observation"].shape == (51 + 168 * 9,)
    else:
        expected_state = 66 if variant == "tcn_mode" else 51
        assert captured["observation"]["state"].shape == (expected_state,)
        assert captured["observation"]["forecast"].shape == (168, 9)


def test_forecast_gym_interfaces_are_exported():
    from sim.environment import (
        ForecastGymEnv as ExportedForecastGymEnv,
        forecast_policy_observation,
        make_forecast_ppo_policy,
    )

    assert ExportedForecastGymEnv is ForecastGymEnv
    assert callable(forecast_policy_observation)
    assert callable(make_forecast_ppo_policy)


def test_core_environment_import_does_not_require_optional_rl_dependencies():
    source_root = Path(__file__).parents[1] / "src"
    code = """
import builtins

real_import = builtins.__import__

def import_without_rl(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'gymnasium'}:
        raise ImportError(f'blocked optional dependency: {name}')
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_rl
from sim.environment import CCSEnv
assert CCSEnv.__name__ == 'CCSEnv'
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=source_root.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
