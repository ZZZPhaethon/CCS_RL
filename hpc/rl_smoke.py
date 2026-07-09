"""Minimal RL smoke test for the Borg HPC environment."""

from __future__ import annotations

import os

import gymnasium
import sb3_contrib
import stable_baselines3
import torch

from sim.control.baselines import greedy_shuttle_policy
from sim.environment.gym_adapter import CCSGymEnv
from sim.metrics import evaluate
from sim.train import make_native_env


def main() -> None:
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device", torch.cuda.get_device_name(0))
    print("gymnasium", gymnasium.__version__)
    print("stable_baselines3", stable_baselines3.__version__)
    print("sb3_contrib", sb3_contrib.__version__)

    weather_rate = float(os.environ.get("WEATHER_WINDOW_RATE_PER_WEEK", "1.0"))
    native_env = make_native_env(
        episode_hours=24,
        warm_start=True,
        scenario="northern_lights_phase1_3vessels",
        include_weather_obs=True,
        weather_mode="window",
        weather_window_rate_per_week=weather_rate,
    )
    env = CCSGymEnv(native_env)
    obs, _info = env.reset(seed=0)
    action = env.action_space.sample()
    _obs, reward, terminated, truncated, _info = env.step(action)
    print("hybrid_action_space", env.action_space)
    print("obs_shape", obs.shape)
    print("weather_observation_layout", native_env.config.weather_observation_layout)
    print("weather_window_rate_per_week", native_env.scenario_generator.config.weather_window_rate_per_week)
    print("one_step_reward", reward)
    print("one_step_done", terminated or truncated)
    assert obs.shape == (55,)
    assert native_env.config.weather_observation_layout == "global"
    assert "hour_of_year_sin" not in native_env.feature_names
    assert "hour_of_year_cos" not in native_env.feature_names

    eval_env = make_native_env(
        episode_hours=24,
        warm_start=False,
        scenario="northern_lights_phase1_3vessels",
        include_weather_obs=True,
        weather_mode="window",
        weather_window_rate_per_week=weather_rate,
    )
    _episodes, summary = evaluate(eval_env, greedy_shuttle_policy, seeds=[101])
    print("storage_rate", summary["storage_rate"]["mean"])
    print("loss_rate", summary["loss_rate"]["mean"])
    print("RL_SMOKE_OK")


if __name__ == "__main__":
    main()
