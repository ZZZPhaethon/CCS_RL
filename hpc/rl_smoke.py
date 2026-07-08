"""Minimal RL smoke test for the Borg HPC environment."""

from __future__ import annotations

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

    env = CCSGymEnv(make_native_env(episode_hours=24, warm_start=True))
    obs, _info = env.reset(seed=0)
    action = env.action_space.sample()
    _obs, reward, terminated, truncated, _info = env.step(action)
    print("hybrid_action_space", env.action_space)
    print("obs_shape", obs.shape)
    print("one_step_reward", reward)
    print("one_step_done", terminated or truncated)

    eval_env = make_native_env(episode_hours=24, warm_start=False)
    _episodes, summary = evaluate(eval_env, greedy_shuttle_policy, seeds=[101])
    print("storage_rate", summary["storage_rate"]["mean"])
    print("loss_rate", summary["loss_rate"]["mean"])
    print("RL_SMOKE_OK")


if __name__ == "__main__":
    main()
