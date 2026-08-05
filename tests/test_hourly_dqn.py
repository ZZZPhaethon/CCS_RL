from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from sim.control.hourly_dqn.gym_env import HourlyJointActionDQNEnv
from sim.control.hourly_dqn.model import (
    joint_action_mask,
    joint_action_table,
    split_flat_action_mask,
)
from sim.control.hourly_dqn.train_hourly_dqn import (
    ReplayBatch,
    double_dqn_loss,
    train_hourly_dqn,
)
from sim.control.hourly_ppo.gym_env import HourlyCentralizedPPOEnv
from sim.control.hourly_ppo.train_hourly_ppo import make_hourly_native_env


def test_joint_action_enumeration_and_mask() -> None:
    table = joint_action_table((2, 3))
    assert table.tolist() == [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
    ]
    split = split_flat_action_mask(
        np.asarray([True, False, False, True, True]),
        (2, 3),
    )
    assert joint_action_mask(split, table).tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
    ]


def test_hourly_dqn_wraps_direct_action_and_preserves_budget() -> None:
    native = make_hourly_native_env(
        episode_hours=4,
        forecast_context_hours=168,
        scenario_protocol="local_formal",
    )
    env = HourlyJointActionDQNEnv(
        HourlyCentralizedPPOEnv(
            native,
            future_summary_windows_h=(168,),
            episode_seed_min=7,
            episode_seed_max=7,
            max_simulator_hour_steps=1,
            include_terminal_cleanup_reward=False,
        )
    )
    observation, info = env.reset(seed=0)
    legal = env.action_masks()
    assert info["episode_seed"] == 7
    assert observation["state"].ndim == 1
    assert np.array_equal(observation["action_mask"], legal)
    assert env.action_space.n == int(np.prod(native.vessel_action_dims))

    next_observation, _reward, terminated, truncated, step_info = env.step(
        int(np.flatnonzero(legal)[0])
    )

    assert next_observation["state"].shape == observation["state"].shape
    assert not terminated
    assert truncated
    assert step_info["simulator_budget_exhausted"]
    assert env.training_simulator_usage()["simulator_step_calls"] == 1


class _ConstantQ(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.values.expand(len(states), -1)


def test_double_dqn_target_masks_illegal_online_argmax() -> None:
    online = _ConstantQ([2.0, 9.0])
    target = _ConstantQ([3.0, 20.0])
    batch = ReplayBatch(
        states=torch.zeros(1, 1),
        actions=torch.tensor([0]),
        rewards=torch.tensor([1.0]),
        next_states=torch.ones(1, 1),
        terminals=torch.tensor([0.0]),
        next_action_masks=torch.tensor([[True, False]]),
    )

    loss = double_dqn_loss(online, target, batch, gamma=1.0)

    # Legal next action 0 gives target 1 + 3 = 4; prediction is 2.
    assert loss.item() == pytest.approx(1.5)


def test_hourly_dqn_smoke_stops_at_exact_physical_budget(tmp_path) -> None:
    run_dir = tmp_path / "dqn"
    result = train_hourly_dqn(
        seed=0,
        episode_hours=4,
        forecast_context_hours=168,
        future_summary_windows_h=(168,),
        scenario_protocol="local_formal",
        batch_size=4,
        num_envs=1,
        device="cpu",
        log_dir=run_dir,
        max_simulator_hour_steps=16,
        training_seed_min=100,
        training_seed_max=200,
        validation_seeds=(300,),
        validation_every_simulator_hour_steps=8,
        hidden_sizes=(16,),
        replay_capacity=32,
        learning_starts=4,
        gradient_steps_per_vector_step=1,
        target_update_interval=4,
        log_every_simulator_hour_steps=8,
    )
    complete = json.loads(
        (result / "training_complete.json").read_text(encoding="utf-8")
    )

    assert complete["simulator_step_calls"] == 16
    assert complete["simulator_hour_steps"] == pytest.approx(16.0)
    assert complete["gradient_updates"] > 0
    assert (result / "masked_double_dqn_best_validation.pt").exists()
    assert (result / "masked_double_dqn_final.pt").exists()
