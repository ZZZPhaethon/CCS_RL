"""Shared simulation and serialization helpers for iterative Q datasets."""

from __future__ import annotations

import numpy as np

from scripts import compare_forecast_encoders_rl as compare

from sim.control.baselines import greedy_shuttle_policy
from sim.environment.event_residual_gym import EventJointResidualGymEnv
from sim.environment.forecast import current_state_feature_names
from sim.environment.forecast_gym import (
    variant_uses_operation_modes,
    variant_uses_sailing_destinations,
)
from sim.environment.vessel_mode import (
    vessel_operation_mode_feature_names,
    vessel_sailing_destination_feature_names,
)


DEFAULT_VARIANT = "future_mlp_mode_destination"


def _compare_args(args):
    return compare.parse_args(
        [
            "train",
            "--variant",
            str(args.variant),
            "--demo-cache",
            "unused-iterative-q-data.npz",
            "--timesteps",
            "0",
            "--bc-only",
            "--episode-hours",
            str(args.episode_hours),
            "--device",
            "cpu",
        ]
    )


def make_native_env(args):
    env = compare.make_experiment_env(_compare_args(args), demonstration=False)
    env.config.reward_scale = float(args.reward_scale)
    return env


def make_event_env(args) -> EventJointResidualGymEnv:
    return EventJointResidualGymEnv(
        make_native_env(args),
        str(args.variant),
        include_episode_progress=True,
        greedy_control_variate=False,
        hourly_gamma=1.0,
    )


def metrics(env) -> dict[str, float]:
    stored_t = float(env.ledger.stored_t)
    return {
        "total_cost_eur": float(env.ledger.total_cost),
        "operating_cost_eur": float(env.ledger.operating_cost),
        "vent_penalty_eur": float(env.ledger.vent_penalty),
        "vented_t": float(env.ledger.vented_t),
        "stored_t": stored_t,
        "unit_cost_eur_per_t": (
            float(env.ledger.total_cost) / stored_t if stored_t > 1e-9 else np.nan
        ),
    }


def greedy_baseline(args, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    """Run Greedy once and retain its hourly economic rewards."""

    env = make_native_env(args)
    env.reset(seed=int(seed))
    rewards = []
    while env.t < env.n_steps:
        _observation, reward, _terminated, _truncated, _info = env.step(
            greedy_shuttle_policy(env)
        )
        rewards.append(float(reward))
    return np.asarray(rewards, dtype=np.float64), metrics(env)


def event_residual_reward(
    raw_event_reward: float,
    greedy_hourly_rewards: np.ndarray,
    start_h: int,
    end_h: int,
) -> float:
    return float(raw_event_reward) - float(greedy_hourly_rewards[start_h:end_h].sum())


def empty_candidate_arrays(wrapper, max_events: int) -> dict[str, np.ndarray]:
    state_shape = wrapper.observation_space["state"].shape
    action_count = int(wrapper.action_space.n)
    return {
        "states": np.zeros((max_events, *state_shape), dtype=np.float32),
        "action_masks": np.zeros((max_events, action_count), dtype=bool),
        "actions": np.full(max_events, -1, dtype=np.int16),
        "physical_start_hours": np.full(max_events, -1, dtype=np.int16),
        "event_durations": np.zeros(max_events, dtype=np.int16),
        "event_residual_rewards": np.zeros(max_events, dtype=np.float32),
        "return_to_go": np.zeros(max_events, dtype=np.float32),
        "valid_steps": np.zeros(max_events, dtype=bool),
    }


def advance_follow(
    wrapper: EventJointResidualGymEnv,
    greedy_hourly_rewards: np.ndarray,
) -> tuple[dict, float, bool]:
    start_h = int(wrapper.env.t)
    observation, raw_reward, terminated, truncated, _info = wrapper.step(
        wrapper.follow_action()
    )
    end_h = int(wrapper.env.t)
    residual_reward = event_residual_reward(
        raw_reward, greedy_hourly_rewards, start_h, end_h
    )
    return observation, residual_reward, bool(terminated or truncated)


def state_feature_names(wrapper) -> list[str]:
    env = wrapper.env
    names = list(current_state_feature_names(env))
    if variant_uses_operation_modes(wrapper.variant):
        names.extend(vessel_operation_mode_feature_names(env))
    if variant_uses_sailing_destinations(wrapper.variant):
        names.extend(vessel_sailing_destination_feature_names(env))
    for vessel_id, dimension in zip(env.vessel_ids, env.vessel_action_dims):
        names.extend(
            f"greedy_proposal.{vessel_id}.native_action_{index}"
            for index in range(int(dimension))
        )
    names.append("episode_progress")
    return names


def stack_records(records: list[dict[str, object]]) -> dict[str, np.ndarray]:
    array_fields = (
        "states",
        "action_masks",
        "actions",
        "physical_start_hours",
        "event_durations",
        "event_residual_rewards",
        "return_to_go",
        "valid_steps",
    )
    scalar_fields = (
        "scenario_seed",
        "candidate_index",
        "sampling_attempt",
        "target_root_time_h",
        "root_time_h",
        "requested_sequence_events",
        "actual_sequence_events",
        "rollin_residual_return",
        "tail_residual_return",
        "residual_return",
    )
    data = {
        field: np.stack([record[field] for record in records])
        for field in array_fields
    }
    for field in scalar_fields:
        data[field] = np.asarray([record[field] for record in records])
    for prefix, key in (
        ("baseline", "baseline_metrics"),
        ("candidate", "candidate_metrics"),
    ):
        for metric in records[0][key]:
            data[f"{prefix}_{metric}"] = np.asarray(
                [record[key][metric] for record in records], dtype=np.float64
            )
    return data
