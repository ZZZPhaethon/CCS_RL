"""Generate same-state action comparisons after Greedy roll-in."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from experiments import iterative_q_data_common as common
from sim.simulator import SimulatorStepCounter

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--root-fractions", type=float, nargs="+", default=[0.10, 0.30, 0.50, 0.70]
    )
    parser.add_argument("--roots-per-seed", type=int)
    parser.add_argument("--max-two-vessel-actions", type=int, default=8)
    parser.add_argument("--max-three-vessel-actions", type=int, default=4)
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--reward-scale", type=float, default=1e-5)
    parser.add_argument("--dataset-seed", type=int, default=20260723)
    parser.add_argument("--variant", default=common.DEFAULT_VARIANT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    common.add_scenario_protocol_arguments(parser)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("scenario seeds must be unique")
    if not args.root_fractions or any(not 0.0 < value < 1.0 for value in args.root_fractions):
        parser.error("root fractions must lie in (0, 1)")
    if len(set(args.root_fractions)) != len(args.root_fractions):
        parser.error("root fractions must be unique")
    if args.roots_per_seed is not None and not (
        1 <= args.roots_per_seed <= len(args.root_fractions)
    ):
        parser.error("roots per seed must be between 1 and the root fraction count")
    if args.max_two_vessel_actions < 0 or args.max_three_vessel_actions < 0:
        parser.error("sample counts must be non-negative")
    if args.episode_hours <= 0 or args.reward_scale <= 0.0:
        parser.error("episode hours and reward scale must be positive")
    if not 0.0 <= args.hard_scenario_probability <= 1.0:
        parser.error("hard scenario probability must be inside [0, 1]")
    if args.forecast_context_hours < 168:
        parser.error("forecast context hours must be at least 168")
    return args


def select_root_fractions(args, seed: int) -> list[tuple[int, float]]:
    roots = list(enumerate(args.root_fractions))
    count = len(roots) if args.roots_per_seed is None else int(args.roots_per_seed)
    if count == len(roots):
        return roots
    start = int(seed) % len(roots)
    selected = sorted((start + offset) % len(roots) for offset in range(count))
    return [(index, float(args.root_fractions[index])) for index in selected]


def select_dense_actions(wrapper, rng, max_two: int, max_three: int) -> np.ndarray:
    """Return all one-vessel overrides plus sampled higher-order overrides."""

    follow = int(wrapper.follow_action())
    legal = np.flatnonzero(wrapper.action_masks())
    legal = legal[legal != follow]
    residuals = wrapper._joint_action_array[legal]
    counts = (
        residuals != wrapper.residual_env.follow_indices.reshape(1, -1)
    ).sum(axis=1)
    selected = list(legal[counts == 1])
    for count, limit in ((2, int(max_two)), (3, int(max_three))):
        candidates = legal[counts == count]
        if len(candidates) > limit:
            candidates = rng.choice(candidates, size=limit, replace=False)
        selected.extend(int(value) for value in candidates)
    return np.asarray(selected, dtype=np.int64)


def prepare_root(
    args,
    seed,
    target_root_h,
    greedy_rewards,
    simulator_step_counter=None,
):
    wrapper = common.make_event_env(args, simulator_step_counter)
    observation, _info = wrapper.reset_native_seed(int(seed))
    rollin_residual = 0.0
    done = False
    while wrapper.env.t < int(target_root_h) and not done:
        observation, residual_reward, done = common.advance_follow(
            wrapper, greedy_rewards
        )
        rollin_residual += residual_reward
    if done:
        raise RuntimeError("Greedy roll-in reached episode end before dense-action root")
    return wrapper, observation, float(rollin_residual)


def generate_candidate(
    args,
    seed,
    candidate_index,
    action,
    target_root_h,
    root,
    greedy_rewards,
    baseline_metrics,
):
    root_wrapper, root_observation, rollin_residual = root
    wrapper = copy.deepcopy(root_wrapper)
    observation = copy.deepcopy(root_observation)
    arrays = common.empty_candidate_arrays(wrapper, 1)
    action_mask = wrapper.action_masks()
    if not action_mask[int(action)]:
        raise RuntimeError("dense action became illegal after copying the root")
    start_h = int(wrapper.env.t)
    arrays["states"][0] = observation["state"]
    arrays["future_summaries"][0] = common.v4_future_summary(wrapper)
    arrays["action_masks"][0] = action_mask
    arrays["actions"][0] = int(action)
    arrays["physical_start_hours"][0] = start_h
    arrays["valid_steps"][0] = True

    observation, raw_reward, terminated, truncated, info = wrapper.step(int(action))
    end_h = int(wrapper.env.t)
    residual_reward = common.event_residual_reward(
        raw_reward, greedy_rewards, start_h, end_h
    )
    arrays["event_durations"][0] = int(info["event_duration_h"])
    arrays["event_residual_rewards"][0] = residual_reward
    done = bool(terminated or truncated)
    tail_residual = 0.0
    while not done:
        observation, tail_reward, done = common.advance_follow(
            wrapper, greedy_rewards
        )
        tail_residual += tail_reward

    candidate_metrics = common.metrics(wrapper.env)
    total_residual_return = float(rollin_residual + residual_reward + tail_residual)
    expected_residual_return = float(args.reward_scale) * (
        float(baseline_metrics["total_cost_eur"])
        - float(candidate_metrics["total_cost_eur"])
    )
    if not np.isclose(total_residual_return, expected_residual_return, atol=2e-5):
        raise RuntimeError(
            "residual reward is not aligned with full economic cost: "
            f"return={total_residual_return}, expected={expected_residual_return}"
        )
    arrays["return_to_go"][0] = float(residual_reward + tail_residual)
    return {
        **arrays,
        "scenario_seed": int(seed),
        "candidate_index": int(candidate_index),
        "sampling_attempt": 0,
        "target_root_time_h": int(target_root_h),
        "root_time_h": int(root_wrapper.env.t),
        "requested_sequence_events": 1,
        "actual_sequence_events": 1,
        "rollin_residual_return": float(rollin_residual),
        "tail_residual_return": float(tail_residual),
        "residual_return": total_residual_return,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
    }


def generate_dataset(args):
    out_path = Path(args.out_path)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing dataset: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    root_action_counts = []
    simulator_step_counter = SimulatorStepCounter()
    for seed in args.seeds:
        greedy_rewards, baseline_metrics = common.greedy_baseline(
            args,
            int(seed),
            simulator_step_counter,
        )
        candidate_index = 0
        for root_index, root_fraction in select_root_fractions(args, int(seed)):
            target_root_h = int(round(float(root_fraction) * args.episode_hours))
            root = prepare_root(
                args,
                seed,
                target_root_h,
                greedy_rewards,
                simulator_step_counter,
            )
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(args.dataset_seed), int(seed), int(root_index)]
                )
            )
            actions = select_dense_actions(
                root[0], rng, args.max_two_vessel_actions, args.max_three_vessel_actions
            )
            root_action_counts.append(len(actions))
            for action in actions:
                records.append(
                    generate_candidate(
                        args,
                        seed,
                        candidate_index,
                        int(action),
                        target_root_h,
                        root,
                        greedy_rewards,
                        baseline_metrics,
                    )
                )
                candidate_index += 1
    data = common.stack_records(records)
    schema_wrapper = common.make_event_env(args, simulator_step_counter)
    schema_wrapper.reset_native_seed(int(args.seeds[0]))
    metadata = {
        "kind": "iterative_q_greedy_rollin_data",
        "split": str(args.split),
        "scenario_seeds": [int(seed) for seed in args.seeds],
        "candidates_per_seed": None,
        "episode_hours": int(args.episode_hours),
        "observation_variant": str(args.variant),
        "state_feature_names": common.state_feature_names(schema_wrapper),
        "future_feature_names": common.v4_future_feature_names(schema_wrapper),
        "future_summary_representation_id": (
            common.FUTURE_SUMMARY_REPRESENTATION_ID
        ),
        "future_summary_windows_h": list(
            schema_wrapper.future_summary_windows_h
        ),
        "joint_actions": schema_wrapper._joint_action_array.tolist(),
        "follow_indices": schema_wrapper.residual_env.follow_indices.tolist(),
        "follow_action_index": int(schema_wrapper.follow_action()),
        "reward_scale": float(args.reward_scale),
        "objective": "pure economic operating cost plus vent penalty",
        "residual_reward": "scaled Greedy cost minus candidate cost",
        "uses_mpc": False,
        "scenario_protocol": str(args.scenario_protocol),
        "scenario_difficulties": common.scenario_difficulties(args),
        "root_fractions": [float(value) for value in args.root_fractions],
        "roots_per_seed": int(
            len(args.root_fractions)
            if args.roots_per_seed is None
            else args.roots_per_seed
        ),
        "training_simulator_usage": simulator_step_counter.snapshot().as_dict(),
        "configuration": vars(args),
    }
    data["metadata_json"] = np.asarray(json.dumps(metadata, separators=(",", ":")))
    np.savez_compressed(out_path, **data)
    delta = data["candidate_total_cost_eur"] - data["baseline_total_cost_eur"]
    summary = {
        "out_path": str(out_path),
        "candidates": int(len(delta)),
        "roots": int(len(root_action_counts)),
        "mean_actions_per_root": float(np.mean(root_action_counts)),
        "improving_candidates": int((delta < -1e-6).sum()),
        "ties": int((np.abs(delta) <= 1e-6).sum()),
        "worse_candidates": int((delta > 1e-6).sum()),
        "mean_delta_cost_eur": float(delta.mean()),
        "best_delta_cost_eur": float(delta.min()),
        **simulator_step_counter.snapshot().as_dict(),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    return generate_dataset(parse_args(argv))


if __name__ == "__main__":
    main()
