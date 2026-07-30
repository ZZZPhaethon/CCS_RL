"""Generate action comparisons on states visited by a locked iterative policy.

For every configured policy window, the locked policy rolls in to the first
decision event in that window. Each legal sparse joint action is substituted
once, then the same locked policy completes the 720-hour episode. Targets are
the scaled economic saving relative to the locked policy's original action at
that root. MPC is never called.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments import iterative_q_data_common as common
from sim.simulator import SimulatorStepCounter
from experiments.evaluate_iterative_action_q import (
    _load_model,
    expected_q_for_observation,
    select_safe_action,
)
from experiments.generate_iterative_q_greedy_data import select_dense_actions

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-config", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-two-vessel-actions", type=int, default=8)
    parser.add_argument("--max-three-vessel-actions", type=int, default=4)
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--reward-scale", type=float, default=1e-5)
    parser.add_argument("--dataset-seed", type=int, default=20260724)
    parser.add_argument("--variant", default=common.DEFAULT_VARIANT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-indices", type=int, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    common.add_scenario_protocol_arguments(parser)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("scenario seeds must be unique")
    if args.max_two_vessel_actions < 0 or args.max_three_vessel_actions < 0:
        parser.error("sample counts must be non-negative")
    if args.episode_hours <= 0 or args.reward_scale <= 0.0:
        parser.error("episode hours and reward scale must be positive")
    if not 0.0 <= args.hard_scenario_probability <= 1.0:
        parser.error("hard scenario probability must be inside [0, 1]")
    if args.forecast_context_hours < 168:
        parser.error("forecast context hours must be at least 168")
    if args.window_indices is not None and (
        len(set(args.window_indices)) != len(args.window_indices)
        or min(args.window_indices) < 0
    ):
        parser.error("window indices must be unique and non-negative")
    return args


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_policy(args):
    config_path = Path(args.lock_config)
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(configuration["locked_checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    if _checkpoint_sha256(checkpoint_path) != configuration["checkpoint_sha256"]:
        raise ValueError("locked checkpoint SHA256 does not match lock config")
    if configuration.get("uses_mpc_for_training_or_selection") is not False:
        raise ValueError("lock config must explicitly exclude MPC from training")
    device_name = str(args.device)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model_args = argparse.Namespace(checkpoint=str(checkpoint_path), device=device_name)
    model, metadata = _load_model(model_args, device)
    if str(metadata["observation_variant"]) != str(args.variant):
        raise ValueError("locked checkpoint observation variant mismatch")
    return model, metadata, configuration, device, checkpoint_path


def _policy_q(model, observation, env, device) -> np.ndarray:
    return expected_q_for_observation(model, observation, env, device)


def locked_action(
    wrapper, observation, model, metadata, policy_config, gate_state, device
):
    follow = int(metadata["follow_action_index"])
    action, decision = select_safe_action(
        _policy_q(model, observation, wrapper.env, device),
        wrapper.action_masks(),
        follow,
        required_heads=int(policy_config["required_heads"]),
        margin=float(policy_config["residual_margin"]),
    )
    active_window = next(
        (
            index
            for index, (start, end) in enumerate(policy_config["windows_h"])
            if float(start) <= float(wrapper.env.t) <= float(end)
        ),
        None,
    )
    if action != follow and (
        active_window is None or active_window in gate_state["used_windows"]
    ):
        action = follow
    if (
        action != follow
        and gate_state["override_events"] >= int(policy_config["max_overrides"])
    ):
        action = follow
    return int(action), active_window, decision


def update_gate_state(gate_state, action, follow, active_window):
    if int(action) == int(follow):
        return
    gate_state["override_events"] += 1
    if active_window is not None:
        gate_state["used_windows"].add(int(active_window))


def _step(wrapper, action):
    observation, _reward, terminated, truncated, _info = wrapper.step(int(action))
    return observation, bool(terminated or truncated)


def prepare_root(
    args,
    model,
    metadata,
    policy_config,
    seed,
    window_index,
    device,
    simulator_step_counter=None,
):
    wrapper = common.make_event_env(args, simulator_step_counter)
    observation, _info = wrapper.reset_native_seed(int(seed))
    gate_state = {"used_windows": set(), "override_events": 0}
    follow = int(metadata["follow_action_index"])
    window_start, window_end = policy_config["windows_h"][int(window_index)]
    done = False
    while float(wrapper.env.t) < float(window_start) and not done:
        action, active_window, _decision = locked_action(
            wrapper, observation, model, metadata, policy_config, gate_state, device
        )
        observation, done = _step(wrapper, action)
        update_gate_state(gate_state, action, follow, active_window)
    if done or float(wrapper.env.t) > float(window_end):
        return None
    anchor_action, active_window, _decision = locked_action(
        wrapper, observation, model, metadata, policy_config, gate_state, device
    )
    if active_window != int(window_index):
        raise RuntimeError("root is not inside its requested policy window")
    return wrapper, observation, gate_state, int(anchor_action), int(active_window)


def rollout_substitution(
    root, action, model, metadata, policy_config, device
):
    root_wrapper, root_observation, root_gate_state, _anchor, active_window = root
    wrapper = copy.deepcopy(root_wrapper)
    observation = copy.deepcopy(root_observation)
    gate_state = {
        "used_windows": set(root_gate_state["used_windows"]),
        "override_events": int(root_gate_state["override_events"]),
    }
    follow = int(metadata["follow_action_index"])
    observation, done = _step(wrapper, int(action))
    update_gate_state(gate_state, int(action), follow, active_window)
    while not done:
        next_action, next_window, _decision = locked_action(
            wrapper, observation, model, metadata, policy_config, gate_state, device
        )
        observation, done = _step(wrapper, next_action)
        update_gate_state(gate_state, next_action, follow, next_window)
    return common.metrics(wrapper.env)


def _candidate_record(
    wrapper,
    observation,
    seed,
    candidate_index,
    action,
    anchor_action,
    window_index,
    baseline_metrics,
    candidate_metrics,
    reward_scale,
):
    arrays = common.empty_candidate_arrays(wrapper, 1)
    arrays["states"][0] = observation["state"]
    if arrays["future_summaries"].shape[-1] > 0:
        arrays["future_summaries"][0] = common.v4_future_summary(wrapper)
    arrays["action_masks"][0] = wrapper.action_masks()
    arrays["actions"][0] = int(action)
    arrays["physical_start_hours"][0] = int(wrapper.env.t)
    arrays["valid_steps"][0] = True
    target = float(reward_scale) * (
        float(baseline_metrics["total_cost_eur"])
        - float(candidate_metrics["total_cost_eur"])
    )
    arrays["return_to_go"][0] = target
    return {
        **arrays,
        "scenario_seed": int(seed),
        "candidate_index": int(candidate_index),
        "sampling_attempt": 0,
        "target_root_time_h": int(wrapper.env.t),
        "root_time_h": int(wrapper.env.t),
        "requested_sequence_events": 1,
        "actual_sequence_events": 1,
        "rollin_residual_return": 0.0,
        "tail_residual_return": target,
        "residual_return": target,
        "anchor_action": int(anchor_action),
        "window_index": int(window_index),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
    }


def generate_dataset(args):
    out_path = Path(args.out_path)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing dataset: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, metadata, lock, device, checkpoint_path = load_locked_policy(args)
    policy_config = lock["policy"]
    records = []
    root_action_counts = []
    skipped_roots = 0
    simulator_step_counter = SimulatorStepCounter()
    window_indices = (
        list(range(len(policy_config["windows_h"])))
        if args.window_indices is None
        else list(args.window_indices)
    )
    if any(index >= len(policy_config["windows_h"]) for index in window_indices):
        raise ValueError("window index exceeds the configured policy windows")
    for seed in args.seeds:
        candidate_index = 0
        for window_index in window_indices:
            root = prepare_root(
                args,
                model,
                metadata,
                policy_config,
                seed,
                window_index,
                device,
                simulator_step_counter,
            )
            if root is None:
                skipped_roots += 1
                continue
            wrapper, observation, _gate_state, anchor_action, _active_window = root
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(args.dataset_seed), int(seed), int(window_index)]
                )
            )
            dense = select_dense_actions(
                wrapper,
                rng,
                args.max_two_vessel_actions,
                args.max_three_vessel_actions,
            )
            actions = np.unique(
                np.concatenate(
                    (
                        dense,
                        np.asarray(
                            [wrapper.follow_action(), anchor_action], dtype=np.int64
                        ),
                    )
                )
            )
            legal = wrapper.action_masks()
            actions = actions[legal[actions]]
            outcomes = {
                int(action): rollout_substitution(
                    root, int(action), model, metadata, policy_config, device
                )
                for action in actions
            }
            baseline_metrics = outcomes[int(anchor_action)]
            root_action_counts.append(len(actions))
            for action in actions:
                records.append(
                    _candidate_record(
                        wrapper,
                        observation,
                        seed,
                        candidate_index,
                        int(action),
                        anchor_action,
                        window_index,
                        baseline_metrics,
                        outcomes[int(action)],
                        args.reward_scale,
                    )
                )
                candidate_index += 1
    if not records:
        raise RuntimeError("locked policy produced no usable decision roots")
    data = common.stack_records(records)
    data["anchor_action"] = np.asarray(
        [record["anchor_action"] for record in records], dtype=np.int16
    )
    data["window_index"] = np.asarray(
        [record["window_index"] for record in records], dtype=np.int8
    )
    schema_wrapper = common.make_event_env(args, simulator_step_counter)
    schema_wrapper.reset_native_seed(int(args.seeds[0]))
    metadata_json = {
        "kind": "iterative_q_policy_rollin_data",
        "split": str(args.split),
        "scenario_seeds": sorted(
            set(int(record["scenario_seed"]) for record in records)
        ),
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
        "objective": (
            "720 h economic episode cost plus common compact terminal cleanup "
            "operating cost"
        ),
        "residual_reward": "scaled locked-policy anchor cost minus candidate cost",
        "uses_mpc": False,
        "scenario_protocol": str(args.scenario_protocol),
        "scenario_difficulties": common.scenario_difficulties(args),
        "anchors_in_data": True,
        "rollin_policy": str(args.lock_config),
        "rollin_checkpoint": str(checkpoint_path),
        "rollin_checkpoint_sha256": lock["checkpoint_sha256"],
        "policy_windows_h": policy_config["windows_h"],
        "training_simulator_usage": simulator_step_counter.snapshot().as_dict(),
        "configuration": vars(args),
    }
    data["metadata_json"] = np.asarray(
        json.dumps(metadata_json, separators=(",", ":"))
    )
    np.savez_compressed(out_path, **data)
    targets = data["return_to_go"][:, 0]
    summary = {
        "out_path": str(out_path),
        "candidates": int(len(targets)),
        "roots": int(len(root_action_counts)),
        "skipped_roots": int(skipped_roots),
        "mean_actions_per_root": float(np.mean(root_action_counts)),
        "improving_candidates": int((targets > 1e-6).sum()),
        "anchor_or_ties": int((np.abs(targets) <= 1e-6).sum()),
        "worse_candidates": int((targets < -1e-6).sum()),
        "mean_saving_eur": float(targets.mean() / float(args.reward_scale)),
        "best_saving_eur": float(targets.max() / float(args.reward_scale)),
        **simulator_step_counter.snapshot().as_dict(),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    return generate_dataset(parse_args(argv))


if __name__ == "__main__":
    main()
