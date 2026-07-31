"""Evaluate frozen E1 controllers from 720 h through annual horizons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments import evaluate_iterative_action_q as iterative_q_eval
from experiments import iterative_q_data_common as common
from sim.control.baselines import (
    greedy_shuttle_policy,
    make_cluster_shuttle_policy,
)
from sim.metrics import run_recorded_episode


CONTROLLERS = (
    "fixed_assignment",
    "greedy",
    "iterative_q_direct",
    "iterative_q_receding",
)
HORIZONS = (720, 2160, 4320, 8760)
FORMAL_SEEDS = tuple(range(9_000_031, 9_000_061))
FORECAST_CONTEXT_HOURS = 168
ANNUAL_SCENARIO_HOURS = 8760 + FORECAST_CONTEXT_HOURS
BASE_POLICY_WINDOWS = (
    (108, 155),
    (156, 203),
    (204, 251),
    (252, 299),
    (300, 347),
    (348, 395),
    (396, 443),
    (444, 491),
    (492, 539),
    (540, 587),
    (588, 635),
    (636, 680),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=CONTROLLERS, required=True)
    parser.add_argument("--horizon-hours", type=int, choices=HORIZONS, required=True)
    parser.add_argument("--model-seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--iterative-q-model-root",
        type=Path,
        default=(
            REPO_ROOT
            / "experiments_results"
            / "E1"
            / "models"
            / "iterative_q"
        ),
    )
    args = parser.parse_args(argv)
    is_q = args.controller.startswith("iterative_q")
    if is_q and args.model_seed is None:
        parser.error("--model-seed is required for Iterative-Q")
    if not is_q and args.model_seed is not None:
        parser.error("--model-seed only applies to Iterative-Q")
    if not args.seeds:
        parser.error("--seeds must not be empty")
    return args


def expanded_policy_windows(
    horizon_hours: int,
    *,
    cycle_hours: int = 720,
) -> tuple[tuple[int, int], ...]:
    if horizon_hours <= 0 or cycle_hours <= 0:
        raise ValueError("horizon_hours and cycle_hours must be positive")
    windows = []
    cycles = int(math.ceil(horizon_hours / cycle_hours))
    for cycle in range(cycles):
        offset = cycle * cycle_hours
        for start, end in BASE_POLICY_WINDOWS:
            shifted_start = start + offset
            if shifted_start >= horizon_hours:
                continue
            windows.append((shifted_start, min(end + offset, horizon_hours - 1)))
    return tuple(windows)


def receding_episode_progress(
    hour: int | float,
    *,
    cycle_hours: int = 720,
) -> float:
    if cycle_hours <= 0:
        raise ValueError("cycle_hours must be positive")
    return float(hour) % cycle_hours / cycle_hours


def with_receding_progress(observation, hour: int | float):
    if not isinstance(observation, dict) or "state" not in observation:
        raise TypeError("Iterative-Q observation must contain a state branch")
    updated = dict(observation)
    state = np.asarray(observation["state"], dtype=np.float32).copy()
    state[-1] = receding_episode_progress(hour)
    updated["state"] = state
    return updated


def global_episode_progress(
    hour: int | float,
    horizon_hours: int,
) -> float:
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    return float(hour) / horizon_hours


def with_global_progress(
    observation,
    hour: int | float,
    horizon_hours: int,
):
    if not isinstance(observation, dict) or "state" not in observation:
        raise TypeError("Iterative-Q observation must contain a state branch")
    updated = dict(observation)
    state = np.asarray(observation["state"], dtype=np.float32).copy()
    state[-1] = global_episode_progress(hour, horizon_hours)
    updated["state"] = state
    return updated


def q_policy_windows(
    controller: str,
    horizon_hours: int,
) -> tuple[tuple[int, int], ...]:
    if controller not in {"iterative_q_direct", "iterative_q_receding"}:
        raise ValueError(f"unsupported Iterative-Q controller: {controller}")
    return expanded_policy_windows(horizon_hours)


def _environment_args(horizon_hours: int) -> SimpleNamespace:
    return SimpleNamespace(
        episode_hours=int(horizon_hours),
        scenario_episode_hours=ANNUAL_SCENARIO_HOURS,
        forecast_context_hours=FORECAST_CONTEXT_HOURS,
        scenario_protocol="unified_window_v1",
        hard_scenario_probability=0.5,
        stress_level="medium",
        reward_scale=0.00001,
        variant="future_mlp_mode",
    )


def _cleanup_cost(env) -> float:
    return common._terminal_cleanup_cost_for_state(
        env,
        env.cost_model.parameters,
    )


def _baseline_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    env_args = _environment_args(args.horizon_hours)
    env = common.make_native_env(env_args)
    policy = (
        make_cluster_shuttle_policy(env)
        if args.controller == "fixed_assignment"
        else greedy_shuttle_policy
    )
    rows = []
    for seed in args.seeds:
        record = run_recorded_episode(
            env,
            policy,
            controller=args.controller,
            seed=int(seed),
            terminal_cleanup_cost=_cleanup_cost,
        ).as_dict()
        rows.append(
            {
                "controller": args.controller,
                "model_seed": "",
                "test_seed": int(seed),
                "horizon_hours": int(args.horizon_hours),
                "scenario_hours": ANNUAL_SCENARIO_HOURS,
                "event_count": "",
                "override_events": "",
                "episode_wall_time_s": float(record["wall_clock_seconds"]),
                "episode_total_cost_eur": float(record["episode_total_cost"]),
                "terminal_cleanup_operating_cost_eur": float(
                    record["terminal_cleanup_operating_cost"]
                ),
                "total_cost_eur": float(record["total_cost"]),
                "normalized_cost_eur_per_720h": (
                    float(record["total_cost"]) * 720.0 / args.horizon_hours
                ),
                "captured_t": float(record["captured_t"]),
                "stored_t": float(record["stored_t"]),
                "vented_t": float(record["vented_t"]),
                "normalized_vented_t_per_720h": (
                    float(record["vented_t"]) * 720.0 / args.horizon_hours
                ),
                "storage_rate": float(record["storage_rate"]),
                "unit_cost_eur_per_captured_t": (
                    float(record["total_cost"]) / float(record["captured_t"])
                ),
            }
        )
    return rows


def _q_checkpoint(model_seed: int, model_root: Path) -> Path:
    return (
        model_root
        / f"g60_p4_model_seed_{model_seed}"
        / "iterative_action_q.pt"
    )


def _q_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    env_args = _environment_args(args.horizon_hours)
    checkpoint = _q_checkpoint(
        int(args.model_seed),
        args.iterative_q_model_root,
    )
    load_args = iterative_q_eval.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(args.output_dir / "_unused"),
            "--eval-seeds",
            *(str(seed) for seed in args.seeds),
            "--episode-hours",
            str(args.horizon_hours),
            "--reward-scale",
            "0.00001",
            "--gates",
            "unused:4:0.40:12:108-155",
            "--scenario-protocol",
            "unified_window_v1",
            "--hard-scenario-probability",
            "0.5",
            "--forecast-context-hours",
            "168",
            "--device",
            "cpu",
        ]
    )
    device = torch.device("cpu")
    model, metadata = iterative_q_eval._load_model(load_args, device)
    variant = str(metadata["observation_variant"])
    if metadata["state_feature_names"][-1] != "episode_progress":
        raise ValueError(
            "temporal deployment requires episode_progress as the final state feature"
        )
    env_args.variant = variant
    wrapper = common.make_event_env(env_args)
    follow_index = int(metadata["follow_action_index"])
    receding = args.controller == "iterative_q_receding"
    windows = q_policy_windows(args.controller, args.horizon_hours)
    max_overrides = len(windows)

    rows = []
    for seed in args.seeds:
        started_at = time.perf_counter()
        observation, _info = wrapper.reset_native_seed(int(seed))
        done = False
        event_count = 0
        override_events = 0
        used_windows: set[int] = set()
        while not done:
            if receding:
                observation = with_receding_progress(
                    observation,
                    wrapper.env.t,
                )
            else:
                observation = with_global_progress(
                    observation,
                    wrapper.env.t,
                    args.horizon_hours,
                )
            expected_q = iterative_q_eval.expected_q_for_observation(
                model,
                observation,
                wrapper.env,
                device,
            )
            action, _decision = iterative_q_eval.select_safe_action(
                expected_q,
                wrapper.action_masks(),
                follow_index,
                required_heads=4,
                margin=0.40,
                uncertainty_beta=0.0,
            )
            active_window = next(
                (
                    index
                    for index, (start, end) in enumerate(windows)
                    if start <= wrapper.env.t <= end
                ),
                None,
            )
            if (
                action != follow_index
                and (active_window is None or active_window in used_windows)
            ):
                action = follow_index
            if action != follow_index and override_events >= max_overrides:
                action = follow_index
            if action != follow_index and active_window is not None:
                used_windows.add(active_window)
            observation, _reward, terminated, truncated, _info = wrapper.step(
                action
            )
            event_count += 1
            override_events += int(action != follow_index)
            done = bool(terminated or truncated)

        metrics = common.metrics(wrapper.env)
        captured_t = float(wrapper.env.cumulative_captured_t)
        rows.append(
            {
                "controller": args.controller,
                "model_seed": int(args.model_seed),
                "test_seed": int(seed),
                "horizon_hours": int(args.horizon_hours),
                "scenario_hours": ANNUAL_SCENARIO_HOURS,
                "event_count": event_count,
                "override_events": override_events,
                "episode_wall_time_s": time.perf_counter() - started_at,
                "episode_total_cost_eur": metrics["episode_total_cost_eur"],
                "terminal_cleanup_operating_cost_eur": metrics[
                    "terminal_cleanup_operating_cost_eur"
                ],
                "total_cost_eur": metrics["total_cost_eur"],
                "normalized_cost_eur_per_720h": (
                    metrics["total_cost_eur"] * 720.0 / args.horizon_hours
                ),
                "captured_t": captured_t,
                "stored_t": metrics["stored_t"],
                "vented_t": metrics["vented_t"],
                "normalized_vented_t_per_720h": (
                    metrics["vented_t"] * 720.0 / args.horizon_hours
                ),
                "storage_rate": (
                    metrics["stored_t"] / captured_t if captured_t > 0.0 else 1.0
                ),
                "unit_cost_eur_per_captured_t": (
                    metrics["total_cost_eur"] / captured_t
                    if captured_t > 0.0
                    else np.nan
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = (
        _q_rows(args)
        if args.controller.startswith("iterative_q")
        else _baseline_rows(args)
    )
    output_csv = args.output_dir / "per_episode.csv"
    _write_csv(output_csv, rows)
    metadata = {
        "experiment": "E7 temporal-horizon generalization",
        "controller": args.controller,
        "model_seed": args.model_seed,
        "test_seeds": [int(seed) for seed in args.seeds],
        "execution_horizon_hours": args.horizon_hours,
        "scenario_hours": ANNUAL_SCENARIO_HOURS,
        "forecast_context_hours": FORECAST_CONTEXT_HOURS,
        "nested_scenario_prefix": True,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "device": "cpu",
        "receding_adapter": (
            {
                "cycle_hours": 720,
                "episode_progress": "(t mod 720) / 720",
                "policy_windows": [list(window) for window in expanded_policy_windows(args.horizon_hours)],
                "weights_changed": False,
            }
            if args.controller == "iterative_q_receding"
            else None
        ),
        "direct_policy_windows": (
            [
                list(window)
                for window in q_policy_windows(
                    args.controller,
                    args.horizon_hours,
                )
            ]
            if args.controller == "iterative_q_direct"
            else None
        ),
        "direct_global_adapter": (
            {
                "cycle_hours": 720,
                "episode_progress": "t / H",
                "policy_windows": [
                    list(window)
                    for window in q_policy_windows(
                        args.controller,
                        args.horizon_hours,
                    )
                ],
                "physical_state_reset": False,
                "weights_changed": False,
            }
            if args.controller == "iterative_q_direct"
            else None
        ),
        "records": len(rows),
    }
    if args.controller.startswith("iterative_q"):
        checkpoint = _q_checkpoint(
            int(args.model_seed),
            args.iterative_q_model_root,
        )
        checkpoint_payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_metadata = checkpoint_payload["metadata"]
        metadata["checkpoint"] = str(checkpoint)
        metadata["checkpoint_sha256"] = hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
        metadata["observation_schema"] = {
            "source_state_features": len(
                checkpoint_metadata.get(
                    "source_state_feature_names",
                    checkpoint_metadata["state_feature_names"],
                )
            ),
            "state_features": len(checkpoint_metadata["state_feature_names"]),
            "excluded_state_features": checkpoint_metadata.get(
                "excluded_state_feature_names",
                [],
            ),
        }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_csv


def main() -> None:
    output = run(parse_args())
    print(f"E7_SHARD_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
