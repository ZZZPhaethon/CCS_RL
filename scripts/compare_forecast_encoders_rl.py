"""Reproducible MPC-BC + PPO comparison for forecast observation encoders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import numpy as np

from sim.control.baselines import greedy_shuttle_policy, idle_policy
from sim.control.demonstrations import (
    collect_mpc_demonstrations,
    load_demonstrations,
    merge_demonstration_shards,
    save_demonstrations,
)
from sim.control.imitation import (
    action_dimension_weights,
    apply_replan_action_weight,
    behavior_clone,
    behavior_clone_balanced_decisions,
    decision_only_action_weights,
    make_kickstart_callback,
)
from sim.control.native_mpc import RollingNativeMpcController
from sim.environment.forecast import current_state_feature_names, forecast_channel_names
from sim.environment.forecast_encoder import (
    BalancedEdgeGNNForecastExtractor,
    BalancedEdgeGNNFutureMLPForecastExtractor,
    EdgeGNNForecastExtractor,
    EntityResidualEdgeGNNForecastExtractor,
    FixedScaleEdgeGNNForecastExtractor,
    FixedScaleLargerMLPForecastExtractor,
    FixedScaleTCNForecastExtractor,
    FutureConditionedEdgeGNNForecastExtractor,
    FutureMLPForecastExtractor,
    GNNForecastExtractor,
    GatedPastMLPForecastExtractor,
    GatedResidualEdgeGNNForecastExtractor,
    LargerMLPForecastExtractor,
    PastMLPForecastExtractor,
    StableTCNForecastExtractor,
    TCNForecastExtractor,
)
from sim.environment.forecast_gym import (
    ForecastGymEnv,
    forecast_policy_observation,
    variant_base_encoder,
    variant_uses_learned_plan_context,
    variant_uses_oracle_candidate,
    variant_uses_operation_modes,
    variant_uses_past,
    variant_uses_zero_past,
)
from sim.environment.gym_adapter import flat_action_mask, native_action_from_flat
from sim.environment.past import PAST_HOURS, PastObservationBuffer
from sim.environment.vessel_mode import (
    VESSEL_OPERATION_MODES,
    vessel_operation_mode_feature_names,
    vessel_sailing_destination_feature_names,
)
from sim.metrics import EpisodeMetrics, run_episode
from sim.train import make_native_env

try:
    from sb3_contrib import MaskablePPO
except ImportError:  # pragma: no cover - exercised only in incomplete installations
    MaskablePPO = None


FORMAL_SCENARIO = "northern_lights_phase1_3vessels"
FORECAST_HORIZON_H = 168
FORECAST_CHANNELS = 9
FORMAL_VARIANTS = ("state", "state_mode", "tcn", "tcn_mode")

RESULT_COLUMNS = (
    "vented_t",
    "policy",
    "family",
    "variant",
    "stage",
    "deterministic",
    "model_seed",
    "eval_seed",
    "emitter_inventory_t",
    "vessel_inventory_t",
    "terminal_inventory_t",
    "in_transit_t",
    "captured_t",
    "stored_t",
    "loss_rate",
    "storage_rate",
    "operating_cost",
    "vent_penalty",
    "total_cost",
    "cost_per_stored_t",
    "total_cost_per_stored_t",
    "throttle_hours",
    "well_switch_count",
    "berth_wait_vessel_hours",
    "pressure_risk_hours",
    "min_pressure_margin_fraction",
    "longest_venting_streak_hours",
    "total_reward",
    "episode_runtime_s",
    "mean_inference_latency_s",
    "trainable_parameters",
    "demonstration_exact_match",
    "demonstration_action_accuracy",
)

SUMMARY_METRICS = (
    "vented_t",
    "emitter_inventory_t",
    "vessel_inventory_t",
    "terminal_inventory_t",
    "in_transit_t",
    "captured_t",
    "stored_t",
    "loss_rate",
    "storage_rate",
    "operating_cost",
    "vent_penalty",
    "total_cost",
    "cost_per_stored_t",
    "total_cost_per_stored_t",
    "throttle_hours",
    "well_switch_count",
    "berth_wait_vessel_hours",
    "pressure_risk_hours",
    "min_pressure_margin_fraction",
    "longest_venting_streak_hours",
    "total_reward",
    "episode_runtime_s",
    "mean_inference_latency_s",
    "trainable_parameters",
    "demonstration_exact_match",
)


def demonstration_task_seeds(task_id: int) -> tuple[int, ...]:
    task_id = int(task_id)
    if task_id < 0 or task_id > 9:
        raise ValueError(f"demonstration task ID must be in 0..9, got {task_id}")
    start = task_id * 10
    return tuple(range(start, start + 10))


def formal_training_task(task_id: int) -> tuple[str, int]:
    task_id = int(task_id)
    if task_id < 0 or task_id > 19:
        raise ValueError(f"formal training task ID must be in 0..19, got {task_id}")
    return FORMAL_VARIANTS[task_id % len(FORMAL_VARIANTS)], task_id // len(FORMAL_VARIANTS)


def bc_objective_training_task(task_id: int) -> tuple[str, int]:
    task_id = int(task_id)
    variants = ("state_mode", "tcn_mode")
    if task_id < 0 or task_id > 9:
        raise ValueError(f"BC objective task ID must be in 0..9, got {task_id}")
    return variants[task_id % len(variants)], task_id // len(variants)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _add_locked_protocol_defaults(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.set_defaults(
        scenario=FORMAL_SCENARIO,
        forecast_horizon_h=FORECAST_HORIZON_H,
        weather_mode="block",
        reward_mode="economic",
        mpc_objective_mode="economic",
        carbon_price_eur_per_t=80.0,
        store_reward_eur_per_t=0.0,
        vent_penalty_weight=1.0,
        operating_cost_weight=1.0,
        enforce_full_load_dispatch=False,
        require_empty_terminal_departure=True,
        capture_noise_std=0.30,
        initial_inventory_fill_max=0.5,
        leg_wave_slowdown_multiplier=1.0,
        leg_wave_speed_factor_floor=0.0,
        weather_window_rate_per_week=1.0,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    demos = commands.add_parser("generate-demos")
    demos.add_argument("--demo-cache", required=True)
    demos.add_argument("--demo-seeds", type=int, nargs="+", required=True)
    demos.add_argument("--teacher", choices=("mpc", "greedy"), default="mpc")
    _add_locked_protocol_defaults(demos)

    merge = commands.add_parser("merge-demos")
    merge.add_argument("--shards", nargs="+", required=True)
    merge.add_argument("--demo-cache", required=True)
    merge.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    merge.add_argument("--episode-hours", type=int, default=720)

    train = commands.add_parser("train")
    train.add_argument(
        "--variant",
        choices=(
            "state",
            "flat",
            "tcn",
            "state_mode",
            "tcn_mode",
            "tcn_mode_destination",
            "gnn_mode_destination",
            "larger_mlp_mode_destination",
            "edge_gnn_mode_destination",
            "future_mlp",
            "future_mlp_mode",
            "future_mlp_mode_destination",
            "gated_past24_mlp_mode_destination",
            "past24_mlp_mode_destination",
            "past24_zero_mlp_mode_destination",
            "balanced_edge_gnn_mode_destination",
            "balanced_edge_gnn_future_mlp_mode_destination",
            "future_conditioned_edge_gnn_mode_destination",
            "gated_residual_edge_gnn_mode_destination",
            "entity_residual_edge_gnn_mode_destination",
            "fixed_scale_larger_mlp_mode_destination",
            "fixed_scale_edge_gnn_mode_destination",
            "stable_tcn_mode_destination",
            "fixed_scale_tcn_mode_destination",
            "fixed_scale_tcn_mode_destination_replan_phase",
            "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate",
            "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context",
        ),
        required=True,
    )
    train.add_argument("--demo-cache", required=True)
    train.add_argument("--heldout-demo-cache")
    train.add_argument("--timesteps", type=_nonnegative_int, default=100_000)
    train.add_argument("--bc-epochs", type=int, default=20)
    train.add_argument(
        "--bc-objective",
        choices=("current", "decision_only", "decision_balanced"),
        default="current",
    )
    train.add_argument("--bc-only", action="store_true")
    train.add_argument("--n-steps", type=int, default=512)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--bc-batch-size", type=int, default=256)
    train.add_argument("--bc-lr", type=float, default=1e-3)
    train.add_argument("--bc-label-smoothing", type=float, default=0.0)
    train.add_argument("--model-seed", type=int, default=0)
    train.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    train.add_argument("--device", default="auto")
    train.add_argument("--verbose", type=int, default=1)
    train.add_argument("--progress-bar", action="store_true")
    train.add_argument("--imitation-only", action="store_true")
    train.add_argument("--out-dir", default="output/rl_forecast")
    train.add_argument("--replan-action-weight", type=float)
    train.set_defaults(
        gamma=0.999,
        learning_rate=3e-4,
        nonwait_weight=10.0,
        replan_action_weight=1.0,
        kickstart_coef=1.0,
    )
    _add_locked_protocol_defaults(train)

    report_parser = commands.add_parser("report")
    report_parser.add_argument("--out-dir", default="output/rl_forecast")
    return parser.parse_args(argv)


def make_experiment_env(args, demonstration: bool = False):
    logical_hours = int(args.episode_hours)
    horizon = int(args.forecast_horizon_h)
    episode_hours = logical_hours + horizon + 1 if demonstration else logical_hours
    context_hours = 0 if demonstration else horizon + 1
    return make_native_env(
        episode_hours=episode_hours,
        scenario_context_hours=context_hours,
        warm_start=True,
        scenario=FORMAL_SCENARIO,
        weather_mode="block",
        include_weather_obs=False,
        reward_mode="economic",
        carbon_price_eur_per_t=80.0,
        store_reward_eur_per_t=0.0,
        vent_penalty_weight=1.0,
        operating_cost_weight=1.0,
        enforce_full_load_dispatch=False,
        require_empty_terminal_departure=True,
    )


class ExperimentEnvFactory:
    def __init__(self, args):
        self.args = args

    def __call__(self, demonstration: bool = False):
        return make_experiment_env(self.args, demonstration=demonstration)

    def metadata(self) -> dict[str, object]:
        env = self()
        channels = list(forecast_channel_names(env))
        state_names = list(current_state_feature_names(env))
        return {
            "scenario": FORMAL_SCENARIO,
            "episode_hours": int(self.args.episode_hours),
            "horizon_h": int(self.args.forecast_horizon_h),
            "forecast_shape": [int(self.args.forecast_horizon_h), len(channels)],
            "forecast_channels": channels,
            "forecast_schema_version": 4,
            "forecast_capture_source": "uncapped_hourly_profile_times_availability",
            "forecast_start_offset_h": 0,
            "forecast_end_offset_h": int(self.args.forecast_horizon_h) - 1,
            "state_feature_names": state_names,
            "state_size": len(state_names),
            "operation_mode_feature_names": list(vessel_operation_mode_feature_names(env)),
            "operation_mode_shape": [len(env.vessel_ids), len(VESSEL_OPERATION_MODES)],
            "vessel_destination_feature_names": list(
                vessel_sailing_destination_feature_names(env)
            ),
            "vessel_destination_shape": [
                len(env.vessel_ids),
                len(env.terminal_ids) + len(env.emitter_ids),
            ],
            "action_dimensions": [*env.vessel_action_dims, *env.well_rate_action_dims],
            "vessel_action_dimensions": list(env.vessel_action_dims),
            "well_rate_action_dimensions": list(env.well_rate_action_dims),
            "weather_mode": "block",
            "weather_observation_layout": env.config.weather_observation_layout,
            "include_weather_obs": False,
            "reward": {
                "mode": "economic",
                "carbon_price_eur_per_t": 80.0,
                "store_reward_eur_per_t": 0.0,
                "vent_penalty_weight": 1.0,
                "operating_cost_weight": 1.0,
            },
            "partial_load_dispatch": True,
            "require_empty_terminal_departure": True,
            "warm_start": True,
            "scenario_context_hours": int(self.args.forecast_horizon_h) + 1,
            "emitter_buffer_capacity_t": {
                emitter_id: float(env.network.entities[emitter_id].buffer_capacity_t)
                for emitter_id in env.emitter_ids
            },
            "disturbance_defaults": {
                "capture_noise_std": 0.30,
                "initial_inventory_fill_max": 0.5,
                "leg_wave_slowdown_multiplier": 1.0,
                "leg_wave_speed_factor_floor": 0.0,
                "weather_window_rate_per_week": 1.0,
            },
        }


def model_policy_config(variant: str):
    extractor_classes = {
        "gnn": GNNForecastExtractor,
        "larger_mlp": LargerMLPForecastExtractor,
        "edge_gnn": EdgeGNNForecastExtractor,
        "future_mlp": FutureMLPForecastExtractor,
        "gated_past_mlp": GatedPastMLPForecastExtractor,
        "past_mlp": PastMLPForecastExtractor,
        "balanced_edge_gnn": BalancedEdgeGNNForecastExtractor,
        "balanced_edge_gnn_future_mlp": BalancedEdgeGNNFutureMLPForecastExtractor,
        "future_conditioned_edge_gnn": FutureConditionedEdgeGNNForecastExtractor,
        "gated_residual_edge_gnn": GatedResidualEdgeGNNForecastExtractor,
        "entity_residual_edge_gnn": EntityResidualEdgeGNNForecastExtractor,
        "fixed_scale_larger_mlp": FixedScaleLargerMLPForecastExtractor,
        "fixed_scale_edge_gnn": FixedScaleEdgeGNNForecastExtractor,
        "stable_tcn": StableTCNForecastExtractor,
        "fixed_scale_tcn": FixedScaleTCNForecastExtractor,
    }
    base_encoder = variant_base_encoder(variant)
    if base_encoder in extractor_classes:
        extractor_kwargs = {
            "state_features": 64,
            "forecast_features": 64,
        }
        if base_encoder in {"gated_past_mlp", "past_mlp"}:
            extractor_kwargs["past_features"] = 64
        return "MultiInputPolicy", {
            "features_extractor_class": extractor_classes[base_encoder],
            "features_extractor_kwargs": extractor_kwargs,
        }
    if base_encoder == "tcn":
        return "MultiInputPolicy", {
            "features_extractor_class": TCNForecastExtractor,
            "features_extractor_kwargs": {
                "state_features": 64,
                "forecast_features": 64,
            },
        }
    if base_encoder in {"state", "flat"}:
        return "MlpPolicy", {}
    raise ValueError(f"unknown variant: {variant}")


def policy_manifest(variant: str) -> dict[str, object]:
    policy_name, policy_kwargs = model_policy_config(variant)
    if policy_name == "MlpPolicy":
        return {"name": policy_name, "features_extractor": None}
    return {
        "name": policy_name,
        "features_extractor": policy_kwargs["features_extractor_class"].__name__,
        **policy_kwargs["features_extractor_kwargs"],
    }


def checkpoint_path(args, stage: str) -> Path:
    if stage not in {"bc", "ppo"}:
        raise ValueError(f"unknown checkpoint stage: {stage}")
    return Path(args.out_dir) / (
        f"{stage}_{args.variant}{_bc_objective_suffix(args)}_seed{args.model_seed}.zip"
    )


def results_path(args) -> Path:
    return Path(args.out_dir) / (
        f"results_{args.variant}{_bc_objective_suffix(args)}_seed{args.model_seed}.csv"
    )


def run_manifest_path(args) -> Path:
    return Path(args.out_dir) / (
        f"run_{args.variant}{_bc_objective_suffix(args)}_seed{args.model_seed}.manifest.json"
    )


def _bc_objective_suffix(args) -> str:
    objective = getattr(args, "bc_objective", "current")
    return "" if objective == "current" else f"_{objective}"


def demo_manifest_path(cache_path: Path) -> Path:
    return Path(f"{cache_path}.manifest.json")


def normalize_demo_cache_path(path) -> Path:
    cache_path = Path(path)
    if cache_path.suffix != ".npz":
        cache_path = Path(f"{cache_path}.npz")
    return cache_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    override = os.environ.get("GIT_COMMIT")
    if override:
        return override
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        detail = result.stderr.strip() or "git returned no commit"
        raise RuntimeError(f"cannot determine git commit: {detail}")
    return commit


def _write_bytes_immutable(path: Path, content: bytes) -> None:
    path = Path(path)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"refusing to replace {path}: existing file has different content")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_json_immutable(path: Path, payload: dict[str, object]) -> None:
    content = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_bytes_immutable(path, content)


def write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=RESULT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})
    _write_bytes_immutable(path, buffer.getvalue().encode("utf-8"))


def generate_demos(args) -> dict[str, object]:
    cache_path = normalize_demo_cache_path(args.demo_cache)
    if cache_path.exists():
        raise FileExistsError(f"refusing to overwrite existing demonstration cache: {cache_path}")
    batch = collect_mpc_demonstrations(
        ExperimentEnvFactory(args),
        seeds=args.demo_seeds,
        episode_hours=args.episode_hours,
        teacher_policy=(greedy_shuttle_policy if args.teacher == "greedy" else None),
        mpc_objective_mode=args.mpc_objective_mode,
    )
    save_demonstrations(batch, cache_path)
    if not cache_path.exists():
        raise RuntimeError(f"demonstration writer did not create requested cache: {cache_path}")
    manifest = {
        "kind": f"{args.teacher}_demonstration_cache",
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": file_sha256(cache_path),
        "git_commit": git_commit(),
        "demo_seeds": [int(seed) for seed in args.demo_seeds],
        "environment": batch.metadata,
        "collection": {
            "episode_hours": int(args.episode_hours),
            "forecast_horizon_h": int(args.forecast_horizon_h),
            "demonstration_native_episode_hours": int(args.episode_hours + args.forecast_horizon_h + 1),
            "scenario_context_hours": 0,
            "teacher": (
                "RollingNativeMpcController"
                if args.teacher == "mpc"
                else "greedy_shuttle_policy"
            ),
            "teacher_objective_mode": (
                args.mpc_objective_mode if args.teacher == "mpc" else None
            ),
            "replan_every_h": 24 if args.teacher == "mpc" else None,
            "planning_horizon_h": 168 if args.teacher == "mpc" else None,
            "replay_validation": "exact",
        },
    }
    write_json_immutable(demo_manifest_path(cache_path), manifest)
    return manifest


def merge_demos(args) -> dict[str, object]:
    cache_path = normalize_demo_cache_path(args.demo_cache)
    manifest_path = demo_manifest_path(cache_path)
    if cache_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite merged demonstration artifacts: {cache_path}"
        )

    shard_paths = [normalize_demo_cache_path(path) for path in args.shards]
    shards = [load_demonstrations(path, None) for path in shard_paths]
    merged = merge_demonstration_shards(
        shards,
        expected_seeds=args.expected_seeds,
        episode_hours=args.episode_hours,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_path.stem}-",
        suffix=".npz",
        dir=cache_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        save_demonstrations(merged, temporary_path)
        load_demonstrations(temporary_path, merged.metadata)
        temporary_path.replace(cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    manifest = {
        "kind": "merged_mpc_demonstration_cache",
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": file_sha256(cache_path),
        "git_commit": git_commit(),
        "demo_seeds": sorted(int(seed) for seed in args.expected_seeds),
        "row_count": int(len(merged.state)),
        "episode_hours": int(args.episode_hours),
        "environment": merged.metadata,
        "shards": [
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for path in shard_paths
        ],
    }
    write_json_immutable(manifest_path, manifest)
    return manifest


def count_trainable_parameters(model) -> int:
    return int(sum(parameter.numel() for parameter in model.policy.parameters() if parameter.requires_grad))


def demonstration_accuracy(model, observations, actions, masks) -> tuple[float, list[float]]:
    predicted, _state = model.predict(observations, deterministic=True, action_masks=masks)
    predicted = np.asarray(predicted, dtype=np.int64)
    expected = np.asarray(actions, dtype=np.int64)
    if predicted.shape != expected.shape:
        raise ValueError(
            f"demonstration prediction shape mismatch: expected {expected.shape}, actual {predicted.shape}"
        )
    correct = predicted == expected
    return float(correct.all(axis=1).mean()), [float(value) for value in correct.mean(axis=0)]


def should_emit_baselines(args) -> bool:
    objective = getattr(args, "bc_objective", "current")
    is_primary_run = (
        (args.variant == "state" and objective == "current")
        or (args.variant == "future_mlp" and objective == "decision_only")
    )
    return is_primary_run and int(args.model_seed) == 0


def metric_result_row(
    metrics: EpisodeMetrics,
    *,
    policy: str,
    family: str,
    variant: str,
    stage: str,
    deterministic: bool,
    model_seed: int,
    eval_seed: int,
    episode_runtime_s: float,
    mean_inference_latency_s: float,
    trainable_parameters: int,
    demonstration_exact_match,
    demonstration_action_accuracy,
) -> dict[str, object]:
    values = {
        "vented_t": metrics.vented_t,
        "policy": policy,
        "family": family,
        "variant": variant,
        "stage": stage,
        "deterministic": bool(deterministic),
        "model_seed": int(model_seed),
        "eval_seed": int(eval_seed),
        "emitter_inventory_t": metrics.emitter_inventory_t,
        "vessel_inventory_t": metrics.vessel_inventory_t,
        "terminal_inventory_t": metrics.terminal_inventory_t,
        "in_transit_t": metrics.in_transit_t,
        "captured_t": metrics.captured_t,
        "stored_t": metrics.stored_t,
        "loss_rate": metrics.loss_rate,
        "storage_rate": metrics.storage_rate,
        "operating_cost": metrics.operating_cost,
        "vent_penalty": metrics.vent_penalty,
        "total_cost": metrics.total_cost,
        "cost_per_stored_t": metrics.cost_per_stored_t,
        "total_cost_per_stored_t": metrics.total_cost_per_stored_t,
        "throttle_hours": metrics.throttle_hours,
        "well_switch_count": metrics.well_switch_count,
        "berth_wait_vessel_hours": metrics.berth_wait_vessel_hours,
        "pressure_risk_hours": metrics.pressure_risk_hours,
        "min_pressure_margin_fraction": metrics.min_pressure_margin_fraction,
        "longest_venting_streak_hours": metrics.longest_venting_streak_hours,
        "total_reward": metrics.total_reward,
        "episode_runtime_s": episode_runtime_s,
        "mean_inference_latency_s": mean_inference_latency_s,
        "trainable_parameters": int(trainable_parameters),
        "demonstration_exact_match": demonstration_exact_match,
        "demonstration_action_accuracy": json.dumps(
            demonstration_action_accuracy, separators=(",", ":")
        ),
    }
    return {column: values[column] for column in RESULT_COLUMNS}


def _timed_episode(env, policy, seed: int):
    inference_seconds = 0.0
    calls = 0

    def timed_policy(native_env):
        nonlocal inference_seconds, calls
        start = time.perf_counter()
        action = policy(native_env)
        inference_seconds += time.perf_counter() - start
        calls += 1
        return action

    start = time.perf_counter()
    metrics = run_episode(env, timed_policy, seed=seed)
    runtime = time.perf_counter() - start
    return metrics, runtime, inference_seconds / calls if calls else 0.0


def evaluate_learned_stage(
    args,
    model,
    *,
    stage: str,
    trainable_parameters: int,
    demonstration_exact_match: float,
    demonstration_action_accuracy: list[float],
) -> list[dict[str, object]]:
    rows = []
    for deterministic in (False, True):
        for eval_seed in args.eval_seeds:
            if hasattr(model, "set_random_seed"):
                model.set_random_seed(int(eval_seed))
            env = ExperimentEnvFactory(args)()
            past_buffer = (
                PastObservationBuffer(
                    len(current_state_feature_names(env))
                    + 5 * len(env.vessel_ids)
                    + len(env.vessel_ids)
                    * (len(env.terminal_ids) + len(env.emitter_ids)),
                    [*env.vessel_action_dims, *env.well_rate_action_dims],
                    hours=PAST_HOURS,
                )
                if variant_uses_past(args.variant)
                else None
            )

            def policy(native_env):
                observation = forecast_policy_observation(
                    native_env,
                    args.variant,
                    past_observation=(
                        past_buffer.observation(
                            zero=variant_uses_zero_past(args.variant)
                        )
                        if past_buffer is not None
                        else None
                    ),
                )
                masks = flat_action_mask(
                    native_env.vessel_action_mask(), native_env.well_rate_action_mask()
                )
                action, _state = model.predict(
                    observation,
                    deterministic=deterministic,
                    action_masks=masks,
                )
                native_action = native_action_from_flat(native_env, action)
                if past_buffer is not None:
                    past_buffer.append(observation["state"], np.asarray(action))
                return native_action

            metrics, runtime, latency = _timed_episode(env, policy, int(eval_seed))
            rows.append(
                metric_result_row(
                    metrics,
                    policy=f"learned_{args.variant}",
                    family="learned",
                    variant=args.variant,
                    stage=stage,
                    deterministic=deterministic,
                    model_seed=args.model_seed,
                    eval_seed=eval_seed,
                    episode_runtime_s=runtime,
                    mean_inference_latency_s=latency,
                    trainable_parameters=trainable_parameters,
                    demonstration_exact_match=demonstration_exact_match,
                    demonstration_action_accuracy=demonstration_action_accuracy,
                )
            )
    return rows


def evaluate_reference_rows(args) -> list[dict[str, object]]:
    if not should_emit_baselines(args):
        return []
    rows = []
    for name in ("idle", "greedy", "RollingNativeMpcController"):
        for eval_seed in args.eval_seeds:
            env = ExperimentEnvFactory(args)()
            if name == "idle":
                policy = idle_policy
            elif name == "greedy":
                policy = greedy_shuttle_policy
            else:
                policy = RollingNativeMpcController(
                    env,
                    replan_every=24,
                    planning_horizon_h=168,
                    objective_mode=args.mpc_objective_mode,
                )
            metrics, runtime, latency = _timed_episode(env, policy, int(eval_seed))
            rows.append(
                metric_result_row(
                    metrics,
                    policy=name,
                    family="reference",
                    variant="",
                    stage="reference",
                    deterministic=True,
                    model_seed=args.model_seed,
                    eval_seed=eval_seed,
                    episode_runtime_s=runtime,
                    mean_inference_latency_s=latency,
                    trainable_parameters=0,
                    demonstration_exact_match="",
                    demonstration_action_accuracy=[],
                )
            )
    return rows


def _ensure_new_run_outputs(args) -> None:
    paths = [
        checkpoint_path(args, "bc"),
        results_path(args),
        run_manifest_path(args),
    ]
    if not args.bc_only:
        paths.append(checkpoint_path(args, "ppo"))
    collisions = [str(path) for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing output collision: {', '.join(collisions)}")


def train_variant(args) -> dict[str, object]:
    cache_path = normalize_demo_cache_path(args.demo_cache)
    cache_sha = file_sha256(cache_path)
    factory = ExperimentEnvFactory(args)
    metadata = factory.metadata()
    batch = load_demonstrations(cache_path, metadata)
    heldout_batch = None
    heldout_cache_sha256 = None
    if args.heldout_demo_cache:
        heldout_cache_path = normalize_demo_cache_path(args.heldout_demo_cache)
        heldout_batch = load_demonstrations(
            heldout_cache_path,
            metadata,
        )
        heldout_cache_sha256 = file_sha256(heldout_cache_path)
    observations = batch.observations(args.variant)
    native_env = make_experiment_env(args, demonstration=False)
    result = _train_loaded_batch(
        args,
        batch=batch,
        observations=observations,
        native_env=native_env,
        metadata=metadata,
        cache_sha256=cache_sha,
        heldout_batch=heldout_batch,
        heldout_cache_sha256=heldout_cache_sha256,
    )
    return result


def _train_loaded_batch(
    args,
    *,
    batch,
    observations,
    native_env,
    metadata,
    cache_sha256: str,
    heldout_batch=None,
    heldout_cache_sha256: str | None = None,
) -> dict[str, object]:
    if MaskablePPO is None:
        raise ImportError("train requires sb3-contrib")
    if args.bc_objective != "current" and not args.bc_only:
        raise ValueError("decision-only BC objectives require --bc-only")
    if args.replan_action_weight != 1.0 and args.bc_objective != "decision_only":
        raise ValueError("replan action weighting requires --bc-objective decision_only")
    if args.imitation_only and not args.bc_only:
        raise ValueError("--imitation-only requires --bc-only")
    if not 0.0 <= args.bc_label_smoothing < 1.0:
        raise ValueError("--bc-label-smoothing must be in [0, 1)")
    if args.bc_label_smoothing and args.bc_objective == "decision_balanced":
        raise ValueError(
            "label smoothing is not implemented for decision-balanced BC"
        )
    if (
        variant_uses_oracle_candidate(args.variant)
        or variant_uses_learned_plan_context(args.variant)
    ) and not args.imitation_only:
        raise ValueError("plan-context variants require --imitation-only")
    _ensure_new_run_outputs(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    policy_name, policy_kwargs = model_policy_config(args.variant)
    if variant_uses_oracle_candidate(args.variant):
        if batch.candidate_names is None or len(batch.candidate_names) != 8:
            raise ValueError("oracle candidate training requires eight candidate names")
        if heldout_batch is not None and heldout_batch.candidate_names != batch.candidate_names:
            raise ValueError("train and held-out candidate names must match")
    gym_env = ForecastGymEnv(
        native_env,
        args.variant,
        oracle_candidate_index=(0 if variant_uses_oracle_candidate(args.variant) else None),
        learned_plan_context=(
            np.zeros(8, dtype=np.float32)
            if variant_uses_learned_plan_context(args.variant)
            else None
        ),
    )
    model = MaskablePPO(
        policy_name,
        gym_env,
        policy_kwargs=policy_kwargs,
        seed=args.model_seed,
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        verbose=args.verbose,
    )
    action_dimensions = metadata.get("action_dimensions")
    if args.bc_objective != "current" and not isinstance(action_dimensions, list):
        raise ValueError("environment metadata must contain action_dimensions")
    vessel_count = len(native_env.vessel_ids)
    sampler_audit = None
    if args.bc_objective == "current":
        weights = action_dimension_weights(
            batch.actions,
            vessel_count=vessel_count,
            nonwait_weight=args.nonwait_weight,
        )
        behavior_clone(
            model,
            observations,
            batch.actions,
            masks=batch.masks,
            weights=weights,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            lr=args.bc_lr,
            label_smoothing=args.bc_label_smoothing,
            log=bool(args.verbose),
        )
    elif args.bc_objective == "decision_only":
        weights = decision_only_action_weights(
            batch.actions,
            batch.masks,
            action_dimensions,
            vessel_count,
            nonwait_weight=args.nonwait_weight,
        )
        if args.replan_action_weight != 1.0:
            weights = apply_replan_action_weight(
                weights,
                batch.hours,
                vessel_count,
                args.replan_action_weight,
            )
        behavior_clone(
            model,
            observations,
            batch.actions,
            masks=batch.masks,
            weights=weights,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            lr=args.bc_lr,
            label_smoothing=args.bc_label_smoothing,
            log=bool(args.verbose),
        )
    else:
        weights = None
        sampler_audit = behavior_clone_balanced_decisions(
            model,
            observations,
            batch.actions,
            batch.masks,
            action_dimensions,
            vessel_count,
            epochs=args.bc_epochs,
            row_batch_size=args.bc_batch_size,
            lr=args.bc_lr,
            seed=args.model_seed,
            log=bool(args.verbose),
        )
    parameter_count = count_trainable_parameters(model)
    bc_accuracy, bc_dimension_accuracy = demonstration_accuracy(
        model, observations, batch.actions, batch.masks
    )
    bc_path = checkpoint_path(args, "bc")
    model.save(str(bc_path))
    rows = []
    if not args.imitation_only:
        rows = evaluate_learned_stage(
            args,
            model,
            stage="bc",
            trainable_parameters=parameter_count,
            demonstration_exact_match=bc_accuracy,
            demonstration_action_accuracy=bc_dimension_accuracy,
        )

    ppo_accuracy = None
    ppo_dimension_accuracy = None
    ppo_path = None
    if not args.bc_only and args.timesteps > 0:
        model.set_random_seed(int(args.model_seed))
        callback = make_kickstart_callback(
            observations,
            batch.actions,
            batch.masks,
            weights,
            total_timesteps=args.timesteps,
            coef0=args.kickstart_coef,
            n_batches=4,
            batch_size=args.bc_batch_size,
            lr=args.bc_lr,
            verbose=args.verbose,
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=args.progress_bar,
        )
    if not args.bc_only:
        ppo_accuracy, ppo_dimension_accuracy = demonstration_accuracy(
            model, observations, batch.actions, batch.masks
        )
        ppo_path = checkpoint_path(args, "ppo")
        model.save(str(ppo_path))
        rows.extend(
            evaluate_learned_stage(
                args,
                model,
                stage="ppo",
                trainable_parameters=parameter_count,
                demonstration_exact_match=ppo_accuracy,
                demonstration_action_accuracy=ppo_dimension_accuracy,
            )
        )
    if not args.imitation_only:
        rows.extend(evaluate_reference_rows(args))
    write_results_csv(results_path(args), rows)

    demo_seeds = sorted({int(seed) for seed in np.asarray(batch.seeds).tolist()})
    manifest = {
        "kind": "forecast_encoder_training_run",
        "variant": args.variant,
        "model_seed": int(args.model_seed),
        "demo_seeds": demo_seeds,
        "eval_seeds": [int(seed) for seed in args.eval_seeds],
        "demo_cache_path": str(normalize_demo_cache_path(args.demo_cache).resolve()),
        "demo_cache_sha256": cache_sha256,
        "heldout_demo_cache_path": (
            str(normalize_demo_cache_path(args.heldout_demo_cache).resolve())
            if args.heldout_demo_cache
            else None
        ),
        "heldout_demo_cache_sha256": heldout_cache_sha256,
        "git_commit": commit,
        "checkpoints": {
            "bc": str(bc_path),
            "ppo": str(ppo_path) if ppo_path is not None else None,
        },
        "results_csv": str(results_path(args)),
        "environment": metadata,
        "policy": policy_manifest(args.variant),
        "bc": {
            "objective": args.bc_objective,
            "epochs": int(args.bc_epochs),
            "batch_size": int(args.bc_batch_size),
            "learning_rate": float(args.bc_lr),
            "label_smoothing": float(args.bc_label_smoothing),
            "nonwait_action_dimension_weight": (
                1.0 if args.bc_objective == "decision_balanced"
                else float(args.nonwait_weight)
            ),
            "forced_vessel_weight": 1.0 if args.bc_objective == "current" else 0.0,
            "replan_action_weight": float(args.replan_action_weight),
            "replan_weight_scope": "non_forced_vessel_dimensions_at_phase_zero",
            "decision_sampling": (
                "balanced_wait_dispatch_pairs"
                if args.bc_objective == "decision_balanced"
                else "uniform_rows"
            ),
            "sampler_audit": sampler_audit,
            "imitation_only": bool(args.imitation_only),
        },
        "ppo": {
            "timesteps": int(args.timesteps),
            "gamma": float(args.gamma),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "learn_skipped": bool(args.timesteps == 0 or args.bc_only),
            "explicitly_skipped": bool(args.bc_only),
        },
        "kickstart": {
            "coefficient": float(args.kickstart_coef),
            "decay": "linear",
            "n_batches": 4,
            "batch_size": int(args.bc_batch_size),
            "learning_rate": float(args.bc_lr),
        },
        "device_request": args.device,
        "verbose": int(args.verbose),
        "progress_bar": bool(args.progress_bar),
        "trainable_parameters": parameter_count,
        "demonstration_accuracy": {
            "bc_exact_match": bc_accuracy,
            "bc_action_dimensions": bc_dimension_accuracy,
            "ppo_exact_match": ppo_accuracy,
            "ppo_action_dimensions": ppo_dimension_accuracy,
        },
    }
    write_json_immutable(run_manifest_path(args), manifest)
    return manifest


def _bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid deterministic value: {value!r}")


MODE_PAIRS = {"state_mode": "state", "tcn_mode": "tcn"}


def _pairing_maps(rows):
    learned = [row for row in rows if row.get("family") == "learned"]
    maps = {variant: {} for variant in FORMAL_VARIANTS}
    for row in learned:
        variant = row.get("variant")
        if variant not in maps:
            continue
        key = (
            row["stage"],
            _bool_value(row["deterministic"]),
            int(row["model_seed"]),
            int(row["eval_seed"]),
        )
        if key in maps[variant]:
            raise ValueError(f"duplicate pairing key for {variant}: {key}")
        maps[variant][key] = float(row["vented_t"])
    for mode_variant, baseline_variant in MODE_PAIRS.items():
        baseline_keys = set(maps[baseline_variant])
        variant_keys = set(maps[mode_variant])
        if not baseline_keys:
            raise ValueError(f"missing paired keys: no {baseline_variant} rows")
        if variant_keys != baseline_keys:
            missing = sorted(baseline_keys - variant_keys)
            extra = sorted(variant_keys - baseline_keys)
            raise ValueError(
                f"missing paired keys for {mode_variant}: "
                f"missing={missing}, unmatched={extra}"
            )
    return maps


def _numeric_values(rows, field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field, "")
        if value not in {"", None}:
            values.append(float(value))
    return values


_T_975 = {
    1: 12.7062047364,
    2: 4.30265272975,
    3: 3.18244630528,
    4: 2.77644510520,
    5: 2.57058183564,
    6: 2.44691184879,
    7: 2.36462425101,
    8: 2.30600413520,
    9: 2.26215716285,
    10: 2.22813885196,
    11: 2.20098516008,
    12: 2.17881282966,
    13: 2.16036865646,
    14: 2.14478668792,
    15: 2.13144954556,
    16: 2.11990529922,
    17: 2.10981557783,
    18: 2.10092204024,
    19: 2.09302405441,
    20: 2.08596344727,
    21: 2.07961384473,
    22: 2.07387306790,
    23: 2.06865761042,
    24: 2.06389856163,
    25: 2.05953855275,
    26: 2.05552943864,
    27: 2.05183051648,
    28: 2.04840714180,
    29: 2.04522964213,
    30: 2.04227245630,
}


def _model_uncertainty(values: list[float]) -> tuple[object, object]:
    if len(values) < 2:
        return "", ""
    sample_sd = statistics.stdev(values)
    critical = _T_975.get(len(values) - 1, 1.95996398454)
    return sample_sd, critical * sample_sd / np.sqrt(len(values))


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_paired_report(rows: list[dict[str, object]], out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairing = _pairing_maps(rows)
    episode_groups = {}
    for row in rows:
        key = (
            row.get("policy", ""),
            row.get("family", ""),
            row.get("variant", ""),
            row.get("stage", ""),
            _bool_value(row.get("deterministic", True)),
            int(row.get("model_seed", 0)),
        )
        episode_groups.setdefault(key, []).append(row)

    episode_rows = []
    for (
        policy,
        family,
        variant,
        stage,
        deterministic,
        model_seed,
    ), group_rows in sorted(episode_groups.items()):
        summary = {
            "policy": policy,
            "family": family,
            "variant": variant,
            "stage": stage,
            "deterministic": deterministic,
            "model_seed": model_seed,
            "eval_episodes": len(group_rows),
        }
        for metric in SUMMARY_METRICS:
            values = _numeric_values(group_rows, metric)
            summary[f"{metric}_mean"] = statistics.fmean(values) if values else ""
        episode_rows.append(summary)

    model_groups = {}
    for row in episode_rows:
        key = tuple(
            row[field]
            for field in ("policy", "family", "variant", "stage", "deterministic")
        )
        model_groups.setdefault(key, []).append(row)

    summary_rows = []
    for (policy, family, variant, stage, deterministic), model_rows in sorted(
        model_groups.items()
    ):
        summary = {
            "policy": policy,
            "family": family,
            "variant": variant,
            "stage": stage,
            "deterministic": deterministic,
            "model_seeds": len(model_rows),
            "eval_episodes": sum(int(row["eval_episodes"]) for row in model_rows),
        }
        for metric in SUMMARY_METRICS:
            values = _numeric_values(model_rows, f"{metric}_mean")
            model_sd, interval = _model_uncertainty(values)
            summary[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            summary[f"{metric}_model_sd"] = model_sd
            summary[f"{metric}_ci95_half_width"] = interval
        summary.update(
            {
                "paired_baseline_variant": "",
                "paired_model_seeds": "",
                "paired_vented_delta_mean": "",
                "paired_vented_delta_model_sd": "",
                "paired_vented_delta_ci95_half_width": "",
            }
        )
        if family == "learned" and variant in MODE_PAIRS:
            baseline = MODE_PAIRS[variant]
            matching = [
                key
                for key in pairing[variant]
                if key[0] == stage and key[1] == deterministic
            ]
            deltas_by_model = {}
            for key in matching:
                deltas_by_model.setdefault(key[2], []).append(
                    pairing[variant][key] - pairing[baseline][key]
                )
            model_deltas = [
                statistics.fmean(values)
                for _model_seed, values in sorted(deltas_by_model.items())
            ]
            delta_sd, delta_interval = _model_uncertainty(model_deltas)
            summary.update(
                {
                    "paired_baseline_variant": baseline,
                    "paired_model_seeds": len(model_deltas),
                    "paired_vented_delta_mean": statistics.fmean(model_deltas),
                    "paired_vented_delta_model_sd": delta_sd,
                    "paired_vented_delta_ci95_half_width": delta_interval,
                }
            )
        summary_rows.append(summary)

    episode_path = out_dir / "forecast_encoder_episode_summary.csv"
    _write_dict_rows(episode_path, episode_rows)
    summary_path = out_dir / "forecast_encoder_summary.csv"
    _write_dict_rows(summary_path, summary_rows)

    markdown_path = out_dir / "forecast_encoder_summary.md"
    lines = [
        "# Forecast Encoder RL Comparison",
        "",
        "Lower venting is the primary outcome. End inventory and operating cost are secondary diagnostics.",
        "",
        "Means and 95% intervals use evaluation-seed means within each model seed, then sample uncertainty across model seeds.",
        "Paired deltas are mode variant minus its matched base encoder on exact stage, determinism, model-seed, and evaluation-seed keys.",
        "",
        "| policy | stage | deterministic | model seeds | vented t mean | paired mode delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for summary in summary_rows:
        lines.append(
            f"| {summary['policy']} | {summary['stage']} | {summary['deterministic']} | "
            f"{summary['model_seeds']} | {summary['vented_t_mean']} | "
            f"{summary['paired_vented_delta_mean']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path, markdown_path


def report(args):
    rows = []
    result_paths = sorted(Path(args.out_dir).glob("results_*.csv"))
    if not result_paths:
        raise FileNotFoundError(f"no results_*.csv files found in {args.out_dir}")
    _validate_report_manifests(result_paths)
    for path in result_paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return write_paired_report(rows, Path(args.out_dir))


def _validate_report_manifests(result_paths: list[Path]) -> None:
    required_equal = (
        "kind",
        "demo_cache_path",
        "demo_cache_sha256",
        "git_commit",
        "environment",
        "demo_seeds",
        "eval_seeds",
        "bc",
        "ppo",
        "kickstart",
        "device_request",
    )
    manifests = []
    for result_path in result_paths:
        run_name = result_path.name.replace("results_", "run_", 1)
        manifest_path = result_path.with_name(
            f"{Path(run_name).stem}.manifest.json"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"run manifest is missing for {result_path.name}: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid run manifest {manifest_path}: {error}") from error
        if not isinstance(manifest, dict):
            raise ValueError(f"invalid run manifest {manifest_path}: expected a JSON object")
        missing = [field for field in required_equal if field not in manifest]
        if missing:
            raise ValueError(f"run manifest {manifest_path} is missing fields: {missing}")
        manifests.append((manifest_path, manifest))

    reference_path, reference = manifests[0]
    for manifest_path, manifest in manifests[1:]:
        for field in required_equal:
            if manifest[field] != reference[field]:
                raise ValueError(
                    f"run manifest mismatch for {field}: "
                    f"{reference_path} != {manifest_path}"
                )


def main(argv=None):
    args = parse_args(argv)
    if args.command == "generate-demos":
        return generate_demos(args)
    if args.command == "merge-demos":
        return merge_demos(args)
    if args.command == "train":
        return train_variant(args)
    if args.command == "report":
        return report(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
