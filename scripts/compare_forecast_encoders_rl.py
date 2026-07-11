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
import time

import numpy as np

from sim.control.baselines import greedy_shuttle_policy, idle_policy
from sim.control.demonstrations import (
    collect_mpc_demonstrations,
    load_demonstrations,
    save_demonstrations,
)
from sim.control.imitation import (
    action_dimension_weights,
    behavior_clone,
    make_kickstart_callback,
)
from sim.control.native_mpc import RollingNativeMpcController
from sim.environment.forecast import current_state_feature_names, forecast_channel_names
from sim.environment.forecast_encoder import TCNForecastExtractor
from sim.environment.forecast_gym import ForecastGymEnv, make_forecast_ppo_policy
from sim.metrics import EpisodeMetrics, run_episode
from sim.train import make_native_env

try:
    from sb3_contrib import MaskablePPO
except ImportError:  # pragma: no cover - exercised only in incomplete installations
    MaskablePPO = None


FORMAL_SCENARIO = "northern_lights_phase1_3vessels"
FORECAST_HORIZON_H = 168
FORECAST_CHANNELS = 9

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
        reward_mode="vent_first",
        vent_first_vent_eur_per_t=10_000.0,
        overflow_risk_eur_per_t=100.0,
        overflow_risk_lookahead_h=24.0,
        operating_cost_weight=1.0,
        enforce_full_load_dispatch=False,
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
    _add_locked_protocol_defaults(demos)

    train = commands.add_parser("train")
    train.add_argument("--variant", choices=("state", "flat", "tcn"), required=True)
    train.add_argument("--demo-cache", required=True)
    train.add_argument("--timesteps", type=_nonnegative_int, default=100_000)
    train.add_argument("--bc-epochs", type=int, default=20)
    train.add_argument("--n-steps", type=int, default=512)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--bc-batch-size", type=int, default=256)
    train.add_argument("--model-seed", type=int, default=0)
    train.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    train.add_argument("--device", default="auto")
    train.add_argument("--verbose", type=int, default=1)
    train.add_argument("--progress-bar", action="store_true")
    train.add_argument("--out-dir", default="output/rl_forecast")
    train.set_defaults(
        gamma=0.999,
        learning_rate=3e-4,
        bc_lr=1e-3,
        nonwait_weight=10.0,
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
        reward_mode="vent_first",
        vent_first_vent_eur_per_t=10_000.0,
        overflow_risk_eur_per_t=100.0,
        overflow_risk_lookahead_h=24.0,
        operating_cost_weight=1.0,
        enforce_full_load_dispatch=False,
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
            "state_feature_names": state_names,
            "state_size": len(state_names),
            "action_dimensions": [*env.vessel_action_dims, *env.well_rate_action_dims],
            "vessel_action_dimensions": list(env.vessel_action_dims),
            "well_rate_action_dimensions": list(env.well_rate_action_dims),
            "weather_mode": "block",
            "weather_observation_layout": env.config.weather_observation_layout,
            "include_weather_obs": False,
            "reward": {
                "mode": "vent_first",
                "vent_eur_per_t": 10_000.0,
                "overflow_risk_eur_per_t": 100.0,
                "overflow_risk_lookahead_h": 24.0,
                "operating_cost_weight": 1.0,
            },
            "partial_load_dispatch": True,
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
    if variant == "tcn":
        return "MultiInputPolicy", {
            "features_extractor_class": TCNForecastExtractor,
            "features_extractor_kwargs": {
                "state_features": 64,
                "forecast_features": 64,
            },
        }
    if variant in {"state", "flat"}:
        return "MlpPolicy", {}
    raise ValueError(f"unknown variant: {variant}")


def checkpoint_path(args, stage: str) -> Path:
    if stage not in {"bc", "ppo"}:
        raise ValueError(f"unknown checkpoint stage: {stage}")
    return Path(args.out_dir) / f"{stage}_{args.variant}_seed{args.model_seed}.zip"


def results_path(args) -> Path:
    return Path(args.out_dir) / f"results_{args.variant}_seed{args.model_seed}.csv"


def run_manifest_path(args) -> Path:
    return Path(args.out_dir) / f"run_{args.variant}_seed{args.model_seed}.manifest.json"


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
    )
    save_demonstrations(batch, cache_path)
    if not cache_path.exists():
        raise RuntimeError(f"demonstration writer did not create requested cache: {cache_path}")
    manifest = {
        "kind": "mpc_demonstration_cache",
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
            "teacher": "RollingNativeMpcController",
            "replan_every_h": 24,
            "planning_horizon_h": 168,
            "replay_validation": "exact",
        },
    }
    write_json_immutable(demo_manifest_path(cache_path), manifest)
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
    return args.variant == "state" and int(args.model_seed) == 0


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
            policy = make_forecast_ppo_policy(
                model,
                args.variant,
                deterministic=deterministic,
            )
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
        checkpoint_path(args, "ppo"),
        results_path(args),
        run_manifest_path(args),
    ]
    collisions = [str(path) for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing output collision: {', '.join(collisions)}")


def train_variant(args) -> dict[str, object]:
    cache_path = normalize_demo_cache_path(args.demo_cache)
    cache_sha = file_sha256(cache_path)
    factory = ExperimentEnvFactory(args)
    metadata = factory.metadata()
    batch = load_demonstrations(cache_path, metadata)
    observations = batch.observations(args.variant)
    native_env = make_experiment_env(args, demonstration=False)
    result = _train_loaded_batch(
        args,
        batch=batch,
        observations=observations,
        native_env=native_env,
        metadata=metadata,
        cache_sha256=cache_sha,
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
) -> dict[str, object]:
    if MaskablePPO is None:
        raise ImportError("train requires sb3-contrib")
    _ensure_new_run_outputs(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    policy_name, policy_kwargs = model_policy_config(args.variant)
    gym_env = ForecastGymEnv(native_env, args.variant)
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
    weights = action_dimension_weights(
        batch.actions,
        vessel_count=len(native_env.vessel_ids),
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
        log=bool(args.verbose),
    )
    parameter_count = count_trainable_parameters(model)
    bc_accuracy, bc_dimension_accuracy = demonstration_accuracy(
        model, observations, batch.actions, batch.masks
    )
    bc_path = checkpoint_path(args, "bc")
    model.save(str(bc_path))
    rows = evaluate_learned_stage(
        args,
        model,
        stage="bc",
        trainable_parameters=parameter_count,
        demonstration_exact_match=bc_accuracy,
        demonstration_action_accuracy=bc_dimension_accuracy,
    )

    if args.timesteps > 0:
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
        "git_commit": commit,
        "checkpoints": {"bc": str(bc_path), "ppo": str(ppo_path)},
        "results_csv": str(results_path(args)),
        "environment": metadata,
        "policy": (
            {
                "name": "MultiInputPolicy",
                "features_extractor": "TCNForecastExtractor",
                "state_features": 64,
                "forecast_features": 64,
            }
            if args.variant == "tcn"
            else {"name": "MlpPolicy", "features_extractor": None}
        ),
        "bc": {
            "epochs": int(args.bc_epochs),
            "batch_size": int(args.bc_batch_size),
            "learning_rate": float(args.bc_lr),
            "nonwait_action_dimension_weight": float(args.nonwait_weight),
        },
        "ppo": {
            "timesteps": int(args.timesteps),
            "gamma": float(args.gamma),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "learn_skipped": bool(args.timesteps == 0),
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


def _pairing_maps(rows):
    learned = [row for row in rows if row.get("family") == "learned"]
    maps = {variant: {} for variant in ("state", "flat", "tcn")}
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
    state_keys = set(maps["state"])
    if not state_keys:
        raise ValueError("missing paired keys: no state rows")
    for variant in ("flat", "tcn"):
        variant_keys = set(maps[variant])
        if variant_keys != state_keys:
            missing = sorted(state_keys - variant_keys)
            extra = sorted(variant_keys - state_keys)
            raise ValueError(
                f"missing paired keys for {variant}: missing={missing}, unmatched={extra}"
            )
    return maps


def _numeric_values(rows, field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field, "")
        if value not in {"", None}:
            values.append(float(value))
    return values


def write_paired_report(rows: list[dict[str, object]], out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairing = _pairing_maps(rows)
    grouped = {}
    for row in rows:
        key = (
            row.get("policy", ""),
            row.get("family", ""),
            row.get("variant", ""),
            row.get("stage", ""),
            _bool_value(row.get("deterministic", True)),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (policy, family, variant, stage, deterministic), group_rows in sorted(grouped.items()):
        summary = {
            "policy": policy,
            "family": family,
            "variant": variant,
            "stage": stage,
            "deterministic": deterministic,
            "episodes": len(group_rows),
        }
        vented_values = _numeric_values(group_rows, "vented_t")
        summary["vented_t_mean"] = statistics.fmean(vented_values) if vented_values else ""
        summary["vented_t_std"] = (
            statistics.pstdev(vented_values)
            if len(vented_values) > 1
            else (0.0 if vented_values else "")
        )
        summary["paired_episodes"] = ""
        summary["paired_vented_delta_vs_state_mean"] = ""
        summary["paired_vented_delta_vs_state_std"] = ""
        if family == "learned":
            keys = [
                key for key in pairing[variant]
                if key[0] == stage and key[1] == deterministic
            ]
            deltas = [pairing[variant][key] - pairing["state"][key] for key in keys]
            summary["paired_episodes"] = len(deltas)
            summary["paired_vented_delta_vs_state_mean"] = statistics.fmean(deltas)
            summary["paired_vented_delta_vs_state_std"] = (
                statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
            )
        for metric in SUMMARY_METRICS:
            if metric == "vented_t":
                continue
            values = _numeric_values(group_rows, metric)
            summary[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            summary[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else "")
        summary_rows.append(summary)

    summary_path = out_dir / "forecast_encoder_summary.csv"
    fields = list(summary_rows[0]) if summary_rows else []
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(summary_rows)

    markdown_path = out_dir / "forecast_encoder_summary.md"
    lines = [
        "# Forecast Encoder RL Comparison",
        "",
        "Lower venting is the primary outcome. End inventory and operating cost are secondary diagnostics.",
        "",
        "Paired deltas are variant minus state on exact stage, determinism, model-seed, and evaluation-seed keys.",
        "",
        "| policy | stage | deterministic | episodes | vented t mean | paired delta vs state |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for summary in summary_rows:
        lines.append(
            f"| {summary['policy']} | {summary['stage']} | {summary['deterministic']} | "
            f"{summary['episodes']} | {summary['vented_t_mean']} | "
            f"{summary['paired_vented_delta_vs_state_mean']} |"
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
    if args.command == "train":
        return train_variant(args)
    if args.command == "report":
        return report(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
