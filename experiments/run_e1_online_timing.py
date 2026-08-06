"""Benchmark E1 online episode wall time on one controlled CPU platform."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments import evaluate_iterative_action_q as iterative_q_eval
from experiments import smoke_test_paper_controllers as paper_controllers
from sim.control.event_based.residual_rl_v4 import evaluate_ppo as event_ppo_eval
from sim.control.event_based.rl import evaluate_high_level_ppo as high_ppo_eval


METHODS = (
    "fixed_assignment",
    "greedy",
    "ppo_hourly",
    "ppo_high_level",
    "ppo_event_residual",
    "iterative_action_q_g60_p4",
)
LEARNED_METHODS = set(METHODS[2:])
FORMAL_SEEDS = tuple(range(9_000_031, 9_000_061))
POLICY_WINDOWS = (
    "108-155,156-203,204-251,252-299,300-347,348-395,"
    "396-443,444-491,492-539,540-587,588-635,636-680"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
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
    if args.method in LEARNED_METHODS and args.model_seed is None:
        parser.error("--model-seed is required for learned methods")
    if args.method not in LEARNED_METHODS and args.model_seed is not None:
        parser.error("--model-seed only applies to learned methods")
    if not args.seeds:
        parser.error("--seeds must not be empty")
    return args


def _model_dir(method: str, model_seed: int) -> Path:
    return (
        REPO_ROOT
        / "experiments_results"
        / "E1"
        / "models"
        / {
            "ppo_hourly": "ppo_hourly",
            "ppo_high_level": "ppo_high_level",
            "ppo_event_residual": "ppo_event_residual",
            "iterative_action_q_g60_p4": "iterative_q",
        }[method]
        / (
            f"g60_p4_model_seed_{model_seed}"
            if method == "iterative_action_q_g60_p4"
            else f"model_seed_{model_seed}"
        )
    )


def _config_path(method: str, model_seed: int) -> Path:
    result_root = REPO_ROOT / "experiments_results" / "E1" / "algorithms"
    mapping = {
        "ppo_hourly": (
            "formal_ppo_hourly_seeds_9000031-9000060_run02",
            "ppo_hourly",
        ),
        "ppo_high_level": (
            "formal_ppo_high_level_seeds_9000031-9000060_run01",
            "ppo_high_level",
        ),
        "ppo_event_residual": (
            "formal_ppo_event_residual_seeds_9000031-9000060_run01",
            "ppo_event_residual",
        ),
    }
    root_name, algorithm_name = mapping[method]
    return (
        result_root
        / root_name
        / algorithm_name
        / f"model_seed_{model_seed}"
        / "config.json"
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _normalize(
    method: str,
    model_seed: int | None,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "algorithm": method,
                "model_seed": "" if model_seed is None else int(model_seed),
                "test_seed": int(row["seed"]),
                "episode_hours": 720,
                "decision_count": int(
                    row.get(
                        "decisions",
                        row.get(
                            "event_count",
                            row.get("controller_decision_calls", 0),
                        ),
                    )
                ),
                "episode_wall_time_s": float(row["wall_clock_seconds"]),
                "total_cost_eur": float(
                    row.get("total_cost_eur", row.get("total_cost"))
                ),
                "vented_t": float(row["vented_t"]),
                "stored_t": float(row["stored_t"]),
            }
        )
    return normalized


def _run_baseline(args: argparse.Namespace) -> list[dict[str, object]]:
    controller_args = SimpleNamespace(
        online_episode_hours=720,
        forecast_context_hours=168,
    )
    return [
        paper_controllers._run_simple_controller(
            SimpleNamespace(
                **vars(controller_args),
                seed=int(seed),
            ),
            args.method,
        )
        for seed in args.seeds
    ]


def _run_hourly(args: argparse.Namespace) -> list[dict[str, object]]:
    from sim.control.hourly_ppo.evaluate_hourly_ppo import evaluate_checkpoint

    model_seed = int(args.model_seed)
    evaluation_dir = args.output_dir / "_hourly_evaluation"
    output = evaluate_checkpoint(
        run_dir=_config_path(args.method, model_seed).parent,
        model_name=str(
            (
                _model_dir(args.method, model_seed)
                / "ppo_hourly_best_validation.zip"
            ).resolve()
        ),
        seeds=tuple(int(seed) for seed in args.seeds),
        out_dir=evaluation_dir,
    )
    with (output / "evaluation.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def _prepared_ppo_run(
    args: argparse.Namespace,
    *,
    checkpoint_name: str,
) -> Path:
    model_seed = int(args.model_seed)
    run_dir = args.output_dir / "_ppo_run"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(_config_path(args.method, model_seed), run_dir / "config.json")
    shutil.copy2(
        _model_dir(args.method, model_seed) / f"{checkpoint_name}.zip",
        run_dir / f"{checkpoint_name}.zip",
    )
    return run_dir


def _run_high_level(args: argparse.Namespace) -> list[dict[str, object]]:
    run_dir = _prepared_ppo_run(
        args,
        checkpoint_name="ppo_high_level_best_validation",
    )
    records, _summary = high_ppo_eval.evaluate_run(
        run_dir,
        seeds=tuple(int(seed) for seed in args.seeds),
        executor="rule",
        model_choice="best",
    )
    return records


def _run_event_residual(args: argparse.Namespace) -> list[dict[str, object]]:
    run_dir = _prepared_ppo_run(
        args,
        checkpoint_name="event_residual_e1_best_validation",
    )
    records, _summary = event_ppo_eval.evaluate_run(
        run_dir,
        seeds=tuple(int(seed) for seed in args.seeds),
        model_choice="best",
        hard_scenario_probability=0.0,
    )
    return records


def _run_iterative_q(args: argparse.Namespace) -> list[dict[str, object]]:
    model_seed = int(args.model_seed)
    checkpoint = (
        args.iterative_q_model_root
        / f"g60_p4_model_seed_{model_seed}"
        / "iterative_action_q.pt"
    )
    eval_args = iterative_q_eval.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(args.output_dir / "_unused_q_output"),
            "--eval-seeds",
            *(str(seed) for seed in args.seeds),
            "--episode-hours",
            "720",
            "--reward-scale",
            "0.00001",
            "--gates",
            f"timing:4:0.40:12:{POLICY_WINDOWS}",
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
    device = iterative_q_eval.torch.device("cpu")
    model, metadata = iterative_q_eval._load_model(eval_args, device)
    variant = str(metadata["observation_variant"])
    baselines = {
        int(seed): iterative_q_eval.greedy_metrics(
            eval_args,
            variant,
            int(seed),
        )
        for seed in args.seeds
    }
    return iterative_q_eval.evaluate_gate(
        eval_args,
        model,
        metadata,
        eval_args.gates[0],
        baselines,
        device,
        event_env_factory=lambda: iterative_q_eval.make_event_env(
            eval_args,
            variant,
            greedy_control_variate=False,
        ),
    )


def run(args: argparse.Namespace) -> Path:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runners = {
        "fixed_assignment": _run_baseline,
        "greedy": _run_baseline,
        "ppo_hourly": _run_hourly,
        "ppo_high_level": _run_high_level,
        "ppo_event_residual": _run_event_residual,
        "iterative_action_q_g60_p4": _run_iterative_q,
    }
    raw_rows = runners[args.method](args)
    rows = _normalize(args.method, args.model_seed, raw_rows)
    output_csv = args.output_dir / "timing_records.csv"
    _write_rows(output_csv, rows)
    metadata = {
        "algorithm": args.method,
        "model_seed": args.model_seed,
        "test_seeds": [int(seed) for seed in args.seeds],
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "device": "cpu",
        "metric": (
            "wall time from scenario reset through the completed 720 h "
            "closed-loop rollout and terminal cleanup calculation"
        ),
        "records": len(rows),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_csv


def main() -> None:
    output = run(parse_args())
    print(f"E1_TIMING_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
