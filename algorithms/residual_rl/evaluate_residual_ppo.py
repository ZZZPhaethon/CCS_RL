"""Evaluate validation-best or final residual PPO on fixed seeds.

在固定 seeds 上评估验证集最优或最终残差 PPO。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from algorithms.rl.reward import HighLevelRewardConfig

from .evaluation import evaluate_seeds, validation_metrics
from .factory import make_residual_native_env


def evaluate_run(
    run_dir: Path,
    *,
    seeds: tuple[int, ...],
    model_choice: str = "best",
    hard_scenario_probability: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate one residual run and persist seed-level results.

    评估一次残差训练，并保存逐 seed 结果。
    """
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    if model_choice not in {"best", "final"}:
        raise ValueError("model_choice must be 'best' or 'final'.")
    model_name = (
        "ppo_residual_best_validation"
        if model_choice == "best"
        else "ppo_residual_final"
    )
    model_path = run_dir / model_name
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Residual PPO model not found: {model_path}.zip")
    probability = (
        float(config["hard_scenario_probability"])
        if hard_scenario_probability is None
        else float(hard_scenario_probability)
    )
    reward = HighLevelRewardConfig(**config["high_level_reward"])
    env = make_residual_native_env(
        scenario=str(config["scenario"]),
        episode_hours=int(config["episode_hours"]),
        forecast_context_hours=int(config["forecast_context_hours"]),
        decision_interval_h=float(config["decision_interval_h"]),
        event_triggered=bool(config["event_triggered"]),
        weather_mode=str(config["weather_mode"]),
        hard_scenario_probability=probability,
        reward=reward,
    )
    model = PPO.load(model_path, device="cpu")
    records = evaluate_seeds(model, env, seeds)
    summary = validation_metrics(
        records,
        cvar_tail_fraction=float(config["cvar_tail_fraction"]),
        tail_vent_penalty_eur_per_t=float(
            config["tail_vent_penalty_eur_per_t"]
        ),
    )
    numeric_keys = (
        "decisions",
        "mean_decision_interval_h",
        "interventions_applied",
        "intervention_rate",
        "episode_reward",
        "captured_t",
        "stored_t",
        "vented_t",
        "storage_rate",
        "operating_cost_eur",
        "total_cost_eur",
        "unit_total_cost_eur_per_t",
        "hard_violations",
        "wall_clock_seconds",
    )
    summary.update(
        {
            f"{key}_mean": sum(float(row[key]) for row in records)
            / len(records)
            for key in numeric_keys
        }
    )
    seed_label = "-".join(str(seed) for seed in seeds)
    output_dir = run_dir / "evaluation" / (
        f"{model_choice}__hardprob{probability:g}__seeds_{seed_label}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "model_choice": model_choice,
                "model_path": str(model_path),
                "hard_scenario_probability": probability,
                "seeds": list(seeds),
                "summary": summary,
                "per_seed": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        scalar_columns = [
            key
            for key, value in records[0].items()
            if not isinstance(value, dict)
        ]
        writer = csv.DictWriter(stream, fieldnames=scalar_columns)
        writer.writeheader()
        writer.writerows(
            {
                key: value
                for key, value in record.items()
                if key in scalar_columns
            }
            for record in records
        )
    return records, summary


def main() -> None:
    """Evaluate one residual training directory from the command line.

    从命令行评估一个残差训练目录。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
    )
    parser.add_argument(
        "--model",
        choices=("best", "final"),
        default="best",
    )
    parser.add_argument(
        "--hard-scenario-probability",
        type=float,
        default=None,
        help=(
            "Override the training mixture; use 0 for the original standard "
            "benchmark. / 覆盖训练难度混合；原始标准基准请使用 0。"
        ),
    )
    args = parser.parse_args()
    records, summary = evaluate_run(
        args.run_dir,
        seeds=tuple(args.seeds),
        model_choice=args.model,
        hard_scenario_probability=args.hard_scenario_probability,
    )
    for record in records:
        print(
            "Residual PPO evaluation | "
            f"seed={record['seed']} | "
            f"stored={record['stored_t']:,.1f} t | "
            f"vented={record['vented_t']:,.1f} t | "
            f"total_cost=EUR {record['total_cost_eur']:,.0f} | "
            f"intervention_rate={100.0 * record['intervention_rate']:.1f}%",
            flush=True,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
