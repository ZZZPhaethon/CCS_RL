"""Evaluate validation-best or final masked residual PPO.

评估验证集最优或最终的掩码残差 PPO。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO

from sim.control.event_based.rl.reward import HighLevelRewardConfig

from .evaluation import evaluate_seeds, validation_metrics
from .factory import make_masked_residual_native_env


def evaluate_run(
    run_dir: Path,
    *,
    seeds: tuple[int, ...],
    model_choice: str = "best",
    hard_scenario_probability: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate one v2 run and save per-seed diagnostics.

    评估一次 v2 训练，并保存逐 seed 诊断。
    """
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    if model_choice not in {"best", "final"}:
        raise ValueError("model_choice must be 'best' or 'final'.")
    model_stem = (
        "maskable_residual_v2_best_validation"
        if model_choice == "best"
        else "maskable_residual_v2_final"
    )
    model_path = run_dir / model_stem
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Model not found: {model_path}.zip")
    probability = (
        float(config["hard_scenario_probability"])
        if hard_scenario_probability is None
        else float(hard_scenario_probability)
    )
    env = make_masked_residual_native_env(
        scenario=str(config["scenario"]),
        episode_hours=int(config["episode_hours"]),
        forecast_context_hours=int(config["forecast_context_hours"]),
        decision_interval_h=float(config["decision_interval_h"]),
        event_triggered=bool(config["event_triggered"]),
        weather_mode=str(config["weather_mode"]),
        hard_scenario_probability=probability,
        reward=HighLevelRewardConfig(**config["high_level_reward"]),
    )
    model = MaskablePPO.load(model_path, device="cpu")
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
        "selected_interventions",
        "selected_intervention_rate",
        "feasible_intervention_decisions",
        "changed_decisions",
        "effective_intervention_rate",
        "changed_native_steps",
        "local_avoided_vent_t",
        "local_incremental_stored_t",
        "local_total_cost_saving_eur",
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

    label = "-".join(str(seed) for seed in seeds)
    output_dir = run_dir / "evaluation" / (
        f"{model_choice}__hardprob{probability:g}__seeds_{label}"
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
    scalar_columns = [
        key
        for key, value in records[0].items()
        if not isinstance(value, dict)
    ]
    with (output_dir / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
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
    """Evaluate a v2 training directory from the command line.

    从命令行评估一个 v2 训练目录。
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
    )
    args = parser.parse_args()
    records, summary = evaluate_run(
        args.run_dir,
        seeds=tuple(args.seeds),
        model_choice=args.model,
        hard_scenario_probability=args.hard_scenario_probability,
    )
    for row in records:
        print(
            "Masked residual v2 | "
            f"seed={row['seed']} | stored={row['stored_t']:,.1f} t | "
            f"vented={row['vented_t']:,.1f} t | "
            f"cost=EUR {row['total_cost_eur']:,.0f} | "
            f"selected={100.0 * row['selected_intervention_rate']:.1f}% | "
            f"effective={100.0 * row['effective_intervention_rate']:.1f}%",
            flush=True,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

