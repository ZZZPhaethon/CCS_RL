"""Evaluate a trained residual PPO v3 model.

评估已训练的 residual PPO v3 模型。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO

from algorithms.rl.reward import HighLevelRewardConfig
from algorithms.residual_rl_v2.evaluation import (
    evaluate_seeds,
    validation_metrics,
)
from algorithms.residual_rl_v2.factory import (
    make_masked_residual_native_env,
)

from .factory import make_risk_gated_native_env
from .risk_gate import AdaptiveRiskGateConfig


def evaluate_run(
    run_dir: Path,
    *,
    seeds: tuple[int, ...],
    model_choice: str = "best",
    hard_scenario_probability: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate one v3 model and persist diagnostics.

    评估一个 v3 模型并保存诊断结果。
    """
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    algorithm = str(config.get("algorithm"))
    supported = {
        "maskable_residual_ppo_v2",
        "maskable_residual_ppo_v3",
    }
    if algorithm not in supported:
        raise ValueError(
            f"Unsupported residual algorithm: {algorithm!r}."
        )
    if model_choice not in {"best", "final"}:
        raise ValueError("model_choice must be 'best' or 'final'.")
    version = "v3" if algorithm.endswith("_v3") else "v2"
    model_stem = (
        f"maskable_residual_{version}_best_validation"
        if model_choice == "best"
        else f"maskable_residual_{version}_final"
    )
    model_path = run_dir / model_stem
    model = MaskablePPO.load(model_path, device="cpu")
    common = {
        "scenario": str(config["scenario"]),
        "episode_hours": int(config["episode_hours"]),
        "forecast_context_hours": int(
            config["forecast_context_hours"]
        ),
        "decision_interval_h": float(config["decision_interval_h"]),
        "event_triggered": bool(config["event_triggered"]),
        "weather_mode": str(config["weather_mode"]),
        "hard_scenario_probability": float(
            hard_scenario_probability
        ),
        "reward": HighLevelRewardConfig(
            **config["high_level_reward"]
        ),
    }
    if version == "v3":
        env = make_risk_gated_native_env(
            gate=AdaptiveRiskGateConfig(**config["risk_gate"]),
            gate_mode=str(config["risk_gate_mode"]),
            outside_risk_intervention_penalty=float(
                config["outside_risk_intervention_penalty"]
            ),
            **common,
        )
    else:
        env = make_masked_residual_native_env(**common)
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
        "selected_intervention_rate",
        "effective_intervention_rate",
        "episode_reward",
        "captured_t",
        "stored_t",
        "vented_t",
        "storage_rate",
        "total_cost_eur",
        "unit_total_cost_eur_per_t",
        "hard_violations",
        "wall_clock_seconds",
    )
    summary.update(
        {
            f"{key}_mean": sum(
                float(record[key]) for record in records
            )
            / len(records)
            for key in numeric_keys
        }
    )

    label = _seed_label(seeds)
    output_dir = run_dir / "evaluation" / (
        f"{model_choice}__hardprob{hard_scenario_probability:g}"
        f"__seeds_{label}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "model_choice": model_choice,
                "algorithm": algorithm,
                "model_path": str(model_path),
                "hard_scenario_probability": (
                    hard_scenario_probability
                ),
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
    columns = [
        key
        for key, value in records[0].items()
        if not isinstance(value, dict)
    ]
    with (output_dir / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {
                key: value
                for key, value in record.items()
                if key in columns
            }
            for record in records
        )
    return records, summary


def _seed_label(seeds: tuple[int, ...]) -> str:
    """Return a Windows-safe compact label for an evaluation seed set.

    为评估 seed 集合返回适合 Windows 路径的紧凑标签。
    """
    if len(seeds) <= 5:
        return "-".join(str(seed) for seed in seeds)
    return f"{min(seeds)}-{max(seeds)}__n{len(seeds)}"


def main() -> None:
    """Evaluate v3 from a terminal.

    从终端评估 v3。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(4_000_001, 4_000_021)),
    )
    parser.add_argument(
        "--model",
        choices=("best", "final"),
        default="best",
    )
    parser.add_argument(
        "--hard-scenario-probability",
        type=float,
        default=0.0,
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
            "Residual PPO v3 | "
            f"seed={row['seed']} | "
            f"stored={row['stored_t']:,.1f} t | "
            f"vented={row['vented_t']:,.1f} t | "
            f"cost=EUR {row['total_cost_eur']:,.0f}",
            flush=True,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
