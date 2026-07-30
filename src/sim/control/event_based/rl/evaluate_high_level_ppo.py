"""Evaluate a trained high-level PPO with comparable physical cost metrics.

使用可比的物理与成本指标评估已训练的高层 PPO。
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from .observation_encoder import FORECAST_WINDOWS_H
from .reward import HARD_VIOLATION_CODES, HighLevelRewardConfig
from .train_high_level_ppo import make_high_level_native_env


def evaluate_run(
    run_dir: Path,
    *,
    seeds: Iterable[int] = range(1, 6),
    executor: str = "rule",
    model_choice: str = "best",
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate one saved PPO deterministically and persist seed-level metrics.

    对一个已保存 PPO 进行确定性评估，并保存各随机种子的指标。
    """
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError(
            "PPO evaluation requires stable-baselines3 and sb3-contrib."
        ) from exc
    try:
        from tqdm.auto import tqdm
        progress_write = tqdm.write
    except ImportError:  # pragma: no cover - depends on the runtime environment.
        progress_write = print

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    reward_config = HighLevelRewardConfig(**config["high_level_reward"])
    env = make_high_level_native_env(
        scenario=str(config["scenario"]),
        episode_hours=int(config["episode_hours"]),
        forecast_context_hours=int(config.get("forecast_context_hours", 72)),
        future_summary_windows_h=tuple(
            int(value)
            for value in config.get(
                "future_summary_windows_h",
                FORECAST_WINDOWS_H,
            )
        ),
        decision_interval_h=float(config["decision_interval_h"]),
        event_triggered=bool(config.get("event_triggered", False)),
        weather_mode=str(config.get("weather_mode", "window")),
        warm_start=bool(config.get("warm_start", False)),
        scenario_protocol=str(config.get("scenario_protocol", "local_formal")),
        executor=executor,
        reward=reward_config,
    )
    if model_choice not in {"best", "final"}:
        raise ValueError("model_choice must be 'best' or 'final'.")
    model_stem = (
        "ppo_high_level_best_validation"
        if model_choice == "best"
        else "ppo_high_level_final"
    )
    model = MaskablePPO.load(run_dir / model_stem, device="cpu")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("At least one evaluation seed is required.")

    records: list[dict[str, Any]] = []
    for position, seed in enumerate(seed_values, start=1):
        record = _evaluate_seed(model, env, seed)
        records.append(record)
        progress_write(
            "Evaluate PPO | "
            f"{position}/{len(seed_values)} seeds | "
            f"seed={seed} | stored={record['stored_t']:.1f} t | "
            f"vented={record['vented_t']:.1f} t | "
            f"total_cost=EUR {record['total_cost_eur']:.1f}"
        )
    numeric_keys = (
        "decisions",
        "mean_decision_interval_h",
        "episode_reward",
        "captured_t",
        "stored_t",
        "vented_t",
        "storage_rate",
        "episode_vessel_fuel_eur",
        "episode_conditioning_eur",
        "episode_reconditioning_eur",
        "episode_loading_eur",
        "episode_unloading_eur",
        "episode_operating_cost_eur",
        "episode_vent_penalty_eur",
        "episode_storage_shortfall_penalty_eur",
        "terminal_cleanup_operating_cost_eur",
        "operating_cost_eur",
        "episode_total_cost_eur",
        "total_cost_eur",
        "unit_total_cost_eur_per_t",
        "hard_violations",
        "wall_clock_seconds",
    )
    summary = {
        key: sum(float(record[key]) for record in records) / len(records)
        for key in numeric_keys
    }
    output_dir = run_dir / (
        "evaluation" if executor == "rule" else f"evaluation_{executor}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_label = "-".join(str(seed) for seed in seed_values)
    json_path = output_dir / f"seeds_{seed_label}.json"
    csv_path = output_dir / f"seeds_{seed_label}.csv"
    json_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "seeds": list(seed_values),
                "mean": summary,
                "per_seed": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                key
                for key in records[0]
                if key not in {"actions", "triggers"}
            ],
        )
        writer.writeheader()
        writer.writerows(
            {
                key: value
                for key, value in record.items()
                if key not in {"actions", "triggers"}
            }
            for record in records
        )
    return records, summary


def _evaluate_seed(model, env, seed: int) -> dict[str, Any]:
    """Run one complete deterministic physical episode.

    运行一个完整的确定性物理回合。
    """
    started_at = perf_counter()
    observation = env.reset(seed=seed)
    total_reward = 0.0
    decisions = 0
    elapsed_hours = 0.0
    actions: Counter[str] = Counter()
    triggers: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    done = False
    while not done:
        action, _state = model.predict(
            observation,
            deterministic=True,
            action_masks=env.action_masks(),
        )
        observation, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        decisions += 1
        elapsed_hours += float(info["elapsed_hours"])
        actions[str(info["action_label"])] += 1
        triggers[str(info["decision_trigger"])] += 1
        violations.update(info["violation_counts"])
        done = terminated or truncated

    physical_env = env.env
    stored_t = float(physical_env.cumulative_stored_t)
    captured_t = float(physical_env.cumulative_captured_t)
    episode_operating_cost = float(
        physical_env.ledger.operating_cost
    )
    episode_total_cost = float(physical_env.ledger.total_cost)
    cleanup_cost = float(
        info.get("terminal_cleanup_operating_cost_eur", 0.0)
    )
    operating_cost = episode_operating_cost + cleanup_cost
    total_cost = episode_total_cost + cleanup_cost
    hard_violations = sum(
        int(count)
        for code, count in violations.items()
        if code in HARD_VIOLATION_CODES
    )
    return {
        "seed": seed,
        "decisions": decisions,
        "mean_decision_interval_h": elapsed_hours / max(1, decisions),
        "episode_reward": total_reward,
        "captured_t": captured_t,
        "stored_t": stored_t,
        "vented_t": float(physical_env.ledger.vented_t),
        "storage_rate": stored_t / captured_t if captured_t > 1e-9 else 0.0,
        "episode_vessel_fuel_eur": float(
            physical_env.ledger.vessel_fuel
        ),
        "episode_conditioning_eur": float(
            physical_env.ledger.conditioning
        ),
        "episode_reconditioning_eur": float(
            physical_env.ledger.reconditioning
        ),
        "episode_loading_eur": float(physical_env.ledger.loading),
        "episode_unloading_eur": float(physical_env.ledger.unloading),
        "episode_operating_cost_eur": episode_operating_cost,
        "episode_vent_penalty_eur": float(
            physical_env.ledger.vent_penalty
        ),
        "episode_storage_shortfall_penalty_eur": float(
            physical_env.ledger.storage_shortfall_penalty
        ),
        "terminal_cleanup_operating_cost_eur": cleanup_cost,
        "operating_cost_eur": operating_cost,
        "episode_total_cost_eur": episode_total_cost,
        "total_cost_eur": total_cost,
        "unit_total_cost_eur_per_t": (
            total_cost / stored_t if stored_t > 1e-9 else float("nan")
        ),
        "hard_violations": hard_violations,
        "wall_clock_seconds": perf_counter() - started_at,
        "actions": dict(actions),
        "triggers": dict(triggers),
    }


def main() -> None:
    """Evaluate a timestamped training directory from the command line.

    从命令行评估一个带时间戳的训练目录。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--executor",
        choices=("rule", "mpc"),
        default="rule",
    )
    parser.add_argument(
        "--model",
        choices=("best", "final"),
        default="best",
    )
    args = parser.parse_args()
    _records, summary = evaluate_run(
        args.run_dir,
        seeds=args.seeds,
        executor=args.executor,
        model_choice=args.model,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    output_dir = args.run_dir / (
        "evaluation"
        if args.executor == "rule"
        else f"evaluation_{args.executor}"
    )
    print(f"Saved evaluation under: {output_dir}")


if __name__ == "__main__":
    main()
