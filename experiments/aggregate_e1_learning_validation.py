"""Aggregate E1 validation-only results for the three learning algorithms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


MODEL_SEEDS = (0, 1, 2)
VALIDATION_SEEDS = tuple(range(8_100_001, 8_100_021))
TRAINING_BUDGET = 9_505_319
BOOTSTRAP_SAMPLES = 10_000


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def _paired_bootstrap_ci(
    deltas: list[float],
    *,
    rng: random.Random,
) -> tuple[float, float]:
    estimates = [
        mean(rng.choice(deltas) for _ in deltas)
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return _percentile(estimates, 0.025), _percentile(
        estimates,
        0.975,
    )


def _hierarchical_bootstrap_ci(
    by_model_seed: dict[int, list[float]],
    *,
    rng: random.Random,
) -> tuple[float, float]:
    seed_ids = sorted(by_model_seed)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled: list[float] = []
        for _model_seed in seed_ids:
            selected_seed = rng.choice(seed_ids)
            deltas = by_model_seed[selected_seed]
            sampled.extend(rng.choice(deltas) for _ in deltas)
        estimates.append(mean(sampled))
    return _percentile(estimates, 0.025), _percentile(
        estimates,
        0.975,
    )


def _validation_curve(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in (run_dir / "validation").glob(
        "simulator_hour_*.json"
    ):
        payload = _load_json(path)
        if "mean_total_cost_eur" in payload:
            cost = float(payload["mean_total_cost_eur"])
        else:
            cost = float(
                payload["metrics"]["mean_total_cost_eur"]
            )
        records.append(
            {
                "calls": float(
                    payload["training_simulator_hour_steps"]
                ),
                "cost": cost,
            }
        )
    return sorted(records, key=lambda row: row["calls"])


def _greedy_reference(
    iterative_root: Path,
) -> dict[int, dict[str, float]]:
    expected: dict[int, dict[str, float]] = {}
    for model_seed in MODEL_SEEDS:
        rows = _read_csv(
            iterative_root
            / "eval"
            / f"fixed_single168_s{model_seed}"
            / "evaluation.csv"
        )
        if [int(row["seed"]) for row in rows] != list(
            VALIDATION_SEEDS
        ):
            raise ValueError(
                f"Unexpected Iterative-Q validation seeds for "
                f"model seed {model_seed}."
            )
        for row in rows:
            seed = int(row["seed"])
            candidate = {
                "episode_total_cost_eur": float(
                    row["greedy_episode_total_cost_eur"]
                ),
                "terminal_cleanup_operating_cost_eur": float(
                    row[
                        "greedy_terminal_cleanup_operating_cost_eur"
                    ]
                ),
                "total_cost_eur": float(
                    row["greedy_total_cost_eur"]
                ),
                "operating_cost_eur": float(
                    row["greedy_operating_cost_eur"]
                ),
                "vented_t": float(row["greedy_vented_t"]),
                "stored_t": float(row["greedy_stored_t"]),
                "unit_cost_eur_per_t": float(
                    row["greedy_unit_cost_eur_per_t"]
                ),
            }
            if seed in expected:
                for key, value in candidate.items():
                    if not math.isclose(
                        expected[seed][key],
                        value,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    ):
                        raise ValueError(
                            f"Greedy reference drift for seed {seed}, "
                            f"field {key}."
                        )
            else:
                expected[seed] = candidate
    return expected


def _ppo_rows(
    *,
    algorithm: str,
    root: Path,
    greedy: dict[int, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    for model_seed in MODEL_SEEDS:
        run_dir = root / f"model_seed_{model_seed}"
        best = _load_json(run_dir / "validation" / "best.json")
        complete = _load_json(run_dir / "training_complete.json")
        records = best["per_seed"]
        if [int(row["seed"]) for row in records] != list(
            VALIDATION_SEEDS
        ):
            raise ValueError(
                f"Unexpected {algorithm} validation seeds for "
                f"model seed {model_seed}."
            )
        for record in records:
            seed = int(record["seed"])
            total_cost = float(record["total_cost_eur"])
            stored_t = float(record["stored_t"])
            unit_cost = float(
                record.get(
                    "unit_total_cost_eur_per_t",
                    total_cost / stored_t,
                )
            )
            baseline = greedy[seed]
            episode_rows.append(
                _episode_row(
                    algorithm=algorithm,
                    model_seed=model_seed,
                    seed=seed,
                    episode_cost=float(
                        record["episode_total_cost_eur"]
                    ),
                    cleanup_cost=float(
                        record[
                            "terminal_cleanup_operating_cost_eur"
                        ]
                    ),
                    total_cost=total_cost,
                    unit_cost=unit_cost,
                    vented_t=float(record["vented_t"]),
                    stored_t=stored_t,
                    captured_t=float(record["captured_t"]),
                    greedy=baseline,
                )
            )
        curve = _validation_curve(run_dir)
        stability.append(
            {
                "algorithm": algorithm,
                "model_seed": model_seed,
                "training_simulator_calls": int(
                    complete["simulator_step_calls"]
                ),
                "budget_fraction": float(
                    complete["simulator_budget_fraction"]
                ),
                "hard_cap_reached": bool(
                    complete.get(
                        "simulator_budget_stop_reached",
                        complete["simulator_budget_exhausted"],
                    )
                ),
                "stop_state": str(complete["state"]),
                "best_checkpoint_simulator_calls": float(
                    best["training_simulator_hour_steps"]
                ),
                "validation_checkpoint_count": len(curve),
                "validation_cost_start_eur": curve[0]["cost"],
                "validation_cost_end_eur": curve[-1]["cost"],
                "validation_cost_best_eur": min(
                    row["cost"] for row in curve
                ),
                "validation_cost_range_eur": max(
                    row["cost"] for row in curve
                )
                - min(row["cost"] for row in curve),
                "epochs_or_updates": None,
                "validation_pairwise_accuracy": None,
            }
        )
    return episode_rows, stability


def _episode_row(
    *,
    algorithm: str,
    model_seed: int,
    seed: int,
    episode_cost: float,
    cleanup_cost: float,
    total_cost: float,
    unit_cost: float,
    vented_t: float,
    stored_t: float,
    captured_t: float | None,
    greedy: dict[str, float],
) -> dict[str, Any]:
    delta = total_cost - greedy["total_cost_eur"]
    return {
        "algorithm": algorithm,
        "model_seed": model_seed,
        "validation_seed": seed,
        "episode_total_cost_eur": episode_cost,
        "terminal_cleanup_operating_cost_eur": cleanup_cost,
        "total_cost_eur": total_cost,
        "unit_cost_eur_per_t": unit_cost,
        "vented_t": vented_t,
        "stored_t": stored_t,
        "captured_t": captured_t,
        "greedy_episode_total_cost_eur": greedy[
            "episode_total_cost_eur"
        ],
        "greedy_terminal_cleanup_operating_cost_eur": greedy[
            "terminal_cleanup_operating_cost_eur"
        ],
        "greedy_total_cost_eur": greedy["total_cost_eur"],
        "greedy_unit_cost_eur_per_t": greedy[
            "unit_cost_eur_per_t"
        ],
        "greedy_vented_t": greedy["vented_t"],
        "greedy_stored_t": greedy["stored_t"],
        "delta_total_cost_eur": delta,
        "paired_outcome": (
            "win"
            if delta < -1e-6
            else "loss"
            if delta > 1e-6
            else "tie"
        ),
    }


def _iterative_rows(
    *,
    iterative_root: Path,
    greedy: dict[int, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    for model_seed in MODEL_SEEDS:
        eval_dir = (
            iterative_root
            / "eval"
            / f"fixed_single168_s{model_seed}"
        )
        rows = _read_csv(eval_dir / "evaluation.csv")
        for row in rows:
            seed = int(row["seed"])
            episode_rows.append(
                _episode_row(
                    algorithm="Iterative Action-Q",
                    model_seed=model_seed,
                    seed=seed,
                    episode_cost=float(
                        row["episode_total_cost_eur"]
                    ),
                    cleanup_cost=float(
                        row[
                            "terminal_cleanup_operating_cost_eur"
                        ]
                    ),
                    total_cost=float(row["total_cost_eur"]),
                    unit_cost=float(row["unit_cost_eur_per_t"]),
                    vented_t=float(row["vented_t"]),
                    stored_t=float(row["stored_t"]),
                    captured_t=None,
                    greedy=greedy[seed],
                )
            )
        training = _load_json(
            iterative_root
            / f"fixed_single168_s{model_seed}_p3"
            / "summary.json"
        )
        history = training["history"]
        final_validation = training["final_validation"]
        stability.append(
            {
                "algorithm": "Iterative Action-Q",
                "model_seed": model_seed,
                "training_simulator_calls": TRAINING_BUDGET,
                "budget_fraction": 1.0,
                "hard_cap_reached": True,
                "stop_state": "early_stopping_on_fixed_G0_G2_bank",
                "best_checkpoint_simulator_calls": TRAINING_BUDGET,
                "validation_checkpoint_count": len(history),
                "validation_cost_start_eur": None,
                "validation_cost_end_eur": None,
                "validation_cost_best_eur": None,
                "validation_cost_range_eur": None,
                "epochs_or_updates": len(history),
                "validation_pairwise_accuracy": float(
                    final_validation["pairwise_accuracy"]
                ),
            }
        )
    return episode_rows, stability


def _summaries(
    episode_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_model: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    algorithms = (
        "Hourly Centralized Maskable PPO",
        "Event-Residual PPO",
        "Iterative Action-Q",
    )
    for algorithm_index, algorithm in enumerate(algorithms):
        by_seed: dict[int, list[dict[str, Any]]] = {}
        for model_seed in MODEL_SEEDS:
            rows = [
                row
                for row in episode_rows
                if row["algorithm"] == algorithm
                and row["model_seed"] == model_seed
            ]
            by_seed[model_seed] = rows
            deltas = [
                float(row["delta_total_cost_eur"])
                for row in rows
            ]
            low, high = _paired_bootstrap_ci(
                deltas,
                rng=random.Random(
                    20_260_728 + algorithm_index * 10 + model_seed
                ),
            )
            training = next(
                row
                for row in stability_rows
                if row["algorithm"] == algorithm
                and row["model_seed"] == model_seed
            )
            per_model.append(
                {
                    "algorithm": algorithm,
                    "model_seed": model_seed,
                    "mean_total_cost_eur": mean(
                        float(row["total_cost_eur"]) for row in rows
                    ),
                    "median_total_cost_eur": median(
                        float(row["total_cost_eur"]) for row in rows
                    ),
                    "mean_unit_cost_eur_per_t": mean(
                        float(row["unit_cost_eur_per_t"])
                        for row in rows
                    ),
                    "mean_vented_t": mean(
                        float(row["vented_t"]) for row in rows
                    ),
                    "mean_stored_t": mean(
                        float(row["stored_t"]) for row in rows
                    ),
                    "mean_delta_vs_greedy_eur": mean(deltas),
                    "paired_bootstrap_95pct_ci_low_eur": low,
                    "paired_bootstrap_95pct_ci_high_eur": high,
                    "wins": sum(
                        row["paired_outcome"] == "win"
                        for row in rows
                    ),
                    "ties": sum(
                        row["paired_outcome"] == "tie"
                        for row in rows
                    ),
                    "losses": sum(
                        row["paired_outcome"] == "loss"
                        for row in rows
                    ),
                    "training_simulator_calls": training[
                        "training_simulator_calls"
                    ],
                    "budget_fraction": training["budget_fraction"],
                    "best_checkpoint_simulator_calls": training[
                        "best_checkpoint_simulator_calls"
                    ],
                }
            )
        model_rows = [
            row for row in per_model if row["algorithm"] == algorithm
        ]
        hierarchical_low, hierarchical_high = (
            _hierarchical_bootstrap_ci(
                {
                    model_seed: [
                        float(row["delta_total_cost_eur"])
                        for row in rows
                    ]
                    for model_seed, rows in by_seed.items()
                },
                rng=random.Random(20_260_828 + algorithm_index),
            )
        )
        total_wins = sum(int(row["wins"]) for row in model_rows)
        total_ties = sum(int(row["ties"]) for row in model_rows)
        total_losses = sum(int(row["losses"]) for row in model_rows)
        aggregate.append(
            {
                "algorithm": algorithm,
                "model_seeds": "0,1,2",
                "validation_episodes": sum(
                    len(rows) for rows in by_seed.values()
                ),
                "mean_total_cost_eur": mean(
                    float(row["mean_total_cost_eur"])
                    for row in model_rows
                ),
                "between_model_seed_sd_total_cost_eur": stdev(
                    float(row["mean_total_cost_eur"])
                    for row in model_rows
                ),
                "median_of_model_seed_mean_total_cost_eur": median(
                    float(row["mean_total_cost_eur"])
                    for row in model_rows
                ),
                "mean_unit_cost_eur_per_t": mean(
                    float(row["mean_unit_cost_eur_per_t"])
                    for row in model_rows
                ),
                "mean_vented_t": mean(
                    float(row["mean_vented_t"])
                    for row in model_rows
                ),
                "mean_stored_t": mean(
                    float(row["mean_stored_t"])
                    for row in model_rows
                ),
                "mean_delta_vs_greedy_eur": mean(
                    float(row["mean_delta_vs_greedy_eur"])
                    for row in model_rows
                ),
                "hierarchical_bootstrap_95pct_ci_low_eur": (
                    hierarchical_low
                ),
                "hierarchical_bootstrap_95pct_ci_high_eur": (
                    hierarchical_high
                ),
                "wins": total_wins,
                "ties": total_ties,
                "losses": total_losses,
                "win_rate": total_wins
                / (total_wins + total_ties + total_losses),
                "training_simulator_calls_mean": mean(
                    float(row["training_simulator_calls"])
                    for row in model_rows
                ),
                "training_simulator_calls_between_seed_sd": stdev(
                    float(row["training_simulator_calls"])
                    for row in model_rows
                ),
            }
        )
    return per_model, aggregate


def _write_readme(
    path: Path,
    aggregate: list[dict[str, Any]],
) -> None:
    lines = [
        "# E1 三种学习算法 validation-only 比较",
        "",
        "所有结果使用 model seeds 0/1/2 和 controller-validation "
        "seeds 8100001–8100020。未访问 formal test seeds。",
        "",
        "| 算法 | 总成本 EUR | 单位成本 EUR/t | Vent t | Stored t | "
        "相对 Greedy EUR | 胜率 | Calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['algorithm']} | "
            f"{row['mean_total_cost_eur']:,.0f} | "
            f"{row['mean_unit_cost_eur_per_t']:.2f} | "
            f"{row['mean_vented_t']:,.1f} | "
            f"{row['mean_stored_t']:,.1f} | "
            f"{row['mean_delta_vs_greedy_eur']:+,.0f} | "
            f"{100.0 * row['win_rate']:.1f}% | "
            f"{row['training_simulator_calls_mean']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "总成本均包含 720 h episode cost 与 common compact "
            "terminal cleanup operating cost。置信区间见 "
            "`aggregate.json`；训练曲线与停止信息见 "
            "`training_stability.csv`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-root", type=Path, required=True)
    parser.add_argument("--iterative-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    greedy = _greedy_reference(args.iterative_root)
    central_rows, central_stability = _ppo_rows(
        algorithm="Hourly Centralized Maskable PPO",
        root=args.e1_root / "centralized_maskable_ppo",
        greedy=greedy,
    )
    event_rows, event_stability = _ppo_rows(
        algorithm="Event-Residual PPO",
        root=args.e1_root / "event_residual_ppo",
        greedy=greedy,
    )
    iterative_rows, iterative_stability = _iterative_rows(
        iterative_root=args.iterative_root,
        greedy=greedy,
    )
    episode_rows = central_rows + event_rows + iterative_rows
    stability_rows = (
        central_stability + event_stability + iterative_stability
    )
    per_model, aggregate = _summaries(
        episode_rows,
        stability_rows,
    )

    _write_csv(out_dir / "validation_per_episode.csv", episode_rows)
    _write_csv(out_dir / "per_model_seed.csv", per_model)
    _write_csv(out_dir / "aggregate.csv", aggregate)
    _write_csv(out_dir / "training_stability.csv", stability_rows)
    (out_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "validation_only": True,
                "validation_seeds": list(VALIDATION_SEEDS),
                "formal_test_accessed": False,
                "training_budget_per_model_seed": TRAINING_BUDGET,
                "reported_total_cost": (
                    "720 h episode total cost + common compact "
                    "terminal cleanup operating cost"
                ),
                "per_model_seed": per_model,
                "aggregate": aggregate,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_readme(out_dir / "README.md", aggregate)


if __name__ == "__main__":
    main()
