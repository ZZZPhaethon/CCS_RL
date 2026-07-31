"""Aggregate E1 formal-test results for the two PPO algorithms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


MODEL_SEEDS = (0, 1, 2)
FORMAL_SEEDS = tuple(range(9_000_031, 9_000_061))
BOOTSTRAP_SAMPLES = 10_000
ALGORITHMS = {
    "Centralized Maskable PPO": "centralized_maskable_ppo",
    "Event-Residual PPO": "event_residual_ppo",
}


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
    model_seeds = sorted(by_model_seed)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled: list[float] = []
        for _model_seed in model_seeds:
            selected = rng.choice(model_seeds)
            deltas = by_model_seed[selected]
            sampled.extend(rng.choice(deltas) for _ in deltas)
        estimates.append(mean(sampled))
    return _percentile(estimates, 0.025), _percentile(
        estimates,
        0.975,
    )


def _assert_cost_identity(
    *,
    total: float,
    episode: float,
    cleanup: float,
    label: str,
) -> None:
    if not math.isclose(
        total,
        episode + cleanup,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"Cleanup identity failed for {label}.")


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(
            _contains_key(child, target) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, target) for child in value)
    return False


def _load_greedy(root: Path) -> dict[int, dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    for seed in FORMAL_SEEDS:
        rows = _read_csv(root / f"seed_{seed}" / "per_controller.csv")
        if len(rows) != 1:
            raise ValueError(f"Expected one Greedy row for seed {seed}.")
        row = rows[0]
        if row["controller"] != "greedy" or row["run_status"] != "completed":
            raise ValueError(f"Invalid Greedy result for seed {seed}.")
        episode = float(row["episode_total_cost"])
        cleanup = float(row["terminal_cleanup_operating_cost"])
        total = float(row["total_cost"])
        _assert_cost_identity(
            total=total,
            episode=episode,
            cleanup=cleanup,
            label=f"Greedy seed {seed}",
        )
        records[seed] = {
            "episode_total_cost_eur": episode,
            "terminal_cleanup_operating_cost_eur": cleanup,
            "total_cost_eur": total,
            "unit_cost_eur_per_t": float(
                row["total_cost_per_stored_t"]
            ),
            "vented_t": float(row["vented_t"]),
            "stored_t": float(row["stored_t"]),
            "captured_t": float(row["captured_t"]),
        }
    return records


def _checkpoint_sha256(run_dir: Path) -> str:
    checkpoints = list(run_dir.glob("*.zip"))
    if len(checkpoints) != 1:
        raise ValueError(
            f"Expected one checkpoint copy under {run_dir}, "
            f"found {len(checkpoints)}."
        )
    digest = hashlib.sha256()
    with checkpoints[0].open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_algorithm_rows(
    *,
    formal_root: Path,
    algorithm: str,
    directory_name: str,
    greedy: dict[int, dict[str, float]],
) -> tuple[
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    episode_rows: list[dict[str, Any]] = []
    provenance: dict[int, dict[str, Any]] = {}
    for model_seed in MODEL_SEEDS:
        run_dir = formal_root / directory_name / f"model_seed_{model_seed}"
        payload = _load_json(run_dir / "results.json")
        records = payload["per_seed"]
        actual_seeds = tuple(int(row["seed"]) for row in records)
        if actual_seeds != FORMAL_SEEDS:
            raise ValueError(
                f"Unexpected {algorithm} seeds for model seed "
                f"{model_seed}: {actual_seeds!r}"
            )
        config = _load_json(run_dir / "config.json")
        if int(config["episode_hours"]) != 720:
            raise ValueError(f"Unexpected episode hours under {run_dir}.")
        if int(config["forecast_context_hours"]) != 168:
            raise ValueError(f"Unexpected forecast context under {run_dir}.")
        if list(config["future_summary_windows_h"]) != [168]:
            raise ValueError(f"Unexpected future summary under {run_dir}.")
        if _contains_key(config, "valid_fraction"):
            raise ValueError(f"Excluded field present under {run_dir}.")

        checkpoint_hash = _checkpoint_sha256(run_dir)
        audit = _load_json(run_dir / "audit.json")
        if int(audit["episodes"]) != len(FORMAL_SEEDS):
            raise ValueError(f"Invalid task audit under {run_dir}.")
        provenance[model_seed] = {
            "checkpoint_sha256": checkpoint_hash,
            "slurm_array_job_id": str(audit["slurm_array_job_id"]),
            "slurm_array_task_id": str(audit["slurm_array_task_id"]),
        }

        for record in records:
            seed = int(record["seed"])
            episode = float(record["episode_total_cost_eur"])
            cleanup = float(
                record["terminal_cleanup_operating_cost_eur"]
            )
            total = float(record["total_cost_eur"])
            _assert_cost_identity(
                total=total,
                episode=episode,
                cleanup=cleanup,
                label=f"{algorithm} model seed {model_seed}, seed {seed}",
            )
            baseline = greedy[seed]
            delta = total - baseline["total_cost_eur"]
            episode_rows.append(
                {
                    "algorithm": algorithm,
                    "model_seed": model_seed,
                    "test_seed": seed,
                    "episode_total_cost_eur": episode,
                    "terminal_cleanup_operating_cost_eur": cleanup,
                    "total_cost_eur": total,
                    "unit_cost_eur_per_t": float(
                        record["unit_total_cost_eur_per_t"]
                    ),
                    "vented_t": float(record["vented_t"]),
                    "stored_t": float(record["stored_t"]),
                    "captured_t": float(record["captured_t"]),
                    "greedy_episode_total_cost_eur": baseline[
                        "episode_total_cost_eur"
                    ],
                    "greedy_terminal_cleanup_operating_cost_eur": (
                        baseline[
                            "terminal_cleanup_operating_cost_eur"
                        ]
                    ),
                    "greedy_total_cost_eur": baseline["total_cost_eur"],
                    "greedy_unit_cost_eur_per_t": baseline[
                        "unit_cost_eur_per_t"
                    ],
                    "greedy_vented_t": baseline["vented_t"],
                    "greedy_stored_t": baseline["stored_t"],
                    "delta_total_cost_eur": delta,
                    "paired_outcome": (
                        "win"
                        if delta < -1e-6
                        else "loss"
                        if delta > 1e-6
                        else "tie"
                    ),
                }
            )
    return episode_rows, provenance


def _summaries(
    episode_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_model: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        by_model_seed: dict[int, list[dict[str, Any]]] = {}
        for model_seed in MODEL_SEEDS:
            rows = [
                row
                for row in episode_rows
                if row["algorithm"] == algorithm
                and row["model_seed"] == model_seed
            ]
            if len(rows) != len(FORMAL_SEEDS):
                raise ValueError(
                    f"Missing rows for {algorithm}, model seed {model_seed}."
                )
            by_model_seed[model_seed] = rows
            deltas = [
                float(row["delta_total_cost_eur"]) for row in rows
            ]
            low, high = _paired_bootstrap_ci(
                deltas,
                rng=random.Random(
                    20_260_729 + algorithm_index * 10 + model_seed
                ),
            )
            wins = sum(row["paired_outcome"] == "win" for row in rows)
            ties = sum(row["paired_outcome"] == "tie" for row in rows)
            losses = len(rows) - wins - ties
            per_model.append(
                {
                    "algorithm": algorithm,
                    "model_seed": model_seed,
                    "test_episodes": len(rows),
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
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate": wins / len(rows),
                }
            )

        model_rows = [
            row for row in per_model if row["algorithm"] == algorithm
        ]
        all_rows = [
            row
            for rows in by_model_seed.values()
            for row in rows
        ]
        hierarchical_low, hierarchical_high = (
            _hierarchical_bootstrap_ci(
                {
                    model_seed: [
                        float(row["delta_total_cost_eur"])
                        for row in rows
                    ]
                    for model_seed, rows in by_model_seed.items()
                },
                rng=random.Random(20_260_829 + algorithm_index),
            )
        )
        wins = sum(row["paired_outcome"] == "win" for row in all_rows)
        ties = sum(row["paired_outcome"] == "tie" for row in all_rows)
        losses = len(all_rows) - wins - ties
        aggregate.append(
            {
                "algorithm": algorithm,
                "model_seeds": "0,1,2",
                "test_episodes": len(all_rows),
                "mean_total_cost_eur": mean(
                    float(row["total_cost_eur"]) for row in all_rows
                ),
                "between_model_seed_sd_total_cost_eur": stdev(
                    float(row["mean_total_cost_eur"])
                    for row in model_rows
                ),
                "mean_unit_cost_eur_per_t": mean(
                    float(row["unit_cost_eur_per_t"])
                    for row in all_rows
                ),
                "mean_vented_t": mean(
                    float(row["vented_t"]) for row in all_rows
                ),
                "mean_stored_t": mean(
                    float(row["stored_t"]) for row in all_rows
                ),
                "mean_delta_vs_greedy_eur": mean(
                    float(row["delta_total_cost_eur"])
                    for row in all_rows
                ),
                "hierarchical_bootstrap_95pct_ci_low_eur": (
                    hierarchical_low
                ),
                "hierarchical_bootstrap_95pct_ci_high_eur": (
                    hierarchical_high
                ),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_rate": wins / len(all_rows),
            }
        )
    return per_model, aggregate


def _greedy_summary(
    greedy: dict[int, dict[str, float]],
) -> dict[str, Any]:
    rows = list(greedy.values())
    return {
        "algorithm": "Greedy",
        "test_episodes": len(rows),
        "mean_total_cost_eur": mean(
            row["total_cost_eur"] for row in rows
        ),
        "mean_unit_cost_eur_per_t": mean(
            row["unit_cost_eur_per_t"] for row in rows
        ),
        "mean_vented_t": mean(row["vented_t"] for row in rows),
        "mean_stored_t": mean(row["stored_t"] for row in rows),
    }


def _write_readme(
    path: Path,
    *,
    greedy: dict[str, Any],
    aggregate: list[dict[str, Any]],
) -> None:
    lines = [
        "# E1 PPO formal-test 汇总",
        "",
        "固定使用 validation 选出的 best checkpoint；测试 seeds 为 "
        "9000031–9000060。总成本包含 terminal cleanup。",
        "",
        "| 方法 | 总成本 EUR | 单位成本 EUR/t | Vent t | Stored t | "
        "相对 Greedy EUR | 胜率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Greedy | {greedy['mean_total_cost_eur']:,.0f} | "
        f"{greedy['mean_unit_cost_eur_per_t']:.2f} | "
        f"{greedy['mean_vented_t']:,.1f} | "
        f"{greedy['mean_stored_t']:,.1f} | 0 | — |",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['algorithm']} | "
            f"{row['mean_total_cost_eur']:,.0f} | "
            f"{row['mean_unit_cost_eur_per_t']:.2f} | "
            f"{row['mean_vented_t']:,.1f} | "
            f"{row['mean_stored_t']:,.1f} | "
            f"{row['mean_delta_vs_greedy_eur']:+,.0f} | "
            f"{100.0 * row['win_rate']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Paired 与 hierarchical bootstrap 95% CI 见 "
            "`per_model_seed.csv` 和 `aggregate.csv`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--greedy-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    greedy = _load_greedy(args.greedy_root)
    episode_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for algorithm, directory_name in ALGORITHMS.items():
        rows, algorithm_provenance = _load_algorithm_rows(
            formal_root=args.formal_root,
            algorithm=algorithm,
            directory_name=directory_name,
            greedy=greedy,
        )
        episode_rows.extend(rows)
        provenance[algorithm] = algorithm_provenance

    per_model, aggregate = _summaries(episode_rows)
    greedy_metrics = _greedy_summary(greedy)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(args.out_dir / "test_per_episode.csv", episode_rows)
    _write_csv(args.out_dir / "per_model_seed.csv", per_model)
    _write_csv(args.out_dir / "aggregate.csv", aggregate)
    (args.out_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "formal_test": True,
                "formal_seeds": list(FORMAL_SEEDS),
                "model_seeds": list(MODEL_SEEDS),
                "model_choice": "best_validation",
                "reported_total_cost": (
                    "720 h episode total cost + common compact "
                    "terminal cleanup operating cost"
                ),
                "greedy": greedy_metrics,
                "per_model_seed": per_model,
                "aggregate": aggregate,
                "provenance": provenance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "protocol_audit.json").write_text(
        json.dumps(
            {
                "formal_seeds_exact": list(FORMAL_SEEDS),
                "model_seeds_exact": list(MODEL_SEEDS),
                "algorithms": list(ALGORITHMS),
                "policy_records_checked": len(episode_rows),
                "greedy_records_checked": len(greedy),
                "cleanup_identity_failures": 0,
                "configuration": {
                    "episode_hours": 720,
                    "read_only_forecast_hours": 168,
                    "future_summary_windows_h": [168],
                    "validity_fraction_feature_present": False,
                    "model_choice": "best_validation",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_readme(
        args.out_dir / "README_ZH.md",
        greedy=greedy_metrics,
        aggregate=aggregate,
    )


if __name__ == "__main__":
    main()
