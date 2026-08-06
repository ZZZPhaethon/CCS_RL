"""Compare random-root and increased-window Iterative-Q ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


TEST_SEEDS = tuple(range(9000031, 9000061))
METRICS = (
    "total_cost_eur",
    "delta_total_cost_eur",
    "vented_t",
    "stored_t",
    "override_events",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--random-root-run", type=Path, required=True)
    parser.add_argument("--windows24-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    return parser.parse_args(argv)


def _read_evaluation(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {
        int(row["seed"]): {metric: float(row[metric]) for metric in METRICS}
        for row in rows
    }
    if tuple(sorted(result)) != TEST_SEEDS:
        raise ValueError(f"unexpected formal test seeds in {path}")
    return result


def _single_eval_path(run_root: Path, condition: str, model_seed: int) -> Path:
    eval_root = run_root / condition / f"model_seed_{model_seed}" / "eval"
    paths = sorted(eval_root.glob("*/evaluation.csv"))
    if len(paths) != 1:
        raise ValueError(f"expected one evaluation.csv below {eval_root}, got {paths}")
    return paths[0]


def _load_matrices(
    root: Path,
    model_seeds: tuple[int, ...],
    condition: str | None = None,
) -> dict[str, np.ndarray]:
    matrices = {
        metric: np.empty((len(model_seeds), len(TEST_SEEDS)), dtype=np.float64)
        for metric in METRICS
    }
    for model_index, model_seed in enumerate(model_seeds):
        path = (
            root / f"model_seed_{model_seed}" / "evaluation.csv"
            if condition is None
            else _single_eval_path(root, condition, model_seed)
        )
        rows = _read_evaluation(path)
        for seed_index, scenario_seed in enumerate(TEST_SEEDS):
            for metric in METRICS:
                matrices[metric][model_index, seed_index] = rows[scenario_seed][metric]
    return matrices


def _hierarchical_ci(
    matrix: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> list[float]:
    replicates = np.empty(int(draws), dtype=np.float64)
    for index in range(len(replicates)):
        model_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        scenario_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        replicates[index] = matrix[np.ix_(model_indices, scenario_indices)].mean()
    return [
        float(np.quantile(replicates, 0.025)),
        float(np.quantile(replicates, 0.975)),
    ]


def _condition_summary(
    matrices: dict[str, np.ndarray],
    rng: np.random.Generator,
    draws: int,
) -> dict[str, object]:
    cost = matrices["total_cost_eur"]
    delta = matrices["delta_total_cost_eur"]
    scenario_delta = delta.mean(axis=0)
    return {
        "episodes": int(cost.size),
        "mean_total_cost_eur": float(cost.mean()),
        "mean_delta_cost_vs_greedy_eur": float(delta.mean()),
        "delta_cost_vs_greedy_ci95_eur": _hierarchical_ci(delta, rng, draws),
        "mean_vented_t": float(matrices["vented_t"].mean()),
        "mean_stored_t": float(matrices["stored_t"].mean()),
        "mean_override_events": float(matrices["override_events"].mean()),
        "wins_vs_greedy": int(np.count_nonzero(scenario_delta < -1e-6)),
        "ties_vs_greedy": int(np.count_nonzero(np.abs(scenario_delta) <= 1e-6)),
        "losses_vs_greedy": int(np.count_nonzero(scenario_delta > 1e-6)),
        "model_seed_mean_total_cost_eur": [
            float(value) for value in cost.mean(axis=1)
        ],
    }


def _comparison_summary(
    condition: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    rng: np.random.Generator,
    draws: int,
) -> dict[str, object]:
    cost_difference = condition["total_cost_eur"] - baseline["total_cost_eur"]
    scenario_difference = cost_difference.mean(axis=0)
    return {
        "mean_cost_difference_eur": float(cost_difference.mean()),
        "mean_cost_difference_pct_of_baseline": float(
            100.0 * cost_difference.mean() / baseline["total_cost_eur"].mean()
        ),
        "cost_difference_ci95_eur": _hierarchical_ci(
            cost_difference, rng, draws
        ),
        "mean_vented_difference_t": float(
            (condition["vented_t"] - baseline["vented_t"]).mean()
        ),
        "mean_stored_difference_t": float(
            (condition["stored_t"] - baseline["stored_t"]).mean()
        ),
        "condition_lower_cost_scenarios": int(
            np.count_nonzero(scenario_difference < -1e-6)
        ),
        "ties": int(np.count_nonzero(np.abs(scenario_difference) <= 1e-6)),
        "baseline_lower_cost_scenarios": int(
            np.count_nonzero(scenario_difference > 1e-6)
        ),
    }


def _budget_summary(
    run_root: Path,
    condition: str,
    model_seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    rows = []
    for model_seed in model_seeds:
        path = run_root / condition / f"model_seed_{model_seed}" / "budget.json"
        budget = json.loads(path.read_text(encoding="utf-8"))
        totals = budget["totals"]
        target = int(budget["target_train_simulator_steps"])
        actual = int(totals["train_simulator_steps"])
        rows.append(
            {
                "model_seed": model_seed,
                "target_train_simulator_steps": target,
                "actual_train_simulator_steps": actual,
                "relative_error_pct": float(100.0 * (actual - target) / target),
                "train_roots": int(totals["train_roots"]),
            }
        )
    return rows


def main(argv=None):
    args = parse_args(argv)
    model_seeds = tuple(args.model_seeds)
    rng = np.random.default_rng(args.bootstrap_seed)
    baseline = _load_matrices(args.baseline_root, model_seeds)
    conditions = {
        "random_root": (args.random_root_run, "random_root"),
        "windows24": (args.windows24_run, "windows24"),
    }
    result = {
        "kind": "iterative_q_root_window_ablation",
        "model_seeds": list(model_seeds),
        "test_seeds": list(TEST_SEEDS),
        "baseline": _condition_summary(
            baseline, rng, args.bootstrap_draws
        ),
        "conditions": {},
    }
    for name, (run_root, directory) in conditions.items():
        matrices = _load_matrices(run_root, model_seeds, directory)
        result["conditions"][name] = {
            "metrics": _condition_summary(matrices, rng, args.bootstrap_draws),
            "vs_formal_baseline": _comparison_summary(
                matrices, baseline, rng, args.bootstrap_draws
            ),
            "budget": _budget_summary(run_root, directory, model_seeds),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "analysis.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
