"""Compare the 0-720 h Iterative-Q windows with the formal 108-680 h baseline."""

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
    parser.add_argument("--full-window-root", type=Path, required=True)
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


def _evaluation_path(root: Path, model_seed: int, nested: bool) -> Path:
    seed_root = root / f"model_seed_{model_seed}"
    if not nested:
        return seed_root / "evaluation.csv"
    paths = sorted((seed_root / "eval").glob("*/evaluation.csv"))
    if len(paths) != 1:
        raise ValueError(f"expected one evaluation.csv below {seed_root / 'eval'}")
    return paths[0]


def _load_matrices(
    root: Path, model_seeds: tuple[int, ...], nested: bool
) -> dict[str, np.ndarray]:
    matrices = {
        metric: np.empty((len(model_seeds), len(TEST_SEEDS)), dtype=np.float64)
        for metric in METRICS
    }
    for model_index, model_seed in enumerate(model_seeds):
        rows = _read_evaluation(_evaluation_path(root, model_seed, nested))
        for seed_index, scenario_seed in enumerate(TEST_SEEDS):
            for metric in METRICS:
                matrices[metric][model_index, seed_index] = rows[scenario_seed][
                    metric
                ]
    return matrices


def _hierarchical_ci(
    matrix: np.ndarray, rng: np.random.Generator, draws: int
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


def _summary(matrices: dict[str, np.ndarray]) -> dict[str, object]:
    delta = matrices["delta_total_cost_eur"]
    return {
        "episodes": int(delta.size),
        "mean_total_cost_eur": float(matrices["total_cost_eur"].mean()),
        "mean_delta_cost_vs_greedy_eur": float(delta.mean()),
        "mean_vented_t": float(matrices["vented_t"].mean()),
        "mean_stored_t": float(matrices["stored_t"].mean()),
        "mean_override_events": float(matrices["override_events"].mean()),
        "wins_vs_greedy": int(np.count_nonzero(delta < -1e-6)),
        "ties_vs_greedy": int(np.count_nonzero(np.abs(delta) <= 1e-6)),
        "losses_vs_greedy": int(np.count_nonzero(delta > 1e-6)),
    }


def _per_model_seed(
    baseline: dict[str, np.ndarray],
    full_window: dict[str, np.ndarray],
    model_seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    rows = []
    for index, model_seed in enumerate(model_seeds):
        cost_change = (
            full_window["total_cost_eur"][index]
            - baseline["total_cost_eur"][index]
        )
        rows.append(
            {
                "model_seed": model_seed,
                "baseline_mean_total_cost_eur": float(
                    baseline["total_cost_eur"][index].mean()
                ),
                "full_window_mean_total_cost_eur": float(
                    full_window["total_cost_eur"][index].mean()
                ),
                "mean_cost_change_eur": float(cost_change.mean()),
                "full_window_mean_delta_vs_greedy_eur": float(
                    full_window["delta_total_cost_eur"][index].mean()
                ),
                "full_window_wins_vs_greedy": int(
                    np.count_nonzero(
                        full_window["delta_total_cost_eur"][index] < -1e-6
                    )
                ),
                "full_window_lower_cost_than_baseline": int(
                    np.count_nonzero(cost_change < -1e-6)
                ),
            }
        )
    return rows


def _budgets(root: Path, model_seeds: tuple[int, ...]) -> list[dict[str, object]]:
    rows = []
    for model_seed in model_seeds:
        path = root / f"model_seed_{model_seed}" / "budget.json"
        budget = json.loads(path.read_text(encoding="utf-8"))
        totals = budget["totals"]
        rows.append(
            {
                "model_seed": model_seed,
                "train_roots": int(totals["train_roots"]),
                "train_simulator_steps": int(totals["train_simulator_steps"]),
                "train_step_relative_error_pct": float(
                    totals["train_step_relative_error_pct"]
                ),
            }
        )
    return rows


def main(argv=None):
    args = parse_args(argv)
    model_seeds = tuple(args.model_seeds)
    rng = np.random.default_rng(args.bootstrap_seed)
    baseline = _load_matrices(args.baseline_root, model_seeds, nested=False)
    full_window = _load_matrices(args.full_window_root, model_seeds, nested=True)
    cost_change = full_window["total_cost_eur"] - baseline["total_cost_eur"]
    result = {
        "kind": "iterative_q_window_extension_0_720_vs_108_680",
        "model_seeds": list(model_seeds),
        "test_seeds": list(TEST_SEEDS),
        "baseline_windows_h": [108, 680],
        "full_windows_h": [0, 720],
        "baseline": _summary(baseline),
        "full_window": _summary(full_window),
        "paired_change_full_window_minus_baseline": {
            "mean_total_cost_eur": float(cost_change.mean()),
            "mean_total_cost_pct": float(
                100.0 * cost_change.mean() / baseline["total_cost_eur"].mean()
            ),
            "total_cost_ci95_eur": _hierarchical_ci(
                cost_change, rng, args.bootstrap_draws
            ),
            "mean_vented_t": float(
                (full_window["vented_t"] - baseline["vented_t"]).mean()
            ),
            "mean_stored_t": float(
                (full_window["stored_t"] - baseline["stored_t"]).mean()
            ),
            "full_window_lower_cost_pairs": int(
                np.count_nonzero(cost_change < -1e-6)
            ),
            "ties": int(np.count_nonzero(np.abs(cost_change) <= 1e-6)),
            "baseline_lower_cost_pairs": int(
                np.count_nonzero(cost_change > 1e-6)
            ),
        },
        "per_model_seed": _per_model_seed(baseline, full_window, model_seeds),
        "budgets": _budgets(args.full_window_root, model_seeds),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "analysis.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
