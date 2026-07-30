"""Compare Iterative-Q checkpoint routers on locked validation seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXPECTED_SEEDS = list(range(8_100_001, 8_100_021))
DEVELOPMENT_SEEDS = EXPECTED_SEEDS[:10]
CONFIRMATION_SEEDS = EXPECTED_SEEDS[10:]


def _read_rows(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = {
        int(row["seed"]): {
            key: float(value)
            for key, value in row.items()
            if key
            in {
                "total_cost_eur",
                "greedy_total_cost_eur",
                "delta_total_cost_eur",
                "vented_t",
                "stored_t",
                "override_events",
            }
        }
        for row in raw
    }
    if sorted(rows) != EXPECTED_SEEDS:
        raise ValueError(f"{path}: unexpected validation seeds")
    return rows


def _values(
    rows: dict[int, dict[str, float]],
    field: str,
    seeds: list[int],
) -> np.ndarray:
    return np.asarray([rows[seed][field] for seed in seeds])


def _paired_interval(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(0)
    means = values[
        rng.integers(0, len(values), size=(10_000, len(values)))
    ].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _metrics(
    rows: dict[int, dict[str, float]],
    reference: dict[int, dict[str, float]],
) -> dict[str, object]:
    costs = _values(rows, "total_cost_eur", EXPECTED_SEEDS)
    greedy_delta = _values(
        rows, "delta_total_cost_eur", EXPECTED_SEEDS
    )
    reference_costs = _values(
        reference, "total_cost_eur", EXPECTED_SEEDS
    )
    paired = costs - reference_costs
    worst_order = np.argsort(greedy_delta)[::-1]
    development = costs[:10] - reference_costs[:10]
    confirmation = costs[10:] - reference_costs[10:]
    return {
        "mean_total_cost_eur": float(costs.mean()),
        "mean_delta_vs_greedy_eur": float(greedy_delta.mean()),
        "worst4_cvar_delta_vs_greedy_eur": float(
            greedy_delta[worst_order[:4]].mean()
        ),
        "mean_difference_vs_p4_eur": float(paired.mean()),
        "mean_difference_vs_p4_95pct_ci_eur": _paired_interval(paired),
        "better_than_p4_seeds": int((paired < -1e-6).sum()),
        "ties_with_p4": int((np.abs(paired) <= 1e-6).sum()),
        "worse_than_p4_seeds": int((paired > 1e-6).sum()),
        "worst_regression_vs_p4_eur": float(paired.max()),
        "largest_improvement_vs_p4_eur": float(-paired.min()),
        "development_mean_difference_vs_p4_eur": float(
            development.mean()
        ),
        "confirmation_mean_difference_vs_p4_eur": float(
            confirmation.mean()
        ),
        "confirmation_improves_mean": bool(confirmation.mean() < 0.0),
        "confirmation_worst_regression_vs_p4_eur": float(
            confirmation.max()
        ),
        "mean_vented_t": float(
            _values(rows, "vented_t", EXPECTED_SEEDS).mean()
        ),
        "mean_stored_t": float(
            _values(rows, "stored_t", EXPECTED_SEEDS).mean()
        ),
        "mean_override_events": float(
            _values(rows, "override_events", EXPECTED_SEEDS).mean()
        ),
    }


def checkpoint_oracle(checkpoint_root: Path) -> dict[str, object]:
    evaluations = {
        path.stem: _read_rows(path)
        for path in sorted(checkpoint_root.glob("*.csv"))
    }
    if not evaluations:
        raise ValueError(f"{checkpoint_root}: no checkpoint evaluations")
    costs = {
        name: _values(rows, "total_cost_eur", EXPECTED_SEEDS)
        for name, rows in evaluations.items()
    }
    names = list(costs)
    matrix = np.stack([costs[name] for name in names], axis=1)
    winners = matrix.argmin(axis=1)
    oracle_costs = matrix.min(axis=1)
    reference = costs["p4"]
    return {
        "checkpoint_mean_cost_eur": {
            name: float(values.mean())
            for name, values in costs.items()
        },
        "oracle_mean_cost_eur": float(oracle_costs.mean()),
        "oracle_gain_vs_p4_eur": float(
            reference.mean() - oracle_costs.mean()
        ),
        "oracle_gain_vs_p4_pct": float(
            100.0
            * (reference.mean() - oracle_costs.mean())
            / reference.mean()
        ),
        "oracle_winner_counts": {
            name: int((winners == index).sum())
            for index, name in enumerate(names)
        },
    }


def analyze(
    result_root: Path,
    reference_name: str = "p4_reference",
    checkpoint_root: Path | None = None,
) -> dict[str, object]:
    evaluations = {
        path.parent.name: _read_rows(path)
        for path in sorted(result_root.glob("*/evaluation.csv"))
    }
    if reference_name not in evaluations:
        raise ValueError(f"missing reference router: {reference_name}")
    reference = evaluations[reference_name]
    results = [
        {
            "router": name,
            **_metrics(rows, reference),
        }
        for name, rows in evaluations.items()
    ]
    results.sort(
        key=lambda row: (
            float(row["confirmation_mean_difference_vs_p4_eur"]),
            float(row["mean_difference_vs_p4_eur"]),
        )
    )
    payload = {
        "kind": "iterative_q_v3_router_validation_analysis",
        "validation_seeds": EXPECTED_SEEDS,
        "development_seeds": DEVELOPMENT_SEEDS,
        "confirmation_seeds": CONFIRMATION_SEEDS,
        "formal_test_accessed": False,
        "reference": reference_name,
        "results": results,
    }
    if checkpoint_root is not None:
        payload["episode_checkpoint_oracle"] = checkpoint_oracle(
            checkpoint_root
        )
    return payload


def _write_csv(path: Path, payload: dict[str, object]) -> None:
    rows = payload["results"]
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--reference", default="p4_reference")
    parser.add_argument("--checkpoint-root", type=Path)
    args = parser.parse_args()
    payload = analyze(
        args.result_root,
        args.reference,
        args.checkpoint_root,
    )
    args.output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_csv, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
