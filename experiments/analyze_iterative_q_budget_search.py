"""Rank fixed-simulator-budget Iterative-Q runs on mean and tail validation risk."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXPECTED_SEEDS = list(range(8_100_001, 8_100_021))
LARGE_REGRESSION_EUR = 100_000.0


def _schedule(path: Path) -> dict[str, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            rows[key] = value
    return rows


def _rows(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = {
        int(row["seed"]): {
            key: float(value)
            for key, value in row.items()
            if key not in {"gate", "seed"}
        }
        for row in raw
    }
    if sorted(rows) != EXPECTED_SEEDS:
        raise ValueError(f"{path}: unexpected validation seeds")
    return rows


def _values(
    rows: dict[int, dict[str, float]], field: str
) -> np.ndarray:
    return np.asarray([rows[seed][field] for seed in EXPECTED_SEEDS])


def _evaluation_metrics(
    rows: dict[int, dict[str, float]]
) -> dict[str, object]:
    delta = _values(rows, "delta_total_cost_eur")
    worst_order = np.argsort(delta)[::-1]
    worst_index = int(worst_order[0])
    greedy = _values(rows, "greedy_total_cost_eur")
    return {
        "mean_total_cost_eur": float(
            _values(rows, "total_cost_eur").mean()
        ),
        "mean_greedy_total_cost_eur": float(greedy.mean()),
        "mean_delta_vs_greedy_eur": float(delta.mean()),
        "relative_saving_vs_greedy_pct": float(
            -100.0 * delta.mean() / greedy.mean()
        ),
        "median_delta_vs_greedy_eur": float(np.median(delta)),
        "wins_vs_greedy": int((delta < -1e-6).sum()),
        "ties_vs_greedy": int((np.abs(delta) <= 1e-6).sum()),
        "losses_vs_greedy": int((delta > 1e-6).sum()),
        "large_regressions_vs_greedy": int(
            (delta > LARGE_REGRESSION_EUR).sum()
        ),
        "worst_delta_vs_greedy_eur": float(delta[worst_index]),
        "worst_seed_vs_greedy": EXPECTED_SEEDS[worst_index],
        "worst4_cvar_delta_vs_greedy_eur": float(
            delta[worst_order[:4]].mean()
        ),
        "mean_operating_cost_eur": float(
            _values(rows, "operating_cost_eur").mean()
        ),
        "mean_vent_penalty_eur": float(
            _values(rows, "vent_penalty_eur").mean()
        ),
        "mean_vented_t": float(_values(rows, "vented_t").mean()),
        "mean_stored_t": float(_values(rows, "stored_t").mean()),
        "mean_unit_cost_eur_per_t": float(
            _values(rows, "unit_cost_eur_per_t").mean()
        ),
        "mean_override_events": float(
            _values(rows, "override_events").mean()
        ),
    }


def _retention(
    final: dict[int, dict[str, float]],
    previous: dict[int, dict[str, float]],
) -> dict[str, object]:
    difference = (
        _values(final, "total_cost_eur")
        - _values(previous, "total_cost_eur")
    )
    worst = int(np.argmax(difference))
    return {
        "mean_final_minus_previous_eur": float(difference.mean()),
        "final_better_seeds": int((difference < -1e-6).sum()),
        "ties": int((np.abs(difference) <= 1e-6).sum()),
        "previous_better_seeds": int((difference > 1e-6).sum()),
        "worst_regression_from_previous_eur": float(
            difference[worst]
        ),
        "worst_regression_seed": EXPECTED_SEEDS[worst],
        "large_regressions_from_previous": int(
            (difference > LARGE_REGRESSION_EUR).sum()
        ),
    }


def _pareto_flags(rows: list[dict[str, object]]) -> None:
    fields = (
        "mean_total_cost_eur",
        "worst_delta_vs_greedy_eur",
        "worst4_cvar_delta_vs_greedy_eur",
    )
    for row in rows:
        row["pareto_mean_tail"] = not any(
            all(float(other[field]) <= float(row[field]) for field in fields)
            and any(
                float(other[field]) < float(row[field])
                for field in fields
            )
            for other in rows
            if other is not row
        )


def analyze(search_root: Path) -> dict[str, object]:
    results = []
    for run_root in sorted((search_root / "runs").iterdir()):
        if not run_root.is_dir():
            continue
        schedule = _schedule(run_root / "schedule.txt")
        name = schedule["config"]
        final_stage = schedule["final_stage"]
        evaluation = _rows(
            run_root / "eval" / name / "evaluation.csv"
        )
        budget = json.loads(
            (run_root / "budget.json").read_text(encoding="utf-8")
        )
        totals = budget["totals"]
        metrics = _evaluation_metrics(evaluation)
        result = {
            "config": name,
            "final_stage": final_stage,
            "g0_train_count": int(schedule["g0_train_count"]),
            "nominal_train_roots": int(schedule["weighted_train_roots"]),
            "actual_train_roots": int(totals["train_roots"]),
            "train_simulator_steps": int(
                totals["train_simulator_steps"]
            ),
            "train_step_relative_error_pct": float(
                totals["train_step_relative_error_pct"]
            ),
            "actual_g0_budget_share_pct": float(
                100.0
                * budget["train"][0]["simulator_step_calls"]
                / totals["train_simulator_steps"]
            ),
            **metrics,
        }
        final_index = int(final_stage[1:])
        all_prior_retention = {}
        for prior_index in range(1, final_index):
            prior_name = f"{name}_p{prior_index}"
            prior_path = (
                run_root
                / "eval"
                / prior_name
                / "evaluation.csv"
            )
            all_prior_retention[f"p{prior_index}"] = _retention(
                evaluation, _rows(prior_path)
            )
        previous_index = final_index - 1
        result["retention"] = all_prior_retention[
            f"p{previous_index}"
        ]
        result["all_prior_retention"] = all_prior_retention
        retention = result["retention"]
        result["stage_retention_feasible"] = (
            all(
                item["large_regressions_from_previous"] == 0
                for item in all_prior_retention.values()
            )
        )
        result["mean_final_minus_previous_eur"] = retention[
            "mean_final_minus_previous_eur"
        ]
        result["previous_better_seeds"] = retention[
            "previous_better_seeds"
        ]
        result["worst_regression_from_previous_eur"] = retention[
            "worst_regression_from_previous_eur"
        ]
        result["large_regressions_from_previous"] = retention[
            "large_regressions_from_previous"
        ]
        worst_prior_stage, worst_prior = max(
            all_prior_retention.items(),
            key=lambda item: item[1][
                "worst_regression_from_previous_eur"
            ],
        )
        result["worst_regression_from_any_prior_eur"] = worst_prior[
            "worst_regression_from_previous_eur"
        ]
        result["worst_regression_from_any_prior_seed"] = worst_prior[
            "worst_regression_seed"
        ]
        result["worst_regression_prior_stage"] = worst_prior_stage
        result["large_regressions_from_any_prior"] = sum(
            int(item["large_regressions_from_previous"])
            for item in all_prior_retention.values()
        )
        result["budget_within_8pct"] = (
            abs(result["train_step_relative_error_pct"]) <= 8.0
        )
        result["tail_feasible"] = (
            result["large_regressions_vs_greedy"] == 0
        )
        result["constrained_feasible"] = (
            result["budget_within_8pct"]
            and result["tail_feasible"]
            and result["stage_retention_feasible"]
        )
        results.append(result)

    if not results:
        raise ValueError(f"{search_root}: no completed runs")
    _pareto_flags(results)
    eligible = [
        row
        for row in results
        if row["constrained_feasible"]
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            float(row["mean_total_cost_eur"]),
            float(row["worst4_cvar_delta_vs_greedy_eur"]),
            float(row["worst_delta_vs_greedy_eur"]),
        ),
    )
    return {
        "kind": "iterative_q_budget_search_validation_analysis",
        "validation_seeds": EXPECTED_SEEDS,
        "formal_test_accessed": False,
        "target_train_simulator_steps": 9_525_119,
        "large_regression_threshold_eur": LARGE_REGRESSION_EUR,
        "configuration_count": len(results),
        "tail_feasible_count": sum(
            bool(row["tail_feasible"]) for row in results
        ),
        "stage_retention_feasible_count": sum(
            bool(row["stage_retention_feasible"]) for row in results
        ),
        "constrained_feasible_count": len(eligible),
        "recommended_by_constrained_mean": (
            ranked[0]["config"] if ranked else None
        ),
        "results": results,
    }


def _write_csv(path: Path, payload: dict[str, object]) -> None:
    fields = [
        "config",
        "final_stage",
        "g0_train_count",
        "nominal_train_roots",
        "actual_train_roots",
        "train_simulator_steps",
        "train_step_relative_error_pct",
        "actual_g0_budget_share_pct",
        "mean_total_cost_eur",
        "mean_delta_vs_greedy_eur",
        "relative_saving_vs_greedy_pct",
        "wins_vs_greedy",
        "losses_vs_greedy",
        "large_regressions_vs_greedy",
        "worst_delta_vs_greedy_eur",
        "worst_seed_vs_greedy",
        "worst4_cvar_delta_vs_greedy_eur",
        "stage_retention_feasible",
        "mean_final_minus_previous_eur",
        "previous_better_seeds",
        "worst_regression_from_previous_eur",
        "large_regressions_from_previous",
        "worst_regression_from_any_prior_eur",
        "worst_regression_from_any_prior_seed",
        "worst_regression_prior_stage",
        "large_regressions_from_any_prior",
        "constrained_feasible",
        "mean_vented_t",
        "mean_stored_t",
        "mean_unit_cost_eur_per_t",
        "mean_override_events",
        "pareto_mean_tail",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["results"]:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.search_root)
    args.output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_csv, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
