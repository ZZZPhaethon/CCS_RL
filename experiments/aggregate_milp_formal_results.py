"""Aggregate the formal Rolling and Full MILP per-seed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


EXPECTED_SEEDS = set(range(9_000_031, 9_000_061))
METRICS = (
    "total_cost",
    "operating_cost",
    "total_cost_per_stored_t",
    "cost_per_stored_t",
    "stored_t",
    "vented_t",
    "captured_t",
    "in_transit_t",
    "terminal_cleanup_operating_cost",
    "wall_clock_seconds",
    "solver_solve_wall_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _read_method(root: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    summaries: list[dict] = []
    for summary_path in sorted(root.glob("seed_*/smoke_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary)
        rows.append(summary["rows"][0])
    seeds = [int(row["seed"]) for row in rows]
    if len(rows) != 30 or len(set(seeds)) != 30 or set(seeds) != EXPECTED_SEEDS:
        raise RuntimeError(
            f"incomplete formal results at {root}: "
            f"rows={len(rows)}, unique={len(set(seeds))}, "
            f"missing={sorted(EXPECTED_SEEDS - set(seeds))}"
        )
    return sorted(rows, key=lambda row: int(row["seed"])), summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_diagnostic(seed: int, item: dict) -> dict:
    row = {"seed": seed}
    for key, value in item.items():
        row[key] = (
            json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
        )
    return row


def _numeric(values) -> list[float]:
    result = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values) -> dict[str, float | int | None]:
    numbers = _numeric(values)
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "p95": None,
        }
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "standard_deviation": (
            statistics.stdev(numbers) if len(numbers) > 1 else 0.0
        ),
        "median": statistics.median(numbers),
        "minimum": min(numbers),
        "maximum": max(numbers),
        "p95": _percentile(numbers, 0.95),
    }


def _metric_summary(rows: list[dict]) -> dict[str, dict]:
    return {
        metric: _stats(row.get(metric) for row in rows)
        for metric in METRICS
    }


def _paired_rows(rolling_rows: list[dict], full_rows: list[dict]) -> list[dict]:
    rolling_by_seed = {int(row["seed"]): row for row in rolling_rows}
    full_by_seed = {int(row["seed"]): row for row in full_rows}
    result = []
    for seed in sorted(EXPECTED_SEEDS):
        rolling = rolling_by_seed[seed]
        full = full_by_seed[seed]
        rolling_cost = rolling.get("total_cost")
        full_cost = full.get("total_cost")
        result.append(
            {
                "seed": seed,
                "rolling_run_status": rolling.get("run_status"),
                "full_run_status": full.get("run_status"),
                "rolling_total_cost": rolling_cost,
                "full_replay_total_cost": full_cost,
                "full_minus_rolling_total_cost": (
                    float(full_cost) - float(rolling_cost)
                    if full_cost is not None and rolling_cost is not None
                    else None
                ),
                "rolling_stored_t": rolling.get("stored_t"),
                "full_stored_t": full.get("stored_t"),
                "rolling_vented_t": rolling.get("vented_t"),
                "full_vented_t": full.get("vented_t"),
                "comparison_scope": (
                    "descriptive only: Rolling is an online controller; "
                    "Full MILP is a time-limited perfect-information offline reference"
                ),
            }
        )
    return result


def run(args: argparse.Namespace) -> Path:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rolling_rows, rolling_summaries = _read_method(args.rolling_root)
    full_rows, full_summaries = _read_method(args.full_root)

    rolling_replans = [
        _flatten_diagnostic(
            int(summary["rows"][0]["seed"]),
            diagnostic,
        )
        for summary in rolling_summaries
        for diagnostic in summary["diagnostics"]["rolling_milp_replans"]
    ]
    full_diagnostics = [
        _flatten_diagnostic(
            int(summary["rows"][0]["seed"]),
            diagnostic,
        )
        for summary in full_summaries
        for diagnostic in summary["diagnostics"]["full_milp_stages"]
    ]

    _write_csv(args.out_dir / "rolling_milp_per_seed.csv", rolling_rows)
    _write_csv(args.out_dir / "full_milp_per_seed.csv", full_rows)
    _write_csv(
        args.out_dir / "rolling_replan_diagnostics.csv",
        rolling_replans,
    )
    _write_csv(
        args.out_dir / "full_solver_diagnostics.csv",
        full_diagnostics,
    )
    _write_csv(
        args.out_dir / "paired_replay_metrics.csv",
        _paired_rows(rolling_rows, full_rows),
    )

    rolling_statuses = Counter(str(row["run_status"]) for row in rolling_rows)
    full_statuses = Counter(str(row["run_status"]) for row in full_rows)
    rolling_solver_statuses = Counter(
        str(item.get("status")) for item in rolling_replans
    )
    full_solver_statuses = Counter(
        str(row.get("solver_status")) for row in full_rows
    )
    rolling_warm_starts = [
        bool(item["warm_start_accepted"])
        for item in rolling_replans
        if item.get("warm_start_accepted") is not None
    ]
    full_warm_starts = [
        bool(row["solver_warm_start_accepted"])
        for row in full_rows
        if row.get("solver_warm_start_accepted") is not None
    ]

    payload = {
        "protocol": "unified_window_v1",
        "formal_attempt": "extended_budget_run03",
        "seed_range_inclusive": [9_000_031, 9_000_060],
        "seed_count": 30,
        "compute_budget": {
            "rolling_time_limit_seconds_per_replan": 600,
            "full_time_limit_seconds_per_seed": 18000,
            "solver_threads_per_process": 4,
        },
        "provenance": {
            "primary_result_status": "promoted_on_2026-07-30",
            "superseded_budgets": {
                "rolling_time_limit_seconds_per_replan": 300,
                "full_time_limit_seconds_per_seed": 7200,
            },
            "superseded_results_retained": True,
            "single_factor_time_limit_comparison": False,
            "source_hashes_differ_from_superseded_run": True,
        },
        "rolling_milp": {
            "evaluation_role": "online_controller",
            "run_status_counts": dict(rolling_statuses),
            "solver_status_counts_across_replans": dict(
                rolling_solver_statuses
            ),
            "episode_count": len(rolling_rows),
            "replan_count": len(rolling_replans),
            "solver_failure_count": sum(
                int(row.get("solver_failure_count", 0))
                for row in rolling_rows
            ),
            "solver_timeout_count": sum(
                int(row.get("solver_timeout_count", 0))
                for row in rolling_rows
            ),
            "warm_start_acceptance": {
                "accepted": sum(rolling_warm_starts),
                "observed": len(rolling_warm_starts),
                "rate": (
                    statistics.fmean(rolling_warm_starts)
                    if rolling_warm_starts
                    else None
                ),
            },
            "relative_gap": _stats(
                item.get("relative_gap") for item in rolling_replans
            ),
            "solve_wall_seconds_per_replan": _stats(
                item.get("solve_wall_s") for item in rolling_replans
            ),
            "metrics": _metric_summary(rolling_rows),
        },
        "full_milp": {
            "evaluation_role": "offline_reference",
            "online_comparable": False,
            "run_status_counts": dict(full_statuses),
            "solver_status_counts": dict(full_solver_statuses),
            "episode_count": len(full_rows),
            "valid_incumbent_count": sum(
                bool(row.get("solver_is_valid")) for row in full_rows
            ),
            "replay_executable_count": sum(
                bool(row.get("replay_is_executable")) for row in full_rows
            ),
            "optimal_count": sum(
                str(row.get("solver_status")).lower() == "optimal"
                for row in full_rows
            ),
            "time_limit_count": sum(
                "time limit" in str(
                    row.get("solver_termination_reason", "")
                ).lower()
                for row in full_rows
            ),
            "warm_start_acceptance": {
                "accepted": sum(full_warm_starts),
                "observed": len(full_warm_starts),
                "rate": (
                    statistics.fmean(full_warm_starts)
                    if full_warm_starts
                    else None
                ),
            },
            "relative_gap": _stats(
                row.get("solver_relative_gap") for row in full_rows
            ),
            "metrics": _metric_summary(full_rows),
        },
        "interpretation": (
            "Rolling MILP is an online controller. Full MILP is a "
            "time-limited perfect-information offline reference; completed "
            "does not imply proven optimality."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "README.md").write_text(
        """# Primary formal MILP results: seeds 9000031-9000060

This directory contains analysis-ready tables derived from the promoted
extended-budget run03: 600 s per Rolling replan and 18,000 s per Full MILP
seed, with four deterministic CPLEX threads per process.

- `rolling_milp_per_seed.csv`: 30 Rolling MILP episode rows.
- `full_milp_per_seed.csv`: 30 time-limited Full MILP rows.
- `rolling_replan_diagnostics.csv`: all 900 Rolling replans.
- `full_solver_diagnostics.csv`: Full MILP solve and MIP-start diagnostics.
- `paired_replay_metrics.csv`: same-seed descriptive replay metrics.
- `summary.json`: completion, solver, gap, warm-start and metric summaries.
- `comparison_summary.json`: paired comparisons against Greedy.
- `comparison_summary.md`: concise human-readable Greedy comparison.

Rolling MILP is an online controller. Full MILP uses perfect information and is
an offline reference, so its paired table is descriptive rather than a direct
online-controller ranking. A completed Full MILP row means that a valid
incumbent was replayed; it does not necessarily mean optimality was proven.

The configuration and job lock is stored in the sibling directory
`../milp_extended_600s_18000s_9000031_9000060_run03_lock/`.

The complete per-seed source artifacts, including action trajectories, remain in
`../E1/algorithms/formal_rolling_milp_h168_r24_t600s_cplex222_seeds_9000031-9000060_run03/` and
`../E5/formal_full_milp_h720_t18000s_cplex222_seeds_9000031-9000060_run03/`.

The superseded 300 s/7,200 s artifacts are retained for provenance. Their
runner and solver source hashes differ from run03, so the old/new comparison
must not be interpreted as a single-factor time-limit experiment.
""",
        encoding="utf-8",
    )
    return args.out_dir


if __name__ == "__main__":
    print(run(parse_args()))
