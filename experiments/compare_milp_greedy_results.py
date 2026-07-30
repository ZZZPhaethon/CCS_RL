"""Compare formal Greedy, Rolling MILP, and Full MILP results by seed."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path


EXPECTED_SEEDS = set(range(9_000_031, 9_000_061))
METRICS = (
    "total_cost",
    "operating_cost",
    "total_cost_per_stored_t",
    "stored_t",
    "vented_t",
    "terminal_cleanup_operating_cost",
    "wall_clock_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--greedy-root", type=Path, required=True)
    parser.add_argument("--rolling-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _load_rows(root: Path, expected_controller: str) -> list[dict]:
    rows = []
    for path in sorted(root.glob("seed_*/smoke_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        row = summary["rows"][0]
        if row["controller"] != expected_controller:
            raise RuntimeError(f"unexpected controller in {path}: {row['controller']}")
        if row["run_status"] != "completed":
            raise RuntimeError(f"incomplete run in {path}: {row['run_status']}")
        rows.append(row)
    seeds = [int(row["seed"]) for row in rows]
    if len(rows) != 30 or set(seeds) != EXPECTED_SEEDS:
        raise RuntimeError(
            f"incomplete {expected_controller} results at {root}: "
            f"rows={len(rows)}, missing={sorted(EXPECTED_SEEDS - set(seeds))}"
        )
    return sorted(rows, key=lambda row: int(row["seed"]))


def _validate_greedy_artifacts(root: Path, rows: list[dict]) -> None:
    for row in rows:
        seed = int(row["seed"])
        seed_dir = root / f"seed_{seed}"
        actions = json.loads(
            (seed_dir / "executed_actions.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (seed_dir / "formal_task_metadata.json").read_text(encoding="utf-8")
        )
        trace = actions["actions_by_controller"]["greedy"]
        if len(trace) != 720 or int(row["executed_action_count"]) != 720:
            raise RuntimeError(f"invalid Greedy action trace for seed {seed}")
        if metadata["run_status"] != "completed" or int(metadata["seed"]) != seed:
            raise RuntimeError(f"invalid Greedy metadata for seed {seed}")
        if not bool(row["terminal_cleanup_included"]):
            raise RuntimeError(f"terminal cleanup missing for Greedy seed {seed}")


def _stats(values) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "standard_deviation": statistics.stdev(numbers),
        "median": statistics.median(numbers),
        "minimum": min(numbers),
        "maximum": max(numbers),
    }


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int = 20_000,
    seed: int = 20_260_729,
) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    return [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]]


def _paired_rows(
    baseline_rows: list[dict],
    method_rows: list[dict],
    method: str,
    scope: str,
) -> list[dict]:
    baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}
    method_by_seed = {int(row["seed"]): row for row in method_rows}
    rows = []
    for seed in sorted(EXPECTED_SEEDS):
        baseline = baseline_by_seed[seed]
        candidate = method_by_seed[seed]
        baseline_cost = float(baseline["total_cost"])
        candidate_cost = float(candidate["total_cost"])
        rows.append(
            {
                "seed": seed,
                "method": method,
                "comparison_scope": scope,
                "greedy_total_cost": baseline_cost,
                "method_total_cost": candidate_cost,
                "method_minus_greedy_total_cost": candidate_cost - baseline_cost,
                "cost_improvement_vs_greedy_percent": (
                    100.0 * (baseline_cost - candidate_cost) / baseline_cost
                ),
                "greedy_stored_t": baseline["stored_t"],
                "method_stored_t": candidate["stored_t"],
                "method_minus_greedy_stored_t": (
                    float(candidate["stored_t"]) - float(baseline["stored_t"])
                ),
                "greedy_vented_t": baseline["vented_t"],
                "method_vented_t": candidate["vented_t"],
                "method_minus_greedy_vented_t": (
                    float(candidate["vented_t"]) - float(baseline["vented_t"])
                ),
            }
        )
    return rows


def _paired_summary(rows: list[dict]) -> dict:
    differences = [
        float(row["method_minus_greedy_total_cost"]) for row in rows
    ]
    improvements = [
        float(row["cost_improvement_vs_greedy_percent"]) for row in rows
    ]
    tolerance = 1e-6
    return {
        "method_minus_greedy_total_cost": {
            **_stats(differences),
            "bootstrap_95_percent_ci_for_mean": _bootstrap_mean_ci(differences),
        },
        "cost_improvement_vs_greedy_percent": {
            **_stats(improvements),
            "bootstrap_95_percent_ci_for_mean": _bootstrap_mean_ci(
                improvements,
                seed=20_260_730,
            ),
        },
        "lower_cost_seed_count": sum(value < -tolerance for value in differences),
        "equal_cost_seed_count": sum(abs(value) <= tolerance for value in differences),
        "higher_cost_seed_count": sum(value > tolerance for value in differences),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _method_summary(rows: list[dict]) -> dict:
    return {
        "episode_count": len(rows),
        "metrics": {
            metric: _stats(row[metric] for row in rows)
            for metric in METRICS
        },
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# Formal Greedy and MILP comparison",
        "",
        "Seeds: 9000031–9000060 (n=30), protocol: unified_window_v1, "
        "720 h execution, terminal cleanup included.",
        "Primary MILP budgets: Rolling 600 s/replan; Full MILP 18,000 s/seed.",
        "",
        "| Method | Role | Total cost, mean ± SD | Stored t, mean ± SD | "
        "Vented t, mean ± SD | Wall time/seed, mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    roles = {
        "greedy": "online baseline",
        "rolling_milp": "online controller",
        "full_milp": "offline perfect-information reference",
    }
    for method in ("greedy", "rolling_milp", "full_milp"):
        metrics = payload["methods"][method]["metrics"]
        lines.append(
            f"| {method} | {roles[method]} | "
            f"{metrics['total_cost']['mean']:,.2f} ± "
            f"{metrics['total_cost']['standard_deviation']:,.2f} | "
            f"{metrics['stored_t']['mean']:,.2f} ± "
            f"{metrics['stored_t']['standard_deviation']:,.2f} | "
            f"{metrics['vented_t']['mean']:,.2f} ± "
            f"{metrics['vented_t']['standard_deviation']:,.2f} | "
            f"{metrics['wall_clock_seconds']['mean']:,.2f} s |"
        )
    lines.extend(
        [
            "",
            "| Paired comparison vs Greedy | Mean cost improvement | "
            "95% bootstrap CI | Lower/equal/higher cost seeds |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("rolling_milp_vs_greedy", "Rolling MILP"),
        ("full_milp_vs_greedy", "Full MILP"),
    ):
        summary = payload["paired_comparisons"][key]
        improvement = summary["cost_improvement_vs_greedy_percent"]
        ci = improvement["bootstrap_95_percent_ci_for_mean"]
        lines.append(
            f"| {label} | {improvement['mean']:.3f}% | "
            f"[{ci[0]:.3f}%, {ci[1]:.3f}%] | "
            f"{summary['lower_cost_seed_count']}/"
            f"{summary['equal_cost_seed_count']}/"
            f"{summary['higher_cost_seed_count']} |"
        )
    lines.extend(
        [
            "",
            "Rolling MILP vs Greedy is the valid online paired comparison. "
            "Full MILP used the complete future trajectory and all 30 runs ended "
            "with time-limited feasible incumbents rather than proven optima, so "
            "its comparison is descriptive only.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = (
        args.out_dir / "greedy_per_seed.csv",
        args.out_dir / "rolling_vs_greedy_paired.csv",
        args.out_dir / "full_vs_greedy_descriptive.csv",
        args.out_dir / "comparison_summary.json",
        args.out_dir / "comparison_summary.md",
    )
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")

    greedy = _load_rows(args.greedy_root, "greedy")
    rolling = _load_rows(args.rolling_root, "rolling_milp")
    full = _load_rows(args.full_root, "full_milp")
    _validate_greedy_artifacts(args.greedy_root, greedy)

    rolling_paired = _paired_rows(
        greedy,
        rolling,
        "rolling_milp",
        "valid online paired comparison",
    )
    full_paired = _paired_rows(
        greedy,
        full,
        "full_milp",
        "descriptive offline perfect-information comparison only",
    )
    payload = {
        "protocol": "unified_window_v1",
        "seed_range_inclusive": [9_000_031, 9_000_060],
        "seed_count": 30,
        "terminal_cleanup_included": True,
        "compute_budget": {
            "rolling_time_limit_seconds_per_replan": 600,
            "full_time_limit_seconds_per_seed": 18000,
            "solver_threads_per_process": 4,
        },
        "provenance": {
            "primary_result_status": "extended_budget_run03_promoted_on_2026-07-30",
            "superseded_results_retained": True,
            "single_factor_time_limit_comparison": False,
            "source_hashes_differ_from_superseded_run": True,
        },
        "methods": {
            "greedy": _method_summary(greedy),
            "rolling_milp": _method_summary(rolling),
            "full_milp": _method_summary(full),
        },
        "paired_comparisons": {
            "rolling_milp_vs_greedy": _paired_summary(rolling_paired),
            "full_milp_vs_greedy": _paired_summary(full_paired),
        },
        "interpretation": {
            "rolling_milp_vs_greedy": "valid online paired comparison",
            "full_milp_vs_greedy": (
                "descriptive only: Full MILP has perfect future information and "
                "all 30 solves produced time-limited feasible incumbents, not "
                "proven optima"
            ),
        },
    }

    _write_csv(targets[0], greedy)
    _write_csv(targets[1], rolling_paired)
    _write_csv(targets[2], full_paired)
    targets[3].write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    targets[4].write_text(_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    run(parse_args())
