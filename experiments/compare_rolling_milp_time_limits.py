"""Compare paired H168-R24 Rolling MILP runs across solver time limits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


SEEDS = set(range(9_000_031, 9_000_061))
LIMITS = (300, 600, 1200)
ROLES = {
    300: "superseded archived non-single-factor baseline",
    600: "formal primary result",
    1200: "time-budget ablation",
}
METRICS = (
    ("total_cost", "Total cost", "EUR"),
    ("operating_cost", "Operating cost", "EUR"),
    ("terminal_cleanup_operating_cost", "Terminal cleanup cost", "EUR"),
    ("stored_t", "Stored CO2", "t"),
    ("vented_t", "Vented CO2", "t"),
    ("captured_t", "Captured CO2", "t"),
    ("wall_clock_seconds", "Wall-clock time", "s"),
    ("solver_solve_wall_seconds", "Solver time", "s"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for limit in LIMITS:
        parser.add_argument(f"--t{limit}-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "p95": percentile(values, 0.95),
    }


def load_run(root: Path, limit: int) -> dict[int, dict[str, object]]:
    run: dict[int, dict[str, object]] = {}
    for summary_path in sorted(root.glob("seed_*/smoke_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        config = payload["configuration"]
        replans = payload["diagnostics"]["rolling_milp_replans"]
        seed = int(row["seed"])
        actions_path = summary_path.parent / "executed_actions.json"
        actions = json.loads(actions_path.read_text(encoding="utf-8"))
        trace = actions["actions_by_controller"]["rolling_milp"]

        checks = {
            "protocol": payload["protocol"] == "unified_window_v1",
            "controller": row["controller"] == "rolling_milp",
            "run_status": row["run_status"] == "completed",
            "solver_threads": int(row["solver_threads"]) == 4,
            "time_limit": float(row["solver_time_limit_seconds_per_replan"])
            == float(limit),
            "planning_horizon": int(config["rolling_planning_horizon_hours"])
            == 168,
            "replan_interval": int(config["rolling_replan_hours"]) == 24,
            "episode_horizon": int(config["online_episode_hours"]) == 720,
            "forecast_context": int(config["forecast_context_hours"]) == 168,
            "fallback": row["fallback_used"] is False,
            "solver_failures": int(row["solver_failure_count"]) == 0,
            "replan_count": len(replans) == 30
            and int(row["solver_replan_count"]) == 30,
            "action_count": len(trace) == 720
            and int(row["executed_action_count"]) == 720,
            "solver_valid": all(item["solver_is_valid"] for item in replans),
            "execution_replay_valid": all(
                item["execution_replay_is_valid"] for item in replans
            ),
        }
        failures = [name for name, passed in checks.items() if not passed]
        if failures:
            raise RuntimeError(f"T{limit} seed {seed} failed: {failures}")
        if seed in run:
            raise RuntimeError(f"duplicate T{limit} seed {seed}")
        run[seed] = {
            "row": row,
            "replans": replans,
            "summary_path": summary_path,
            "actions_path": actions_path,
        }

    if set(run) != SEEDS:
        raise RuntimeError(
            f"T{limit} seed mismatch: missing={sorted(SEEDS - set(run))}, "
            f"extra={sorted(set(run) - SEEDS)}"
        )
    return run


def summarize(limit: int, run: dict[int, dict[str, object]]) -> dict[str, object]:
    rows = [item["row"] for item in run.values()]
    replans = [rp for item in run.values() for rp in item["replans"]]
    statuses = Counter(str(rp["status"]) for rp in replans)
    gaps = [
        float(rp["relative_gap"])
        for rp in replans
        if rp["relative_gap"] is not None
    ]
    return {
        "time_limit_seconds_per_replan": limit,
        "role": ROLES[limit],
        "seed_count": len(run),
        "replan_count": len(replans),
        "solver_status_counts": dict(sorted(statuses.items())),
        "optimal_replan_rate": statuses["Optimal"] / len(replans),
        "all_replans_optimal_seed_count": sum(
            all(rp["status"] == "Optimal" for rp in item["replans"])
            for item in run.values()
        ),
        "solver_valid_replan_count": sum(
            rp["solver_is_valid"] is True for rp in replans
        ),
        "execution_replay_valid_replan_count": sum(
            rp["execution_replay_is_valid"] is True for rp in replans
        ),
        "model_replay_inexact_count": sum(
            rp["model_replay_is_exact"] is False for rp in replans
        ),
        "relative_gap": distribution(gaps),
        "metrics": {
            key: distribution([float(row[key]) for row in rows])
            for key, _label, _unit in METRICS
        },
    }


def paired_rows(runs: dict[int, dict[int, dict[str, object]]]) -> list[dict[str, object]]:
    output = []
    for seed in sorted(SEEDS):
        row: dict[str, object] = {"seed": seed}
        for key, _label, _unit in METRICS:
            for limit in LIMITS:
                row[f"t{limit}_{key}"] = float(runs[limit][seed]["row"][key])
            row[f"t600_minus_t300_{key}"] = (
                float(row[f"t600_{key}"]) - float(row[f"t300_{key}"])
            )
            row[f"t1200_minus_t600_{key}"] = (
                float(row[f"t1200_{key}"]) - float(row[f"t600_{key}"])
            )
        output.append(row)
    return output


def outcome(delta: float) -> str:
    if delta < -1e-6:
        return "win"
    if delta > 1e-6:
        return "loss"
    return "tie"


def aggregate_rows(methods: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for limit in LIMITS:
        method = methods[limit]
        statuses = method["solver_status_counts"]
        row: dict[str, object] = {
            "time_limit_seconds_per_replan": limit,
            "role": method["role"],
            "seed_count": method["seed_count"],
            "optimal_replans": statuses.get("Optimal", 0),
            "integer_feasible_replans": statuses.get("Integer Feasible", 0),
            "optimal_replan_rate": method["optimal_replan_rate"],
            "all_replans_optimal_seed_count": method[
                "all_replans_optimal_seed_count"
            ],
            "relative_gap_mean": method["relative_gap"]["mean"],
            "relative_gap_median": method["relative_gap"]["median"],
            "relative_gap_p95": method["relative_gap"]["p95"],
            "relative_gap_maximum": method["relative_gap"]["maximum"],
        }
        for key, _label, _unit in METRICS:
            row[f"{key}_mean"] = method["metrics"][key]["mean"]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(payload: dict[str, object]) -> str:
    methods = payload["methods"]
    lines = [
        "# Rolling MILP H168-R24 time-limit comparison",
        "",
        "Paired seeds 9000031-9000060; four deterministic CPLEX threads.",
        "T300 is superseded, T600 is the formal result, and T1200 is a "
        "time-budget ablation.",
        "T300 used different runner and solver source hashes, so comparisons "
        "involving T300 are descriptive rather than single-factor time-limit "
        "effects. T600 versus T1200 is the clean time-limit comparison.",
        "",
        "## Episode-level means",
        "",
        "| Metric | T300 | T600 | T1200 |",
        "|---|---:|---:|---:|",
    ]
    for key, label, unit in METRICS:
        values = [methods[str(limit)]["metrics"][key]["mean"] for limit in LIMITS]
        lines.append(
            f"| {label} ({unit}) | {values[0]:.6g} | {values[1]:.6g} "
            f"| {values[2]:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Solver proof status",
            "",
            "| Limit | Optimal | Integer feasible | Optimal rate | "
            "30/30 optimal seeds | Mean gap | P95 gap |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for limit in LIMITS:
        method = methods[str(limit)]
        statuses = method["solver_status_counts"]
        lines.append(
            f"| {limit}s | {statuses.get('Optimal', 0)} "
            f"| {statuses.get('Integer Feasible', 0)} "
            f"| {100 * method['optimal_replan_rate']:.2f}% "
            f"| {method['all_replans_optimal_seed_count']} "
            f"| {100 * method['relative_gap']['mean']:.2f}% "
            f"| {100 * method['relative_gap']['p95']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "All three runs have 900/900 valid solver results and 900/900 "
            "valid execution replays.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    if args.out_dir.exists() and not args.overwrite:
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    roots = {limit: getattr(args, f"t{limit}_root") for limit in LIMITS}
    runs = {limit: load_run(roots[limit], limit) for limit in LIMITS}
    methods = {limit: summarize(limit, runs[limit]) for limit in LIMITS}
    paired = paired_rows(runs)
    aggregates = aggregate_rows(methods)

    paired_cost_outcomes = {}
    for limit in (300, 1200):
        counts = Counter(
            outcome(
                float(runs[limit][seed]["row"]["total_cost"])
                - float(runs[600][seed]["row"]["total_cost"])
            )
            for seed in SEEDS
        )
        paired_cost_outcomes[f"t{limit}_vs_t600"] = {
            "wins": counts["win"],
            "ties": counts["tie"],
            "losses": counts["loss"],
        }

    manifest = []
    for limit in LIMITS:
        for seed in sorted(SEEDS):
            item = runs[limit][seed]
            manifest.append(
                {
                    "time_limit_seconds_per_replan": limit,
                    "seed": seed,
                    "summary_path": str(item["summary_path"]),
                    "summary_sha256": sha256(item["summary_path"]),
                    "actions_path": str(item["actions_path"]),
                    "actions_sha256": sha256(item["actions_path"]),
                }
            )

    payload = {
        "comparison": "Rolling MILP H168-R24 T300/T600/T1200",
        "paired_seed_range_inclusive": [9_000_031, 9_000_060],
        "comparability": {
            "t600_vs_t1200_single_factor_time_limit_comparison": True,
            "t300_vs_later_single_factor_time_limit_comparison": False,
            "t300_caveat": (
                "The superseded T300 run used different runner and solver "
                "source hashes. Its table entries are descriptive only."
            ),
        },
        "methods": {str(limit): methods[limit] for limit in LIMITS},
        "paired_total_cost_outcomes_against_formal_t600": paired_cost_outcomes,
        "validation": {
            "method_count": 3,
            "seed_count_per_method": 30,
            "total_episode_rows": 90,
            "total_replans": 2700,
            "source_manifest_rows": len(manifest),
        },
    }

    aggregate_path = args.out_dir / "aggregate_comparison.csv"
    paired_path = args.out_dir / "paired_per_seed.csv"
    manifest_path = args.out_dir / "source_manifest.csv"
    summary_path = args.out_dir / "summary.json"
    readme_path = args.out_dir / "README.md"
    write_csv(aggregate_path, aggregates)
    write_csv(paired_path, paired)
    write_csv(manifest_path, manifest)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    readme_path.write_text(markdown(payload), encoding="utf-8")
    audited = (aggregate_path, paired_path, manifest_path, summary_path, readme_path)
    (args.out_dir / "audit.json").write_text(
        json.dumps(
            {"output_sha256": {path.name: sha256(path) for path in audited}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return args.out_dir


if __name__ == "__main__":
    print(run(parse_args()))
