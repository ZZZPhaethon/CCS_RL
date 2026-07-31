"""Compare paired Rolling MILP horizon runs on the same formal test seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


EXPECTED_SEEDS = set(range(9_000_031, 9_000_061))
METRICS = (
    ("total_cost", "Total cost", "EUR"),
    ("operating_cost", "Operating cost", "EUR"),
    (
        "terminal_cleanup_operating_cost",
        "Terminal cleanup operating cost",
        "EUR",
    ),
    ("stored_t", "Stored CO2", "t"),
    ("vented_t", "Vented CO2", "t"),
    ("captured_t", "Captured CO2", "t"),
    ("wall_clock_seconds", "Wall-clock time", "s"),
    ("solver_solve_wall_seconds", "Solver time", "s"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h72-root", type=Path, required=True)
    parser.add_argument("--h168-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "standard_deviation": statistics.stdev(values)
        if len(values) > 1
        else 0.0,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "p95": _percentile(values, 0.95),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_run(root: Path, planning_horizon_h: int) -> dict[int, dict[str, object]]:
    results: dict[int, dict[str, object]] = {}
    for summary_path in sorted(root.glob("seed_*/smoke_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        config = payload["configuration"]
        replans = payload["diagnostics"]["rolling_milp_replans"]
        seed = int(row["seed"])
        if seed in results:
            raise RuntimeError(f"duplicate seed {seed} under {root}")

        actions_path = summary_path.parent / "executed_actions.json"
        actions = json.loads(actions_path.read_text(encoding="utf-8"))
        trace = actions["actions_by_controller"]["rolling_milp"]

        if row["controller"] != "rolling_milp":
            raise RuntimeError(f"unexpected controller for seed {seed}")
        if row["run_status"] != "completed":
            raise RuntimeError(f"incomplete seed {seed}: {row['run_status']}")
        if int(row["solver_threads"]) != 4:
            raise RuntimeError(f"unexpected solver threads for seed {seed}")
        if float(row["solver_time_limit_seconds_per_replan"]) != 600.0:
            raise RuntimeError(f"unexpected solver time limit for seed {seed}")
        if int(config["rolling_planning_horizon_hours"]) != planning_horizon_h:
            raise RuntimeError(f"unexpected planning horizon for seed {seed}")
        if int(config["rolling_replan_hours"]) != 24:
            raise RuntimeError(f"unexpected replan interval for seed {seed}")
        if int(config["online_episode_hours"]) != 720:
            raise RuntimeError(f"unexpected episode horizon for seed {seed}")
        if row["fallback_used"] is not False:
            raise RuntimeError(f"fallback used for seed {seed}")
        if int(row["solver_failure_count"]) != 0:
            raise RuntimeError(f"solver failure for seed {seed}")
        if len(replans) != 30 or int(row["solver_replan_count"]) != 30:
            raise RuntimeError(f"unexpected replan count for seed {seed}")
        if len(trace) != 720 or int(row["executed_action_count"]) != 720:
            raise RuntimeError(f"unexpected action count for seed {seed}")
        if not all(item["solver_is_valid"] for item in replans):
            raise RuntimeError(f"invalid solver result for seed {seed}")
        if not all(item["execution_replay_is_valid"] for item in replans):
            raise RuntimeError(f"invalid execution replay for seed {seed}")

        results[seed] = {
            "row": row,
            "replans": replans,
            "summary_path": summary_path,
            "actions_path": actions_path,
        }

    if set(results) != EXPECTED_SEEDS:
        raise RuntimeError(
            f"seed mismatch under {root}: "
            f"missing={sorted(EXPECTED_SEEDS - set(results))}, "
            f"extra={sorted(set(results) - EXPECTED_SEEDS)}"
        )
    return results


def _method_summary(run: dict[int, dict[str, object]]) -> dict[str, object]:
    rows = [item["row"] for item in run.values()]
    replans = [
        replan
        for item in run.values()
        for replan in item["replans"]
    ]
    statuses = Counter(str(item["status"]) for item in replans)
    gaps = [
        float(item["relative_gap"])
        for item in replans
        if item["relative_gap"] is not None
    ]
    solve_times = [float(item["solve_wall_s"]) for item in replans]
    all_optimal_seeds = sum(
        all(replan["status"] == "Optimal" for replan in item["replans"])
        for item in run.values()
    )
    model_replay_inexact = sum(
        item["model_replay_is_exact"] is False for item in replans
    )
    return {
        "episode_count": len(rows),
        "replan_count": len(replans),
        "solver_valid_replan_count": sum(
            item["solver_is_valid"] is True for item in replans
        ),
        "execution_replay_valid_replan_count": sum(
            item["execution_replay_is_valid"] is True for item in replans
        ),
        "solver_status_counts": dict(sorted(statuses.items())),
        "optimal_replan_rate": statuses["Optimal"] / len(replans),
        "all_replans_optimal_seed_count": all_optimal_seeds,
        "relative_gap": _distribution(gaps),
        "solve_wall_seconds_per_replan": _distribution(solve_times),
        "model_replay_inexact_count": model_replay_inexact,
        "metrics": {
            key: _distribution([float(row[key]) for row in rows])
            for key, _label, _unit in METRICS
        },
    }


def _paired_outputs(
    h72: dict[int, dict[str, object]],
    h168: dict[int, dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    paired_rows: list[dict[str, object]] = []
    for seed in sorted(EXPECTED_SEEDS):
        h72_row = h72[seed]["row"]
        h168_row = h168[seed]["row"]
        output: dict[str, object] = {"seed": seed}
        for key, _label, _unit in METRICS:
            h72_value = float(h72_row[key])
            h168_value = float(h168_row[key])
            output[f"h72_{key}"] = h72_value
            output[f"h168_{key}"] = h168_value
            output[f"h72_minus_h168_{key}"] = h72_value - h168_value
        cost_delta = float(output["h72_minus_h168_total_cost"])
        output["h72_outcome_total_cost"] = (
            "win"
            if cost_delta < -1e-6
            else "loss"
            if cost_delta > 1e-6
            else "tie"
        )
        for label, run in (("h72", h72), ("h168", h168)):
            replans = run[seed]["replans"]
            status_counts = Counter(str(item["status"]) for item in replans)
            output[f"{label}_optimal_replans"] = status_counts["Optimal"]
            output[f"{label}_integer_feasible_replans"] = status_counts[
                "Integer Feasible"
            ]
            output[f"{label}_all_replans_optimal"] = (
                status_counts["Optimal"] == 30
            )
            output[f"{label}_mean_relative_gap"] = statistics.fmean(
                float(item["relative_gap"])
                for item in replans
                if item["relative_gap"] is not None
            )
        paired_rows.append(output)

    outcomes = Counter(str(row["h72_outcome_total_cost"]) for row in paired_rows)
    paired_summary: dict[str, object] = {
        "total_cost_outcomes_from_h72_perspective": {
            "wins": outcomes["win"],
            "ties": outcomes["tie"],
            "losses": outcomes["loss"],
        },
        "metrics": {},
    }
    for key, label, unit in METRICS:
        h72_values = [float(row[f"h72_{key}"]) for row in paired_rows]
        h168_values = [float(row[f"h168_{key}"]) for row in paired_rows]
        differences = [
            float(row[f"h72_minus_h168_{key}"]) for row in paired_rows
        ]
        h168_mean = statistics.fmean(h168_values)
        paired_summary["metrics"][key] = {
            "label": label,
            "unit": unit,
            "h72_mean": statistics.fmean(h72_values),
            "h168_mean": h168_mean,
            "h72_minus_h168": _distribution(differences),
            "mean_percent_change_relative_to_h168": (
                100.0 * statistics.fmean(differences) / h168_mean
                if h168_mean
                else None
            ),
        }
    return paired_rows, paired_summary


def _replan_rows(
    label: str,
    run: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in sorted(run):
        for index, item in enumerate(run[seed]["replans"]):
            rows.append(
                {
                    "horizon_label": label,
                    "seed": seed,
                    "replan_index": index,
                    "state_hour": item["state_hour"],
                    "planning_horizon_h": item["planning_horizon_h"],
                    "status": item["status"],
                    "solver_is_valid": item["solver_is_valid"],
                    "execution_replay_is_valid": item[
                        "execution_replay_is_valid"
                    ],
                    "model_replay_is_exact": item["model_replay_is_exact"],
                    "solve_wall_s": item["solve_wall_s"],
                    "relative_gap": item["relative_gap"],
                    "best_bound": item["best_bound"],
                    "first_incumbent_time_s": item.get(
                        "first_incumbent_time_s"
                    ),
                    "warm_start_accepted": item["warm_start_accepted"],
                    "termination_reason": item["termination_reason"],
                }
            )
    return rows


def _source_manifest(
    label: str,
    root: Path,
    run: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for seed in sorted(run):
        summary_path = run[seed]["summary_path"]
        actions_path = run[seed]["actions_path"]
        rows.append(
            {
                "horizon_label": label,
                "seed": seed,
                "summary_path": str(summary_path.relative_to(root.parent)),
                "summary_sha256": _sha256(summary_path),
                "actions_path": str(actions_path.relative_to(root.parent)),
                "actions_sha256": _sha256(actions_path),
            }
        )
    return rows


def _markdown(payload: dict[str, object]) -> str:
    paired = payload["paired_comparison"]
    outcomes = paired["total_cost_outcomes_from_h72_perspective"]
    lines = [
        "# Rolling MILP horizon ablation: H72 vs H168",
        "",
        "Both methods use R24, 600 s per replan, four deterministic CPLEX "
        "threads, and paired seeds 9000031-9000060.",
        "",
        "H72 is a horizon ablation and does not replace the formal H168 result.",
        "",
        "## Paired episode metrics",
        "",
        "| Metric | H72 mean | H168 mean | H72-H168 | Change vs H168 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, _label, _unit in METRICS:
        metric = paired["metrics"][key]
        lines.append(
            f"| {metric['label']} ({metric['unit']}) "
            f"| {metric['h72_mean']:.6g} "
            f"| {metric['h168_mean']:.6g} "
            f"| {metric['h72_minus_h168']['mean']:.6g} "
            f"| {metric['mean_percent_change_relative_to_h168']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "H72 total-cost outcomes: "
            f"{outcomes['wins']} wins, {outcomes['ties']} ties, "
            f"{outcomes['losses']} losses.",
            "",
            "## Solver proof status",
            "",
            "| Horizon | Optimal replans | Integer-feasible replans "
            "| Optimal rate | Seeds with 30/30 optimal | Mean gap | P95 gap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("h72", "h168"):
        method = payload["methods"][label]
        statuses = method["solver_status_counts"]
        lines.append(
            f"| {label.upper()} "
            f"| {statuses.get('Optimal', 0)} "
            f"| {statuses.get('Integer Feasible', 0)} "
            f"| {100.0 * method['optimal_replan_rate']:.2f}% "
            f"| {method['all_replans_optimal_seed_count']} "
            f"| {method['relative_gap']['mean']:.6g} "
            f"| {method['relative_gap']['p95']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The optimization proof status applies to each rolling subproblem. "
            "It does not establish global optimality for the 720 h episode.",
            "",
            "All 900 solver results and all 900 execution replays are valid for "
            "each horizon.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    if args.out_dir.exists() and not args.overwrite:
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    h72 = _load_run(args.h72_root, 72)
    h168 = _load_run(args.h168_root, 168)
    paired_rows, paired_summary = _paired_outputs(h72, h168)
    replans = [
        *_replan_rows("h72", h72),
        *_replan_rows("h168", h168),
    ]
    sources = [
        *_source_manifest("h72", args.h72_root, h72),
        *_source_manifest("h168", args.h168_root, h168),
    ]

    paired_path = args.out_dir / "paired_per_seed.csv"
    replan_path = args.out_dir / "replan_solver_diagnostics.csv"
    source_path = args.out_dir / "source_manifest.csv"
    summary_path = args.out_dir / "summary.json"
    markdown_path = args.out_dir / "README.md"
    _write_csv(paired_path, paired_rows)
    _write_csv(replan_path, replans)
    _write_csv(source_path, sources)

    payload = {
        "comparison": "Rolling MILP H72-R24-T600s vs H168-R24-T600s",
        "interpretation": (
            "H72 is a planning-horizon ablation, not a replacement for the "
            "formal H168 online-controller result."
        ),
        "seed_range_inclusive": [9_000_031, 9_000_060],
        "seed_count": 30,
        "paired": True,
        "common_configuration": {
            "episode_hours": 720,
            "replan_interval_hours": 24,
            "time_limit_seconds_per_replan": 600,
            "solver_threads": 4,
        },
        "methods": {
            "h72": _method_summary(h72),
            "h168": _method_summary(h168),
        },
        "paired_comparison": paired_summary,
        "validation": {
            "h72_completed_episodes": len(h72),
            "h168_completed_episodes": len(h168),
            "h72_replans": 900,
            "h168_replans": 900,
            "paired_rows": len(paired_rows),
            "replan_rows": len(replans),
            "source_manifest_rows": len(sources),
        },
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(payload), encoding="utf-8")

    audited = (
        paired_path,
        replan_path,
        source_path,
        summary_path,
        markdown_path,
    )
    audit = {
        "output_sha256": {
            path.name: _sha256(path)
            for path in audited
        },
        "input_source_manifest_sha256": _sha256(source_path),
        "validation": payload["validation"],
    }
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    return args.out_dir


if __name__ == "__main__":
    print(run(parse_args()))
