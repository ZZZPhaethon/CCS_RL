"""Build the E1 cost-reduction and absolute online-time table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON = (
    REPO_ROOT
    / "experiments_results"
    / "E1"
    / "algorithms"
    / "formal_comparison"
    / "e1_formal_per_episode.csv"
)
DEFAULT_TIMING_ROOT = (
    REPO_ROOT
    / "experiments_results"
    / "E1"
    / "timing"
    / "online_timing_hpc_run01"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments_results" / "E1" / "timing" / "cost_timing_table"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--timing-root", type=Path, default=DEFAULT_TIMING_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paired_reductions(
    rows: list[dict[str, str]],
    algorithm: str,
) -> tuple[np.ndarray, dict[str, dict[int, float]]]:
    fixed = {
        int(row["test_seed"]): float(row["total_cost_eur"])
        for row in rows
        if row["algorithm"] == "fixed_assignment"
    }
    selected = [row for row in rows if row["algorithm"] == algorithm]
    by_model: dict[str, dict[int, float]] = {}
    values = []
    for row in selected:
        test_seed = int(row["test_seed"])
        model_seed = row["model_seed"] or "not_applicable"
        reduction = (
            100.0
            * (fixed[test_seed] - float(row["total_cost_eur"]))
            / fixed[test_seed]
        )
        by_model.setdefault(model_seed, {})[test_seed] = reduction
        values.append(reduction)
    return np.asarray(values, dtype=float), by_model


def _hierarchical_ci(
    by_model: dict[str, dict[int, float]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    model_keys = tuple(sorted(by_model))
    test_seeds = tuple(sorted(next(iter(by_model.values()))))
    expected = set(test_seeds)
    if any(set(values) != expected for values in by_model.values()):
        raise ValueError("cost-reduction grid is incomplete")
    estimates = np.empty(samples, dtype=float)
    learned = model_keys != ("not_applicable",)
    for index in range(samples):
        sampled_models = (
            rng.choice(model_keys, size=len(model_keys), replace=True)
            if learned
            else model_keys
        )
        sampled_seeds = rng.choice(
            test_seeds,
            size=len(test_seeds),
            replace=True,
        )
        estimates[index] = np.mean(
            [
                by_model[str(model)][int(seed)]
                for model in sampled_models
                for seed in sampled_seeds
            ]
        )
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.975)))


def _timing_rows(
    timing_root: Path,
    comparison_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    rows: list[dict[str, str]] = []
    metadata = []
    for path in sorted(timing_root.rglob("timing_records.csv")):
        rows.extend(_read_csv(path))
        metadata_path = path.with_name("metadata.json")
        metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    expected_algorithms = {
        "fixed_assignment",
        "greedy",
        "ppo_hourly",
        "ppo_high_level",
        "ppo_event_residual",
        "iterative_action_q_g60_p4",
    }
    if {row["algorithm"] for row in rows} != expected_algorithms:
        raise ValueError("HPC timing shards do not cover all non-MILP E1 methods")
    if any(item.get("host") != "rootrunner" for item in metadata):
        raise ValueError("all E1 timing shards must run on rootrunner")
    if any(str(item.get("slurm_cpus_per_task")) != "4" for item in metadata):
        raise ValueError("all E1 timing shards must use four CPUs")

    formal = {
        (
            row["algorithm"],
            row["model_seed"],
            int(row["test_seed"]),
        ): float(row["total_cost_eur"])
        for row in comparison_rows
    }
    for row in rows:
        key = (
            row["algorithm"],
            row["model_seed"],
            int(row["test_seed"]),
        )
        if key not in formal:
            raise ValueError(f"timing row is not in the E1 formal grid: {key}")
        if not np.isclose(
            float(row["total_cost_eur"]),
            formal[key],
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"timing replay changed formal cost for {key}")

    rolling = [
        {
            "algorithm": row["algorithm"],
            "model_seed": row["model_seed"],
            "test_seed": row["test_seed"],
            "episode_hours": row["episode_hours"],
            "decision_count": row["decision_count"],
            "episode_wall_time_s": row["wall_clock_seconds"],
            "total_cost_eur": row["total_cost_eur"],
            "vented_t": row["vented_t"],
            "stored_t": row["stored_t"],
        }
        for row in comparison_rows
        if row["algorithm"] == "rolling_milp"
    ]
    rows.extend(rolling)
    return rows, {
        "non_milp_hpc_metadata": metadata,
        "rolling_timing_source": (
            "E1 formal Rolling MILP run03; rootrunner, four CPLEX threads"
        ),
    }


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = _read_csv(args.comparison_csv)
    timing, timing_audit = _timing_rows(args.timing_root, comparison)
    rng = np.random.default_rng(args.bootstrap_seed)

    display_names = {
        row["algorithm"]: row["algorithm_display_name"] for row in comparison
    }
    algorithm_order = [
        "fixed_assignment",
        "greedy",
        "ppo_hourly",
        "ppo_high_level",
        "ppo_event_residual",
        "iterative_action_q_g60_p4",
        "rolling_milp",
    ]
    output_rows: list[dict[str, object]] = []
    for algorithm in algorithm_order:
        cost_rows = [row for row in comparison if row["algorithm"] == algorithm]
        time_values = np.asarray(
            [
                float(row["episode_wall_time_s"])
                for row in timing
                if row["algorithm"] == algorithm
            ],
            dtype=float,
        )
        if algorithm == "fixed_assignment":
            reductions = np.zeros(len(cost_rows), dtype=float)
            ci_low = ci_high = 0.0
        else:
            reductions, by_model = _paired_reductions(comparison, algorithm)
            ci_low, ci_high = _hierarchical_ci(
                by_model,
                samples=args.bootstrap_samples,
                rng=rng,
            )
        output_rows.append(
            {
                "algorithm": algorithm,
                "algorithm_display_name": display_names[algorithm],
                "cost_records": len(cost_rows),
                "mean_total_cost_eur": float(
                    np.mean(
                        [float(row["total_cost_eur"]) for row in cost_rows]
                    )
                ),
                "mean_cost_reduction_vs_fixed_percent": float(
                    np.mean(reductions)
                ),
                "cost_reduction_95pct_ci_low_percent": ci_low,
                "cost_reduction_95pct_ci_high_percent": ci_high,
                "timing_records": len(time_values),
                "mean_episode_wall_time_s": float(np.mean(time_values)),
                "median_episode_wall_time_s": float(np.median(time_values)),
                "p95_episode_wall_time_s": float(
                    np.quantile(time_values, 0.95)
                ),
            }
        )

    csv_path = args.output_dir / "e1_cost_reduction_and_online_time.csv"
    _write_csv(csv_path, output_rows)
    markdown = [
        "| Method | Mean total cost (EUR) | Cost reduction vs Fixed-Assignment (95% CI) | Online solution time, mean / median / P95 (s) |",
        "|---|---:|---:|---:|",
    ]
    for row in output_rows:
        markdown.append(
            "| {algorithm_display_name} | {mean_total_cost_eur:,.0f} | "
            "{mean_cost_reduction_vs_fixed_percent:.2f}% "
            "[{cost_reduction_95pct_ci_low_percent:.2f}, "
            "{cost_reduction_95pct_ci_high_percent:.2f}] | "
            "{mean_episode_wall_time_s:,.3f} / "
            "{median_episode_wall_time_s:,.3f} / "
            "{p95_episode_wall_time_s:,.3f} |".format(**row)
        )
    markdown.extend(
        [
            "",
            "Cost reduction is paired by test seed against Fixed-Assignment. "
            "Positive values indicate lower cost. Time is reported in absolute "
            "seconds and is not converted to a percentage. Online solution time "
            "covers scenario reset through the completed 720 h closed-loop "
            "rollout and terminal-cleanup calculation; it excludes training and "
            "checkpoint loading.",
        ]
    )
    (args.output_dir / "e1_cost_reduction_and_online_time.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    audit = {
        "comparison_csv": str(args.comparison_csv),
        "comparison_sha256": _sha256(args.comparison_csv),
        "timing_root": str(args.timing_root),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "timing_audit": timing_audit,
        "algorithms": algorithm_order,
        "rows": len(output_rows),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_rows


def main() -> None:
    rows = run(parse_args())
    print(f"E1_COST_TIMING_COMPLETE methods={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
