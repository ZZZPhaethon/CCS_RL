from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "experiments_results"
    / "E1"
    / "algorithms"
    / "formal_comparison"
    / "e1_formal_per_episode.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments_results" / "E1" / "figures"

FIXED_ASSIGNMENT = "fixed_assignment"
METHODS = (
    "greedy",
    "ppo_hourly",
    "ppo_high_level",
    "iterative_action_q_g60_p4",
    "rolling_milp",
)
LEARNED_METHODS = {
    "ppo_hourly",
    "ppo_high_level",
    "iterative_action_q_g60_p4",
}
EXPECTED_TEST_SEEDS = tuple(range(9000031, 9000061))
EXPECTED_MODEL_SEEDS = (0, 1, 2)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_RANDOM_STATE = 20260729

DISPLAY_NAMES = {
    "greedy": "Greedy",
    "ppo_hourly": "Hourly PPO",
    "ppo_high_level": "High-level PPO",
    "iterative_action_q_g60_p4": "Iterative Action-Q",
    "rolling_milp": "Rolling MILP (600 s/replan)",
}
METHOD_CLASSES = {
    "greedy": "heuristic",
    "ppo_hourly": "reinforcement_learning",
    "ppo_high_level": "reinforcement_learning",
    "iterative_action_q_g60_p4": "reinforcement_learning",
    "rolling_milp": "optimization",
}
COLORS = {
    "greedy": "#606060",
    "ppo_hourly": "#B4C0E4",
    "ppo_high_level": "#7884B4",
    "iterative_action_q_g60_p4": "#D58CA3",
    "rolling_milp": "#42949E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create E1 Figure 3a using Fixed-Assignment as the paired baseline."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _parse_model_seed(value: str) -> int | None:
    value = value.strip()
    return None if not value else int(value)


def load_costs(input_csv: Path) -> dict[str, dict[int, dict[int | None, float]]]:
    included = {FIXED_ASSIGNMENT, *METHODS}
    costs: dict[str, dict[int, dict[int | None, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            algorithm = row["algorithm"]
            if algorithm not in included:
                continue
            test_seed = int(row["test_seed"])
            model_seed = _parse_model_seed(row["model_seed"])
            if model_seed in costs[algorithm][test_seed]:
                raise ValueError(
                    f"Duplicate row for {algorithm}, test seed {test_seed}, "
                    f"model seed {model_seed}."
                )
            costs[algorithm][test_seed][model_seed] = float(row["total_cost_eur"])

    expected_test_seed_set = set(EXPECTED_TEST_SEEDS)
    for algorithm in (FIXED_ASSIGNMENT, *METHODS):
        actual_test_seeds = set(costs[algorithm])
        if actual_test_seeds != expected_test_seed_set:
            missing = sorted(expected_test_seed_set - actual_test_seeds)
            extra = sorted(actual_test_seeds - expected_test_seed_set)
            raise ValueError(
                f"{algorithm} has incorrect test-seed coverage; "
                f"missing={missing}, extra={extra}."
            )

        expected_model_seeds: set[int | None]
        if algorithm in LEARNED_METHODS:
            expected_model_seeds = set(EXPECTED_MODEL_SEEDS)
        else:
            expected_model_seeds = {None}
        for test_seed in EXPECTED_TEST_SEEDS:
            actual_model_seeds = set(costs[algorithm][test_seed])
            if actual_model_seeds != expected_model_seeds:
                raise ValueError(
                    f"{algorithm}, test seed {test_seed} has model seeds "
                    f"{sorted(str(item) for item in actual_model_seeds)}; expected "
                    f"{sorted(str(item) for item in expected_model_seeds)}."
                )

    return costs


def paired_delta_matrix(
    costs: dict[str, dict[int, dict[int | None, float]]],
    algorithm: str,
) -> np.ndarray:
    model_seeds: tuple[int | None, ...]
    if algorithm in LEARNED_METHODS:
        model_seeds = EXPECTED_MODEL_SEEDS
    else:
        model_seeds = (None,)

    matrix = np.empty((len(model_seeds), len(EXPECTED_TEST_SEEDS)), dtype=float)
    for column, test_seed in enumerate(EXPECTED_TEST_SEEDS):
        fixed_cost = costs[FIXED_ASSIGNMENT][test_seed][None]
        for row, model_seed in enumerate(model_seeds):
            matrix[row, column] = (
                costs[algorithm][test_seed][model_seed] - fixed_cost
            )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{algorithm} contains non-finite paired cost differences.")
    return matrix


def hierarchical_mean_ci(
    delta_matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    model_count, test_count = delta_matrix.shape
    test_indices = rng.integers(
        0, test_count, size=(BOOTSTRAP_DRAWS, test_count)
    )
    if model_count == 1:
        bootstrap_means = delta_matrix[0, test_indices].mean(axis=1)
    else:
        model_indices = rng.integers(
            0, model_count, size=(BOOTSTRAP_DRAWS, model_count)
        )
        resampled = delta_matrix[
            model_indices[:, :, np.newaxis],
            test_indices[:, np.newaxis, :],
        ]
        bootstrap_means = resampled.mean(axis=(1, 2))
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(low), float(high)


def build_source_data(
    costs: dict[str, dict[int, dict[int | None, float]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scenario_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_STATE)

    for algorithm in METHODS:
        delta_matrix = paired_delta_matrix(costs, algorithm)
        scenario_deltas = delta_matrix.mean(axis=0)
        ci_low, ci_high = hierarchical_mean_ci(delta_matrix, rng)

        for index, test_seed in enumerate(EXPECTED_TEST_SEEDS):
            fixed_cost = costs[FIXED_ASSIGNMENT][test_seed][None]
            method_cost = fixed_cost + scenario_deltas[index]
            scenario_rows.append(
                {
                    "algorithm": algorithm,
                    "algorithm_display_name": DISPLAY_NAMES[algorithm],
                    "method_class": METHOD_CLASSES[algorithm],
                    "test_seed": test_seed,
                    "model_instance_count": delta_matrix.shape[0],
                    "mean_total_cost_eur": method_cost,
                    "fixed_assignment_total_cost_eur": fixed_cost,
                    "delta_total_cost_vs_fixed_assignment_eur": scenario_deltas[index],
                }
            )

        tolerance = 1e-9
        summary_rows.append(
            {
                "algorithm": algorithm,
                "algorithm_display_name": DISPLAY_NAMES[algorithm],
                "method_class": METHOD_CLASSES[algorithm],
                "test_scenario_count": len(EXPECTED_TEST_SEEDS),
                "model_instance_count": delta_matrix.shape[0],
                "mean_delta_total_cost_vs_fixed_assignment_eur": float(
                    delta_matrix.mean()
                ),
                "median_scenario_delta_total_cost_vs_fixed_assignment_eur": float(
                    np.median(scenario_deltas)
                ),
                "ci95_low_eur": ci_low,
                "ci95_high_eur": ci_high,
                "wins_vs_fixed_assignment": int(
                    np.count_nonzero(scenario_deltas < -tolerance)
                ),
                "ties_vs_fixed_assignment": int(
                    np.count_nonzero(np.abs(scenario_deltas) <= tolerance)
                ),
                "losses_vs_fixed_assignment": int(
                    np.count_nonzero(scenario_deltas > tolerance)
                ),
            }
        )

    return scenario_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _deterministic_jitter(count: int, amplitude: float = 0.14) -> np.ndarray:
    angles = np.arange(count, dtype=float) * np.pi * (3.0 - np.sqrt(5.0))
    return amplitude * np.sin(angles)


def draw_figure(
    scenario_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    output_dir: Path,
) -> list[Path]:
    scenario_by_algorithm: dict[str, list[float]] = defaultdict(list)
    for row in scenario_rows:
        scenario_by_algorithm[str(row["algorithm"])].append(
            float(row["delta_total_cost_vs_fixed_assignment_eur"]) / 1_000_000.0
        )
    summary_by_algorithm = {
        str(row["algorithm"]): row for row in summary_rows
    }

    width_in = 89.0 / 25.4
    height_in = 80.0 / 25.4
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_positions = np.arange(len(METHODS))[::-1]
    for y, algorithm in zip(y_positions, METHODS):
        values = np.asarray(scenario_by_algorithm[algorithm], dtype=float)
        order = np.argsort(values)
        jitter = np.empty_like(values)
        jitter[order] = _deterministic_jitter(len(values))
        color = COLORS[algorithm]

        ax.scatter(
            values,
            y + jitter,
            s=11,
            color=color,
            alpha=0.42,
            edgecolors="none",
            zorder=2,
        )

        summary = summary_by_algorithm[algorithm]
        mean = (
            float(summary["mean_delta_total_cost_vs_fixed_assignment_eur"])
            / 1_000_000.0
        )
        ci_low = float(summary["ci95_low_eur"]) / 1_000_000.0
        ci_high = float(summary["ci95_high_eur"]) / 1_000_000.0
        ax.errorbar(
            mean,
            y,
            xerr=np.array([[mean - ci_low], [ci_high - mean]]),
            fmt="D",
            markersize=4.3,
            markerfacecolor=color,
            markeredgecolor="#272727",
            markeredgewidth=0.55,
            ecolor="#272727",
            elinewidth=1.05,
            capsize=2.5,
            capthick=0.9,
            zorder=4,
        )

    ax.axvline(0.0, color="#4D4D4D", linewidth=0.85, linestyle=(0, (3, 2)), zorder=1)
    ax.xaxis.grid(True, color="#D8D8D8", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [DISPLAY_NAMES[algorithm] for algorithm in METHODS],
        fontsize=6.3,
    )
    ax.set_ylim(-0.55, len(METHODS) - 0.45)
    ax.set_xlabel(
        "Δ total cost vs Fixed-Assignment (€ million)",
        fontsize=6.2,
        labelpad=4,
    )
    ax.tick_params(axis="x", labelsize=5.8, length=2.5, width=0.6)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    all_values = np.concatenate(
        [np.asarray(scenario_by_algorithm[algorithm]) for algorithm in METHODS]
    )
    ci_values = np.asarray(
        [
            float(row[key]) / 1_000_000.0
            for row in summary_rows
            for key in ("ci95_low_eur", "ci95_high_eur")
        ]
    )
    data_min = min(0.0, float(all_values.min()), float(ci_values.min()))
    data_max = max(0.0, float(all_values.max()), float(ci_values.max()))
    padding = max(0.18, 0.06 * (data_max - data_min))
    ax.set_xlim(data_min - padding, data_max + padding)

    ax.text(
        0.0,
        1.015,
        "Fixed-Assignment reference",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=5.6,
        color="#4D4D4D",
    )
    ax.text(
        0.0,
        -0.26,
        "Points: 30 paired test scenarios; diamonds: mean; bars: 95% CI",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        color="#4D4D4D",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    fig.text(0.015, 0.975, "a", ha="left", va="top", fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.36, right=0.985, top=0.90, bottom=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "figure_3a_fixed_assignment_baseline"
    outputs = [
        base.with_suffix(".pdf"),
        base.with_suffix(".png"),
    ]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    costs = load_costs(args.input_csv)
    scenario_rows, summary_rows = build_source_data(costs)

    source_data_dir = args.output_dir / "source_data"
    scenario_csv = source_data_dir / "figure_3a_paired_scenario_differences.csv"
    summary_csv = source_data_dir / "figure_3a_summary_statistics.csv"
    metadata_json = source_data_dir / "figure_3a_metadata.json"
    write_csv(scenario_csv, scenario_rows)
    write_csv(summary_csv, summary_rows)
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "figure": "Figure 3a",
                "baseline": "Fixed-Assignment Heuristic",
                "test_seeds": [EXPECTED_TEST_SEEDS[0], EXPECTED_TEST_SEEDS[-1]],
                "test_scenario_count": len(EXPECTED_TEST_SEEDS),
                "point_definition": (
                    "Per-test-scenario paired total-cost difference; learned-method "
                    "points average model seeds 0, 1, and 2."
                ),
                "interval_definition": (
                    "95% percentile hierarchical bootstrap confidence interval for "
                    "the mean; test scenarios and learned model instances are "
                    "resampled independently."
                ),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "rolling_milp_time_limit_seconds_per_replan": 600,
                "input_csv": str(args.input_csv.relative_to(REPO_ROOT)),
                "output_formats": ["pdf", "png"],
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    outputs = draw_figure(scenario_rows, summary_rows, args.output_dir)
    print(f"Wrote {scenario_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {metadata_json}")
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
