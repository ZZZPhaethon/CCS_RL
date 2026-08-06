"""Aggregate the one-shot E4 and validation-only gate-sweep follow-ups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.aggregate_e2_e3_e4 import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    MODEL_SEEDS,
    TEST_SEEDS,
    _hierarchical_ci,
    _load_method,
    _save_figure,
    _write_csv,
    _write_json,
)
from experiments.gate_sweep_configs import VALIDATION_SEEDS, gate_records


REPO_ROOT = Path(__file__).resolve().parents[1]
STRESS_LEVELS = ("low", "medium", "high")
GATE_METRICS = (
    "total_cost_eur",
    "greedy_total_cost_eur",
    "vented_t",
    "stored_t",
    "override_events",
    "proposed_override_events",
    "event_count",
)
CURRENT_GATE = "h4_m040_w12_c12"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=("one_shot_e4", "gate_sweep"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "experiments_results",
    )
    return parser.parse_args(argv)


def _method_row(
    method: str,
    stress: str,
    matrices: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, object]:
    cost = matrices["total_cost_eur"]
    greedy = matrices["greedy_total_cost_eur"]
    delta = cost - greedy
    cost_low, cost_high = _hierarchical_ci(cost, rng)
    delta_low, delta_high = _hierarchical_ci(delta, rng)
    scenario_delta = delta.mean(axis=0)
    return {
        "stress_level": stress,
        "method": method,
        "model_seeds": len(MODEL_SEEDS),
        "test_seeds": len(TEST_SEEDS),
        "mean_total_cost_eur": float(cost.mean()),
        "total_cost_ci95_low_eur": cost_low,
        "total_cost_ci95_high_eur": cost_high,
        "mean_delta_cost_vs_greedy_eur": float(delta.mean()),
        "delta_cost_ci95_low_eur": delta_low,
        "delta_cost_ci95_high_eur": delta_high,
        "mean_vented_t": float(matrices["vented_t"].mean()),
        "mean_stored_t": float(matrices["stored_t"].mean()),
        "wins_vs_greedy": int(np.count_nonzero(scenario_delta < -1e-6)),
        "ties_vs_greedy": int(
            np.count_nonzero(np.abs(scenario_delta) <= 1e-6)
        ),
        "losses_vs_greedy": int(np.count_nonzero(scenario_delta > 1e-6)),
    }


def aggregate_one_shot_e4(results_root: Path) -> None:
    e4_root = results_root / "E4"
    p4_root = (
        e4_root / "formal_stress_seeds_9000031-9000060_run01"
    )
    one_root = (
        e4_root
        / "formal_one_shot_stress_seeds_9000031-9000060_run01"
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    pairwise_rows = []
    for stress in STRESS_LEVELS:
        p4 = _load_method(p4_root / stress)
        one = _load_method(one_root / stress)
        rows.append(_method_row("iterative_p4", stress, p4, rng))
        rows.append(_method_row("one_shot_matched", stress, one, rng))
        greedy = p4["greedy_total_cost_eur"]
        greedy_low, greedy_high = _hierarchical_ci(greedy[:1], rng)
        rows.append(
            {
                "stress_level": stress,
                "method": "greedy",
                "model_seeds": 0,
                "test_seeds": len(TEST_SEEDS),
                "mean_total_cost_eur": float(greedy[0].mean()),
                "total_cost_ci95_low_eur": greedy_low,
                "total_cost_ci95_high_eur": greedy_high,
                "mean_delta_cost_vs_greedy_eur": 0.0,
                "delta_cost_ci95_low_eur": 0.0,
                "delta_cost_ci95_high_eur": 0.0,
                "mean_vented_t": float(p4["greedy_vented_t"][0].mean()),
                "mean_stored_t": float(p4["greedy_stored_t"][0].mean()),
                "wins_vs_greedy": 0,
                "ties_vs_greedy": len(TEST_SEEDS),
                "losses_vs_greedy": 0,
            }
        )
        difference = one["total_cost_eur"] - p4["total_cost_eur"]
        low, high = _hierarchical_ci(difference, rng)
        scenario_difference = difference.mean(axis=0)
        pairwise_rows.append(
            {
                "stress_level": stress,
                "contrast": "one_shot_minus_iterative_p4",
                "mean_difference_eur": float(difference.mean()),
                "ci95_low_eur": low,
                "ci95_high_eur": high,
                "iterative_p4_lower_scenarios": int(
                    np.count_nonzero(scenario_difference > 1e-6)
                ),
                "ties": int(
                    np.count_nonzero(
                        np.abs(scenario_difference) <= 1e-6
                    )
                ),
                "one_shot_lower_scenarios": int(
                    np.count_nonzero(scenario_difference < -1e-6)
                ),
            }
        )

    table_dir = e4_root / "tables"
    _write_csv(table_dir / "table_s2_action_q_stress_comparison.csv", rows)
    _write_json(
        table_dir / "table_s2_action_q_stress_comparison.json",
        {
            "kind": "one_shot_vs_iterative_p4_stress_comparison",
            "rows": rows,
            "pairwise": pairwise_rows,
        },
    )
    _write_csv(
        table_dir / "table_s2_action_q_stress_pairwise.csv",
        pairwise_rows,
    )

    x = np.arange(len(STRESS_LEVELS))
    fig, (ax_cost, ax_delta) = plt.subplots(
        1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    colors = {
        "greedy": "#666666",
        "iterative_p4": "#C77C96",
        "one_shot_matched": "#6E9FB8",
    }
    labels = {
        "greedy": "Greedy",
        "iterative_p4": "Iterative P4",
        "one_shot_matched": "One-shot matched",
    }
    for method in ("greedy", "iterative_p4", "one_shot_matched"):
        selected = [row for row in rows if row["method"] == method]
        mean = np.asarray(
            [row["mean_total_cost_eur"] for row in selected]
        ) / 1_000_000.0
        low = np.asarray(
            [row["total_cost_ci95_low_eur"] for row in selected]
        ) / 1_000_000.0
        high = np.asarray(
            [row["total_cost_ci95_high_eur"] for row in selected]
        ) / 1_000_000.0
        ax_cost.errorbar(
            x,
            mean,
            yerr=[mean - low, high - mean],
            marker="o",
            capsize=2.5,
            color=colors[method],
            label=labels[method],
        )
    ax_cost.set_xticks(x, ["Low", "Medium", "High"])
    ax_cost.set_ylabel("Total cost (million EUR)")
    ax_cost.set_xlabel("Composite disturbance stress")
    ax_cost.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax_cost.legend(fontsize=6)

    delta_mean = np.asarray(
        [row["mean_difference_eur"] for row in pairwise_rows]
    ) / 1_000.0
    delta_low = np.asarray(
        [row["ci95_low_eur"] for row in pairwise_rows]
    ) / 1_000.0
    delta_high = np.asarray(
        [row["ci95_high_eur"] for row in pairwise_rows]
    ) / 1_000.0
    ax_delta.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    ax_delta.errorbar(
        x,
        delta_mean,
        yerr=[delta_mean - delta_low, delta_high - delta_mean],
        color="#343434",
        marker="D",
        capsize=2.5,
    )
    ax_delta.set_xticks(x, ["Low", "Medium", "High"])
    ax_delta.set_ylabel("One-shot - P4 cost (thousand EUR)")
    ax_delta.set_xlabel("Composite disturbance stress")
    ax_delta.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax_cost.text(
        -0.16, 1.05, "a", transform=ax_cost.transAxes,
        fontsize=8, fontweight="bold", va="bottom"
    )
    ax_delta.text(
        -0.16, 1.05, "b", transform=ax_delta.transAxes,
        fontsize=8, fontweight="bold", va="bottom"
    )
    fig.tight_layout()
    figure_dir = e4_root / "figures"
    _save_figure(fig, figure_dir / "figure_5b_action_q_stress_comparison")
    _write_csv(
        figure_dir / "source_data" / "figure_5b_source_data.csv",
        rows,
    )
    _write_json(
        figure_dir / "source_data" / "figure_5b_metadata.json",
        {
            "figure": "Figure 5b",
            "claim": (
                "Test whether the matched Greedy-only Action-Q model retains "
                "the frozen iterative P4 model's stress robustness."
            ),
            "test_seed_range_inclusive": [TEST_SEEDS[0], TEST_SEEDS[-1]],
            "test_seed_count": len(TEST_SEEDS),
            "model_seeds": list(MODEL_SEEDS),
            "interval_definition": (
                "95% percentile hierarchical bootstrap confidence interval; "
                "test scenarios and model instances are resampled "
                "independently."
            ),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_definition": (
                "Three independently trained model seeds crossed with 30 "
                "locked formal-test scenario seeds per stress level."
            ),
            "center_statistic": "Grand mean across model and scenario seeds.",
            "baseline_definition": (
                "Frozen Greedy deployment policy evaluated on the identical "
                "formal-test scenarios."
            ),
            "train_validation_test_split": (
                "Frozen training models; formal test seeds 9000031-9000060 "
                "used descriptively, with no gate or model selection."
            ),
            "multiple_comparison_correction": (
                "Not applied; three pre-specified descriptive stress levels."
            ),
            "iterative_q_state_feature_exclusions": ["hour_of_week"],
            "one_shot_state_feature_exclusions": ["hour_of_week"],
            "state_feature_count": 93,
            "formats": ["pdf", "png"],
            "png_dpi": 300,
        },
    )


def _load_gate_method(root: Path) -> dict[str, dict[str, np.ndarray]]:
    configs = {record["name"] for record in gate_records()}
    matrices = {
        gate: {
            metric: np.empty(
                (len(MODEL_SEEDS), len(VALIDATION_SEEDS)), dtype=float
            )
            for metric in GATE_METRICS
        }
        for gate in configs
    }
    for model_index, model_seed in enumerate(MODEL_SEEDS):
        path = root / f"model_seed_{model_seed}" / "evaluation.csv"
        seen: set[tuple[str, int]] = set()
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                gate = row["gate"]
                seed = int(row["seed"])
                if gate not in configs or seed not in VALIDATION_SEEDS:
                    raise ValueError(
                        f"unexpected gate/seed in {path}: {gate}/{seed}"
                    )
                key = (gate, seed)
                if key in seen:
                    raise ValueError(f"duplicate {key} in {path}")
                seen.add(key)
                test_index = VALIDATION_SEEDS.index(seed)
                for metric in GATE_METRICS:
                    matrices[gate][metric][model_index, test_index] = float(
                        row[metric]
                    )
        expected = {
            (gate, seed) for gate in configs for seed in VALIDATION_SEEDS
        }
        if seen != expected:
            raise ValueError(
                f"incomplete validation gate coverage in {path}: "
                f"missing={len(expected - seen)}, extra={len(seen - expected)}"
            )
    return matrices


def _gate_summary_row(
    method: str,
    gate: dict[str, object],
    matrices: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, object]:
    cost = matrices["total_cost_eur"]
    delta = cost - matrices["greedy_total_cost_eur"]
    cost_low, cost_high = _hierarchical_ci(cost, rng)
    delta_low, delta_high = _hierarchical_ci(delta, rng)
    scenario_cost = cost.mean(axis=0)
    return {
        "method": method,
        "gate": gate["name"],
        "required_heads": gate["required_heads"],
        "margin": gate["margin"],
        "max_overrides": gate["max_overrides"],
        "window_scheme": gate["window_scheme"],
        "window_count": (
            0 if gate["windows"] is None else len(gate["windows"])
        ),
        "model_seeds": len(MODEL_SEEDS),
        "validation_seeds": len(VALIDATION_SEEDS),
        "mean_total_cost_eur": float(cost.mean()),
        "total_cost_ci95_low_eur": cost_low,
        "total_cost_ci95_high_eur": cost_high,
        "p90_scenario_total_cost_eur": float(
            np.quantile(scenario_cost, 0.9)
        ),
        "mean_delta_cost_vs_greedy_eur": float(delta.mean()),
        "delta_cost_ci95_low_eur": delta_low,
        "delta_cost_ci95_high_eur": delta_high,
        "mean_vented_t": float(matrices["vented_t"].mean()),
        "mean_stored_t": float(matrices["stored_t"].mean()),
        "mean_override_events": float(
            matrices["override_events"].mean()
        ),
        "mean_proposed_override_events": float(
            matrices["proposed_override_events"].mean()
        ),
        "mean_event_count": float(matrices["event_count"].mean()),
        "cap_hit_fraction": float(
            np.mean(
                matrices["override_events"]
                >= float(gate["max_overrides"])
            )
        ),
    }


def _annotated_heatmap(ax, matrix, title, fmt, cmap, center=None):
    if center is None:
        image = ax.imshow(matrix, cmap=cmap, aspect="auto")
    else:
        maximum = max(
            abs(float(np.nanmin(matrix) - center)),
            abs(float(np.nanmax(matrix) - center)),
        )
        image = ax.imshow(
            matrix,
            cmap=cmap,
            aspect="auto",
            vmin=center - maximum,
            vmax=center + maximum,
        )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            red, green, blue, _ = image.cmap(
                image.norm(matrix[row, column])
            )
            luminance = (
                0.2126 * red + 0.7152 * green + 0.0722 * blue
            )
            ax.text(
                column,
                row,
                format(matrix[row, column], fmt),
                ha="center",
                va="center",
                fontsize=5.2,
                color="white" if luminance < 0.48 else "black",
            )
    ax.set_title(title, fontsize=7)
    ax.set_xticks(range(4), ["0", "0.1", "0.2", "0.4"])
    ax.set_yticks(range(4), ["2/5", "3/5", "4/5", "5/5"])
    ax.set_xlabel("Q margin")
    ax.set_ylabel("Required heads")
    return image


def aggregate_gate_sweep(results_root: Path) -> None:
    root = (
        results_root
        / "E2"
        / "validation_gate_sweep_seeds_8100001-8100020_run01"
    )
    records = gate_records()
    by_name = {record["name"]: record for record in records}
    methods = {
        "iterative_p4": _load_gate_method(root / "iterative_p4"),
        "one_shot_matched": _load_gate_method(root / "one_shot_matched"),
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    pairwise_rows = []
    for gate in records:
        name = str(gate["name"])
        for method, matrices in methods.items():
            rows.append(
                _gate_summary_row(method, gate, matrices[name], rng)
            )
        difference = (
            methods["one_shot_matched"][name]["total_cost_eur"]
            - methods["iterative_p4"][name]["total_cost_eur"]
        )
        low, high = _hierarchical_ci(difference, rng)
        pairwise_rows.append(
            {
                "gate": name,
                "required_heads": gate["required_heads"],
                "margin": gate["margin"],
                "max_overrides": gate["max_overrides"],
                "window_scheme": gate["window_scheme"],
                "mean_one_shot_minus_p4_eur": float(difference.mean()),
                "ci95_low_eur": low,
                "ci95_high_eur": high,
            }
        )
    vs_current_rows = []
    current_rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    for gate in records:
        name = str(gate["name"])
        for method, matrices in methods.items():
            difference = (
                matrices[name]["total_cost_eur"]
                - matrices[CURRENT_GATE]["total_cost_eur"]
            )
            low, high = _hierarchical_ci(difference, current_rng)
            vs_current_rows.append(
                {
                    "method": method,
                    "gate": name,
                    "required_heads": gate["required_heads"],
                    "margin": gate["margin"],
                    "max_overrides": gate["max_overrides"],
                    "window_scheme": gate["window_scheme"],
                    "current_gate": CURRENT_GATE,
                    "mean_candidate_minus_current_eur": float(
                        difference.mean()
                    ),
                    "ci95_low_eur": low,
                    "ci95_high_eur": high,
                }
            )
    table_dir = results_root / "E2" / "tables"
    _write_csv(table_dir / "validation_gate_sweep_summary.csv", rows)
    _write_csv(
        table_dir / "validation_gate_sweep_pairwise.csv", pairwise_rows
    )
    _write_csv(
        table_dir / "validation_gate_sweep_vs_current.csv",
        vs_current_rows,
    )
    best = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        best[method] = min(
            selected, key=lambda row: float(row["mean_total_cost_eur"])
        )
    _write_json(
        table_dir / "validation_gate_sweep_summary.json",
        {
            "kind": "validation_only_gate_sweep",
            "validation_seed_range_inclusive": [
                VALIDATION_SEEDS[0],
                VALIDATION_SEEDS[-1],
            ],
            "formal_test_used_for_selection": False,
            "configurations": records,
            "rows": rows,
            "pairwise": pairwise_rows,
            "paired_vs_current": vs_current_rows,
            "best_by_validation_mean": best,
        },
    )

    confidence = [
        record
        for record in records
        if record["window_scheme"] == "w12"
        and record["max_overrides"] == 12
    ]
    row_lookup = {
        (row["method"], row["gate"]): row for row in rows
    }
    pair_lookup = {row["gate"]: row for row in pairwise_rows}
    p4_matrix = np.empty((4, 4))
    one_matrix = np.empty((4, 4))
    pair_matrix = np.empty((4, 4))
    for gate in confidence:
        row_index = int(gate["required_heads"]) - 2
        column_index = (0.0, 0.1, 0.2, 0.4).index(
            float(gate["margin"])
        )
        p4_matrix[row_index, column_index] = (
            float(
                row_lookup[("iterative_p4", gate["name"])][
                    "mean_total_cost_eur"
                ]
            )
            / 1_000_000.0
        )
        one_matrix[row_index, column_index] = (
            float(
                row_lookup[("one_shot_matched", gate["name"])][
                    "mean_total_cost_eur"
                ]
            )
            / 1_000_000.0
        )
        pair_matrix[row_index, column_index] = (
            float(pair_lookup[gate["name"]]["mean_one_shot_minus_p4_eur"])
            / 1_000.0
        )

    fig = plt.figure(figsize=(7.2, 5.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    ax_p4 = fig.add_subplot(grid[0, 0])
    ax_one = fig.add_subplot(grid[0, 1])
    ax_pair = fig.add_subplot(grid[0, 2])
    ax_capacity = fig.add_subplot(grid[1, :])
    _annotated_heatmap(
        ax_p4, p4_matrix, "Iterative P4 cost (EUR M)", ".2f", "Blues_r"
    )
    _annotated_heatmap(
        ax_one, one_matrix, "One-shot cost (EUR M)", ".2f", "Blues_r"
    )
    _annotated_heatmap(
        ax_pair,
        pair_matrix,
        "One-shot - P4 (EUR k)",
        ".0f",
        "coolwarm",
        center=0.0,
    )

    method_style = {
        "iterative_p4": ("Iterative P4", "#C77C96", "o"),
        "one_shot_matched": ("One-shot matched", "#6E9FB8", "s"),
    }
    scheme_marker = {"global": "X", "w12": "o", "w24": "^", "w48": "D"}
    capacity_gates = [
        record
        for record in records
        if record["required_heads"] == 3
        and record["margin"] == 0.1
    ]
    for method, (label, color, _) in method_style.items():
        for gate in capacity_gates:
            row = row_lookup[(method, gate["name"])]
            ax_capacity.scatter(
                row["mean_override_events"],
                float(row["mean_total_cost_eur"]) / 1_000_000.0,
                color=color,
                marker=scheme_marker[str(gate["window_scheme"])],
                s=25,
                alpha=0.85,
            )
        ax_capacity.scatter([], [], color=color, label=label)
    for scheme, marker in scheme_marker.items():
        ax_capacity.scatter(
            [],
            [],
            color="#555555",
            marker=marker,
            label=scheme,
        )
    ax_capacity.set_xlabel("Mean override events per episode")
    ax_capacity.set_ylabel("Validation total cost (million EUR)")
    ax_capacity.grid(color="#D9D9D9", linewidth=0.45)
    ax_capacity.legend(
        ncol=3, fontsize=5.7, loc="best", handletextpad=0.35
    )
    for label, axis in zip(
        ("a", "b", "c", "d"),
        (ax_p4, ax_one, ax_pair, ax_capacity),
    ):
        axis.text(
            -0.2 if axis is not ax_capacity else -0.06,
            1.08 if axis is not ax_capacity else 1.03,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
        )
    fig.tight_layout()
    figure_dir = results_root / "E2" / "figures"
    _save_figure(fig, figure_dir / "validation_gate_sweep")
    _write_csv(
        figure_dir / "source_data" / "validation_gate_sweep_source_data.csv",
        rows,
    )
    _write_csv(
        figure_dir
        / "source_data"
        / "validation_gate_sweep_pairwise_source_data.csv",
        pairwise_rows,
    )
    _write_csv(
        figure_dir
        / "source_data"
        / "validation_gate_sweep_vs_current_source_data.csv",
        vs_current_rows,
    )
    _write_json(
        figure_dir / "source_data" / "validation_gate_sweep_metadata.json",
        {
            "figure": "Validation-only gate sweep",
            "claim": (
                "Determine whether relaxed confidence and intervention "
                "constraints reveal an Iterative P4 advantage without using "
                "formal-test outcomes for gate selection."
            ),
            "validation_seed_range_inclusive": [
                VALIDATION_SEEDS[0],
                VALIDATION_SEEDS[-1],
            ],
            "validation_seed_count": len(VALIDATION_SEEDS),
            "model_seeds": list(MODEL_SEEDS),
            "configurations": len(by_name),
            "exploratory_multiple_comparisons": True,
            "interval_definition": (
                "95% percentile hierarchical bootstrap confidence interval; "
                "validation scenarios and model instances are resampled "
                "independently."
            ),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_definition": (
                "Three independently trained model seeds crossed with 20 "
                "locked controller-validation scenario seeds per gate."
            ),
            "center_statistic": "Grand mean across model and validation seeds.",
            "baseline_definition": (
                "Greedy deployment policy evaluated on the identical "
                "validation scenarios."
            ),
            "train_validation_test_split": (
                "Gate comparisons use controller-validation seeds "
                "8100001-8100020 only; formal-test seeds are prohibited."
            ),
            "multiple_comparison_correction": (
                "None; the 31-gate scan is explicitly exploratory and "
                "validation-only."
            ),
            "formats": ["pdf", "png"],
            "png_dpi": 300,
        },
    )


def main(argv=None):
    args = parse_args(argv)
    if args.analysis == "one_shot_e4":
        aggregate_one_shot_e4(args.results_root)
    else:
        aggregate_gate_sweep(args.results_root)


if __name__ == "__main__":
    main()
