"""Aggregate the locked E2-E4 Iterative Action-Q experiments and draw figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SEEDS = tuple(range(9000031, 9000061))
MODEL_SEEDS = (0, 1, 2)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260730
STAGE_CALLS = {
    "p1": 5_639_992,
    "p2": 6_400_212,
    "p3": 7_607_197,
    "p4": 9_526_297,
}

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
    parser.add_argument("experiment", choices=("E2", "E3", "E4"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "experiments_results",
    )
    return parser.parse_args(argv)


def _read_evaluation(path: Path) -> dict[int, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            if seed in rows:
                raise ValueError(f"duplicate seed {seed} in {path}")
            rows[seed] = {
                key: float(row[key])
                for key in (
                    "total_cost_eur",
                    "vented_t",
                    "stored_t",
                    "greedy_total_cost_eur",
                    "greedy_vented_t",
                    "greedy_stored_t",
                )
            }
    if set(rows) != set(TEST_SEEDS):
        missing = sorted(set(TEST_SEEDS) - set(rows))
        extra = sorted(set(rows) - set(TEST_SEEDS))
        raise ValueError(
            f"incorrect test-seed coverage in {path}: "
            f"missing={missing}, extra={extra}"
        )
    return rows


def _load_method(root: Path) -> dict[str, np.ndarray]:
    metrics = (
        "total_cost_eur",
        "vented_t",
        "stored_t",
        "greedy_total_cost_eur",
        "greedy_vented_t",
        "greedy_stored_t",
    )
    matrices = {
        metric: np.empty((len(MODEL_SEEDS), len(TEST_SEEDS)), dtype=float)
        for metric in metrics
    }
    for model_index, model_seed in enumerate(MODEL_SEEDS):
        rows = _read_evaluation(
            root / f"model_seed_{model_seed}" / "evaluation.csv"
        )
        for test_index, test_seed in enumerate(TEST_SEEDS):
            for metric in metrics:
                matrices[metric][model_index, test_index] = rows[test_seed][
                    metric
                ]
    for metric, matrix in matrices.items():
        if not np.isfinite(matrix).all():
            raise ValueError(f"{root} contains non-finite {metric}")
    return matrices


def _hierarchical_ci(
    matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    model_count, test_count = matrix.shape
    test_indices = rng.integers(
        0, test_count, size=(BOOTSTRAP_DRAWS, test_count)
    )
    if model_count == 1:
        means = matrix[0, test_indices].mean(axis=1)
    else:
        model_indices = rng.integers(
            0, model_count, size=(BOOTSTRAP_DRAWS, model_count)
        )
        resampled = matrix[
            model_indices[:, :, None],
            test_indices[:, None, :],
        ]
        means = resampled.mean(axis=(1, 2))
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _summary_row(
    key: str,
    label: str,
    matrices: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, object]:
    cost = matrices["total_cost_eur"]
    greedy = matrices["greedy_total_cost_eur"]
    delta = cost - greedy
    ci_low, ci_high = _hierarchical_ci(cost, rng)
    delta_low, delta_high = _hierarchical_ci(delta, rng)
    scenario_delta = delta.mean(axis=0)
    return {
        "method": key,
        "display_name": label,
        "model_seeds": len(MODEL_SEEDS),
        "test_seeds": len(TEST_SEEDS),
        "mean_total_cost_eur": float(cost.mean()),
        "total_cost_ci95_low_eur": ci_low,
        "total_cost_ci95_high_eur": ci_high,
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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _save_figure(fig, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def _metadata(figure: str, claim: str) -> dict[str, object]:
    return {
        "figure": figure,
        "claim": claim,
        "test_seed_range_inclusive": [TEST_SEEDS[0], TEST_SEEDS[-1]],
        "test_seed_count": len(TEST_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "interval_definition": (
            "95% percentile hierarchical bootstrap confidence interval; "
            "test scenarios and model instances are resampled independently."
        ),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "formats": ["pdf", "png"],
        "png_dpi": 300,
    }


def aggregate_e2(results_root: Path) -> None:
    root = results_root / "E2"
    formal = (
        root
        / "formal_iterative_q_stages_seeds_9000031-9000060_run01"
    )
    matched_root = (
        root
        / "formal_one_shot_matched_seeds_9000031-9000060_run01"
    )
    method_specs = [
        ("p1", "One-shot original / P1", formal / "p1"),
        ("p2", "Iterative Q P2", formal / "p2"),
        ("p3", "Iterative Q P3", formal / "p3"),
        ("p4", "Iterative Q P4", formal / "p4"),
        ("one_shot_matched", "One-shot matched", matched_root),
    ]
    budget_path = root / "training_one_shot_matched_run01" / "budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    matched_calls = int(budget["train_simulator_step_calls"])
    matched_roots = int(budget["train_roots"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    matrices_by_method = {}
    for key, label, path in method_specs:
        matrices = _load_method(path)
        matrices_by_method[key] = matrices
        row = _summary_row(key, label, matrices, rng)
        row["training_data_source"] = (
            "Greedy roll-in only"
            if key in {"p1", "one_shot_matched"}
            else "Greedy plus policy-induced roots"
        )
        row["cumulative_simulator_calls"] = (
            matched_calls if key == "one_shot_matched" else STAGE_CALLS[key]
        )
        row["cumulative_roots"] = {
            "p1": 2159,
            "p2": 2447,
            "p3": 2879,
            "p4": 3599,
            "one_shot_matched": matched_roots,
        }[key]
        rows.append(row)

    table_dir = root / "tables"
    _write_csv(table_dir / "table_4_iteration_ablation.csv", rows)
    _write_json(
        table_dir / "table_4_iteration_ablation.json",
        {
            "kind": "E2_iteration_ablation",
            "rows": rows,
            "one_shot_original_alias": "P1",
        },
    )

    stage_rows = rows[:4]
    x = np.asarray(
        [row["cumulative_simulator_calls"] for row in stage_rows],
        dtype=float,
    ) / 1_000_000.0
    y = np.asarray(
        [row["mean_total_cost_eur"] for row in stage_rows],
        dtype=float,
    ) / 1_000_000.0
    low = np.asarray(
        [row["total_cost_ci95_low_eur"] for row in stage_rows],
        dtype=float,
    ) / 1_000_000.0
    high = np.asarray(
        [row["total_cost_ci95_high_eur"] for row in stage_rows],
        dtype=float,
    ) / 1_000_000.0
    matched = rows[-1]
    fig, ax = plt.subplots(figsize=(89 / 25.4, 72 / 25.4))
    ax.errorbar(
        x,
        y,
        yerr=np.vstack((y - low, high - y)),
        color="#C77991",
        marker="o",
        markersize=4,
        linewidth=1.2,
        capsize=2.5,
        label="Iterative aggregation",
    )
    for xi, yi, label in zip(x, y, ("P1", "P2", "P3", "P4")):
        ax.annotate(
            label,
            (xi, yi),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=6,
        )
    matched_x = float(matched["cumulative_simulator_calls"]) / 1_000_000.0
    matched_y = float(matched["mean_total_cost_eur"]) / 1_000_000.0
    matched_low = (
        float(matched["total_cost_ci95_low_eur"]) / 1_000_000.0
    )
    matched_high = (
        float(matched["total_cost_ci95_high_eur"]) / 1_000_000.0
    )
    ax.errorbar(
        [matched_x],
        [matched_y],
        yerr=[[matched_y - matched_low], [matched_high - matched_y]],
        color="#545454",
        marker="D",
        markersize=4,
        capsize=2.5,
        label="One-shot matched",
    )
    ax.set_xlabel("Cumulative simulator calls (million)")
    ax.set_ylabel("Total cost (€ million)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax.legend(fontsize=5.8)
    fig.tight_layout()
    figure_dir = root / "figures"
    _save_figure(
        fig, figure_dir / "supplementary_figure_s2_iteration_ablation"
    )
    source_rows = [
        {
            "method": row["method"],
            "cumulative_simulator_calls": row[
                "cumulative_simulator_calls"
            ],
            "mean_total_cost_eur": row["mean_total_cost_eur"],
            "ci95_low_eur": row["total_cost_ci95_low_eur"],
            "ci95_high_eur": row["total_cost_ci95_high_eur"],
        }
        for row in rows
    ]
    _write_csv(
        figure_dir
        / "source_data"
        / "supplementary_figure_s2_source_data.csv",
        source_rows,
    )
    _write_json(
        figure_dir
        / "source_data"
        / "supplementary_figure_s2_metadata.json",
        _metadata(
            "Supplementary Figure S2",
            "Compare policy-induced iterative aggregation with a "
            "Greedy-only one-shot model at the same simulator budget.",
        ),
    )


def _checkpoint_stats(path: Path) -> tuple[int, int]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    parameters = sum(
        int(value.numel())
        for value in checkpoint["model_state_dict"].values()
    )
    metadata = checkpoint["metadata"]
    configuration = checkpoint["configuration"]
    dimension = len(metadata["state_feature_names"])
    q_head = configuration["q_head"]
    if q_head == "iterative_action_q_future_summary":
        dimension += len(metadata["future_feature_names"])
    elif q_head == "iterative_action_q_future_168":
        dimension += int(metadata["forecast_horizon_h"]) * len(
            metadata["forecast_feature_names"]
        )
    return dimension, parameters


def aggregate_e3(results_root: Path) -> None:
    root = results_root / "E3"
    formal = (
        root
        / "formal_future_information_seeds_9000031-9000060_run01"
    )
    specs = [
        ("state_only", "State-only"),
        ("structured_summary_168", "168 h structured summary"),
        ("full_sequence_168", "Full 168 h sequence"),
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    matrices_by_method = {
        key: _load_method(formal / key) for key, _label in specs
    }
    rows = []
    for key, label in specs:
        row = _summary_row(key, label, matrices_by_method[key], rng)
        if key == "structured_summary_168":
            local_checkpoint = (
                results_root
                / "E1"
                / "models"
                / "iterative_q"
                / "g60_p4_model_seed_0"
                / "iterative_action_q.pt"
            )
            remote_source_checkpoint = (
                REPO_ROOT
                / "output"
                / "iterative_q_budget_search"
                / "runs"
                / "g60_p4"
                / "p4"
                / "iterative_action_q.pt"
            )
            checkpoint = (
                local_checkpoint
                if local_checkpoint.is_file()
                else remote_source_checkpoint
            )
        else:
            checkpoint = (
                root
                / "training_future_information_run01"
                / key
                / "model_seed_0"
                / "p4"
                / "iterative_action_q.pt"
            )
        dimension, parameters = _checkpoint_stats(checkpoint)
        row["representation_dimension"] = dimension
        row["parameter_count"] = parameters
        state_cost = matrices_by_method["state_only"]["total_cost_eur"]
        delta = matrices_by_method[key]["total_cost_eur"] - state_cost
        delta_low, delta_high = _hierarchical_ci(delta, rng)
        row["mean_delta_cost_vs_state_only_eur"] = float(delta.mean())
        row["delta_cost_vs_state_only_ci95_low_eur"] = delta_low
        row["delta_cost_vs_state_only_ci95_high_eur"] = delta_high
        rows.append(row)

    table_dir = root / "tables"
    _write_csv(table_dir / "table_5_future_information_ablation.csv", rows)
    _write_json(
        table_dir / "table_5_future_information_ablation.json",
        {"kind": "E3_future_information_ablation", "rows": rows},
    )

    comparison_rows = rows[1:]
    fig, ax = plt.subplots(figsize=(89 / 25.4, 61 / 25.4))
    colors = ("#C77991", "#668F9E")
    for y, row, color in zip((1, 0), comparison_rows, colors):
        method = str(row["method"])
        delta = (
            matrices_by_method[method]["total_cost_eur"]
            - matrices_by_method["state_only"]["total_cost_eur"]
        )
        scenario_delta = delta.mean(axis=0) / 1_000_000.0
        jitter = np.linspace(-0.12, 0.12, len(TEST_SEEDS))
        ax.scatter(
            scenario_delta,
            y + jitter,
            s=10,
            color=color,
            alpha=0.4,
            edgecolors="none",
        )
        mean = float(row["mean_delta_cost_vs_state_only_eur"]) / 1_000_000.0
        low = (
            float(row["delta_cost_vs_state_only_ci95_low_eur"])
            / 1_000_000.0
        )
        high = (
            float(row["delta_cost_vs_state_only_ci95_high_eur"])
            / 1_000_000.0
        )
        ax.errorbar(
            mean,
            y,
            xerr=[[mean - low], [high - mean]],
            fmt="D",
            color="#292929",
            markerfacecolor=color,
            markersize=4,
            capsize=2.5,
        )
    ax.axvline(0, color="#555555", linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set_yticks((1, 0))
    ax.set_yticklabels([row["display_name"] for row in comparison_rows])
    ax.set_xlabel("Δ total cost vs State-only (€ million)")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.45)
    fig.tight_layout()
    figure_dir = root / "figures"
    _save_figure(
        fig, figure_dir / "supplementary_figure_s3_future_information"
    )
    source_rows = [
        {
            "method": row["method"],
            "mean_delta_cost_vs_state_only_eur": row[
                "mean_delta_cost_vs_state_only_eur"
            ],
            "ci95_low_eur": row[
                "delta_cost_vs_state_only_ci95_low_eur"
            ],
            "ci95_high_eur": row[
                "delta_cost_vs_state_only_ci95_high_eur"
            ],
        }
        for row in comparison_rows
    ]
    _write_csv(
        figure_dir
        / "source_data"
        / "supplementary_figure_s3_source_data.csv",
        source_rows,
    )
    _write_json(
        figure_dir
        / "source_data"
        / "supplementary_figure_s3_metadata.json",
        _metadata(
            "Supplementary Figure S3",
            "Test whether structured or full-sequence future information "
            "changes paired cost relative to State-only.",
        ),
    )


def aggregate_e4(results_root: Path) -> None:
    root = results_root / "E4"
    formal = root / "formal_stress_seeds_9000031-9000060_run01"
    stress_levels = ("low", "medium", "high")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    q_matrices = {}
    greedy_costs = {}
    for stress in stress_levels:
        matrices = _load_method(formal / stress)
        q_matrices[stress] = matrices
        greedy_matrix = matrices["greedy_total_cost_eur"]
        if not np.allclose(greedy_matrix, greedy_matrix[:1], atol=1e-6):
            raise ValueError(f"Greedy results differ by model seed at {stress}")
        greedy_costs[stress] = greedy_matrix[:1]
        q_row = _summary_row(
            f"iterative_q_{stress}",
            f"Iterative Action-Q ({stress})",
            matrices,
            rng,
        )
        q_row["controller"] = "iterative_action_q"
        q_row["stress_level"] = stress
        rows.append(q_row)
        greedy_ci = _hierarchical_ci(greedy_costs[stress], rng)
        rows.append(
            {
                "method": f"greedy_{stress}",
                "display_name": f"Greedy ({stress})",
                "model_seeds": 0,
                "test_seeds": len(TEST_SEEDS),
                "mean_total_cost_eur": float(greedy_costs[stress].mean()),
                "total_cost_ci95_low_eur": greedy_ci[0],
                "total_cost_ci95_high_eur": greedy_ci[1],
                "mean_delta_cost_vs_greedy_eur": 0.0,
                "delta_cost_ci95_low_eur": 0.0,
                "delta_cost_ci95_high_eur": 0.0,
                "mean_vented_t": float(
                    matrices["greedy_vented_t"][0].mean()
                ),
                "mean_stored_t": float(
                    matrices["greedy_stored_t"][0].mean()
                ),
                "wins_vs_greedy": 0,
                "ties_vs_greedy": len(TEST_SEEDS),
                "losses_vs_greedy": 0,
                "controller": "greedy",
                "stress_level": stress,
            }
        )

    table_dir = root / "tables"
    _write_csv(table_dir / "supplementary_table_s2_stress.csv", rows)
    _write_json(
        table_dir / "supplementary_table_s2_stress.json",
        {"kind": "E4_frozen_model_stress_test", "rows": rows},
    )

    fig, ax = plt.subplots(figsize=(89 / 25.4, 72 / 25.4))
    positions = np.arange(len(stress_levels))
    for controller, label, color in (
        ("greedy", "Greedy", "#606060"),
        ("iterative_action_q", "Iterative Action-Q", "#C77991"),
    ):
        selected = [
            next(
                row
                for row in rows
                if row["controller"] == controller
                and row["stress_level"] == stress
            )
            for stress in stress_levels
        ]
        y = np.asarray(
            [row["mean_total_cost_eur"] for row in selected], dtype=float
        ) / 1_000_000.0
        low = np.asarray(
            [row["total_cost_ci95_low_eur"] for row in selected],
            dtype=float,
        ) / 1_000_000.0
        high = np.asarray(
            [row["total_cost_ci95_high_eur"] for row in selected],
            dtype=float,
        ) / 1_000_000.0
        ax.errorbar(
            positions,
            y,
            yerr=np.vstack((y - low, high - y)),
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.2,
            capsize=2.5,
            label=label,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(("Low", "Medium", "High"))
    ax.set_xlabel("Composite disturbance stress")
    ax.set_ylabel("Total cost (€ million)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax.legend(fontsize=5.8)
    fig.tight_layout()
    figure_dir = root / "figures"
    _save_figure(fig, figure_dir / "figure_5_stress_robustness")
    source_rows = [
        {
            "controller": row["controller"],
            "stress_level": row["stress_level"],
            "mean_total_cost_eur": row["mean_total_cost_eur"],
            "ci95_low_eur": row["total_cost_ci95_low_eur"],
            "ci95_high_eur": row["total_cost_ci95_high_eur"],
        }
        for row in rows
    ]
    _write_csv(
        figure_dir / "source_data" / "figure_5_source_data.csv",
        source_rows,
    )
    _write_json(
        figure_dir / "source_data" / "figure_5_metadata.json",
        _metadata(
            "Figure 5",
            "Evaluate degradation of the frozen E1 model and Greedy from "
            "Low through High composite disturbance stress.",
        ),
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    {
        "E2": aggregate_e2,
        "E3": aggregate_e3,
        "E4": aggregate_e4,
    }[args.experiment](args.results_root)


if __name__ == "__main__":
    main()
