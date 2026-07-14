from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output" / "rl_forecast" / "gnn_attribution_3x2"
NEW_EVAL = BASE / "eval_101_120"
SMALL_EVAL = ROOT / "output" / "rl_forecast" / "aligned_forecast_v4_bc" / "eval_101_120"
VARIANTS = {
    "small_original": "tcn_mode_destination",
    "small_fixed": "fixed_scale_tcn_mode_destination",
    "large_original": "larger_mlp_mode_destination",
    "large_fixed": "fixed_scale_larger_mlp_mode_destination",
    "edge_original": "edge_gnn_mode_destination",
    "edge_fixed": "fixed_scale_edge_gnn_mode_destination",
}
LABELS = {
    "small_original": "Small MLP + Original TCN",
    "small_fixed": "Small MLP + FixedScale TCN",
    "large_original": "Large MLP + Original TCN",
    "large_fixed": "Large MLP + FixedScale TCN",
    "edge_original": "Edge-GNN + Original TCN",
    "edge_fixed": "Edge-GNN + FixedScale TCN",
}
METRICS = (
    "vented_t",
    "stored_t",
    "operating_cost",
    "total_cost",
    "cost_per_stored_t",
    "total_cost_per_stored_t",
)
T_CRIT_95_DF4 = 2.7764451051977987


def result_path(key: str, seed: int) -> Path:
    variant = VARIANTS[key]
    root = SMALL_EVAL if key.startswith("small_") else NEW_EVAL
    return root / f"results_{variant}_seed{seed}.csv"


def read_rows(key: str, seed: int):
    with result_path(key, seed).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def seed_means():
    values = {}
    for key in VARIANTS:
        for seed in range(5):
            rows = read_rows(key, seed)
            if len(rows) != 40:
                raise ValueError(f"expected 40 rows for {key} seed {seed}, got {len(rows)}")
            for deterministic in (True, False):
                selected = [
                    row for row in rows
                    if (row["deterministic"].lower() == "true") == deterministic
                ]
                if len(selected) != 20:
                    raise ValueError(
                        f"expected 20 mode rows for {key} seed {seed}, got {len(selected)}"
                    )
                values[key, seed, deterministic] = {
                    metric: statistics.mean(float(row[metric]) for row in selected)
                    for metric in METRICS
                }
    return values


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def ci95(values):
    mean, sd = mean_sd(values)
    half = T_CRIT_95_DF4 * sd / math.sqrt(len(values))
    return mean, mean - half, mean + half


def write_aggregate(values):
    rows = []
    for key in VARIANTS:
        for deterministic in (True, False):
            row = {
                "key": key,
                "variant": VARIANTS[key],
                "label": LABELS[key],
                "deterministic": deterministic,
            }
            for metric in METRICS:
                metric_values = [
                    values[key, seed, deterministic][metric]
                    for seed in range(5)
                ]
                mean, sd = mean_sd(metric_values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
            rows.append(row)
    destination = BASE / "aggregate_metrics.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def contrast_values(values, left, right, deterministic, metric):
    return [
        values[left, seed, deterministic][metric]
        - values[right, seed, deterministic][metric]
        for seed in range(5)
    ]


def interaction_values(values, deterministic, metric):
    return [
        (
            values["edge_fixed", seed, deterministic][metric]
            - values["edge_original", seed, deterministic][metric]
        )
        - (
            values["large_fixed", seed, deterministic][metric]
            - values["large_original", seed, deterministic][metric]
        )
        for seed in range(5)
    ]


def write_contrasts(values):
    definitions = {
        "future_effect_small": ("small_fixed", "small_original"),
        "future_effect_large": ("large_fixed", "large_original"),
        "future_effect_edge": ("edge_fixed", "edge_original"),
        "capacity_effect_original": ("large_original", "small_original"),
        "capacity_effect_fixed": ("large_fixed", "small_fixed"),
        "graph_effect_original": ("edge_original", "large_original"),
        "graph_effect_fixed": ("edge_fixed", "large_fixed"),
    }
    rows = []
    for deterministic in (True, False):
        for metric in METRICS:
            for name, (left, right) in definitions.items():
                differences = contrast_values(
                    values, left, right, deterministic, metric
                )
                mean, lower, upper = ci95(differences)
                rows.append(
                    {
                        "contrast": name,
                        "left": left,
                        "right": right,
                        "deterministic": deterministic,
                        "metric": metric,
                        "mean_difference": mean,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                    }
                )
            differences = interaction_values(values, deterministic, metric)
            mean, lower, upper = ci95(differences)
            rows.append(
                {
                    "contrast": "graph_by_future_interaction",
                    "left": "(edge_fixed-edge_original)",
                    "right": "(large_fixed-large_original)",
                    "deterministic": deterministic,
                    "metric": metric,
                    "mean_difference": mean,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    destination = BASE / "paired_contrasts.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def fmt(value):
    return f"{value:,.3f}"


def markdown_table(aggregate, deterministic):
    selected = [row for row in aggregate if row["deterministic"] == deterministic]
    lines = [
        "| Method | Vented t | Stored t | Operating EUR | Total EUR | Operating EUR/t | Total EUR/t |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| "
            + row["label"]
            + " | "
            + " | ".join(
                f"{fmt(row[f'{metric}_mean'])} +/- {fmt(row[f'{metric}_sd'])}"
                for metric in METRICS
            )
            + " |"
        )
    return lines


def write_summary(aggregate, contrasts):
    lines = [
        "# GNN attribution 3x2 BC results",
        "",
        "All methods use forecast schema v4, decision-only BC, model seeds 0--4, "
        "and paired evaluation seeds 101--120. Values are mean +/- sample SD "
        "across the five model-seed means.",
        "",
        "## Deterministic",
        "",
        *markdown_table(aggregate, True),
        "",
        "## Stochastic",
        "",
        *markdown_table(aggregate, False),
        "",
        "## Primary paired deterministic contrasts",
        "",
        "Differences are left minus right; intervals are two-sided 95% t intervals "
        "over five paired model-seed means.",
        "",
        "| Contrast | Metric | Mean difference | 95% CI |",
        "|---|---|---:|---:|",
    ]
    primary = {
        "future_effect_small",
        "future_effect_large",
        "future_effect_edge",
        "graph_effect_original",
        "graph_effect_fixed",
        "graph_by_future_interaction",
    }
    shown_metrics = {
        "vented_t",
        "stored_t",
        "total_cost",
        "total_cost_per_stored_t",
    }
    for row in contrasts:
        if (
            row["deterministic"]
            and row["contrast"] in primary
            and row["metric"] in shown_metrics
        ):
            lines.append(
                f"| {row['contrast']} | {row['metric']} | "
                f"{fmt(row['mean_difference'])} | "
                f"[{fmt(row['ci95_lower'])}, {fmt(row['ci95_upper'])}] |"
            )
    audit_path = BASE / "forecast_use_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## Forecast-use diagnostics",
                "",
                "| Method | Active seeds | Feature L2 | Input gradient L2 | Shuffle TV | Argmax change |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for key, variant in VARIANTS.items():
            rows = [row for row in audit if row["variant"] == variant]
            feature_l2 = statistics.mean(
                row["forecast_feature_l2"] for row in rows
            )
            gradient_l2 = statistics.mean(
                row["forecast_input_gradient_l2"] for row in rows
            )
            shuffle_tv = statistics.mean(
                row["mean_probability_tv"] for row in rows
            )
            argmax_change = statistics.mean(
                row["argmax_row_change_rate"] for row in rows
            )
            active_seeds = sum(
                row["forecast_feature_l2"] > 0.0
                and row["forecast_input_gradient_l2"] > 0.0
                for row in rows
            )
            lines.append(
                f"| {LABELS[key]} | "
                + f"{active_seeds}/5 | {feature_l2:.3f} | {gradient_l2:.3e} | "
                + f"{shuffle_tv:.4f} | {100.0 * argmax_change:.2f}%"
                + " |"
            )
    destination = BASE / "summary.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    values = seed_means()
    aggregate = write_aggregate(values)
    contrasts = write_contrasts(values)
    write_summary(aggregate, contrasts)
    print(BASE / "summary.md")


if __name__ == "__main__":
    main()
