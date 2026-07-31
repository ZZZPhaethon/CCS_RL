"""Aggregate P2-P4 validation-only progression for retained H3 routes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.aggregate_iterative_h3_sampler_validation import (
    COLORS,
    GATES,
    LABELS,
    MODEL_SEEDS,
    VALIDATION_SEEDS,
    _hierarchical_mean_ci,
    _matrix,
    _metrics,
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False

STAGES = ("p2", "p3", "p4")
PRIMARY_GATE = "h4_m040_w12_c12"
STAGE_MARKERS = {"p2": "o", "p3": "s", "p4": "^"}
REFERENCE_METHODS = ("one_shot_matched", "iterative_p4")
REFERENCE_LABELS = {
    "one_shot_matched": "Existing one-shot",
    "iterative_p4": "Existing G60-P4",
}
REFERENCE_STYLES = {
    "one_shot_matched": "--",
    "iterative_p4": ":",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--reference-root",
        default=(
            "experiments_results/E2/"
            "validation_gate_sweep_seeds_8100001-8100020_run01"
        ),
    )
    return parser.parse_args(argv)


def _load_rows(run_root, variant, model_seed, stage):
    eval_dir = (
        run_root
        / "validation"
        / variant
        / f"model_seed_{model_seed}"
    )
    if stage != "p2":
        eval_dir = eval_dir / stage
    summary = json.loads(
        (eval_dir / "summary.json").read_text(encoding="utf-8")
    )
    if summary.get("validation_only") is not True:
        raise ValueError(f"non-validation result found in {eval_dir}")
    if tuple(summary["eval_seeds"]) != VALIDATION_SEEDS:
        raise ValueError(f"unexpected validation seeds in {eval_dir}")
    with (eval_dir / "evaluation.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["model_seed"] = model_seed
        row["seed"] = int(row["seed"])
        for key in (
            "delta_total_cost_eur",
            "vented_t",
            "override_events",
            "stored_t",
            "unit_cost_eur_per_t",
        ):
            row[key] = float(row[key])
    return rows


def _write_metrics(path, metric_rows):
    fields = [
        "variant",
        "stage",
        "gate",
        "episodes",
        "mean_delta_total_cost_eur",
        "ci_low_eur",
        "ci_high_eur",
        "p90_delta_total_cost_eur",
        "win_fraction_vs_greedy",
        "mean_vented_t",
        "mean_override_events",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metric_rows:
            metrics = row["metrics"]
            low, high = metrics[
                "mean_delta_total_cost_95pct_hierarchical_ci_eur"
            ]
            writer.writerow(
                {
                    "variant": row["variant"],
                    "stage": row["stage"],
                    "gate": row["gate"],
                    "episodes": metrics["episodes"],
                    "mean_delta_total_cost_eur": (
                        metrics["mean_delta_total_cost_eur"]
                    ),
                    "ci_low_eur": low,
                    "ci_high_eur": high,
                    "p90_delta_total_cost_eur": (
                        metrics["p90_delta_total_cost_eur"]
                    ),
                    "win_fraction_vs_greedy": (
                        metrics["win_fraction_vs_greedy"]
                    ),
                    "mean_vented_t": metrics["mean_vented_t"],
                    "mean_override_events": (
                        metrics["mean_override_events"]
                    ),
                }
            )


def _load_reference_rows(reference_root, method):
    rows = []
    for model_seed in MODEL_SEEDS:
        eval_dir = reference_root / method / f"model_seed_{model_seed}"
        summary = json.loads(
            (eval_dir / "summary.json").read_text(encoding="utf-8")
        )
        if summary.get("validation_only") is not True:
            raise ValueError(f"non-validation reference: {eval_dir}")
        if tuple(summary["eval_seeds"]) != VALIDATION_SEEDS:
            raise ValueError(f"unexpected reference seeds: {eval_dir}")
        with (eval_dir / "evaluation.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            loaded = list(csv.DictReader(handle))
        for row in loaded:
            if row["gate"] != PRIMARY_GATE:
                continue
            row["model_seed"] = model_seed
            row["seed"] = int(row["seed"])
            for key in (
                "delta_total_cost_eur",
                "vented_t",
                "override_events",
                "stored_t",
                "unit_cost_eur_per_t",
            ):
                row[key] = float(row[key])
            rows.append(row)
    expected = len(MODEL_SEEDS) * len(VALIDATION_SEEDS)
    if len(rows) != expected:
        raise ValueError(f"incomplete reference rows for {method}")
    return rows


def _plot(run_root, variants, metrics, references):
    figure_dir = run_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(183 / 25.4, 75 / 25.4),
        gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]},
    )
    ax_mean, ax_tail, ax_safety = axes
    x = np.arange(len(STAGES), dtype=float)
    offsets = {variants[0]: -0.05, variants[1]: 0.05}
    for variant in variants:
        means = []
        lows = []
        highs = []
        tails = []
        for stage in STAGES:
            row = metrics[(variant, stage, PRIMARY_GATE)]
            mean = row["mean_delta_total_cost_eur"] / 1000.0
            low, high = row[
                "mean_delta_total_cost_95pct_hierarchical_ci_eur"
            ]
            means.append(mean)
            lows.append(mean - low / 1000.0)
            highs.append(high / 1000.0 - mean)
            tails.append(row["p90_delta_total_cost_eur"] / 1000.0)
            ax_safety.scatter(
                row["mean_override_events"],
                row["mean_vented_t"],
                marker=STAGE_MARKERS[stage],
                s=30,
                color=COLORS[variant],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax_safety.annotate(
                stage.upper(),
                (row["mean_override_events"], row["mean_vented_t"]),
                xytext=(3, 3),
                textcoords="offset points",
                color=COLORS[variant],
                fontsize=6,
            )
        ax_mean.errorbar(
            x + offsets[variant],
            means,
            yerr=[lows, highs],
            marker="o",
            ms=4.5,
            lw=1.2,
            capsize=2.5,
            color=COLORS[variant],
            label=LABELS[variant],
        )
        ax_tail.plot(
            x,
            tails,
            marker="o",
            ms=4.5,
            lw=1.2,
            color=COLORS[variant],
            label=LABELS[variant],
        )
    for method in REFERENCE_METHODS:
        reference = references[method]
        ax_mean.axhline(
            reference["mean_delta_total_cost_eur"] / 1000.0,
            color="#8F8F8F",
            lw=1.0,
            ls=REFERENCE_STYLES[method],
            label=REFERENCE_LABELS[method],
        )
        ax_tail.axhline(
            reference["p90_delta_total_cost_eur"] / 1000.0,
            color="#8F8F8F",
            lw=1.0,
            ls=REFERENCE_STYLES[method],
        )
        ax_safety.scatter(
            reference["mean_override_events"],
            reference["mean_vented_t"],
            marker="x" if method == "one_shot_matched" else "D",
            s=26,
            color="#606060",
            linewidth=1.0,
            zorder=3,
        )
        ax_safety.annotate(
            "One-shot" if method == "one_shot_matched" else "Old P4",
            (
                reference["mean_override_events"],
                reference["mean_vented_t"],
            ),
            xytext=(3, -8),
            textcoords="offset points",
            color="#606060",
            fontsize=5.8,
        )
    for axis in (ax_mean, ax_tail):
        axis.axhline(0.0, color="#8F8F8F", lw=0.8, ls="--")
        axis.set_xticks(x)
        axis.set_xticklabels([stage.upper() for stage in STAGES])
    ax_mean.set_ylabel("Mean cost difference vs Greedy (€k/episode)")
    ax_mean.set_title("Recursive validation gain", loc="left", fontweight="bold")
    ax_mean.legend(fontsize=6.3, loc="best")
    ax_tail.set_ylabel("p90 cost difference vs Greedy (€k)")
    ax_tail.set_title("Tail cost", loc="left", fontweight="bold")
    ax_safety.set_xlabel("Mean override events")
    ax_safety.set_ylabel("Mean vented CO$_2$ (t)")
    ax_safety.set_title("Safety–intervention", loc="left", fontweight="bold")
    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(
            -0.16,
            1.05,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8,
        )
    fig.tight_layout(pad=1.0)
    base = figure_dir / "recursive_validation_progression"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def run(args):
    run_root = Path(args.run_root)
    selection = json.loads(
        (run_root / "p2_selection.json").read_text(encoding="utf-8")
    )
    selected = selection["selected_reweight_variant"]
    variants = ("b_gate_only", selected)
    rows_by_key = {}
    metrics_by_key = {}
    metric_rows = []
    bootstrap_seed = 0
    for variant in variants:
        for stage in STAGES:
            all_rows = []
            for model_seed in MODEL_SEEDS:
                all_rows.extend(
                    _load_rows(run_root, variant, model_seed, stage)
                )
            for gate in GATES:
                rows = [row for row in all_rows if row["gate"] == gate]
                expected = len(MODEL_SEEDS) * len(VALIDATION_SEEDS)
                if len(rows) != expected:
                    raise ValueError(
                        f"incomplete evaluation for {variant}/{stage}/{gate}"
                    )
                key = (variant, stage, gate)
                rows_by_key[key] = rows
                metrics_by_key[key] = _metrics(rows, bootstrap_seed)
                bootstrap_seed += 1
                metric_rows.append(
                    {
                        "variant": variant,
                        "stage": stage,
                        "gate": gate,
                        "metrics": metrics_by_key[key],
                    }
                )

    stage_changes = []
    for variant in variants:
        p2 = _matrix(
            rows_by_key[(variant, "p2", PRIMARY_GATE)],
            "delta_total_cost_eur",
        )
        for stage in ("p3", "p4"):
            current = _matrix(
                rows_by_key[(variant, stage, PRIMARY_GATE)],
                "delta_total_cost_eur",
            )
            paired = current - p2
            stage_changes.append(
                {
                    "variant": variant,
                    "stage": stage,
                    "mean_change_vs_p2_eur": float(paired.mean()),
                    "mean_change_vs_p2_95pct_hierarchical_ci_eur": (
                        _hierarchical_mean_ci(
                            paired, 100 + len(stage_changes)
                        )
                    ),
                }
            )
    reference_root = Path(args.reference_root)
    reference_rows = {
        method: _load_reference_rows(reference_root, method)
        for method in REFERENCE_METHODS
    }
    references = {
        method: _metrics(rows, 200 + index)
        for index, (method, rows) in enumerate(reference_rows.items())
    }
    final_comparisons = []
    for variant in variants:
        p4 = _matrix(
            rows_by_key[(variant, "p4", PRIMARY_GATE)],
            "delta_total_cost_eur",
        )
        for method in REFERENCE_METHODS:
            reference = _matrix(
                reference_rows[method], "delta_total_cost_eur"
            )
            paired = p4 - reference
            final_comparisons.append(
                {
                    "variant": variant,
                    "reference": method,
                    "p4_minus_reference_mean_cost_eur": float(
                        paired.mean()
                    ),
                    "p4_minus_reference_mean_cost_95pct_hierarchical_ci_eur": (
                        _hierarchical_mean_ci(
                            paired, 300 + len(final_comparisons)
                        )
                    ),
                }
            )
    baseline_p4 = _matrix(
        rows_by_key[("b_gate_only", "p4", PRIMARY_GATE)],
        "delta_total_cost_eur",
    )
    selected_p4 = _matrix(
        rows_by_key[(selected, "p4", PRIMARY_GATE)],
        "delta_total_cost_eur",
    )
    retained_variant_p4_difference = selected_p4 - baseline_p4
    retained_variant_comparison = {
        "variant": selected,
        "reference_variant": "b_gate_only",
        "p4_minus_reference_mean_cost_eur": float(
            retained_variant_p4_difference.mean()
        ),
        "p4_minus_reference_mean_cost_95pct_hierarchical_ci_eur": (
            _hierarchical_mean_ci(retained_variant_p4_difference, 400)
        ),
    }
    result = {
        "kind": "iterative_h3_recursive_validation_summary",
        "formal_test_access": False,
        "retained_variants": list(variants),
        "primary_gate": PRIMARY_GATE,
        "replicates": "3 model seeds x 20 controller-validation seeds",
        "interval": (
            "95% hierarchical bootstrap over model and scenario seeds"
        ),
        "metrics": metric_rows,
        "paired_stage_changes": stage_changes,
        "fixed_validation_references": references,
        "p4_paired_reference_comparisons": final_comparisons,
        "retained_variant_p4_comparison": retained_variant_comparison,
    }
    (run_root / "recursive_validation_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_metrics(
        run_root / "recursive_validation_metrics.csv", metric_rows
    )
    report = [
        "# Iterative H3 recursive validation",
        "",
        "Validation-only P2–P4 comparison; no formal test stage is included.",
        "",
    ]
    for row in stage_changes:
        low, high = row[
            "mean_change_vs_p2_95pct_hierarchical_ci_eur"
        ]
        report.append(
            f"- `{row['variant']}` {row['stage'].upper()} minus P2: "
            f"{row['mean_change_vs_p2_eur'] / 1000.0:+.1f} kEUR/episode "
            f"(95% hierarchical CI {low / 1000.0:+.1f} to "
            f"{high / 1000.0:+.1f})."
        )
    report.extend(["", "## Fixed validation references", ""])
    for row in final_comparisons:
        low, high = row[
            "p4_minus_reference_mean_cost_95pct_hierarchical_ci_eur"
        ]
        report.append(
            f"- `{row['variant']}` P4 minus `{row['reference']}`: "
            f"{row['p4_minus_reference_mean_cost_eur'] / 1000.0:+.1f} "
            f"kEUR/episode (95% hierarchical CI "
            f"{low / 1000.0:+.1f} to {high / 1000.0:+.1f})."
        )
    low, high = retained_variant_comparison[
        "p4_minus_reference_mean_cost_95pct_hierarchical_ci_eur"
    ]
    report.extend(
        [
            "",
            "## Retained-variant comparison",
            "",
            f"- `{selected}` P4 minus `b_gate_only` P4: "
            f"{retained_variant_comparison['p4_minus_reference_mean_cost_eur'] / 1000.0:+.1f} "
            f"kEUR/episode (95% hierarchical CI "
            f"{low / 1000.0:+.1f} to {high / 1000.0:+.1f}).",
        ]
    )
    (run_root / "recursive_validation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    _plot(run_root, variants, metrics_by_key, references)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
