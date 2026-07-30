"""Aggregate validation-only P2 sampler ablations and make paper-ready plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


VARIANTS = (
    "b_gate_only",
    "c_dedup_balanced",
    "d_dedup_advantage",
)
GATES = ("h4_m040_w12_c12", "h3_m040_w12_c12")
MODEL_SEEDS = (0, 1, 2)
VALIDATION_SEEDS = tuple(range(8100001, 8100021))
COLORS = {
    "b_gate_only": "#484878",
    "c_dedup_balanced": "#7884B4",
    "d_dedup_advantage": "#C7849A",
}
LABELS = {
    "b_gate_only": "B  H3 collection",
    "c_dedup_balanced": "C  + dedup/balance",
    "d_dedup_advantage": "D  + advantage strata",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    return parser.parse_args(argv)


def _load_rows(run_root, variant, model_seed):
    eval_dir = (
        run_root
        / "validation"
        / variant
        / f"model_seed_{model_seed}"
    )
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


def _matrix(rows, key):
    by_pair = {
        (int(row["model_seed"]), int(row["seed"])): float(row[key])
        for row in rows
    }
    return np.asarray(
        [
            [by_pair[(model_seed, seed)] for seed in VALIDATION_SEEDS]
            for model_seed in MODEL_SEEDS
        ],
        dtype=np.float64,
    )


def _hierarchical_mean_ci(matrix, seed=0):
    rng = np.random.default_rng(seed)
    replicates = np.empty(20_000, dtype=np.float64)
    for index in range(len(replicates)):
        model_indices = rng.integers(
            0, matrix.shape[0], size=matrix.shape[0]
        )
        scenario_indices = rng.integers(
            0, matrix.shape[1], size=matrix.shape[1]
        )
        replicates[index] = matrix[np.ix_(
            model_indices, scenario_indices
        )].mean()
    return [
        float(np.quantile(replicates, 0.025)),
        float(np.quantile(replicates, 0.975)),
    ]


def _metrics(rows, bootstrap_seed):
    delta = _matrix(rows, "delta_total_cost_eur")
    vented = _matrix(rows, "vented_t")
    overrides = _matrix(rows, "override_events")
    return {
        "episodes": int(delta.size),
        "model_seeds": len(MODEL_SEEDS),
        "controller_validation_seeds": len(VALIDATION_SEEDS),
        "mean_delta_total_cost_eur": float(delta.mean()),
        "mean_delta_total_cost_95pct_hierarchical_ci_eur": (
            _hierarchical_mean_ci(delta, bootstrap_seed)
        ),
        "p90_delta_total_cost_eur": float(np.quantile(delta, 0.9)),
        "median_delta_total_cost_eur": float(np.median(delta)),
        "win_fraction_vs_greedy": float(np.mean(delta < -1e-6)),
        "mean_vented_t": float(vented.mean()),
        "mean_override_events": float(overrides.mean()),
        "model_seed_mean_delta_total_cost_eur": [
            float(value) for value in delta.mean(axis=1)
        ],
    }


def _write_metrics_csv(path, metric_rows):
    fields = [
        "variant",
        "gate",
        "episodes",
        "mean_delta_total_cost_eur",
        "ci_low_eur",
        "ci_high_eur",
        "p90_delta_total_cost_eur",
        "median_delta_total_cost_eur",
        "win_fraction_vs_greedy",
        "mean_vented_t",
        "mean_override_events",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metric_rows:
            metrics = row["metrics"]
            ci = metrics[
                "mean_delta_total_cost_95pct_hierarchical_ci_eur"
            ]
            writer.writerow(
                {
                    "variant": row["variant"],
                    "gate": row["gate"],
                    "episodes": metrics["episodes"],
                    "mean_delta_total_cost_eur": (
                        metrics["mean_delta_total_cost_eur"]
                    ),
                    "ci_low_eur": ci[0],
                    "ci_high_eur": ci[1],
                    "p90_delta_total_cost_eur": (
                        metrics["p90_delta_total_cost_eur"]
                    ),
                    "median_delta_total_cost_eur": (
                        metrics["median_delta_total_cost_eur"]
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


def _plot(run_root, metrics_by_key):
    figure_dir = run_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_mean, ax_tail) = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 82 / 25.4),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    x = np.arange(len(VARIANTS), dtype=float)
    gate_offsets = {
        "h4_m040_w12_c12": -0.10,
        "h3_m040_w12_c12": 0.10,
    }
    gate_markers = {
        "h4_m040_w12_c12": "o",
        "h3_m040_w12_c12": "s",
    }
    gate_labels = {
        "h4_m040_w12_c12": "H4 deployment gate",
        "h3_m040_w12_c12": "H3 deployment gate",
    }
    for gate in GATES:
        for index, variant in enumerate(VARIANTS):
            metrics = metrics_by_key[(variant, gate)]
            mean = metrics["mean_delta_total_cost_eur"] / 1000.0
            low, high = metrics[
                "mean_delta_total_cost_95pct_hierarchical_ci_eur"
            ]
            ax_mean.errorbar(
                x[index] + gate_offsets[gate],
                mean,
                yerr=[
                    [mean - low / 1000.0],
                    [high / 1000.0 - mean],
                ],
                fmt=gate_markers[gate],
                ms=5,
                mfc=(
                    COLORS[variant]
                    if gate == "h4_m040_w12_c12"
                    else "white"
                ),
                mec=COLORS[variant],
                ecolor=COLORS[variant],
                elinewidth=1.1,
                capsize=2.5,
                zorder=3,
            )
            ax_tail.scatter(
                metrics["mean_override_events"],
                metrics["p90_delta_total_cost_eur"] / 1000.0,
                marker=gate_markers[gate],
                s=28,
                facecolor=(
                    COLORS[variant]
                    if gate == "h4_m040_w12_c12"
                    else "white"
                ),
                edgecolor=COLORS[variant],
                linewidth=1.0,
                zorder=3,
            )
            if gate == "h4_m040_w12_c12":
                ax_tail.annotate(
                    variant[0].upper(),
                    (
                        metrics["mean_override_events"],
                        metrics["p90_delta_total_cost_eur"] / 1000.0,
                    ),
                    xytext=(4, 3),
                    textcoords="offset points",
                    color=COLORS[variant],
                    fontsize=7,
                    fontweight="bold",
                )
    ax_mean.axhline(0.0, color="#8F8F8F", lw=0.8, ls="--", zorder=1)
    ax_mean.set_xticks(x)
    ax_mean.set_xticklabels(
        [LABELS[variant] for variant in VARIANTS],
        rotation=15,
        ha="right",
    )
    ax_mean.set_ylabel("Mean cost difference vs Greedy (€k/episode)")
    ax_mean.set_title("Validation performance", loc="left", fontweight="bold")
    ax_tail.axhline(0.0, color="#8F8F8F", lw=0.8, ls="--", zorder=1)
    ax_tail.set_xlabel("Mean override events")
    ax_tail.set_ylabel("p90 cost difference vs Greedy (€k)")
    ax_tail.set_title("Tail–intervention trade-off", loc="left", fontweight="bold")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=gate_markers[gate],
            color="#606060",
            markerfacecolor=(
                "#606060" if gate == "h4_m040_w12_c12" else "white"
            ),
            lw=0,
            label=gate_labels[gate],
        )
        for gate in GATES
    ]
    ax_mean.legend(handles=handles, loc="best", fontsize=6.5)
    ax_mean.text(
        -0.14,
        1.05,
        "a",
        transform=ax_mean.transAxes,
        fontweight="bold",
        fontsize=8,
    )
    ax_tail.text(
        -0.18,
        1.05,
        "b",
        transform=ax_tail.transAxes,
        fontweight="bold",
        fontsize=8,
    )
    fig.tight_layout(pad=1.0)
    base = figure_dir / "p2_validation_sampling_comparison"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def run(args):
    run_root = Path(args.run_root)
    rows_by_key = {}
    metrics_by_key = {}
    metric_rows = []
    counter = 0
    for variant in VARIANTS:
        all_rows = []
        for model_seed in MODEL_SEEDS:
            all_rows.extend(_load_rows(run_root, variant, model_seed))
        for gate in GATES:
            rows = [row for row in all_rows if row["gate"] == gate]
            if len(rows) != len(MODEL_SEEDS) * len(VALIDATION_SEEDS):
                raise ValueError(
                    f"incomplete evaluation for {variant}/{gate}"
                )
            rows_by_key[(variant, gate)] = rows
            metrics = _metrics(rows, counter)
            counter += 1
            metrics_by_key[(variant, gate)] = metrics
            metric_rows.append(
                {"variant": variant, "gate": gate, "metrics": metrics}
            )

    primary_gate = "h4_m040_w12_c12"
    reweight_candidates = (
        "c_dedup_balanced",
        "d_dedup_advantage",
    )
    winner = min(
        reweight_candidates,
        key=lambda variant: metrics_by_key[
            (variant, primary_gate)
        ]["mean_delta_total_cost_eur"],
    )
    other = next(
        variant for variant in reweight_candidates if variant != winner
    )
    winner_delta = _matrix(
        rows_by_key[(winner, primary_gate)], "delta_total_cost_eur"
    )
    other_delta = _matrix(
        rows_by_key[(other, primary_gate)], "delta_total_cost_eur"
    )
    paired = winner_delta - other_delta
    baseline_metrics = metrics_by_key[("b_gate_only", primary_gate)]
    winner_metrics = metrics_by_key[(winner, primary_gate)]
    selection = {
        "selection_gate": primary_gate,
        "baseline_retained": "b_gate_only",
        "selected_reweight_variant": winner,
        "other_reweight_variant": other,
        "selected_minus_other_mean_cost_eur": float(paired.mean()),
        "selected_minus_other_mean_cost_95pct_hierarchical_ci_eur": (
            _hierarchical_mean_ci(paired, 91)
        ),
        "selected_minus_baseline_mean_cost_eur": float(
            winner_metrics["mean_delta_total_cost_eur"]
            - baseline_metrics["mean_delta_total_cost_eur"]
        ),
        "selected_minus_baseline_p90_cost_eur": float(
            winner_metrics["p90_delta_total_cost_eur"]
            - baseline_metrics["p90_delta_total_cost_eur"]
        ),
        "selected_minus_baseline_mean_vented_t": float(
            winner_metrics["mean_vented_t"]
            - baseline_metrics["mean_vented_t"]
        ),
        "formal_test_access": False,
    }
    result = {
        "kind": "iterative_h3_sampler_p2_validation_summary",
        "figure_contract": {
            "core_conclusion": (
                "Determine whether deduplication, stage balancing, and "
                "advantage strata improve H3-collected iterative-Q data "
                "without worsening tail cost or intervention intensity."
            ),
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "replicates": (
                "3 model seeds x 20 locked controller-validation seeds"
            ),
            "interval": (
                "95% hierarchical bootstrap over model and scenario seeds"
            ),
            "baseline": "Greedy controller",
        },
        "formal_test_access": False,
        "metrics": metric_rows,
        "selection": selection,
    }
    summary_path = run_root / "p2_validation_summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_metrics_csv(run_root / "p2_validation_metrics.csv", metric_rows)
    (run_root / "p2_selection.json").write_text(
        json.dumps(selection, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Iterative H3 collection and sampler P2 validation",
        "",
        "This is a validation-only comparison; no formal test stage is included.",
        "",
        f"- Baseline retained: `b_gate_only`",
        f"- Selected reweight route: `{winner}`",
        (
            "- Selected minus other reweight route: "
            f"{paired.mean() / 1000.0:+.1f} kEUR/episode"
        ),
        (
            "- Selected minus H3-only baseline: "
            f"{selection['selected_minus_baseline_mean_cost_eur'] / 1000.0:+.1f} "
            "kEUR/episode"
        ),
        "",
        "Primary selection uses the common H4/M0.4/W12/C12 deployment gate; "
        "H3 deployment is a fixed secondary diagnostic.",
    ]
    (run_root / "p2_validation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    _plot(run_root, metrics_by_key)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
