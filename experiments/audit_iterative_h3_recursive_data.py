"""Audit G1-G3 data quality and cross-stage duplication for retained routes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.audit_iterative_sampling_data import _stage_quality
from scripts.train_iterative_action_q import (
    _combined_dataset,
    _load_collection,
    root_sampling_weights,
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

MODEL_SEEDS = (0, 1, 2)
STAGES = ("g0", "g1", "g2", "g3")
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


def _write_stage_csv(path, rows):
    fields = [
        "variant",
        "model_seed",
        "stage",
        "roots",
        "candidates",
        "improving_candidate_fraction",
        "tie_candidate_fraction",
        "roots_with_any_improvement_fraction",
        "roots_with_strong_improvement_fraction",
        "best_saving_eur_median",
        "best_saving_eur_p90",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in fields} for row in rows
        )


def _plot(run_root, variants, stage_rows):
    figure_dir = run_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_stages = ("g1", "g2", "g3")
    x = np.arange(len(plot_stages), dtype=float)
    panels = (
        (
            "roots_with_any_improvement_fraction",
            "Roots with any improvement (%)",
        ),
        (
            "roots_with_strong_improvement_fraction",
            "Roots with ≥€40k improvement (%)",
        ),
        ("tie_candidate_fraction", "Tie candidates (%)"),
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(183 / 25.4, 68 / 25.4), sharex=True
    )
    for variant in variants:
        for axis, (metric, _ylabel) in zip(axes, panels):
            means = []
            minimums = []
            maximums = []
            for stage in plot_stages:
                values = np.asarray(
                    [
                        row[metric] * 100.0
                        for row in stage_rows
                        if row["variant"] == variant
                        and row["stage"] == stage
                    ]
                )
                means.append(float(values.mean()))
                minimums.append(float(values.min()))
                maximums.append(float(values.max()))
            means_array = np.asarray(means)
            lower_error = np.maximum(
                means_array - np.asarray(minimums), 0.0
            )
            upper_error = np.maximum(
                np.asarray(maximums) - means_array, 0.0
            )
            axis.errorbar(
                x,
                means_array,
                yerr=[lower_error, upper_error],
                marker="o",
                ms=4,
                lw=1.2,
                capsize=2.5,
                color=COLORS[variant],
                label=LABELS[variant],
            )
    for label, axis, (_metric, ylabel) in zip(
        ("a", "b", "c"), axes, panels
    ):
        axis.set_xticks(x)
        axis.set_xticklabels([stage.upper() for stage in plot_stages])
        axis.set_ylabel(ylabel)
        axis.text(
            -0.16,
            1.05,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8,
        )
    axes[0].legend(fontsize=6.3, loc="best")
    fig.tight_layout(pad=1.0)
    base = figure_dir / "recursive_data_quality"
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
    stage_rows = []
    branch_audits = []
    training_sampling = []
    for variant in variants:
        for model_seed in MODEL_SEEDS:
            branch = (
                run_root
                / "branches"
                / variant
                / f"model_seed_{model_seed}"
            )
            paths = [
                branch / stage / "train_merged.npz" for stage in STAGES
            ]
            rows = _load_collection([str(path) for path in paths])
            all_seeds = np.concatenate(
                [
                    np.asarray(data["scenario_seed"]).reshape(-1)
                    for data, _metadata in rows
                ]
            )
            if np.any(all_seeds >= 8_000_000):
                raise ValueError(
                    "training data contains controller-evaluation seeds"
                )
            follow_index = int(rows[0][1]["follow_action_index"])
            _combined, datasets = _combined_dataset(
                rows,
                follow_index,
                "shared_future_summary",
                return_parts=True,
            )
            for stage, (data, metadata), dataset in zip(
                STAGES, rows, datasets
            ):
                quality = _stage_quality(
                    data, metadata, dataset, stage
                )
                quality.update(
                    {"variant": variant, "model_seed": model_seed}
                )
                stage_rows.append(quality)
            checkpoint = torch.load(
                branch / "p4" / "iterative_action_q.pt",
                map_location="cpu",
                weights_only=False,
            )
            _weights, duplicate_audit = root_sampling_weights(
                datasets,
                checkpoint["normalization"],
                stage_sampling_temperature=0.5,
                near_duplicate_weighting="inverse_cluster",
            )
            branch_audits.append(
                {
                    "variant": variant,
                    "model_seed": model_seed,
                    "all_stage_near_duplicate_audit": (
                        duplicate_audit["near_duplicate"]
                    ),
                }
            )
            for model_stage in ("p3", "p4"):
                summary = json.loads(
                    (
                        branch / model_stage / "summary.json"
                    ).read_text(encoding="utf-8")
                )
                training_sampling.append(
                    {
                        "variant": variant,
                        "model_seed": model_seed,
                        "model_stage": model_stage,
                        "root_sampling": summary["root_sampling"],
                    }
                )
    result = {
        "kind": "iterative_h3_recursive_data_quality",
        "formal_test_access": False,
        "retained_variants": list(variants),
        "stage_quality": stage_rows,
        "branch_duplicate_audits": branch_audits,
        "training_sampling_audits": training_sampling,
    }
    (run_root / "recursive_data_quality.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_stage_csv(
        run_root / "recursive_data_quality.csv", stage_rows
    )
    _plot(run_root, variants, stage_rows)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
