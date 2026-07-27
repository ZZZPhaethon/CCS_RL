"""Summarize future-residual Iterative Q experiments across model seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = (
    "state_only",
    "concat_summary_168",
    "frozen_scale025",
    "frozen_scale100",
    "tune_scale025_dropout25",
    "tune_scale025_dropout25_mean",
)
METRICS = (
    "total_cost_eur",
    "vented_t",
    "stored_t",
    "unit_cost_eur_per_t",
    "override_events",
)
T_CRITICAL_95_DF4 = 2.776445


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def _evaluation_path(
    baseline_root: Path, adapter_root: Path, method: str, model_seed: int
) -> Path:
    run = f"model_seed_{model_seed}"
    if method == "state_only":
        return baseline_root / "state_only" / run / "eval_formal" / "evaluation.csv"
    if method == "concat_summary_168":
        return baseline_root / "summary_168" / run / "eval_formal" / "evaluation.csv"
    if method == "tune_scale025_dropout25_mean":
        return (
            adapter_root
            / "tune_scale025_dropout25"
            / run
            / "eval_future_mean"
            / "evaluation.csv"
        )
    return adapter_root / method / run / "eval_formal" / "evaluation.csv"


def _load_rows(baseline_root: Path, adapter_root: Path):
    result = {method: [] for method in METHODS}
    expected_eval_seeds = None
    for method in METHODS:
        for model_seed in range(5):
            path = _evaluation_path(
                baseline_root, adapter_root, method, model_seed
            )
            with path.open(newline="", encoding="utf-8") as handle:
                episodes = list(csv.DictReader(handle))
            eval_seeds = tuple(int(row["seed"]) for row in episodes)
            if expected_eval_seeds is None:
                expected_eval_seeds = eval_seeds
            if eval_seeds != expected_eval_seeds:
                raise ValueError(f"evaluation seeds differ in {path}")
            row = {
                "method": method,
                "model_seed": model_seed,
                "episodes": len(episodes),
            }
            for metric in METRICS:
                row[metric] = float(
                    np.mean([float(item[metric]) for item in episodes])
                )
            result[method].append(row)
    return result


def _metric_summary(rows, metric):
    values = np.asarray([row[metric] for row in rows])
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _paired_cost(rows_by_method, method, reference):
    values = np.asarray(
        [
            rows_by_method[method][seed]["total_cost_eur"]
            - rows_by_method[reference][seed]["total_cost_eur"]
            for seed in range(5)
        ]
    )
    mean = float(values.mean())
    half_width = float(
        T_CRITICAL_95_DF4 * values.std(ddof=1) / np.sqrt(len(values))
    )
    return {
        "comparison": f"{method} - {reference}",
        "mean_delta_total_cost_eur": mean,
        "sample_sd_eur": float(values.std(ddof=1)),
        "replicate_level_95pct_t_ci_eur": [mean - half_width, mean + half_width],
        "method_wins": int(np.sum(values < 0.0)),
        "ties": int(np.sum(values == 0.0)),
        "reference_wins": int(np.sum(values > 0.0)),
        "model_seed_deltas_eur": values.tolist(),
    }


def run(args):
    baseline_root = Path(args.baseline_root)
    adapter_root = Path(args.adapter_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_method = _load_rows(baseline_root, adapter_root)
    result = {
        "kind": "iterative_q_future_adapter_analysis",
        "methods": {
            method: {
                metric: _metric_summary(rows, metric) for metric in METRICS
            }
            for method, rows in rows_by_method.items()
        },
        "paired_cost_comparisons": [
            _paired_cost(rows_by_method, method, reference)
            for method, reference in (
                ("concat_summary_168", "state_only"),
                ("frozen_scale025", "state_only"),
                ("frozen_scale025", "concat_summary_168"),
                ("frozen_scale100", "state_only"),
                ("frozen_scale100", "concat_summary_168"),
                ("tune_scale025_dropout25", "state_only"),
                ("tune_scale025_dropout25", "concat_summary_168"),
                ("tune_scale025_dropout25_mean", "state_only"),
                (
                    "tune_scale025_dropout25",
                    "tune_scale025_dropout25_mean",
                ),
            )
        ],
    }
    with (out_dir / "replicate_means.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ["method", "model_seed", "episodes", *METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            writer.writerows(rows_by_method[method])
    (out_dir / "analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main():
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
