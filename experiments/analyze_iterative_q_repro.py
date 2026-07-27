"""Summarize Iterative Q model-seed and root-resampling experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ("state_only", "original_14d", "summary_168")
METRICS = (
    "total_cost_eur",
    "vented_t",
    "stored_t",
    "unit_cost_eur_per_t",
)
T_CRITICAL_95_DF4 = 2.776445


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def _load_replicates(run_root: Path, axis: str, method: str):
    rows = []
    for path in sorted((run_root / axis / method).glob("*/eval_formal/evaluation.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            episodes = list(csv.DictReader(handle))
        seeds = tuple(int(row["seed"]) for row in episodes)
        row = {
            "axis": axis,
            "method": method,
            "run": path.parents[1].name,
            "episodes": len(episodes),
            "eval_seeds": seeds,
        }
        for metric in METRICS:
            row[metric] = float(np.mean([float(item[metric]) for item in episodes]))
        rows.append(row)
    if len(rows) != 5:
        raise ValueError(f"expected 5 {axis}/{method} runs, found {len(rows)}")
    if len({row["eval_seeds"] for row in rows}) != 1:
        raise ValueError(f"{axis}/{method} does not use one fixed evaluation set")
    return rows


def _metric_summary(rows, metric):
    values = np.asarray([row[metric] for row in rows])
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _paired_cost(rows_by_method, left, right):
    left_rows = {row["run"]: row for row in rows_by_method[left]}
    right_rows = {row["run"]: row for row in rows_by_method[right]}
    if set(left_rows) != set(right_rows):
        raise ValueError(f"unpaired runs for {left} and {right}")
    deltas = np.asarray(
        [
            left_rows[run]["total_cost_eur"] - right_rows[run]["total_cost_eur"]
            for run in sorted(left_rows)
        ]
    )
    mean = float(deltas.mean())
    half_width = float(
        T_CRITICAL_95_DF4 * deltas.std(ddof=1) / np.sqrt(len(deltas))
    )
    return {
        "comparison": f"{left} - {right}",
        "mean_delta_total_cost_eur": mean,
        "sample_sd_eur": float(deltas.std(ddof=1)),
        "replicate_level_95pct_t_ci_eur": [mean - half_width, mean + half_width],
        "left_wins": int(np.sum(deltas < 0.0)),
        "ties": int(np.sum(deltas == 0.0)),
        "right_wins": int(np.sum(deltas > 0.0)),
        "replicate_deltas_eur": deltas.tolist(),
    }


def run(args):
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"kind": "iterative_q_reproducibility_analysis", "axes": {}}
    raw_rows = []
    for axis in ("model_seed", "root_sample"):
        rows_by_method = {
            method: _load_replicates(run_root, axis, method)
            for method in METHODS
        }
        raw_rows.extend(
            row for method_rows in rows_by_method.values() for row in method_rows
        )
        result["axes"][axis] = {
            "methods": {
                method: {
                    metric: _metric_summary(rows, metric) for metric in METRICS
                }
                for method, rows in rows_by_method.items()
            },
            "paired_cost_comparisons": [
                _paired_cost(rows_by_method, "original_14d", "state_only"),
                _paired_cost(rows_by_method, "summary_168", "state_only"),
                _paired_cost(rows_by_method, "summary_168", "original_14d"),
            ],
        }

    with (out_dir / "replicate_means.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ["axis", "method", "run", "episodes", *METRICS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({field: row[field] for field in fields})
    (out_dir / "analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main():
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
