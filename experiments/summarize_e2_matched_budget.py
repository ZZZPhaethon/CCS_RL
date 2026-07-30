"""Write and validate the E2 Greedy-only matched-budget dataset summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--target-simulator-calls", type=int, required=True)
    parser.add_argument("--max-relative-error-pct", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv=None) -> dict[str, object]:
    args = parse_args(argv)
    with np.load(args.train_data, allow_pickle=False) as loaded:
        metadata = json.loads(str(loaded["metadata_json"]))
    actual = int(
        metadata["training_simulator_usage"]["simulator_step_calls"]
    )
    target = int(args.target_simulator_calls)
    error_pct = 100.0 * (actual - target) / target
    summary = {
        "kind": "E2_one_shot_matched_budget",
        "train_data": str(args.train_data),
        "train_seed_range_inclusive": [
            min(metadata["scenario_seeds"]),
            max(metadata["scenario_seeds"]),
        ],
        "train_seed_count": len(metadata["scenario_seeds"]),
        "train_roots": int(metadata["root_count"]),
        "train_candidates": int(metadata["candidate_count"]),
        "train_simulator_step_calls": actual,
        "target_simulator_step_calls": target,
        "relative_error_pct": error_pct,
        "max_relative_error_pct": float(args.max_relative_error_pct),
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if abs(error_pct) > float(args.max_relative_error_pct):
        raise ValueError(
            f"matched budget error {error_pct:.3f}% exceeds "
            f"{args.max_relative_error_pct:.3f}%"
        )
    return summary


if __name__ == "__main__":
    main()
