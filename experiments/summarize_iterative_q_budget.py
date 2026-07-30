"""Summarize exact-root and physical-simulator use for one Iterative-Q run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as loaded:
        return json.loads(str(loaded["metadata_json"]))


def _stage_row(run_root: Path, stage: str, split: str) -> dict[str, object]:
    path = run_root / stage / f"{split}_merged.npz"
    metadata = _metadata(path)
    usage = metadata.get("training_simulator_usage")
    if usage is None:
        raise ValueError(f"{path}: missing aggregated simulator usage")
    if "root_count" not in metadata:
        raise ValueError(f"{path}: missing root_count")
    return {
        "stage": stage,
        "split": split,
        "roots": int(metadata["root_count"]),
        "candidates": int(metadata["candidate_count"]),
        "scenario_seeds": len(metadata["scenario_seeds"]),
        "simulator_step_calls": int(usage["simulator_step_calls"]),
        "simulator_simulated_hours": float(
            usage["simulator_simulated_hours"]
        ),
        "simulator_hour_steps": float(usage["simulator_hour_steps"]),
    }


def summarize(run_root: Path, target_train_steps: int) -> dict[str, object]:
    stages = []
    index = 0
    while (run_root / f"g{index}" / "train_merged.npz").is_file():
        stages.append(f"g{index}")
        index += 1
    if not stages:
        raise ValueError(f"{run_root}: no merged training stages")

    train = [_stage_row(run_root, stage, "train") for stage in stages]
    validation = [
        _stage_row(run_root, stage, "validation") for stage in stages
    ]
    train_steps = sum(row["simulator_step_calls"] for row in train)
    validation_steps = sum(
        row["simulator_step_calls"] for row in validation
    )
    return {
        "kind": "iterative_q_physical_simulator_budget",
        "run_root": str(run_root),
        "stages": stages,
        "target_train_simulator_steps": int(target_train_steps),
        "train": train,
        "validation": validation,
        "totals": {
            "train_roots": sum(row["roots"] for row in train),
            "validation_roots": sum(row["roots"] for row in validation),
            "train_candidates": sum(row["candidates"] for row in train),
            "validation_candidates": sum(
                row["candidates"] for row in validation
            ),
            "train_simulator_steps": train_steps,
            "validation_simulator_steps": validation_steps,
            "all_data_simulator_steps": train_steps + validation_steps,
            "train_step_error": train_steps - target_train_steps,
            "train_step_relative_error_pct": (
                100.0 * (train_steps - target_train_steps)
                / target_train_steps
            ),
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--target-train-steps", type=int, default=9_525_119
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    payload = summarize(args.run_root, args.target_train_steps)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
