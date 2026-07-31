"""Audit iterative-Q roots and the proposed sampling distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_iterative_action_q import (
    _combined_dataset,
    _load_collection,
    dataset_normalization,
    root_sampling_weights,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", nargs="+", required=True)
    parser.add_argument("--stage-names", nargs="+", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--normalization-checkpoint")
    parser.add_argument(
        "--observation-input", default="shared_future_summary"
    )
    args = parser.parse_args(argv)
    if len(args.train_data) != len(args.stage_names):
        parser.error("train data and stage names must have equal lengths")
    return args


def _stage_quality(data, metadata, dataset, stage_name):
    reward_scale = float(metadata["reward_scale"])
    returns_eur = (
        np.asarray(data["return_to_go"][:, 0], dtype=np.float64)
        / reward_scale
    )
    seeds = np.unique(data["scenario_seed"]).astype(int)
    return {
        "stage": stage_name,
        "scenario_seed_min": int(seeds.min()),
        "scenario_seed_max": int(seeds.max()),
        "scenario_seed_count": int(len(seeds)),
        "roots": int(len(dataset)),
        "candidates": int(len(returns_eur)),
        "improving_candidate_fraction": float(
            np.mean(returns_eur > 1e-6)
        ),
        "tie_candidate_fraction": float(
            np.mean(np.abs(returns_eur) <= 1e-6)
        ),
        "roots_with_any_improvement_fraction": float(
            np.mean(dataset.root_best_saving_eur > 1e-6)
        ),
        "roots_with_strong_improvement_fraction": float(
            np.mean(dataset.root_best_saving_eur >= 40000.0)
        ),
        "best_saving_eur_median": float(
            np.median(dataset.root_best_saving_eur)
        ),
        "best_saving_eur_p90": float(
            np.quantile(dataset.root_best_saving_eur, 0.9)
        ),
    }


def run(args):
    rows = _load_collection(args.train_data)
    follow_index = int(rows[0][1]["follow_action_index"])
    for _data, metadata in rows[1:]:
        if int(metadata["follow_action_index"]) != follow_index:
            raise ValueError("follow action index differs across stages")
    if args.normalization_checkpoint:
        checkpoint = torch.load(
            args.normalization_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        normalization = checkpoint["normalization"]
    else:
        normalization = dataset_normalization(
            rows, observation_input=args.observation_input
        )
    _combined, datasets = _combined_dataset(
        rows,
        follow_index,
        args.observation_input,
        return_parts=True,
    )
    stage_quality = [
        _stage_quality(data, metadata, dataset, stage_name)
        for (data, metadata), dataset, stage_name in zip(
            rows, datasets, args.stage_names
        )
    ]
    _c_weights, c_audit = root_sampling_weights(
        datasets,
        normalization,
        stage_sampling_temperature=0.5,
        near_duplicate_weighting="inverse_cluster",
    )
    _d_weights, d_audit = root_sampling_weights(
        datasets,
        normalization,
        stage_sampling_temperature=0.5,
        near_duplicate_weighting="inverse_cluster",
        root_advantage_weighting="stratified",
    )
    all_seeds = np.concatenate(
        [np.asarray(data["scenario_seed"]).reshape(-1) for data, _ in rows]
    )
    if np.any(all_seeds >= 8_000_000):
        raise ValueError(
            "training data unexpectedly contains controller-evaluation seeds"
        )
    result = {
        "kind": "iterative_q_sampling_data_audit",
        "formal_test_access": False,
        "observation_input": args.observation_input,
        "normalization_checkpoint": args.normalization_checkpoint,
        "stages": stage_quality,
        "c_dedup_balanced_sampling": c_audit,
        "d_dedup_advantage_sampling": d_audit,
    }
    out_path = Path(args.out_path)
    if out_path.exists():
        raise FileExistsError(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
