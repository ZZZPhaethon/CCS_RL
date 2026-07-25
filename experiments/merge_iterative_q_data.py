"""Validate and merge shards produced by iterative-Q experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--expected-split", choices=("train", "validation", "test"))
    parser.add_argument("--expected-seeds", type=int, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def merge_shards(args) -> dict[str, object]:
    out_path = Path(args.out_path)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing dataset: {out_path}")

    shards = []
    metadata_rows = []
    for path_text in args.shards:
        path = Path(path_text)
        with np.load(path, allow_pickle=False) as loaded:
            metadata = json.loads(str(loaded["metadata_json"]))
            arrays = {key: loaded[key].copy() for key in loaded.files if key != "metadata_json"}
        shards.append(arrays)
        metadata_rows.append(metadata)

    reference = metadata_rows[0]
    for metadata in metadata_rows:
        for field in (
            "kind",
            "split",
            "episode_hours",
            "observation_variant",
            "state_feature_names",
            "joint_actions",
            "follow_indices",
            "follow_action_index",
            "reward_scale",
            "objective",
            "residual_reward",
            "uses_mpc",
        ):
            if metadata[field] != reference[field]:
                raise ValueError(f"shard metadata mismatch for {field}")
    if args.expected_split and reference["split"] != args.expected_split:
        raise ValueError(
            f"expected split {args.expected_split}, got {reference['split']}"
        )

    key_set = set(shards[0])
    if any(set(shard) != key_set for shard in shards[1:]):
        raise ValueError("shards have different array fields")
    for shard in shards:
        rows = len(shard["scenario_seed"])
        for key, value in shard.items():
            if value.ndim == 0 or len(value) != rows:
                raise ValueError(f"array {key} does not have candidate first dimension")

    merged = {key: np.concatenate([shard[key] for shard in shards]) for key in key_set}
    actual_seeds = sorted(set(int(seed) for seed in merged["scenario_seed"]))
    shard_seeds = [
        int(seed)
        for metadata in metadata_rows
        for seed in metadata["scenario_seeds"]
    ]
    if len(shard_seeds) != len(set(shard_seeds)):
        raise ValueError("scenario seeds overlap between shards")
    if actual_seeds != sorted(shard_seeds):
        raise ValueError("metadata scenario seeds do not match array contents")
    if args.expected_seeds is not None and actual_seeds != sorted(args.expected_seeds):
        raise ValueError("merged scenario seeds do not match --expected-seeds")

    metadata = dict(reference)
    metadata["scenario_seeds"] = actual_seeds
    metadata["candidates_per_seed"] = None
    metadata["merged_shards"] = [str(path) for path in args.shards]
    metadata["candidate_count"] = int(len(merged["scenario_seed"]))
    merged["metadata_json"] = np.asarray(json.dumps(metadata, separators=(",", ":")))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **merged)
    delta = merged["candidate_total_cost_eur"] - merged["baseline_total_cost_eur"]
    summary = {
        "out_path": str(out_path),
        "shards": len(shards),
        "scenario_seeds": len(actual_seeds),
        "candidates": len(delta),
        "improving_candidates": int((delta < -1e-6).sum()),
        "ties": int((np.abs(delta) <= 1e-6).sum()),
        "worse_candidates": int((delta > 1e-6).sum()),
        "mean_delta_cost_eur": float(delta.mean()),
        "best_delta_cost_eur": float(delta.min()),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    return merge_shards(parse_args(argv))


if __name__ == "__main__":
    main()
