import json

import numpy as np
import pytest

from experiments.merge_iterative_q_data import merge_shards, parse_args


def _write_shard(path, split, seeds, delta):
    metadata = {
        "kind": "greedy_rollin_event_counterfactual",
        "split": split,
        "scenario_seeds": seeds,
        "episode_hours": 720,
        "observation_variant": "future_mlp_mode_destination",
        "state_feature_names": ["x"],
        "forecast_channel_names": ["f"],
        "joint_actions": [[0]],
        "follow_indices": [0],
        "follow_action_index": 0,
        "reward_scale": 1e-5,
        "objective": "economic",
        "residual_reward": "greedy minus candidate",
        "uses_mpc": False,
    }
    baseline = np.full(len(seeds), 100.0)
    np.savez_compressed(
        path,
        scenario_seed=np.asarray(seeds),
        candidate_total_cost_eur=baseline + np.asarray(delta),
        baseline_total_cost_eur=baseline,
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_merge_shards_validates_split_seeds_and_concatenates(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    output = tmp_path / "merged.npz"
    _write_shard(first, "train", [1, 2], [-5.0, 3.0])
    _write_shard(second, "train", [3], [-2.0])
    args = parse_args(
        [
            "--shards",
            str(first),
            str(second),
            "--out-path",
            str(output),
            "--expected-split",
            "train",
            "--expected-seeds",
            "1",
            "2",
            "3",
        ]
    )
    summary = merge_shards(args)
    assert summary["candidates"] == 3
    assert summary["improving_candidates"] == 2
    with np.load(output, allow_pickle=False) as data:
        assert data["scenario_seed"].tolist() == [1, 2, 3]


def test_merge_shards_rejects_overlapping_scenario_seeds(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_shard(first, "train", [1], [0.0])
    _write_shard(second, "train", [1], [0.0])
    args = parse_args(
        [
            "--shards",
            str(first),
            str(second),
            "--out-path",
            str(tmp_path / "merged.npz"),
        ]
    )
    with pytest.raises(ValueError, match="overlap"):
        merge_shards(args)
