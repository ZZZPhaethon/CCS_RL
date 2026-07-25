import json

import numpy as np
import torch

from scripts.train_iterative_action_q import (
    GroupedDenseActionDataset,
    dataset_normalization,
    parse_args,
    run,
    selected_action_quantiles,
)


def _feature_names():
    names = ["global.fill"]
    for vessel in ("a", "b", "c"):
        names.extend(
            [
                f"{vessel}.cargo",
                f"greedy_proposal.{vessel}.native_action_0",
            ]
        )
    return names


def _dataset(path, split, seeds):
    rows = []
    for seed in seeds:
        for root_index, hour in enumerate((120, 360)):
            state = np.asarray(
                [seed / 100.0, root_index, 0.1, 0.2, 0.3, 0.4, 0.5],
                dtype=np.float32,
            )
            for action, value in ((0, 0.4 + 0.1 * root_index), (1, -0.3)):
                rows.append((seed, hour, state, action, value))
    metadata = {
        "kind": "iterative_q_greedy_rollin_data",
        "split": split,
        "scenario_seeds": list(seeds),
        "episode_hours": 720,
        "observation_variant": "future_mlp_mode_destination",
        "state_feature_names": _feature_names(),
        "joint_actions": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 2]],
        "follow_action_index": 3,
        "reward_scale": 1e-5,
        "uses_mpc": False,
    }
    np.savez_compressed(
        path,
        states=np.asarray([[row[2]] for row in rows], dtype=np.float32),
        actions=np.asarray([[row[3]] for row in rows], dtype=np.int16),
        return_to_go=np.asarray([[row[4]] for row in rows], dtype=np.float32),
        scenario_seed=np.asarray([row[0] for row in rows]),
        root_time_h=np.asarray([row[1] for row in rows]),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_grouped_dense_dataset_appends_follow_and_deduplicates():
    state = np.asarray([[[1.0]], [[1.0]], [[1.0]]])
    data = {
        "scenario_seed": np.asarray([1, 1, 1]),
        "root_time_h": np.asarray([10, 10, 10]),
        "states": state,
        "actions": np.asarray([[3], [4], [3]]),
        "return_to_go": np.asarray([[-1.0], [2.0], [-1.0]]),
    }
    dataset = GroupedDenseActionDataset(data, follow_action_index=7)
    _state, actions, targets, valid, root_hour = dataset[0]
    assert list(actions[valid]) == [3, 4, 7]
    assert list(targets[valid]) == [-1.0, 2.0, 0.0]
    assert root_hour == 10


def test_dataset_normalization_is_self_contained(tmp_path):
    path = tmp_path / "train.npz"
    _dataset(path, "train", [10, 11])
    from scripts.train_iterative_action_q import _load_collection

    normalization = dataset_normalization(_load_collection([str(path)]))
    assert normalization["state_mean"].shape == (7,)
    assert np.all(normalization["state_std"] >= 1e-5)
    assert normalization["return_scale"] >= 1.0


def test_selected_action_quantiles_shape():
    q = torch.randn(2, 3, 4, 5)
    actions = torch.tensor([[0, 2], [1, -1]])
    selected = selected_action_quantiles(q, actions)
    assert selected.shape == (2, 2, 3, 5)


def test_training_from_greedy_data_requires_no_legacy_checkpoint(tmp_path):
    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    out_dir = tmp_path / "out"
    _dataset(train_path, "train", [10, 11])
    _dataset(validation_path, "validation", [20, 21])
    args = parse_args(
        [
            "--train-data",
            str(train_path),
            "--validation-data",
            str(validation_path),
            "--out-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--patience",
            "1",
            "--batch-size",
            "2",
            "--heads",
            "2",
            "--quantiles",
            "3",
            "--action-embedding-size",
            "4",
            "--action-feature-size",
            "8",
            "--device",
            "cpu",
        ]
    )
    summary = run(args)
    checkpoint = torch.load(
        out_dir / "iterative_action_q.pt", map_location="cpu", weights_only=False
    )
    assert summary["loaded_pretrained_tensors"] == 0
    assert checkpoint["configuration"]["q_head"] == "iterative_action_q"
    assert checkpoint["configuration"]["observation_input"] == "state_only"
