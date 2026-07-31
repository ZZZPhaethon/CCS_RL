import json

import numpy as np
import pytest
import torch

from scripts.train_iterative_action_q import (
    GroupedDenseActionDataset,
    dataset_normalization,
    exclude_state_features,
    parse_args,
    root_sampling_weights,
    run,
    selective_previous_policy_anchor_loss,
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


def _dataset(path, split, seeds, *, include_future=False, include_forecast=False):
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
    arrays = {
        "states": np.asarray([[row[2]] for row in rows], dtype=np.float32),
        "actions": np.asarray([[row[3]] for row in rows], dtype=np.int16),
        "return_to_go": np.asarray(
            [[row[4]] for row in rows], dtype=np.float32
        ),
        "scenario_seed": np.asarray([row[0] for row in rows]),
        "root_time_h": np.asarray([row[1] for row in rows]),
    }
    if include_future:
        metadata["future_feature_names"] = [
            f"future_{index}" for index in range(7)
        ]
        arrays["future_summaries"] = np.asarray(
            [
                [[row[0] / 100.0, row[1] / 720.0, *([0.5] * 5)]]
                for row in rows
            ],
            dtype=np.float32,
        )
    if include_forecast:
        metadata["forecast_feature_names"] = [
            *[f"forecast_{index}" for index in range(9)],
            "valid_horizon",
        ]
        metadata["forecast_horizon_h"] = 168
        forecasts = np.zeros((len(rows), 1, 168, 10), dtype=np.float32)
        for index, row in enumerate(rows):
            valid_horizon = min(168, 720 - row[1])
            forecasts[index, 0, :valid_horizon, :9] = row[0] / 100.0
            forecasts[index, 0, :valid_horizon, 9] = 1.0
        arrays["future_forecasts"] = forecasts
    np.savez_compressed(
        path,
        **arrays,
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
    _state, future, actions, targets, valid, root_hour, anchor_action = dataset[0]
    assert future.shape == (0,)
    assert list(actions[valid]) == [3, 4, 7]
    assert list(targets[valid]) == [-1.0, 2.0, 0.0]
    assert root_hour == 10
    assert anchor_action == -1


def test_previous_policy_anchor_marks_only_exactly_protected_roots():
    expected = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [100.0, 100.0]],
        ],
        requires_grad=True,
    )
    actions = torch.tensor([[0, 1], [0, 1]])
    targets = torch.tensor([[0.0, 0.3], [0.0, 0.5]])
    valid = torch.ones_like(actions, dtype=torch.bool)
    anchors = torch.tensor([0, 0])

    loss, metrics = selective_previous_policy_anchor_loss(
        expected,
        actions,
        targets,
        valid,
        anchors,
        release_margin=0.4,
        temperature=0.5,
    )
    loss.backward()

    assert metrics == {
        "protected_roots": 1,
        "released_roots": 1,
        "agreement_roots": 0,
        "effective_weight": 1.0,
        "weighted_agreement": 0.0,
    }
    assert expected.grad[0].abs().sum() > 0
    assert expected.grad[1].abs().sum() == 0


def test_previous_policy_anchor_linear_weights_match_euro_schedule():
    exact_improvements = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    expected = torch.zeros((6, 2, 2), requires_grad=True)
    actions = torch.tensor([[0, 1]]).expand(6, -1)
    targets = torch.stack(
        (torch.zeros_like(exact_improvements), exact_improvements), dim=1
    )
    valid = torch.ones_like(actions, dtype=torch.bool)
    anchors = torch.zeros(6, dtype=torch.int64)

    loss, metrics = selective_previous_policy_anchor_loss(
        expected,
        actions,
        targets,
        valid,
        anchors,
        release_margin=0.5,
        temperature=0.5,
        weighting="linear",
    )
    loss.backward()

    assert metrics["protected_roots"] == 5
    assert metrics["released_roots"] == 1
    assert np.isclose(metrics["effective_weight"], 3.0)
    assert np.isclose(metrics["weighted_agreement"], 3.0)
    assert expected.grad[-1].abs().sum() == 0


def test_previous_policy_anchor_plateau_linear_weights_match_euro_schedule():
    exact_improvements = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    expected = torch.zeros((6, 2, 2), requires_grad=True)
    actions = torch.tensor([[0, 1]]).expand(6, -1)
    targets = torch.stack(
        (torch.zeros_like(exact_improvements), exact_improvements), dim=1
    )
    valid = torch.ones_like(actions, dtype=torch.bool)
    anchors = torch.zeros(6, dtype=torch.int64)

    loss, metrics = selective_previous_policy_anchor_loss(
        expected,
        actions,
        targets,
        valid,
        anchors,
        release_margin=0.5,
        temperature=0.5,
        weighting="plateau_linear",
        plateau_margin=0.2,
    )
    loss.backward()

    assert metrics["protected_roots"] == 5
    assert metrics["released_roots"] == 1
    assert np.isclose(metrics["effective_weight"], 4.0)
    assert np.isclose(metrics["weighted_agreement"], 4.0)
    assert expected.grad[-1].abs().sum() == 0


def test_p0_anchor_requires_explicit_non_neural_teacher_opt_in(tmp_path):
    base = [
        "--train-data",
        str(tmp_path / "train.npz"),
        "--validation-data",
        str(tmp_path / "validation.npz"),
        "--out-dir",
        str(tmp_path / "out"),
        "--previous-policy-anchor-coefficient",
        "1.0",
    ]
    with pytest.raises(SystemExit):
        parse_args(base)

    args = parse_args(
        [*base, "--allow-anchor-without-initial-checkpoint"]
    )
    assert args.initial_checkpoint is None
    assert args.allow_anchor_without_initial_checkpoint is True


def test_grouped_dense_dataset_stratified_root_sampling_is_reproducible():
    rows = []
    for seed in range(20):
        for root_hour in (24, 72, 120):
            for action in (0, 1):
                rows.append((seed, root_hour, action))
    data = {
        "scenario_seed": np.asarray([row[0] for row in rows]),
        "root_time_h": np.asarray([row[1] for row in rows]),
        "states": np.asarray(
            [[[row[0], row[1]]] for row in rows], dtype=np.float32
        ),
        "actions": np.asarray([[row[2]] for row in rows]),
        "return_to_go": np.asarray(
            [[row[0] - row[2]] for row in rows], dtype=np.float32
        ),
    }
    first = GroupedDenseActionDataset(
        data, root_sample_fraction=0.8, root_sample_seed=17
    )
    repeated = GroupedDenseActionDataset(
        data, root_sample_fraction=0.8, root_sample_seed=17
    )
    different = GroupedDenseActionDataset(
        data, root_sample_fraction=0.8, root_sample_seed=18
    )
    assert len(first) == 48
    assert first.root_keys == repeated.root_keys
    assert first.root_keys != different.root_keys
    assert sorted(set(first.root_hours)) == [24, 72, 120]


def _root_dataset(states, returns, *, seed_offset=0, reward_scale=1e-5):
    count = len(states)
    data = {
        "scenario_seed": np.arange(seed_offset, seed_offset + count),
        "root_time_h": np.full(count, 120),
        "states": np.asarray(states, dtype=np.float32)[:, None, :],
        "actions": np.zeros((count, 1), dtype=np.int16),
        "return_to_go": np.asarray(returns, dtype=np.float32)[:, None],
    }
    return GroupedDenseActionDataset(data, reward_scale=reward_scale)


def test_root_sampling_temperature_uses_sqrt_stage_mass():
    first = _root_dataset(
        [[1.0], [2.0], [3.0], [4.0]],
        [0.0, 0.0, 0.0, 0.0],
    )
    second = _root_dataset([[5.0]], [0.0], seed_offset=10)

    weights, audit = root_sampling_weights(
        [first, second],
        {"state_mean": np.zeros(1), "state_std": np.ones(1)},
        stage_sampling_temperature=0.5,
    )

    assert np.isclose(weights[:4].sum(), 2.0 / 3.0)
    assert np.isclose(weights[4:].sum(), 1.0 / 3.0)
    assert np.isclose(
        audit["stages"][0]["sampling_probability"], 2.0 / 3.0
    )


def test_near_duplicate_weights_cluster_matching_cross_stage_roots():
    first = _root_dataset([[1.0, 2.0]], [0.0])
    second = _root_dataset([[1.0, 2.0]], [0.0], seed_offset=10)

    weights, audit = root_sampling_weights(
        [first, second],
        {"state_mean": np.zeros(2), "state_std": np.ones(2)},
        near_duplicate_weighting="inverse_cluster",
    )

    assert np.allclose(weights, [0.5, 0.5])
    assert audit["near_duplicate"]["clusters"] == 1
    assert audit["near_duplicate"]["largest_cluster"] == 2
    assert audit["near_duplicate"]["cross_stage_clusters"] == 1
    assert audit["near_duplicate"]["roots_in_cross_stage_clusters"] == 2


def test_advantage_stratification_reweights_roots_within_stage():
    dataset = _root_dataset(
        [[1.0], [2.0], [3.0]],
        [0.0, 0.2, 0.6],
    )

    weights, audit = root_sampling_weights(
        [dataset],
        {"state_mean": np.zeros(1), "state_std": np.ones(1)},
        root_advantage_weighting="stratified",
        root_advantage_threshold_eur=40000,
        root_no_improvement_weight=0.5,
        root_moderate_improvement_weight=1.0,
        root_strong_improvement_weight=2.0,
    )

    assert np.allclose(weights, np.asarray([0.5, 1.0, 2.0]) / 3.5)
    assert audit["stages"][0]["no_improvement_roots"] == 1
    assert audit["stages"][0]["strong_improvement_roots"] == 1


def test_dataset_normalization_is_self_contained(tmp_path):
    path = tmp_path / "train.npz"
    _dataset(path, "train", [10, 11])
    from scripts.train_iterative_action_q import _load_collection

    normalization = dataset_normalization(_load_collection([str(path)]))
    assert normalization["state_mean"].shape == (7,)
    assert np.all(normalization["state_std"] >= 1e-5)
    assert normalization["return_scale"] >= 1.0


def test_exclude_state_features_projects_arrays_and_metadata(tmp_path):
    path = tmp_path / "train.npz"
    _dataset(path, "train", [10, 11])
    from scripts.train_iterative_action_q import _load_collection

    rows = _load_collection([str(path)])
    original_names = list(rows[0][1]["state_feature_names"])
    exclude_state_features(rows, ["global.fill"])

    data, metadata = rows[0]
    assert data["states"].shape[-1] == len(original_names) - 1
    assert metadata["source_state_feature_names"] == original_names
    assert metadata["state_feature_names"] == original_names[1:]
    assert metadata["excluded_state_feature_names"] == ["global.fill"]


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


def test_training_can_exclude_state_features(tmp_path):
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
            "--exclude-state-features",
            "global.fill",
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

    run(args)
    checkpoint = torch.load(
        out_dir / "iterative_action_q.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["configuration"]["exclude_state_features"] == ["global.fill"]
    assert checkpoint["metadata"]["excluded_state_feature_names"] == ["global.fill"]
    assert checkpoint["metadata"]["source_state_feature_names"] == _feature_names()
    assert checkpoint["metadata"]["state_feature_names"] == _feature_names()[1:]
    assert checkpoint["normalization"]["state_mean"].shape == (6,)


def test_future_training_uses_v4_summary_input(tmp_path):
    train_path = tmp_path / "train_future.npz"
    validation_path = tmp_path / "validation_future.npz"
    out_dir = tmp_path / "out_future"
    _dataset(train_path, "train", [10, 11], include_future=True)
    _dataset(validation_path, "validation", [20, 21], include_future=True)
    args = parse_args(
        [
            "--train-data",
            str(train_path),
            "--validation-data",
            str(validation_path),
            "--out-dir",
            str(out_dir),
            "--observation-input",
            "shared_future_summary",
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
    run(args)
    checkpoint = torch.load(
        out_dir / "iterative_action_q.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert (
        checkpoint["configuration"]["q_head"]
        == "iterative_action_q_future_summary"
    )
    assert checkpoint["normalization"]["future_mean"].shape == (7,)


def test_full_forecast_training_uses_selected_encoder(tmp_path):
    train_path = tmp_path / "train_forecast.npz"
    validation_path = tmp_path / "validation_forecast.npz"
    out_dir = tmp_path / "out_forecast"
    _dataset(train_path, "train", [10, 11], include_forecast=True)
    _dataset(validation_path, "validation", [20, 21], include_forecast=True)
    args = parse_args(
        [
            "--train-data",
            str(train_path),
            "--validation-data",
            str(validation_path),
            "--out-dir",
            str(out_dir),
            "--observation-input",
            "forecast_168",
            "--forecast-encoder",
            "gru",
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
    run(args)
    checkpoint = torch.load(
        out_dir / "iterative_action_q.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["configuration"]["q_head"] == "iterative_action_q_future_168"
    assert checkpoint["configuration"]["forecast_encoder"] == "gru"
    assert checkpoint["normalization"]["forecast_mean"].shape == (9,)


def test_nonoverlapping_forecast_summary_training(tmp_path):
    train_path = tmp_path / "train_summary.npz"
    validation_path = tmp_path / "validation_summary.npz"
    out_dir = tmp_path / "out_summary"
    _dataset(train_path, "train", [10, 11], include_forecast=True)
    _dataset(validation_path, "validation", [20, 21], include_forecast=True)
    args = parse_args(
        [
            "--train-data",
            str(train_path),
            "--validation-data",
            str(validation_path),
            "--out-dir",
            str(out_dir),
            "--observation-input",
            "forecast_summary_bands_24_72_168",
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
    run(args)
    checkpoint = torch.load(
        out_dir / "iterative_action_q.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["configuration"]["q_head"] == "iterative_action_q_future_summary"
    assert checkpoint["metadata"]["future_summary_bands_h"] == [
        [0, 24],
        [24, 72],
        [72, 168],
    ]
    assert checkpoint["normalization"]["future_mean"].shape == (21,)


def test_residual_summary_training_starts_from_state_checkpoint(tmp_path):
    train_path = tmp_path / "train_residual.npz"
    validation_path = tmp_path / "validation_residual.npz"
    state_dir = tmp_path / "state"
    residual_dir = tmp_path / "residual"
    _dataset(train_path, "train", [10, 11], include_forecast=True)
    _dataset(validation_path, "validation", [20, 21], include_forecast=True)
    common = [
        "--train-data",
        str(train_path),
        "--validation-data",
        str(validation_path),
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
    run(parse_args([*common, "--out-dir", str(state_dir)]))
    summary = run(
        parse_args(
            [
                *common,
                "--out-dir",
                str(residual_dir),
                "--initial-checkpoint",
                str(state_dir / "iterative_action_q.pt"),
                "--observation-input",
                "forecast_summary_168",
                "--future-fusion",
                "residual_frozen",
            ]
        )
    )
    checkpoint = torch.load(
        residual_dir / "iterative_action_q.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert summary["loaded_pretrained_tensors"] > 0
    assert (
        checkpoint["configuration"]["q_head"]
        == "iterative_action_q_future_residual_summary"
    )
    trainable = [
        name
        for name, parameter in checkpoint["model_state_dict"].items()
        if name.startswith("future_") and parameter.is_floating_point()
    ]
    assert "future_scale" in trainable
