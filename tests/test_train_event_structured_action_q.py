import argparse
import json

import numpy as np
import pytest
import torch

from scripts.train_event_structured_action_q import (
    GroupedDenseActionDataset,
    _combined_dataset,
    configure_trainable_components,
    model_forecast_input,
    parse_args,
    run,
    selected_action_quantiles,
    selection_score,
)
from sim.control.recurrent_distributional_q import RecurrentBootstrappedQuantileQ


def test_grouped_dense_dataset_pads_same_state_actions():
    data = {
        "scenario_seed": np.asarray([1, 1, 1, 2, 2]),
        "root_time_h": np.asarray([10, 10, 10, 20, 20]),
        "states": np.asarray([[[1.0]], [[1.0]], [[1.0]], [[2.0]], [[2.0]]]),
        "forecasts": np.asarray(
            [[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[2.0]]], [[[2.0]]]]
        ),
        "actions": np.asarray([[3], [4], [5], [6], [7]]),
        "return_to_go": np.asarray([[1.0], [-1.0], [2.0], [0.5], [-0.5]]),
    }
    dataset = GroupedDenseActionDataset(data)
    assert len(dataset) == 2
    _state, _forecast, actions, targets, valid, root_hour = dataset[1]
    assert list(actions) == [6, 7, -1]
    assert list(valid) == [True, True, False]
    assert list(targets[:2]) == [0.5, -0.5]
    assert root_hour == 20


def test_grouped_dense_dataset_appends_follow_with_zero_return():
    data = {
        "scenario_seed": np.asarray([1, 1]),
        "root_time_h": np.asarray([10, 10]),
        "states": np.asarray([[[1.0]], [[1.0]]]),
        "forecasts": np.asarray([[[[1.0]]], [[[1.0]]]]),
        "actions": np.asarray([[3], [4]]),
        "return_to_go": np.asarray([[-1.0], [-2.0]]),
    }
    dataset = GroupedDenseActionDataset(data, follow_action_index=7)
    _state, _forecast, actions, targets, valid, root_hour = dataset[0]
    assert list(actions) == [3, 4, 7]
    assert list(targets) == [-1.0, -2.0, 0.0]
    assert valid.all()
    assert root_hour == 10


def test_grouped_dense_dataset_collapses_consistent_duplicate_actions():
    data = {
        "scenario_seed": np.asarray([1, 1, 1]),
        "root_time_h": np.asarray([10, 10, 10]),
        "states": np.asarray([[[1.0]], [[1.0]], [[1.0]]]),
        "forecasts": np.asarray([[[[1.0]]], [[[1.0]]], [[[1.0]]]]),
        "actions": np.asarray([[3], [4], [3]]),
        "return_to_go": np.asarray([[-1.0], [2.0], [-1.0]]),
    }
    dataset = GroupedDenseActionDataset(data)
    _state, _forecast, actions, targets, valid, _root_hour = dataset[0]
    assert list(actions[valid]) == [3, 4]
    assert list(targets[valid]) == [-1.0, 2.0]


def test_grouped_dense_dataset_rejects_inconsistent_duplicate_targets():
    data = {
        "scenario_seed": np.asarray([1, 1]),
        "root_time_h": np.asarray([10, 10]),
        "states": np.asarray([[[1.0]], [[1.0]]]),
        "forecasts": np.asarray([[[[1.0]]], [[[1.0]]]]),
        "actions": np.asarray([[3], [3]]),
        "return_to_go": np.asarray([[-1.0], [1.0]]),
    }
    with pytest.raises(ValueError, match="inconsistent targets"):
        GroupedDenseActionDataset(data)


def test_combined_dataset_preserves_explicit_policy_anchor_targets():
    old = {
        "scenario_seed": np.asarray([1]),
        "root_time_h": np.asarray([10]),
        "states": np.asarray([[[1.0]]]),
        "forecasts": np.asarray([[[[1.0]]]]),
        "actions": np.asarray([[3]]),
        "return_to_go": np.asarray([[1.0]]),
    }
    policy = {
        "scenario_seed": np.asarray([2, 2]),
        "root_time_h": np.asarray([20, 20]),
        "states": np.asarray([[[2.0]], [[2.0]]]),
        "forecasts": np.asarray([[[[2.0]]], [[[2.0]]]]),
        "actions": np.asarray([[4], [7]]),
        "return_to_go": np.asarray([[0.5], [-0.25]]),
    }
    dataset = _combined_dataset(
        [(old, {}), (policy, {"anchors_in_data": True})],
        follow_action_index=7,
    )
    assert len(dataset) == 2
    _state, _forecast, actions, targets, valid, _hour = dataset[1]
    assert list(actions[valid]) == [4, 7]
    assert list(targets[valid]) == [0.5, -0.25]


def test_selected_action_quantiles_gathers_padded_candidates():
    q = torch.arange(2 * 3 * 8 * 5, dtype=torch.float32).reshape(2, 3, 8, 5)
    actions = torch.tensor([[1, 4], [7, -1]])
    selected = selected_action_quantiles(q, actions)
    assert selected.shape == (2, 2, 3, 5)
    assert torch.equal(selected[0, 1, 2], q[0, 2, 4])
    assert torch.equal(selected[1, 0, 1], q[1, 1, 7])


def test_parser_accepts_multiple_train_and_validation_datasets():
    args = parse_args(
        [
            "--train-data",
            "old_train.npz",
            "policy_train.npz",
            "--validation-data",
            "old_validation.npz",
            "policy_validation.npz",
            "--initial-checkpoint",
            "source.pt",
            "--out-dir",
            "out",
        ]
    )
    assert args.train_data == ["old_train.npz", "policy_train.npz"]
    assert args.validation_data == [
        "old_validation.npz",
        "policy_validation.npz",
    ]
    assert args.trainable_components == "all"


def test_legacy_residual_only_flag_selects_residual_components():
    args = parse_args(
        [
            "--train-data",
            "train.npz",
            "--validation-data",
            "validation.npz",
            "--initial-checkpoint",
            "source.pt",
            "--out-dir",
            "out",
            "--train-action-aligned-residual-only",
        ]
    )
    assert args.trainable_components == "residual_only"


@pytest.mark.parametrize(
    ("mode", "expected_prefixes"),
    [
        ("residual_only", ("window_summary_residual.",)),
        (
            "residual_and_base_head",
            (
                "value.",
                "structured_action_embeddings.",
                "structured_action_fusion.",
                "structured_query.",
                "window_summary_residual.",
            ),
        ),
        (
            "base_head_only",
            (
                "value.",
                "structured_action_embeddings.",
                "structured_action_fusion.",
                "structured_query.",
            ),
        ),
    ],
)
def test_configure_trainable_components(mode, expected_prefixes):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state_encoder = torch.nn.Linear(2, 2)
            self.value = torch.nn.Linear(2, 2)
            self.structured_action_embeddings = torch.nn.ModuleList(
                [torch.nn.Embedding(2, 2)]
            )
            self.structured_action_fusion = torch.nn.Linear(2, 2)
            self.structured_query = torch.nn.Linear(2, 2)
            self.window_summary_residual = torch.nn.Linear(2, 2)

    model = TinyModel()
    configure_trainable_components(model, mode)
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith(expected_prefixes) for name in trainable)
    assert not any(name.startswith("state_encoder.") for name in trainable)


def test_state_only_input_replaces_forecast_with_training_mean():
    class Model:
        forecast_mean = torch.tensor([2.0, 3.0])

    forecast = torch.randn(3, 168, 2)
    result = model_forecast_input(forecast, Model(), "state_only", "cpu")
    assert result.shape == (3, 1, 168, 2)
    assert torch.all(result[..., 0] == 2.0)
    assert torch.all(result[..., 1] == 3.0)


def test_structured_selection_score_rewards_ranking_and_top_action_quality():
    score = selection_score(
        {
            "balanced_sign_accuracy": 0.65,
            "pairwise_accuracy": 0.7,
            "top1_improving_fraction": 0.6,
            "r2": 0.2,
        }
    )
    assert score == pytest.approx(1.96)


def test_structured_selection_score_can_use_top1_mean_return():
    assert selection_score(
        {"top1_mean_return": 1.25}, "top1_mean_return"
    ) == pytest.approx(1.25)


def test_structured_action_training_smoke(tmp_path):
    names = ["global.fill"]
    for vessel in ("a", "b", "c"):
        names.extend([f"{vessel}.cargo", f"greedy_proposal.{vessel}.native_action_0"])
    joint_actions = [
        [left, right, third]
        for left in range(2)
        for right in range(2)
        for third in range(2)
    ]
    metadata = {
        "kind": "dense_same_state_greedy_rollin_event_counterfactual",
        "split": "train",
        "scenario_seeds": [1, 2],
        "state_feature_names": names,
        "forecast_channel_names": [f"f{index}" for index in range(9)],
        "joint_actions": joint_actions,
        "follow_action_index": 7,
        "reward_scale": 1e-5,
        "uses_mpc": False,
    }

    def save_data(path, seeds, roots):
        rows = len(seeds) * 3
        states = np.zeros((rows, 1, len(names)), dtype=np.float32)
        forecasts = np.zeros((rows, 1, 168, 9), dtype=np.float32)
        actions = np.zeros((rows, 1), dtype=np.int64)
        returns = np.zeros((rows, 1), dtype=np.float32)
        scenario = np.zeros(rows, dtype=np.int64)
        root_time = np.zeros(rows, dtype=np.int64)
        row = 0
        for seed, root in zip(seeds, roots):
            for action, target in zip((0, 1, 2), (-1.0, 0.5, 1.5)):
                states[row, 0] = float(seed)
                forecasts[row, 0] = float(seed)
                actions[row, 0] = action
                returns[row, 0] = target
                scenario[row] = seed
                root_time[row] = root
                row += 1
        row_metadata = dict(metadata)
        row_metadata["scenario_seeds"] = list(seeds)
        np.savez(
            path,
            states=states,
            forecasts=forecasts,
            actions=actions,
            return_to_go=returns,
            scenario_seed=scenario,
            root_time_h=root_time,
            metadata_json=np.asarray(json.dumps(row_metadata)),
        )

    train_path = tmp_path / "train.npz"
    validation_path = tmp_path / "validation.npz"
    save_data(train_path, [1, 2], [10, 20])
    save_data(validation_path, [3], [30])
    normalization = {
        "state_mean": np.zeros(len(names), dtype=np.float32),
        "state_std": np.ones(len(names), dtype=np.float32),
        "forecast_mean": np.zeros(9, dtype=np.float32),
        "forecast_std": np.ones(9, dtype=np.float32),
        "return_scale": 1.0,
    }
    source_model = RecurrentBootstrappedQuantileQ(
        names,
        (168, 9),
        len(joint_actions),
        **normalization,
        heads=2,
        quantiles=3,
        prior_scale=0.25,
    )
    checkpoint_path = tmp_path / "source.pt"
    torch.save(
        {
            "model_state_dict": source_model.state_dict(),
            "metadata": metadata,
            "normalization": normalization,
        },
        checkpoint_path,
    )
    args = parse_args(
        [
            "--train-data",
            str(train_path),
            "--validation-data",
            str(validation_path),
            "--initial-checkpoint",
            str(checkpoint_path),
            "--skip-initial-weights",
            "--out-dir",
            str(tmp_path / "out"),
            "--epochs",
            "1",
            "--patience",
            "1",
            "--batch-size",
            "1",
            "--heads",
            "2",
            "--quantiles",
            "3",
            "--action-feature-size",
            "8",
            "--action-embedding-size",
            "4",
            "--device",
            "cpu",
        ]
    )
    summary = run(args)
    assert summary["train_groups"] == 2
    assert summary["validation_groups"] == 1
    assert summary["loaded_pretrained_tensors"] == 0
    assert (tmp_path / "out" / "structured_action_recurrent_q.pt").exists()


@pytest.mark.parametrize(
    "forecast_encoder",
    [
        "small_mlp",
        "temporal_attention",
        "action_aligned",
        "arrival_time",
        "eta_aligned",
        "eta_joint",
        "window_summary_24_72",
        "window_summary_168",
        "window_summary_24_72_168",
        "window_summary_joint_168",
    ],
)
def test_stateless_future_training_smoke(tmp_path, forecast_encoder):
    emitters = ("a", "b", "c")
    vessels = ("ship_a", "ship_b", "ship_c")
    names = [
        "a.fill",
        "b.fill",
        "c.fill",
        "oygarden_terminal.fill",
        "weather.speed_now",
    ]
    for vessel in vessels:
        names.extend(
            f"{vessel}.to_{destination}.travel_hours_now"
            for destination in ("oygarden_terminal", *emitters)
        )
        names.extend(
            f"greedy_proposal.{vessel}.native_action_{action}"
            for action in range(5)
        )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.well",
        "injectivity.well",
        "weather.global_speed_factor",
    ]
    joint_actions = [[action, action, action] for action in range(6)]
    metadata = {
        "kind": "dense_same_state_greedy_rollin_event_counterfactual",
        "scenario_seeds": [1],
        "state_feature_names": names,
        "forecast_channel_names": forecast_names,
        "joint_actions": joint_actions,
        "follow_action_index": 5,
        "reward_scale": 1e-5,
        "episode_hours": 720,
        "uses_mpc": False,
    }

    def save_data(path, seed, root):
        rows = 3
        states = np.zeros((rows, 1, len(names)), dtype=np.float32)
        states[:, 0, names.index("weather.speed_now")] = 1.0
        np.savez(
            path,
            states=states,
            forecasts=np.ones(
                (rows, 1, 168, len(forecast_names)), dtype=np.float32
            ),
            actions=np.asarray([[0], [1], [2]], dtype=np.int64),
            return_to_go=np.asarray([[-1.0], [0.5], [1.5]], dtype=np.float32),
            scenario_seed=np.full(rows, seed, dtype=np.int64),
            root_time_h=np.full(rows, root, dtype=np.int64),
            metadata_json=np.asarray(
                json.dumps({**metadata, "scenario_seeds": [seed]})
            ),
        )

    train_path = tmp_path / f"train_{forecast_encoder}.npz"
    validation_path = tmp_path / f"validation_{forecast_encoder}.npz"
    save_data(train_path, 1, 10)
    save_data(validation_path, 2, 20)
    normalization = {
        "state_mean": np.zeros(len(names), dtype=np.float32),
        "state_std": np.ones(len(names), dtype=np.float32),
        "forecast_mean": np.zeros(len(forecast_names), dtype=np.float32),
        "forecast_std": np.ones(len(forecast_names), dtype=np.float32),
        "return_scale": 1.0,
    }
    checkpoint_path = tmp_path / "normalization.pt"
    torch.save(
        {
            "model_state_dict": {},
            "metadata": metadata,
            "normalization": normalization,
        },
        checkpoint_path,
    )
    args = parse_args(
        [
            "--train-data",
            str(train_path),
            "--validation-data",
            str(validation_path),
            "--initial-checkpoint",
            str(checkpoint_path),
            "--skip-initial-weights",
            "--out-dir",
            str(tmp_path / f"stateless_{forecast_encoder}"),
            "--epochs",
            "1",
            "--patience",
            "1",
            "--batch-size",
            "1",
            "--heads",
            "2",
            "--quantiles",
            "3",
            "--action-feature-size",
            "8",
            "--action-embedding-size",
            "4",
            "--model-architecture",
            "stateless_structured",
            "--observation-input",
            "state_future",
            "--forecast-encoder",
            forecast_encoder,
            "--device",
            "cpu",
        ]
    )
    summary = run(args)
    assert summary["configuration"]["q_head"] == "stateless_structured"
    assert (
        tmp_path
        / f"stateless_{forecast_encoder}"
        / "structured_action_stateless_q.pt"
    ).exists()
    if forecast_encoder == "eta_joint" or forecast_encoder.startswith(
        "window_summary_"
    ):
        from experiments.evaluate_event_recurrent_q_policy import _load_model

        checkpoint = (
            tmp_path
            / f"stateless_{forecast_encoder}"
            / "structured_action_stateless_q.pt"
        )
        loaded, _metadata = _load_model(
            argparse.Namespace(checkpoint=str(checkpoint), device="cpu"),
            torch.device("cpu"),
        )
        if forecast_encoder == "eta_joint":
            assert loaded.eta_joint_q is not None
        elif forecast_encoder == "window_summary_joint_168":
            assert loaded.window_summary_joint_q is not None
        else:
            assert loaded.window_summary_residual is not None
