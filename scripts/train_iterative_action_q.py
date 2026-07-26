"""Train iterative action-value models on paired same-state outcomes."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from sim.control.iterative_action_q import (
    IterativeActionQuantileQ,
    IterativeForecastActionQuantileQ,
    IterativeFutureActionQuantileQ,
    quantile_huber_loss,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", nargs="+", required=True)
    parser.add_argument("--validation-data", nargs="+", required=True)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--observation-input",
        choices=("state_only", "v4_future_24_72", "forecast_168"),
        default="state_only",
    )
    parser.add_argument(
        "--forecast-encoder",
        choices=("small_mlp", "tcn", "gru"),
        default="small_mlp",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--quantiles", type=int, default=51)
    parser.add_argument("--prior-scale", type=float, default=0.25)
    parser.add_argument("--action-embedding-size", type=int, default=16)
    parser.add_argument("--action-feature-size", type=int, default=64)
    parser.add_argument("--bootstrap-probability", type=float, default=0.8)
    parser.add_argument("--improving-sample-weight", type=float, default=2.5)
    parser.add_argument("--ranking-coefficient", type=float, default=1.0)
    parser.add_argument("--listwise-coefficient", type=float, default=1.0)
    parser.add_argument("--classification-coefficient", type=float, default=1.0)
    parser.add_argument("--follow-anchor-coefficient", type=float, default=0.5)
    parser.add_argument("--pairwise-min-cost-eur", type=float, default=10000.0)
    parser.add_argument("--ranking-temperature", type=float, default=0.5)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=("composite", "top1_mean_return"),
        default="composite",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if min(
        args.epochs,
        args.patience,
        args.batch_size,
        args.heads,
        args.quantiles,
        args.action_embedding_size,
        args.action_feature_size,
    ) <= 0:
        parser.error("training and model sizes must be positive")
    if min(
        args.encoder_learning_rate,
        args.head_learning_rate,
        args.improving_sample_weight,
        args.ranking_temperature,
    ) <= 0.0:
        parser.error("learning rates, weights, and temperature must be positive")
    if not 0.0 < args.bootstrap_probability <= 1.0:
        parser.error("bootstrap probability must be in (0, 1]")
    if min(
        args.ranking_coefficient,
        args.listwise_coefficient,
        args.classification_coefficient,
        args.follow_anchor_coefficient,
        args.pairwise_min_cost_eur,
    ) < 0.0:
        parser.error("loss coefficients and pairwise threshold must be non-negative")
    return args


def _load(path: str):
    required_fields = (
        "states",
        "actions",
        "return_to_go",
        "scenario_seed",
        "root_time_h",
    )
    with np.load(path, allow_pickle=False) as loaded:
        data = {field: loaded[field].copy() for field in required_fields}
        if "future_summaries" in loaded:
            data["future_summaries"] = loaded["future_summaries"].copy()
        if "future_forecasts" in loaded:
            data["future_forecasts"] = loaded["future_forecasts"].copy()
        metadata = json.loads(str(loaded["metadata_json"]))
    return data, metadata


def _load_collection(paths):
    return [_load(path) for path in paths]


def regression_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    error = predicted - actual
    denominator = float(np.square(actual - actual.mean()).sum())
    actual_positive = actual > 1e-6
    predicted_positive = predicted > 1e-6
    actual_nonpositive = ~actual_positive
    true_positive_rate = (
        float(predicted_positive[actual_positive].mean())
        if actual_positive.any()
        else np.nan
    )
    true_negative_rate = (
        float((~predicted_positive[actual_nonpositive]).mean())
        if actual_nonpositive.any()
        else np.nan
    )
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": (
            float(1.0 - np.square(error).sum() / denominator)
            if denominator > 0.0
            else np.nan
        ),
        "pearson": float(np.corrcoef(actual, predicted)[0, 1]),
        "sign_accuracy": float((actual_positive == predicted_positive).mean()),
        "balanced_sign_accuracy": float(
            np.nanmean([true_positive_rate, true_negative_rate])
        ),
        "improving_fraction": float(actual_positive.mean()),
        "top_decile_improving_fraction": float(
            actual_positive[
                np.argsort(predicted)[-max(1, int(np.ceil(0.10 * len(actual)))) :]
            ].mean()
        ),
    }


def dataset_normalization(
    rows, observation_input: str = "state_only"
) -> dict[str, np.ndarray | float]:
    """Compute P1 normalization directly from its Greedy training data."""

    unique_root_states = []
    for data, _metadata in rows:
        keys = np.stack((data["scenario_seed"], data["root_time_h"]), axis=1)
        _unique, first_indices = np.unique(keys, axis=0, return_index=True)
        unique_root_states.append(
            data["states"][np.sort(first_indices), 0].astype(np.float32)
        )
    states = np.concatenate(unique_root_states)
    returns = np.concatenate(
        [data["return_to_go"][:, 0].astype(np.float32) for data, _metadata in rows]
    )
    normalization = {
        "state_mean": states.mean(axis=0),
        "state_std": np.maximum(states.std(axis=0), 1e-5),
        "return_scale": max(float(returns.std()), 1.0),
    }
    if observation_input == "v4_future_24_72":
        unique_root_futures = []
        for data, _metadata in rows:
            if "future_summaries" not in data:
                raise ValueError("future-aware training requires future_summaries")
            keys = np.stack((data["scenario_seed"], data["root_time_h"]), axis=1)
            _unique, first_indices = np.unique(keys, axis=0, return_index=True)
            unique_root_futures.append(
                data["future_summaries"][
                    np.sort(first_indices), 0
                ].astype(np.float32)
            )
        futures = np.concatenate(unique_root_futures)
        normalization.update(
            {
                "future_mean": futures.mean(axis=0),
                "future_std": np.maximum(futures.std(axis=0), 1e-5),
            }
        )
    elif observation_input == "forecast_168":
        unique_root_forecasts = []
        for data, _metadata in rows:
            if "future_forecasts" not in data:
                raise ValueError("forecast-aware training requires future_forecasts")
            keys = np.stack((data["scenario_seed"], data["root_time_h"]), axis=1)
            _unique, first_indices = np.unique(keys, axis=0, return_index=True)
            unique_root_forecasts.append(
                data["future_forecasts"][
                    np.sort(first_indices), 0
                ].astype(np.float32)
            )
        forecasts = np.concatenate(unique_root_forecasts)
        valid = forecasts[..., -1] > 0.5
        values = forecasts[..., :-1][valid]
        normalization.update(
            {
                "forecast_mean": values.mean(axis=0),
                "forecast_std": np.maximum(values.std(axis=0), 1e-5),
            }
        )
    return normalization


class GroupedDenseActionDataset(Dataset):
    def __init__(
        self,
        data,
        follow_action_index: int | None = None,
        observation_input: str = "state_only",
    ):
        keys = np.stack((data["scenario_seed"], data["root_time_h"]), axis=1)
        self.groups = []
        self.root_hours = []
        for key in np.unique(keys, axis=0):
            indices = np.flatnonzero(np.all(keys == key, axis=1))
            reference_state = data["states"][indices[0], 0]
            if not np.allclose(data["states"][indices, 0], reference_state):
                raise ValueError("same-root state observations are not identical")
            if observation_input == "v4_future_24_72":
                if "future_summaries" not in data:
                    raise ValueError(
                        "future-aware training requires future_summaries"
                    )
                reference_future = data["future_summaries"][indices[0], 0]
                if not np.allclose(
                    data["future_summaries"][indices, 0], reference_future
                ):
                    raise ValueError(
                        "same-root future summaries are not identical"
                    )
            elif observation_input == "forecast_168":
                if "future_forecasts" not in data:
                    raise ValueError(
                        "forecast-aware training requires future_forecasts"
                    )
                reference_future = data["future_forecasts"][indices[0], 0]
                if not np.allclose(
                    data["future_forecasts"][indices, 0], reference_future
                ):
                    raise ValueError(
                        "same-root future forecasts are not identical"
                    )
            else:
                reference_future = np.empty(0, dtype=np.float32)
            actions = data["actions"][indices, 0].astype(np.int64)
            targets = data["return_to_go"][indices, 0].astype(np.float32)
            if len(actions) != len(np.unique(actions)):
                keep = []
                for action in np.unique(actions):
                    matches = np.flatnonzero(actions == action)
                    if not np.allclose(targets[matches], targets[matches[0]]):
                        raise ValueError(
                            "same-root duplicate actions have inconsistent targets"
                        )
                    keep.append(int(matches[0]))
                keep.sort()
                actions = actions[keep]
                targets = targets[keep]
            if follow_action_index is not None:
                if int(follow_action_index) in actions:
                    raise ValueError("dense action data unexpectedly contains FOLLOW")
                actions = np.concatenate((actions, [int(follow_action_index)]))
                targets = np.concatenate((targets, [0.0])).astype(np.float32)
            self.groups.append(
                (
                    reference_state.astype(np.float32),
                    reference_future.astype(np.float32),
                    actions,
                    targets,
                )
            )
            self.root_hours.append(int(key[1]))
        self.max_actions = max(len(group[2]) for group in self.groups)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        state, future, actions, targets = self.groups[index]
        padded_actions = np.full(self.max_actions, -1, dtype=np.int64)
        padded_targets = np.zeros(self.max_actions, dtype=np.float32)
        valid = np.zeros(self.max_actions, dtype=bool)
        padded_actions[: len(actions)] = actions
        padded_targets[: len(actions)] = targets
        valid[: len(actions)] = True
        return (
            state,
            future,
            padded_actions,
            padded_targets,
            valid,
            self.root_hours[index],
        )


def _combined_dataset(rows, follow_action_index, observation_input):
    datasets = [
        GroupedDenseActionDataset(
            data,
            None if metadata.get("anchors_in_data", False) else follow_action_index,
            observation_input,
        )
        for data, metadata in rows
    ]
    max_actions = max(dataset.max_actions for dataset in datasets)
    for dataset in datasets:
        dataset.max_actions = max_actions
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def selected_action_quantiles(q: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Gather padded candidate actions from [batch, head, action, quantile] Q."""

    safe = actions.clamp(min=0)
    index = safe[:, None, :, None].expand(-1, q.shape[1], -1, q.shape[3])
    return q.gather(2, index).permute(0, 2, 1, 3)


def model_quantiles(model, state, future, device):
    states = state[:, None].to(device)
    if isinstance(
        model, (IterativeFutureActionQuantileQ, IterativeForecastActionQuantileQ)
    ):
        futures = future[:, None].to(device)
        return model(states, futures)
    return model(states)


def evaluate(
    model,
    loader,
    device,
    reward_scale,
    pairwise_min_cost_eur,
):
    actual_rows = []
    predicted_rows = []
    pair_correct = []
    selected_returns = []
    oracle_returns = []
    by_root = {}
    model.eval()
    minimum_return = float(pairwise_min_cost_eur) * float(reward_scale)
    with torch.no_grad():
        for state, future, actions, targets, valid, root_hours in loader:
            batch = len(state)
            q = model_quantiles(model, state, future, device)
            chosen = selected_action_quantiles(q[:, 0], actions.to(device))
            predicted = (
                chosen.mean(dim=(-1, -2)).cpu().numpy() * float(model.return_scale)
            )
            targets_np = targets.numpy()
            valid_np = valid.numpy()
            for row in range(batch):
                row_valid = valid_np[row]
                row_actual = targets_np[row, row_valid]
                row_predicted = predicted[row, row_valid]
                actual_rows.append(row_actual)
                predicted_rows.append(row_predicted)
                selected_returns.append(float(row_actual[np.argmax(row_predicted)]))
                oracle_returns.append(float(row_actual.max()))
                for left in range(len(row_actual)):
                    for right in range(left + 1, len(row_actual)):
                        difference = row_actual[left] - row_actual[right]
                        if abs(difference) >= minimum_return:
                            predicted_difference = row_predicted[left] - row_predicted[right]
                            pair_correct.append(bool(difference * predicted_difference > 0.0))
                bucket = by_root.setdefault(
                    int(root_hours[row]),
                    {"selected": [], "oracle": [], "pairs": []},
                )
                bucket["selected"].append(selected_returns[-1])
                bucket["oracle"].append(oracle_returns[-1])
                for left in range(len(row_actual)):
                    for right in range(left + 1, len(row_actual)):
                        difference = row_actual[left] - row_actual[right]
                        if abs(difference) >= minimum_return:
                            predicted_difference = row_predicted[left] - row_predicted[right]
                            bucket["pairs"].append(
                                bool(difference * predicted_difference > 0.0)
                            )
    actual = np.concatenate(actual_rows)
    predicted = np.concatenate(predicted_rows)
    metrics = regression_metrics(actual, predicted)
    metrics.update(
        {
            "pairwise_accuracy": float(np.mean(pair_correct)),
            "top1_improving_fraction": float((np.asarray(selected_returns) > 1e-6).mean()),
            "top1_non_worse_fraction": float(
                (np.asarray(selected_returns) >= -1e-6).mean()
            ),
            "top1_mean_return": float(np.mean(selected_returns)),
            "oracle_improving_fraction": float((np.asarray(oracle_returns) > 1e-6).mean()),
            "oracle_mean_return": float(np.mean(oracle_returns)),
            "mean_regret": float(
                np.mean(np.asarray(oracle_returns) - np.asarray(selected_returns))
            ),
            "groups": len(selected_returns),
        }
    )
    metrics["by_root_hour"] = {
        str(hour): {
            "groups": len(bucket["selected"]),
            "pairwise_accuracy": (
                float(np.mean(bucket["pairs"])) if bucket["pairs"] else None
            ),
            "top1_improving_fraction": float(
                (np.asarray(bucket["selected"]) > 1e-6).mean()
            ),
            "top1_non_worse_fraction": float(
                (np.asarray(bucket["selected"]) >= -1e-6).mean()
            ),
            "top1_mean_return": float(np.mean(bucket["selected"])),
            "oracle_improving_fraction": float(
                (np.asarray(bucket["oracle"]) > 1e-6).mean()
            ),
        }
        for hour, bucket in sorted(by_root.items())
    }
    return metrics


def selection_score(metrics, metric="composite"):
    if metric == "top1_mean_return":
        return float(metrics["top1_mean_return"])
    if metric != "composite":
        raise ValueError(f"unknown checkpoint selection metric: {metric}")
    return float(
        metrics["balanced_sign_accuracy"]
        + metrics["pairwise_accuracy"]
        + metrics["top1_improving_fraction"]
        + 0.05 * metrics["r2"]
    )


def run(args):
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    train_rows = _load_collection(args.train_data)
    validation_rows = _load_collection(args.validation_data)
    train_metadata = train_rows[0][1]
    all_rows = train_rows + validation_rows
    for key in ("state_feature_names", "joint_actions"):
        if any(metadata[key] != train_metadata[key] for _data, metadata in all_rows):
            raise ValueError(f"dataset schema mismatch for {key}")
    if args.observation_input == "v4_future_24_72":
        if "future_feature_names" not in train_metadata:
            raise ValueError("future-aware training requires future feature names")
        if any(
            metadata.get("future_feature_names")
            != train_metadata["future_feature_names"]
            for _data, metadata in all_rows
        ):
            raise ValueError("dataset schema mismatch for future_feature_names")
    elif args.observation_input == "forecast_168":
        if "forecast_feature_names" not in train_metadata:
            raise ValueError("forecast-aware training requires forecast feature names")
        if any(
            metadata.get("forecast_feature_names")
            != train_metadata["forecast_feature_names"]
            for _data, metadata in all_rows
        ):
            raise ValueError("dataset schema mismatch for forecast_feature_names")
    for _data, metadata in all_rows:
        if float(metadata["reward_scale"]) != float(train_metadata["reward_scale"]):
            raise ValueError("dataset reward scales do not match")
        if metadata.get("uses_mpc") is not False:
            raise ValueError("training datasets must explicitly exclude MPC")
    train_seeds = set().union(
        *(set(metadata["scenario_seeds"]) for _data, metadata in train_rows)
    )
    validation_seeds = set().union(
        *(set(metadata["scenario_seeds"]) for _data, metadata in validation_rows)
    )
    if train_seeds & validation_seeds:
        raise ValueError("train and validation scenario seeds overlap")
    source = None
    normalization = dataset_normalization(train_rows, args.observation_input)
    if args.initial_checkpoint:
        source = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=False
        )
        for key in ("state_feature_names", "joint_actions"):
            if source["metadata"][key] != train_metadata[key]:
                raise ValueError(f"initial checkpoint schema mismatch for {key}")
        if source["configuration"].get(
            "observation_input", "state_only"
        ) != args.observation_input:
            raise ValueError("initial checkpoint observation input mismatch")
        if (
            args.observation_input == "forecast_168"
            and source["configuration"].get("forecast_encoder")
            != args.forecast_encoder
        ):
            raise ValueError("initial checkpoint forecast encoder mismatch")
        normalization_keys = ["state_mean", "state_std", "return_scale"]
        if args.observation_input == "v4_future_24_72":
            normalization_keys.extend(("future_mean", "future_std"))
        elif args.observation_input == "forecast_168":
            normalization_keys.extend(("forecast_mean", "forecast_std"))
        normalization = {
            key: source["normalization"][key] for key in normalization_keys
        }
    follow_index = int(train_metadata["follow_action_index"])
    train_dataset = _combined_dataset(
        train_rows, follow_index, args.observation_input
    )
    validation_dataset = _combined_dataset(
        validation_rows, follow_index, args.observation_input
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.model_seed),
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model_arguments = {
        **normalization,
        "heads": args.heads,
        "quantiles": args.quantiles,
        "prior_scale": args.prior_scale,
        "action_embedding_size": args.action_embedding_size,
        "action_feature_size": args.action_feature_size,
    }
    if args.observation_input == "v4_future_24_72":
        if "future_feature_names" not in train_metadata:
            raise ValueError("future-aware training requires future feature names")
        model = IterativeFutureActionQuantileQ(
            train_metadata["state_feature_names"],
            train_metadata["future_feature_names"],
            train_metadata["joint_actions"],
            **model_arguments,
        ).to(device)
    elif args.observation_input == "forecast_168":
        model = IterativeForecastActionQuantileQ(
            train_metadata["state_feature_names"],
            train_metadata["forecast_feature_names"],
            train_metadata["joint_actions"],
            forecast_horizon_h=int(train_metadata["forecast_horizon_h"]),
            forecast_encoder=args.forecast_encoder,
            **model_arguments,
        ).to(device)
    else:
        model = IterativeActionQuantileQ(
            train_metadata["state_feature_names"],
            train_metadata["joint_actions"],
            **model_arguments,
        ).to(device)
    compatible = {}
    if source is not None:
        target_state = model.state_dict()
        compatible = {
            key: value
            for key, value in source["model_state_dict"].items()
            if key in target_state and target_state[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
    head_parameters = []
    base_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("structured_"):
            head_parameters.append(parameter)
        else:
            base_parameters.append(parameter)
    parameter_groups = []
    if base_parameters:
        parameter_groups.append(
            {"params": base_parameters, "lr": args.encoder_learning_rate}
        )
    if head_parameters:
        parameter_groups.append(
            {"params": head_parameters, "lr": args.head_learning_rate}
        )
    optimizer = torch.optim.Adam(parameter_groups)
    bootstrap_rng = torch.Generator(device=device).manual_seed(args.model_seed + 1)
    reward_scale = float(train_metadata["reward_scale"])
    return_scale = float(model.return_scale)
    history = []
    best_score = -float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for state, future, actions, targets, valid, _root_hours in train_loader:
            batch = len(state)
            q = model_quantiles(model, state, future, device)
            root_q = q[:, 0]
            actions_device = actions.to(device)
            targets_device = targets.to(device)
            valid_device = valid.to(device)
            chosen = selected_action_quantiles(root_q, actions_device)
            normalized_targets = targets_device / return_scale
            target_quantiles = normalized_targets[:, :, None, None].expand(
                -1, -1, model.heads, 1
            )
            behavior_losses = quantile_huber_loss(chosen, target_quantiles)
            bootstrap = torch.rand(
                (batch, model.heads), generator=bootstrap_rng, device=device
            ) < float(args.bootstrap_probability)
            missing = ~bootstrap.any(dim=1)
            if missing.any():
                bootstrap[missing, 0] = True
            behavior_mask = valid_device[:, :, None] & bootstrap[:, None, :]
            sample_weights = torch.where(
                targets_device > 1e-6,
                torch.full_like(targets_device, float(args.improving_sample_weight)),
                torch.ones_like(targets_device),
            )
            behavior_loss = (
                behavior_losses * sample_weights[:, :, None]
            )[behavior_mask].mean()

            expected = chosen.mean(dim=-1) * return_scale
            target_difference = targets_device[:, :, None] - targets_device[:, None, :]
            predicted_difference = expected[:, :, None, :] - expected[:, None, :, :]
            pair_valid = valid_device[:, :, None] & valid_device[:, None, :]
            pair_valid &= torch.triu(
                torch.ones_like(pair_valid, dtype=torch.bool), diagonal=1
            )
            pair_valid &= target_difference.abs() >= (
                float(args.pairwise_min_cost_eur) * reward_scale
            )
            pair_mask = pair_valid[:, :, :, None] & bootstrap[:, None, None, :]
            pair_losses = F.softplus(
                -target_difference.sign()[:, :, :, None]
                * predicted_difference
                / float(args.ranking_temperature)
            )
            ranking_loss = (
                pair_losses[pair_mask].mean()
                if pair_mask.any()
                else pair_losses.sum() * 0.0
            )
            listwise_logits = expected.permute(0, 2, 1) / float(
                args.ranking_temperature
            )
            listwise_logits = listwise_logits.masked_fill(
                ~valid_device[:, None, :], -torch.inf
            )
            best_actions = targets_device.masked_fill(
                ~valid_device, -torch.inf
            ).argmax(dim=1)
            listwise_losses = -F.log_softmax(listwise_logits, dim=-1).gather(
                2, best_actions[:, None, None].expand(-1, model.heads, 1)
            ).squeeze(-1)
            listwise_loss = listwise_losses[bootstrap].mean()
            classification_logits = expected / float(args.ranking_temperature)
            classification_targets = (targets_device > 1e-6).to(expected.dtype)
            classification_losses = F.binary_cross_entropy_with_logits(
                classification_logits,
                classification_targets[:, :, None].expand_as(classification_logits),
                reduction="none",
            )
            classification_loss = classification_losses[behavior_mask].mean()
            follow_q = root_q[:, :, follow_index, :]
            zero = torch.zeros(
                (*follow_q.shape[:-1], 1), device=device, dtype=follow_q.dtype
            )
            anchor_loss = quantile_huber_loss(follow_q, zero).mean()
            loss = (
                behavior_loss
                + float(args.ranking_coefficient) * ranking_loss
                + float(args.listwise_coefficient) * listwise_loss
                + float(args.classification_coefficient) * classification_loss
                + float(args.follow_anchor_coefficient) * anchor_loss
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation = evaluate(
            model,
            validation_loader,
            device,
            reward_scale,
            args.pairwise_min_cost_eur,
        )
        score = selection_score(
            validation, args.checkpoint_selection_metric
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selection_score": score,
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score + 1e-8:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    model.load_state_dict(best_state)
    final_validation = evaluate(
        model,
        validation_loader,
        device,
        reward_scale,
        args.pairwise_min_cost_eur,
    )
    configuration = vars(args).copy()
    configuration["q_head"] = {
        "state_only": "iterative_action_q",
        "v4_future_24_72": "iterative_action_q_future_v4_24_72",
        "forecast_168": "iterative_action_q_future_168",
    }[args.observation_input]
    checkpoint_metadata = dict(train_metadata)
    checkpoint_metadata["training_data_sources"] = list(args.train_data)
    checkpoint_metadata["validation_data_sources"] = list(args.validation_data)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": checkpoint_metadata,
        "normalization": normalization,
        "configuration": configuration,
        "validation": final_validation,
    }
    checkpoint_path = out_dir / "iterative_action_q.pt"
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "kind": "iterative_action_q_training",
        "checkpoint": str(checkpoint_path),
        "train_groups": len(train_dataset),
        "validation_groups": len(validation_dataset),
        "loaded_pretrained_tensors": len(compatible),
        "configuration": configuration,
        "final_validation": final_validation,
        "history": history,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(final_validation, indent=2), flush=True)
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
