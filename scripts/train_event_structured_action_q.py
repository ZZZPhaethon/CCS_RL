"""Train a structured action-value model on paired same-state outcomes."""

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

if __package__:
    from scripts.train_event_recurrent_q_pretrain import regression_metrics
else:  # pragma: no cover
    from train_event_recurrent_q_pretrain import regression_metrics
from sim.control.recurrent_distributional_q import (
    StatelessStructuredActionQuantileQ,
    StructuredActionRecurrentQuantileQ,
    quantile_huber_loss,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", nargs="+", required=True)
    parser.add_argument("--validation-data", nargs="+", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--skip-initial-weights", action="store_true")
    parser.add_argument("--out-dir", required=True)
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
        "--model-architecture",
        choices=("recurrent_structured", "stateless_structured"),
        default="recurrent_structured",
    )
    parser.add_argument(
        "--observation-input",
        choices=("state_future", "state_only"),
        default="state_future",
    )
    parser.add_argument(
        "--forecast-encoder",
        choices=(
            "tcn",
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
        ),
        default="tcn",
    )
    parser.add_argument(
        "--train-action-aligned-residual-only",
        action="store_true",
    )
    parser.add_argument(
        "--trainable-components",
        choices=(
            "all",
            "residual_only",
            "residual_and_base_head",
            "base_head_only",
        ),
        default=None,
    )
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=("composite", "top1_mean_return"),
        default="composite",
    )
    parser.add_argument("--require-trained-checkpoint", action="store_true")
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
    if (
        args.model_architecture == "stateless_structured"
        and args.observation_input == "state_future"
        and args.forecast_encoder
        not in (
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
        )
    ):
        parser.error(
            "stateless state_future training requires small_mlp, "
            "temporal_attention, action_aligned, arrival_time, eta_aligned, "
            "eta_joint, or a window_summary forecast"
        )
    if args.train_action_aligned_residual_only:
        if args.trainable_components not in (None, "residual_only"):
            parser.error(
                "--train-action-aligned-residual-only conflicts with "
                "--trainable-components"
            )
        args.trainable_components = "residual_only"
    elif args.trainable_components is None:
        args.trainable_components = "all"
    return args


_RESIDUAL_PARAMETER_PREFIXES = (
    "action_aligned_residual.",
    "eta_aligned_residual.",
    "small_mlp_residual.",
    "temporal_attention_residual.",
    "arrival_time_residual.",
    "window_summary_residual.",
)
_BASE_HEAD_PARAMETER_PREFIXES = (
    "value.",
    "structured_action_embeddings.",
    "structured_action_fusion.",
    "structured_query.",
)


def configure_trainable_components(model: nn.Module, mode: str) -> None:
    if mode == "all":
        return
    if mode in ("residual_only", "residual_and_base_head"):
        has_residual = any(
            name.startswith(_RESIDUAL_PARAMETER_PREFIXES)
            for name, _parameter in model.named_parameters()
        )
        if not has_residual:
            raise ValueError(
                f"{mode} training requires a forecast residual"
            )
    prefixes = {
        "residual_only": _RESIDUAL_PARAMETER_PREFIXES,
        "residual_and_base_head": (
            *_RESIDUAL_PARAMETER_PREFIXES,
            *_BASE_HEAD_PARAMETER_PREFIXES,
        ),
        "base_head_only": _BASE_HEAD_PARAMETER_PREFIXES,
    }[mode]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))


def _load(path: str):
    fields = (
        "states",
        "forecasts",
        "actions",
        "return_to_go",
        "scenario_seed",
        "root_time_h",
    )
    with np.load(path, allow_pickle=False) as loaded:
        data = {field: loaded[field].copy() for field in fields}
        metadata = json.loads(str(loaded["metadata_json"]))
    return data, metadata


def _load_collection(paths):
    return [_load(path) for path in paths]


class GroupedDenseActionDataset(Dataset):
    def __init__(self, data, follow_action_index: int | None = None):
        keys = np.stack((data["scenario_seed"], data["root_time_h"]), axis=1)
        self.groups = []
        self.root_hours = []
        for key in np.unique(keys, axis=0):
            indices = np.flatnonzero(np.all(keys == key, axis=1))
            reference_state = data["states"][indices[0], 0]
            reference_forecast = data["forecasts"][indices[0], 0]
            if not np.allclose(data["states"][indices, 0], reference_state):
                raise ValueError("same-root state observations are not identical")
            if not np.allclose(data["forecasts"][indices, 0], reference_forecast):
                raise ValueError("same-root forecasts are not identical")
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
                    reference_forecast.astype(np.float32),
                    actions,
                    targets,
                )
            )
            self.root_hours.append(int(key[1]))
        self.max_actions = max(len(group[2]) for group in self.groups)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        state, forecast, actions, targets = self.groups[index]
        padded_actions = np.full(self.max_actions, -1, dtype=np.int64)
        padded_targets = np.zeros(self.max_actions, dtype=np.float32)
        valid = np.zeros(self.max_actions, dtype=bool)
        padded_actions[: len(actions)] = actions
        padded_targets[: len(actions)] = targets
        valid[: len(actions)] = True
        return (
            state,
            forecast,
            padded_actions,
            padded_targets,
            valid,
            self.root_hours[index],
        )


def _combined_dataset(rows, follow_action_index):
    datasets = [
        GroupedDenseActionDataset(
            data,
            None if metadata.get("anchors_in_data", False) else follow_action_index,
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


def model_forecast_input(forecast, model, observation_input, device):
    values = forecast[:, None].to(device)
    if (
        observation_input == "state_only"
        and not getattr(model, "is_stateless", False)
    ):
        values = model.forecast_mean.reshape(1, 1, 1, -1).expand_as(values)
    return values


def model_quantiles(model, state, forecast, observation_input, device):
    states = state[:, None].to(device)
    forecasts = model_forecast_input(
        forecast, model, observation_input, device
    )
    if getattr(model, "is_stateless", False):
        return model(states, forecasts)
    q, _hidden = model(
        states,
        forecasts,
        torch.full((len(state), 1), -1, dtype=torch.long, device=device),
        torch.zeros((len(state), 1), dtype=torch.float32, device=device),
        torch.zeros((len(state), 1), dtype=torch.float32, device=device),
    )
    return q


def evaluate(
    model,
    loader,
    device,
    reward_scale,
    pairwise_min_cost_eur,
    observation_input="state_future",
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
        for state, forecast, actions, targets, valid, root_hours in loader:
            batch = len(state)
            q = model_quantiles(
                model,
                state,
                forecast,
                observation_input,
                device,
            )
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
    train_data, train_metadata = train_rows[0]
    validation_data, validation_metadata = validation_rows[0]
    all_rows = train_rows + validation_rows
    for key in ("state_feature_names", "forecast_channel_names", "joint_actions"):
        if any(metadata[key] != train_metadata[key] for _data, metadata in all_rows):
            raise ValueError(f"dataset schema mismatch for {key}")
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
    source = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    for key in ("state_feature_names", "forecast_channel_names", "joint_actions"):
        if source["metadata"][key] != train_metadata[key]:
            raise ValueError(f"initial checkpoint schema mismatch for {key}")
    normalization = source["normalization"]
    follow_index = int(train_metadata["follow_action_index"])
    train_dataset = _combined_dataset(train_rows, follow_index)
    validation_dataset = _combined_dataset(validation_rows, follow_index)
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
        "forecast_channel_names": train_metadata["forecast_channel_names"],
        "episode_hours": int(train_metadata.get("episode_hours", 720)),
    }
    if args.model_architecture == "stateless_structured":
        model = StatelessStructuredActionQuantileQ(
            train_metadata["state_feature_names"],
            tuple(train_data["forecasts"].shape[2:]),
            train_metadata["joint_actions"],
            forecast_encoder=(
                "state_only"
                if args.observation_input == "state_only"
                else args.forecast_encoder
            ),
            **model_arguments,
        ).to(device)
    else:
        model = StructuredActionRecurrentQuantileQ(
            train_metadata["state_feature_names"],
            tuple(train_data["forecasts"].shape[2:]),
            train_metadata["joint_actions"],
            forecast_encoder=args.forecast_encoder,
            **model_arguments,
        ).to(device)
    compatible = {}
    if not args.skip_initial_weights:
        target_state = model.state_dict()
        compatible = {
            key: value
            for key, value in source["model_state_dict"].items()
            if key in target_state and target_state[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
    configure_trainable_components(model, args.trainable_components)
    head_parameters = []
    base_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(_RESIDUAL_PARAMETER_PREFIXES):
            head_parameters.append(parameter)
        elif (
            args.trainable_components
            in ("residual_and_base_head", "base_head_only")
        ):
            base_parameters.append(parameter)
        elif name.startswith(
            (
                "structured_",
                "action_aligned_residual.",
                "eta_aligned_residual.",
                "small_mlp_residual.",
                "arrival_time_residual.",
                "eta_joint_q.",
                "window_summary_residual.",
                "window_summary_joint_q.",
            )
        ):
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
    if args.trainable_components != "all":
        initial_validation = evaluate(
            model,
            validation_loader,
            device,
            reward_scale,
            args.pairwise_min_cost_eur,
            args.observation_input,
        )
        initial_score = selection_score(
            initial_validation, args.checkpoint_selection_metric
        )
        history.append(
            {
                "epoch": 0,
                "train_loss": 0.0,
                "selection_score": initial_score,
                "validation": initial_validation,
            }
        )
        if not args.require_trained_checkpoint:
            best_score = initial_score
            best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for state, forecast, actions, targets, valid, _root_hours in train_loader:
            batch = len(state)
            q = model_quantiles(
                model,
                state,
                forecast,
                args.observation_input,
                device,
            )
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
            args.observation_input,
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
        args.observation_input,
    )
    configuration = vars(args).copy()
    configuration["q_head"] = (
        "stateless_structured"
        if args.model_architecture == "stateless_structured"
        else "structured"
    )
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
    checkpoint_path = out_dir / (
        "structured_action_stateless_q.pt"
        if args.model_architecture == "stateless_structured"
        else "structured_action_recurrent_q.pt"
    )
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "kind": "structured_action_paired_q_training",
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
