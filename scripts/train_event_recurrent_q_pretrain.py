"""Pretrain recurrent distributional residual Q on counterfactual sequences."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sim.control.recurrent_distributional_q import (
    RecurrentBootstrappedQuantileQ,
    quantile_huber_loss,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--quantiles", type=int, default=51)
    parser.add_argument("--prior-scale", type=float, default=0.25)
    parser.add_argument("--bootstrap-probability", type=float, default=0.8)
    parser.add_argument("--follow-anchor-coefficient", type=float, default=0.25)
    parser.add_argument("--auxiliary-all-steps-coefficient", type=float, default=0.25)
    parser.add_argument("--improving-sample-weight", type=float, default=2.5)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.batch_size, args.heads, args.quantiles) <= 0:
        parser.error("training counts must be positive")
    if not 0.0 < args.bootstrap_probability <= 1.0:
        parser.error("bootstrap probability must be in (0, 1]")
    if (
        args.learning_rate <= 0.0
        or args.follow_anchor_coefficient < 0.0
        or args.auxiliary_all_steps_coefficient < 0.0
        or args.improving_sample_weight < 1.0
    ):
        parser.error("learning rate must be positive and loss coefficients non-negative")
    return args


def _load(path: str) -> tuple[dict[str, np.ndarray], dict]:
    fields = (
        "states",
        "forecasts",
        "actions",
        "event_durations",
        "event_residual_rewards",
        "return_to_go",
        "valid_steps",
        "actual_sequence_events",
        "scenario_seed",
    )
    with np.load(path, allow_pickle=False) as loaded:
        data = {field: loaded[field].copy() for field in fields}
        metadata = json.loads(str(loaded["metadata_json"]))
    return data, metadata


def observation_normalization(data) -> dict[str, np.ndarray | float]:
    valid = data["valid_steps"]
    states = data["states"][valid].astype(np.float32)
    forecasts = data["forecasts"][valid].astype(np.float32)
    returns = data["return_to_go"][valid].astype(np.float32)
    return {
        "state_mean": states.mean(axis=0),
        "state_std": np.maximum(states.std(axis=0), 1e-5),
        "forecast_mean": forecasts.mean(axis=(0, 1)),
        "forecast_std": np.maximum(forecasts.std(axis=(0, 1)), 1e-5),
        "return_scale": max(float(returns.std()), 1.0),
    }


class OfflineSequenceDataset(Dataset):
    def __init__(self, data):
        self.states = data["states"].astype(np.float32)
        self.forecasts = data["forecasts"].astype(np.float32)
        self.actions = data["actions"].astype(np.int64)
        self.rewards = data["event_residual_rewards"].astype(np.float32)
        self.durations = data["event_durations"].astype(np.float32)
        self.targets = data["return_to_go"].astype(np.float32)
        self.valid = data["valid_steps"].astype(bool)
        self.lengths = data["actual_sequence_events"].astype(np.int64)
        self.scenario_seeds = data["scenario_seed"].astype(np.int64)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        previous_actions = np.full_like(self.actions[index], -1)
        previous_rewards = np.zeros_like(self.rewards[index])
        previous_durations = np.zeros_like(self.durations[index])
        previous_actions[1:] = self.actions[index, :-1]
        previous_rewards[1:] = self.rewards[index, :-1]
        previous_durations[1:] = self.durations[index, :-1]
        return (
            self.states[index],
            self.forecasts[index],
            previous_actions,
            previous_rewards,
            previous_durations,
            self.actions[index],
            self.targets[index],
            self.valid[index],
        )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    denominator = float(np.square(actual - actual.mean()).sum())
    actual_positive = actual > 1e-6
    predicted_positive = predicted > 1e-6
    negative = ~actual_positive
    tpr = (
        float(predicted_positive[actual_positive].mean())
        if actual_positive.any()
        else np.nan
    )
    tnr = (
        float((~predicted_positive[negative]).mean()) if negative.any() else np.nan
    )
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - np.square(error).sum() / denominator)
        if denominator > 0.0
        else np.nan,
        "pearson": float(np.corrcoef(actual, predicted)[0, 1]),
        "sign_accuracy": float((actual_positive == predicted_positive).mean()),
        "balanced_sign_accuracy": float(np.nanmean([tpr, tnr])),
        "improving_fraction": float(actual_positive.mean()),
        "top_decile_improving_fraction": float(
            actual_positive[
                np.argsort(predicted)[-max(1, int(np.ceil(0.10 * len(actual)))) :]
            ].mean()
        ),
    }


def validation_selection_score(metrics: dict[str, float]) -> float:
    return float(
        metrics["balanced_sign_accuracy"]
        + 0.1 * metrics["r2"]
        + 0.1 * metrics["top_decile_improving_fraction"]
    )


def _forward_batch(model, batch, device):
    return model(
        batch[0].to(device),
        batch[1].to(device),
        batch[2].to(device),
        batch[3].to(device),
        batch[4].to(device),
    )[0]


def selected_quantiles(q, actions):
    safe_actions = actions.clamp(min=0)
    index = safe_actions[:, :, None, None, None].expand(
        -1, -1, q.shape[2], 1, q.shape[4]
    )
    return q.gather(3, index).squeeze(3)


def evaluate(model, loader, device, return_scale, follow_index):
    predictions = []
    targets = []
    all_predictions = []
    all_targets = []
    root_follow_values = []
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            q = _forward_batch(model, batch, device)
            actions = batch[5].to(device)
            target = batch[6].to(device) / float(return_scale)
            valid = batch[7].to(device)
            chosen = selected_quantiles(q, actions)
            target_quantiles = target[:, :, None, None].expand(
                -1, -1, model.heads, 1
            )
            loss = quantile_huber_loss(chosen, target_quantiles)
            last_indices = valid.sum(dim=1) - 1
            batch_indices = torch.arange(len(last_indices), device=device)
            last_chosen = chosen[batch_indices, last_indices]
            last_target = target[batch_indices, last_indices]
            last_target_quantiles = last_target[:, None, None].expand(
                -1, model.heads, 1
            )
            losses.append(
                quantile_huber_loss(last_chosen, last_target_quantiles).mean().item()
            )
            expected = chosen.mean(dim=(-1, -2)) * float(return_scale)
            predictions.append(expected[batch_indices, last_indices].cpu().numpy())
            targets.append(
                batch[6].to(device)[batch_indices, last_indices].cpu().numpy()
            )
            all_predictions.append(expected[valid].cpu().numpy())
            all_targets.append(batch[6][batch[7]].numpy())
            root_follow_values.append(
                q[:, 0, :, int(follow_index), :].mean(dim=(-1, -2)).cpu().numpy()
                * float(return_scale)
            )
    predicted = np.concatenate(predictions)
    actual = np.concatenate(targets)
    metrics = regression_metrics(actual, predicted)
    metrics["quantile_loss"] = float(np.mean(losses))
    metrics["all_steps"] = regression_metrics(
        np.concatenate(all_targets), np.concatenate(all_predictions)
    )
    metrics["root_follow_abs_mean"] = float(
        np.abs(np.concatenate(root_follow_values)).mean()
    )
    return metrics


def run(args):
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)

    train_data, metadata = _load(args.train_data)
    validation_data, validation_metadata = _load(args.validation_data)
    if metadata["split"] != "train" or validation_metadata["split"] != "validation":
        raise ValueError("offline Q pretraining requires train and validation splits")
    if set(metadata["scenario_seeds"]) & set(validation_metadata["scenario_seeds"]):
        raise ValueError("train and validation scenario seeds overlap")
    for key in ("state_feature_names", "forecast_channel_names", "joint_actions"):
        if metadata[key] != validation_metadata[key]:
            raise ValueError(f"train/validation schema mismatch for {key}")

    initial_checkpoint = None
    if args.initial_checkpoint:
        initial_checkpoint = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=False
        )
        initial_metadata = initial_checkpoint["metadata"]
        for key in ("state_feature_names", "forecast_channel_names", "joint_actions"):
            if initial_metadata[key] != metadata[key]:
                raise ValueError(f"initial checkpoint schema mismatch for {key}")
        initial_configuration = initial_checkpoint["configuration"]
        for key in ("heads", "quantiles", "prior_scale"):
            if float(initial_configuration[key]) != float(getattr(args, key)):
                raise ValueError(f"initial checkpoint configuration mismatch for {key}")
        normalization = initial_checkpoint["normalization"]
    else:
        normalization = observation_normalization(train_data)
    train_dataset = OfflineSequenceDataset(train_data)
    validation_dataset = OfflineSequenceDataset(validation_data)
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
    model = RecurrentBootstrappedQuantileQ(
        metadata["state_feature_names"],
        tuple(train_data["forecasts"].shape[2:]),
        len(metadata["joint_actions"]),
        **normalization,
        heads=args.heads,
        quantiles=args.quantiles,
        prior_scale=args.prior_scale,
    ).to(device)
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    bootstrap_rng = torch.Generator(device=device).manual_seed(args.model_seed + 1)
    follow_index = int(metadata["follow_action_index"])
    initial_validation = evaluate(
        model,
        validation_loader,
        device,
        normalization["return_scale"],
        follow_index,
    )
    best_score = validation_selection_score(initial_validation)
    history = [
        {
            "epoch": 0,
            "train_loss": None,
            "validation": initial_validation,
            "selection_score": best_score,
        }
    ]
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            q = _forward_batch(model, batch, device)
            actions = batch[5].to(device)
            targets = batch[6].to(device) / float(normalization["return_scale"])
            valid = batch[7].to(device)
            chosen = selected_quantiles(q, actions)
            target_quantiles = targets[:, :, None, None].expand(
                -1, -1, model.heads, 1
            )
            losses = quantile_huber_loss(chosen, target_quantiles)
            bootstrap = torch.rand(
                (q.shape[0], model.heads), generator=bootstrap_rng, device=device
            ) < float(args.bootstrap_probability)
            missing = ~bootstrap.any(dim=1)
            if missing.any():
                bootstrap[missing, 0] = True
            weights = valid[:, :, None] & bootstrap[:, None, :]
            improvement_threshold = 1e-6 / float(normalization["return_scale"])
            all_step_sample_weights = torch.where(
                targets > improvement_threshold,
                torch.full_like(targets, float(args.improving_sample_weight)),
                torch.ones_like(targets),
            )
            auxiliary_loss = (
                losses * all_step_sample_weights[:, :, None]
            )[weights].mean()
            last_indices = valid.sum(dim=1) - 1
            batch_indices = torch.arange(q.shape[0], device=device)
            last_chosen = chosen[batch_indices, last_indices]
            last_targets = targets[batch_indices, last_indices]
            last_target_quantiles = last_targets[:, None, None].expand(
                -1, model.heads, 1
            )
            last_losses = quantile_huber_loss(last_chosen, last_target_quantiles)
            sample_weights = torch.where(
                last_targets > improvement_threshold,
                torch.full_like(last_targets, float(args.improving_sample_weight)),
                torch.ones_like(last_targets),
            )
            behavior_loss = (last_losses * sample_weights[:, None])[bootstrap].mean()
            behavior_loss = behavior_loss + float(
                args.auxiliary_all_steps_coefficient
            ) * auxiliary_loss

            root_follow = q[:, 0, :, follow_index, :]
            zero_targets = torch.zeros(
                (*root_follow.shape[:-1], 1), device=device, dtype=root_follow.dtype
            )
            anchor_loss = quantile_huber_loss(root_follow, zero_targets).mean()
            loss = behavior_loss + float(args.follow_anchor_coefficient) * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        validation = evaluate(
            model,
            validation_loader,
            device,
            normalization["return_scale"],
            follow_index,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "validation": validation,
            }
        )
        selection_score = validation_selection_score(validation)
        history[-1]["selection_score"] = float(selection_score)
        print(json.dumps(history[-1]), flush=True)
        if selection_score > best_score + 1e-8:
            best_score = selection_score
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
        normalization["return_scale"],
        follow_index,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
        "normalization": normalization,
        "configuration": vars(args),
        "validation": final_validation,
    }
    torch.save(checkpoint, out_dir / "recurrent_distributional_q_pretrained.pt")
    summary = {
        "kind": "offline_recurrent_distributional_q_pretraining",
        "train_candidates": len(train_dataset),
        "validation_candidates": len(validation_dataset),
        "train_scenario_seeds": len(set(train_dataset.scenario_seeds)),
        "validation_scenario_seeds": len(set(validation_dataset.scenario_seeds)),
        "configuration": vars(args),
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
