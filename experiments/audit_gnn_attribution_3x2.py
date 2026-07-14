from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sb3_contrib import MaskablePPO

from sim.control.demonstrations import load_demonstrations
from sim.control.vessel_diagnostics import masked_vessel_action_probabilities


BASE = ROOT / "output" / "rl_forecast" / "gnn_attribution_3x2"
NEW_RESULTS = BASE / "results"
SMALL_MLP_RESULTS = (
    ROOT / "output" / "rl_forecast" / "aligned_forecast_v4_bc" / "results"
)
CACHE = (
    ROOT
    / "output"
    / "rl_forecast"
    / "corrected_forecast_cache"
    / "destination_mask_heldout_121_140_v4.npz"
)
VARIANTS = (
    "tcn_mode_destination",
    "fixed_scale_tcn_mode_destination",
    "larger_mlp_mode_destination",
    "fixed_scale_larger_mlp_mode_destination",
    "edge_gnn_mode_destination",
    "fixed_scale_edge_gnn_mode_destination",
)


def checkpoint_path(variant: str, seed: int) -> Path:
    root = (
        SMALL_MLP_RESULTS
        if variant in {"tcn_mode_destination", "fixed_scale_tcn_mode_destination"}
        else NEW_RESULTS
    )
    return root / f"bc_{variant}_decision_only_seed{seed}.zip"


def cross_seed_forecast(batch):
    lookup = {
        (int(seed), int(hour)): index
        for index, (seed, hour) in enumerate(zip(batch.seeds, batch.hours))
    }
    seeds = sorted(set(int(value) for value in batch.seeds))
    successor = {seed: seeds[(index + 1) % len(seeds)] for index, seed in enumerate(seeds)}
    indices = np.asarray(
        [lookup[(successor[int(seed)], int(hour))] for seed, hour in zip(batch.seeds, batch.hours)]
    )
    return batch.forecast[indices]


def imitation_metrics(probabilities, actions, masks):
    predicted = np.stack([values.argmax(axis=1) for values in probabilities], axis=1)
    expected = actions[:, :3]
    correct = predicted == expected
    active_correct = []
    dispatch_correct = []
    offset = 0
    for vessel in range(3):
        action_count = probabilities[vessel].shape[1]
        legal_count = masks[:, offset : offset + action_count].sum(axis=1)
        active_correct.extend(correct[legal_count > 1, vessel].tolist())
        dispatch_correct.extend(correct[expected[:, vessel] != 0, vessel].tolist())
        offset += action_count
    return {
        "active_accuracy": float(np.mean(active_correct)),
        "destination_accuracy": float(np.mean(dispatch_correct)),
        "predicted": predicted,
    }


def gradient_metrics(model, observations, sample_count: int = 256):
    device = model.policy.device
    state = torch.as_tensor(
        observations["state"][:sample_count],
        dtype=torch.float32,
        device=device,
    )
    forecast = torch.as_tensor(
        observations["forecast"][:sample_count],
        dtype=torch.float32,
        device=device,
    ).requires_grad_(True)
    model.policy.set_training_mode(False)
    features = model.policy.features_extractor(
        {"state": state, "forecast": forecast}
    )[:, 64:]
    features.square().mean().backward()
    gradient = forecast.grad
    assert gradient is not None
    return {
        "forecast_feature_l2": float(
            torch.linalg.vector_norm(features, dim=1).mean().detach().cpu()
        ),
        "forecast_input_gradient_l2": float(
            torch.linalg.vector_norm(gradient.flatten(1), dim=1).mean().detach().cpu()
        ),
    }


def main():
    batch = load_demonstrations(CACHE, None)
    shuffled_forecast = cross_seed_forecast(batch)
    rows = []
    for variant in VARIANTS:
        actual_observations = batch.observations(variant)
        shuffled_observations = {
            "state": actual_observations["state"],
            "forecast": shuffled_forecast,
        }
        for seed in range(5):
            checkpoint = checkpoint_path(variant, seed)
            model = MaskablePPO.load(checkpoint, device="cuda")
            actual_probabilities = masked_vessel_action_probabilities(
                model, actual_observations, batch.masks, 3
            )
            shuffled_probabilities = masked_vessel_action_probabilities(
                model, shuffled_observations, batch.masks, 3
            )
            actual = imitation_metrics(actual_probabilities, batch.actions, batch.masks)
            shuffled = imitation_metrics(
                shuffled_probabilities,
                batch.actions,
                batch.masks,
            )
            tv = np.mean(
                [
                    0.5 * np.abs(left - right).sum(axis=1).mean()
                    for left, right in zip(actual_probabilities, shuffled_probabilities)
                ]
            )
            rows.append(
                {
                    "variant": variant,
                    "model_seed": seed,
                    **gradient_metrics(model, actual_observations),
                    "actual_active_accuracy": actual["active_accuracy"],
                    "shuffled_active_accuracy": shuffled["active_accuracy"],
                    "actual_destination_accuracy": actual["destination_accuracy"],
                    "shuffled_destination_accuracy": shuffled["destination_accuracy"],
                    "mean_probability_tv": float(tv),
                    "argmax_row_change_rate": float(
                        (actual["predicted"] != shuffled["predicted"]).any(axis=1).mean()
                    ),
                }
            )
            print(f"DONE variant={variant} seed={seed}", flush=True)
    destination = BASE / "forecast_use_audit.json"
    destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
