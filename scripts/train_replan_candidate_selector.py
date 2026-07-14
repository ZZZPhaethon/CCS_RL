"""Train a phase-zero MPC candidate selector and save continuous plan logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from sim.control.demonstrations import load_demonstrations
from sim.control.plan_context import CandidatePlanEncoder


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--heldout-cache", required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def replan_arrays(batch):
    if batch.operation_modes is None or batch.vessel_destinations is None:
        raise ValueError("candidate selector requires operation modes and destinations")
    if batch.plan_candidates is None or batch.candidate_names is None:
        raise ValueError("candidate selector requires plan candidate labels")
    rows = np.asarray(batch.hours) % 24 == 0
    state = np.concatenate(
        (
            batch.state,
            batch.operation_modes.reshape(len(batch.state), -1),
            batch.vessel_destinations.reshape(len(batch.state), -1),
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    return {
        "state": state[rows],
        "forecast": batch.forecast[rows],
        "targets": batch.plan_candidates[rows],
        "seeds": batch.seeds[rows],
        "hours": batch.hours[rows],
    }


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _tensor_observations(arrays, device):
    return {
        "state": torch.as_tensor(arrays["state"], device=device),
        "forecast": torch.as_tensor(arrays["forecast"], device=device),
    }


def _probabilities(model, observations, batch_size: int) -> np.ndarray:
    model.eval()
    total = len(observations["state"])
    values = []
    with torch.no_grad():
        for start in range(0, total, batch_size):
            logits = model(
                {key: value[start : start + batch_size] for key, value in observations.items()}
            )
            values.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(values, axis=0)


def _metrics(probabilities, targets, candidate_names):
    targets = np.asarray(targets, dtype=np.int64)
    predicted = probabilities.argmax(axis=1)
    per_candidate = []
    recalls = []
    for index, name in enumerate(candidate_names):
        selected = targets == index
        accuracy = float(np.mean(predicted[selected] == index))
        recalls.append(accuracy)
        per_candidate.append(
            {
                "index": index,
                "name": name,
                "count": int(selected.sum()),
                "recall": accuracy,
            }
        )
    true_probability = probabilities[np.arange(len(targets)), targets]
    return {
        "accuracy": float(np.mean(predicted == targets)),
        "macro_recall": float(np.mean(recalls)),
        "mean_true_probability": float(np.mean(true_probability)),
        "per_candidate": per_candidate,
    }


def train(args):
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.model_seed)

    train_batch = load_demonstrations(args.train_cache, None)
    heldout_batch = load_demonstrations(args.heldout_cache, None)
    if train_batch.candidate_names != heldout_batch.candidate_names:
        raise ValueError("train and held-out candidate names must match")
    candidate_names = train_batch.candidate_names
    train_arrays = replan_arrays(train_batch)
    heldout_arrays = replan_arrays(heldout_batch)
    device = _device(args.device)
    train_observations = _tensor_observations(train_arrays, device)
    heldout_observations = _tensor_observations(heldout_arrays, device)
    train_targets = torch.as_tensor(train_arrays["targets"], device=device)

    model = CandidatePlanEncoder(
        state_size=train_arrays["state"].shape[1],
        candidate_count=len(candidate_names),
    ).to(device)
    counts = np.bincount(
        train_arrays["targets"],
        minlength=len(candidate_names),
    ).astype(np.float32)
    class_weights = counts.sum() / (len(counts) * counts)
    loss_function = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    model.train()
    total = len(train_targets)
    generator = torch.Generator(device=device).manual_seed(args.model_seed)
    for _epoch in range(args.epochs):
        permutation = torch.randperm(total, generator=generator, device=device)
        for start in range(0, total, args.batch_size):
            indices = permutation[start : start + args.batch_size]
            logits = model(
                {key: value[indices] for key, value in train_observations.items()}
            )
            loss = loss_function(logits, train_targets[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_probabilities = _probabilities(model, train_observations, args.batch_size)
    heldout_probabilities = _probabilities(model, heldout_observations, args.batch_size)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"candidate_selector_seed{args.model_seed}"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_size": int(train_arrays["state"].shape[1]),
            "candidate_names": candidate_names,
        },
        output / f"{stem}.pt",
    )
    np.savez_compressed(
        output / f"{stem}_probabilities.npz",
        train_probabilities=train_probabilities,
        train_seeds=train_arrays["seeds"],
        train_hours=train_arrays["hours"],
        heldout_probabilities=heldout_probabilities,
        heldout_seeds=heldout_arrays["seeds"],
        heldout_hours=heldout_arrays["hours"],
        candidate_names=np.asarray(candidate_names),
    )
    result = {
        "model_seed": args.model_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "class_weights": class_weights.tolist(),
        "train": _metrics(
            train_probabilities,
            train_arrays["targets"],
            candidate_names,
        ),
        "heldout": _metrics(
            heldout_probabilities,
            heldout_arrays["targets"],
            candidate_names,
        ),
    }
    (output / f"{stem}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, ensure_ascii=False))
