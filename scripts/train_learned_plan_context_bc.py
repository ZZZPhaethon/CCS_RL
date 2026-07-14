"""Train native-action BC with cached continuous selector probabilities."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import compare_forecast_encoders_rl as compare
from sim.control.demonstrations import load_demonstrations


VARIANT = "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--heldout-cache", required=True)
    parser.add_argument("--selector-probabilities", required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def expand_plan_context(batch, probabilities, seeds, hours) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    seeds = np.asarray(seeds, dtype=np.int64)
    hours = np.asarray(hours, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 8:
        raise ValueError("selector probabilities must have shape [cycles, 8]")
    if not (len(probabilities) == len(seeds) == len(hours)):
        raise ValueError("selector probabilities, seeds, and hours must align")
    if np.any(hours % 24 != 0):
        raise ValueError("selector probability hours must be replan hours")
    lookup = {
        (int(seed), int(hour)): probability
        for seed, hour, probability in zip(seeds, hours, probabilities)
    }
    origins = np.asarray(batch.hours, dtype=np.int64)
    origins = origins - origins % 24
    try:
        return np.stack(
            [
                lookup[(int(seed), int(origin))]
                for seed, origin in zip(batch.seeds, origins)
            ]
        ).astype(np.float32, copy=False)
    except KeyError as error:
        raise ValueError(f"missing selector probability for cycle {error.args[0]}") from error


def train(args):
    compare_args = compare.parse_args(
        [
            "train",
            "--variant",
            VARIANT,
            "--demo-cache",
            args.train_cache,
            "--heldout-demo-cache",
            args.heldout_cache,
            "--bc-objective",
            "decision_only",
            "--bc-only",
            "--imitation-only",
            "--bc-epochs",
            str(args.epochs),
            "--bc-batch-size",
            "256",
            "--model-seed",
            str(args.model_seed),
            "--device",
            args.device,
            "--out-dir",
            args.out_dir,
            "--verbose",
            "0",
        ]
    )
    factory = compare.ExperimentEnvFactory(compare_args)
    metadata = factory.metadata()
    train_cache = compare.normalize_demo_cache_path(args.train_cache)
    heldout_cache = compare.normalize_demo_cache_path(args.heldout_cache)
    batch = load_demonstrations(train_cache, metadata)
    heldout_batch = load_demonstrations(heldout_cache, metadata)
    with np.load(Path(args.selector_probabilities), allow_pickle=False) as selector:
        candidate_names = tuple(str(value) for value in selector["candidate_names"].tolist())
        if candidate_names != batch.candidate_names or candidate_names != heldout_batch.candidate_names:
            raise ValueError("selector and demonstration candidate names must match")
        train_context = expand_plan_context(
            batch,
            selector["train_probabilities"],
            selector["train_seeds"],
            selector["train_hours"],
        )
        heldout_context = expand_plan_context(
            heldout_batch,
            selector["heldout_probabilities"],
            selector["heldout_seeds"],
            selector["heldout_hours"],
        )
    batch = replace(batch, plan_context=train_context)
    heldout_batch = replace(heldout_batch, plan_context=heldout_context)
    native_env = compare.make_experiment_env(compare_args, demonstration=False)
    return compare._train_loaded_batch(
        compare_args,
        batch=batch,
        observations=batch.observations(VARIANT),
        native_env=native_env,
        metadata=metadata,
        cache_sha256=compare.file_sha256(train_cache),
        heldout_batch=heldout_batch,
        heldout_cache_sha256=compare.file_sha256(heldout_cache),
    )


if __name__ == "__main__":
    train(parse_args())
