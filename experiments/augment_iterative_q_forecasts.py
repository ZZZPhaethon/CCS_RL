"""Add masked 168-hour forecasts to an existing iterative-Q dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments import iterative_q_data_common as common
from sim.environment.forecast import (
    forecast_channel_names,
    masked_future_forecast_observation,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--horizon-h", type=int, default=168)
    args = parser.parse_args(argv)
    if args.horizon_h <= 0:
        parser.error("horizon-h must be positive")
    return args


def augment(args):
    input_path = Path(args.input_path)
    out_path = Path(args.out_path)
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {out_path}")
    with np.load(input_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    metadata = json.loads(str(arrays.pop("metadata_json")))
    configuration = dict(metadata.get("configuration", {}))
    configuration.update(
        {
            "variant": metadata["observation_variant"],
            "episode_hours": int(metadata["episode_hours"]),
            "reward_scale": float(metadata["reward_scale"]),
            "scenario_protocol": metadata.get("scenario_protocol", "q_original"),
            "hard_scenario_probability": float(
                configuration.get("hard_scenario_probability", 0.5)
            ),
            "forecast_context_hours": max(
                int(configuration.get("forecast_context_hours", args.horizon_h)),
                int(args.horizon_h),
            ),
            "seeds": [int(seed) for seed in metadata["scenario_seeds"]],
        }
    )
    env_args = SimpleNamespace(**configuration)
    keys = np.stack((arrays["scenario_seed"], arrays["root_time_h"]), axis=1)
    unique_keys = np.unique(keys, axis=0)
    forecast_by_key = {}
    channel_names = None
    for seed in np.unique(unique_keys[:, 0]):
        env = common.make_native_env(env_args)
        env.reset(seed=int(seed))
        channel_names = list(forecast_channel_names(env))
        for _seed, root_time_h in unique_keys[unique_keys[:, 0] == seed]:
            env.simulator.state.time_h = float(root_time_h)
            forecast_by_key[(int(seed), int(root_time_h))] = np.asarray(
                masked_future_forecast_observation(
                    env, horizon_h=int(args.horizon_h)
                ),
                dtype=np.float32,
            )
    arrays["future_forecasts"] = np.asarray(
        [
            [forecast_by_key[(int(seed), int(root_time_h))]]
            for seed, root_time_h in keys
        ],
        dtype=np.float32,
    )
    metadata["forecast_feature_names"] = [*channel_names, "valid_horizon"]
    metadata["forecast_horizon_h"] = int(args.horizon_h)
    metadata["forecast_padding"] = "zero_after_episode_end"
    metadata["forecast_mask_channel"] = "valid_horizon"
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, separators=(",", ":"))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    summary = {
        "input_path": str(input_path),
        "out_path": str(out_path),
        "candidate_rows": int(len(keys)),
        "unique_roots": int(len(unique_keys)),
        "future_forecast_shape": list(arrays["future_forecasts"].shape),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    return augment(parse_args(argv))


if __name__ == "__main__":
    main()
