"""Evaluate an hourly Masked Double-DQN checkpoint on fixed seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from sim.control.hourly_ppo.train_hourly_ppo import (
    evaluate_seed,
    make_hourly_native_env,
)

from .model import MaskedDoubleDQNPolicy


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_checkpoint(
    *,
    run_dir: Path,
    model_name: str = "best",
    seeds: tuple[int, ...],
    out_dir: Path | None = None,
) -> Path:
    if not seeds:
        raise ValueError("seeds must not be empty")
    config = _load_json(run_dir / "config.json")
    if config.get("paper_name") != "Hourly Masked Double DQN":
        raise ValueError(f"{run_dir} is not an hourly masked Double-DQN run")
    if model_name == "best":
        model_path = run_dir / "masked_double_dqn_best_validation.pt"
    elif model_name == "final":
        model_path = run_dir / "masked_double_dqn_final.pt"
    else:
        model_path = Path(model_name)
        if not model_path.is_absolute():
            model_path = run_dir / model_path

    output = out_dir or run_dir / f"evaluation_{model_name}"
    output.mkdir(parents=True, exist_ok=False)
    policy = MaskedDoubleDQNPolicy.load(model_path, device="cpu")
    env = make_hourly_native_env(
        scenario=str(config["scenario"]),
        episode_hours=int(config["episode_hours"]),
        forecast_context_hours=int(config["forecast_context_hours"]),
        weather_mode=str(config["weather_mode"]),
        warm_start=bool(config["warm_start"]),
        scenario_protocol=str(config["scenario_protocol"]),
        reward_scale=float(config["reward_scale"]),
    )
    windows_h = tuple(int(value) for value in config["future_summary_windows_h"])
    rows = [
        evaluate_seed(
            policy,
            env,
            seed=seed,
            future_summary_windows_h=windows_h,
        )
        for seed in seeds
    ]
    with (output / "evaluation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "paper_name": "Hourly Masked Double DQN",
        "model_path": str(model_path),
        "seeds": list(seeds),
        **{
            f"mean_{key}": mean(float(row[key]) for row in rows)
            for key in (
                "episode_vessel_fuel_eur",
                "episode_conditioning_eur",
                "episode_reconditioning_eur",
                "episode_loading_eur",
                "episode_unloading_eur",
                "episode_operating_cost_eur",
                "episode_vent_penalty_eur",
                "episode_storage_shortfall_penalty_eur",
            )
        },
        "mean_total_cost_eur": mean(
            float(row["total_cost_eur"]) for row in rows
        ),
        "mean_vented_t": mean(float(row["vented_t"]) for row in rows),
        "mean_stored_t": mean(float(row["stored_t"]) for row in rows),
        "per_seed": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="best")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = evaluate_checkpoint(
        run_dir=args.run_dir,
        model_name=args.model,
        seeds=tuple(args.seeds),
        out_dir=args.out_dir,
    )
    print(f"Saved hourly masked Double-DQN evaluation under: {output}")


if __name__ == "__main__":
    main()
