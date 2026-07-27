"""Compare trained Q/v4 policies with Greedy and native MPC."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from experiments import evaluate_iterative_action_q as q_evaluation
from sim.control.baselines import greedy_shuttle_policy
from sim.control.event_based.residual_rl_v2.evaluation import evaluate_seeds
from sim.control.event_based.residual_rl_v3.risk_gate import (
    AdaptiveRiskGateConfig,
)
from sim.control.event_based.residual_rl_v4.factory import (
    make_tail_robust_native_env,
)
from sim.control.event_based.rl.reward import HighLevelRewardConfig
from sim.control.native_mpc import RollingNativeMpcController
from sim.environment.event_residual_gym import EventJointResidualGymEnv


WINDOWS_H = (
    (108.0, 155.0),
    (156.0, 203.0),
    (204.0, 251.0),
    (252.0, 299.0),
    (300.0, 347.0),
    (348.0, 395.0),
    (396.0, 443.0),
    (444.0, 491.0),
    (492.0, 539.0),
    (540.0, 587.0),
    (588.0, 635.0),
    (636.0, 680.0),
)

Q_GATE = {
    "name": "strict4_margin40k_12windows",
    "required_heads": 4,
    "margin": 0.40,
    "max_overrides": 12,
    "min_hour": None,
    "max_hour": None,
    "windows": [list(window) for window in WINDOWS_H],
    "uncertainty_beta": 0.0,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterative-q-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(8_000_001, 8_000_031)),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _v4_config(run_dir: Path) -> dict:
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    if config.get("scenario_protocol") != "unified_window_v1":
        raise ValueError("v4 run does not use unified_window_v1")
    configured = tuple(
        tuple(float(value) for value in window)
        for window in config.get("override_windows_h", ())
    )
    if configured != WINDOWS_H:
        raise ValueError("v4 run does not use the locked 12-window protocol")
    return config


def _residual_env(config: dict):
    return make_tail_robust_native_env(
        scenario=str(config["scenario"]),
        episode_hours=int(config["episode_hours"]),
        forecast_context_hours=int(config["forecast_context_hours"]),
        future_summary_windows_h=tuple(
            int(value)
            for value in config.get(
                "future_summary_windows_h",
                (24, 72),
            )
        ),
        decision_interval_h=float(config["decision_interval_h"]),
        event_triggered=bool(config["event_triggered"]),
        weather_mode=str(config["weather_mode"]),
        scenario_protocol=str(config["scenario_protocol"]),
        hard_scenario_probability=0.0,
        reward=HighLevelRewardConfig(**config["high_level_reward"]),
        gate=AdaptiveRiskGateConfig(**config["risk_gate"]),
        gate_mode=str(config["risk_gate_mode"]),
        outside_risk_intervention_penalty=float(
            config["outside_risk_intervention_penalty"]
        ),
        override_windows_h=WINDOWS_H,
    )


def _physical_env(config: dict):
    return _residual_env(config).env


def _metrics(env) -> dict[str, float]:
    stored = float(env.ledger.stored_t)
    total = float(env.ledger.total_cost)
    return {
        "total_cost_eur": total,
        "operating_cost_eur": float(env.ledger.operating_cost),
        "vent_penalty_eur": float(env.ledger.vent_penalty),
        "vented_t": float(env.ledger.vented_t),
        "stored_t": stored,
        "unit_cost_eur_per_t": (
            total / stored if stored > 1e-9 else float("nan")
        ),
    }


def _run_hourly(config: dict, seed: int, controller: str) -> dict:
    env = _physical_env(config)
    env.reset(seed=int(seed))
    started = time.perf_counter()
    if controller == "greedy":
        policy = greedy_shuttle_policy
    elif controller == "native_mpc":
        policy = RollingNativeMpcController(
            env,
            replan_every=24,
            planning_horizon_h=168,
            objective_mode="economic",
        )
    else:
        raise ValueError(controller)
    while env.t < env.n_steps:
        env.step(policy(env))
    return {
        "controller": controller,
        "seed": int(seed),
        "override_events": 0,
        "wall_clock_seconds": time.perf_counter() - started,
        **_metrics(env),
    }


def _q_rows(args, config, model, metadata, greedy_by_seed):
    variant = str(metadata["observation_variant"])
    q_args = SimpleNamespace(
        eval_seeds=list(args.seeds),
        episode_hours=int(config["episode_hours"]),
        reward_scale=1e-5,
        device=args.device,
    )

    def event_factory():
        return EventJointResidualGymEnv(
            _physical_env(config),
            variant,
            include_episode_progress=True,
            greedy_control_variate=True,
            hourly_gamma=1.0,
        )

    rows = q_evaluation.evaluate_gate(
        q_args,
        model,
        metadata,
        Q_GATE,
        greedy_by_seed,
        torch.device(args.device),
        event_env_factory=event_factory,
    )
    return [
        {
            "controller": "iterative_q",
            "seed": int(row["seed"]),
            "override_events": int(row["override_events"]),
            "wall_clock_seconds": 0.0,
            **{
                key: float(row[key])
                for key in (
                    "total_cost_eur",
                    "operating_cost_eur",
                    "vent_penalty_eur",
                    "vented_t",
                    "stored_t",
                    "unit_cost_eur_per_t",
                )
            },
        }
        for row in rows
    ]


def _v4_rows(config, model, seeds):
    records = evaluate_seeds(model, _residual_env(config), seeds)
    return [
        {
            "controller": "residual_ppo_v4",
            "seed": int(row["seed"]),
            "override_events": int(row["selected_interventions"]),
            "wall_clock_seconds": float(row["wall_clock_seconds"]),
            "total_cost_eur": float(row["total_cost_eur"]),
            "operating_cost_eur": float(row["operating_cost_eur"]),
            "vent_penalty_eur": float(
                row["total_cost_eur"] - row["operating_cost_eur"]
            ),
            "vented_t": float(row["vented_t"]),
            "stored_t": float(row["stored_t"]),
            "unit_cost_eur_per_t": float(
                row["unit_total_cost_eur_per_t"]
            ),
        }
        for row in records
    ]


def _summary(rows):
    output = {}
    for controller in sorted({str(row["controller"]) for row in rows}):
        selected = [
            row for row in rows if row["controller"] == controller
        ]
        output[controller] = {
            "episodes": len(selected),
            **{
                f"mean_{key}": float(
                    np.mean([float(row[key]) for row in selected])
                )
                for key in (
                    "total_cost_eur",
                    "operating_cost_eur",
                    "vent_penalty_eur",
                    "vented_t",
                    "stored_t",
                    "unit_cost_eur_per_t",
                    "override_events",
                )
            },
        }
    return output


def run(args):
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = _v4_config(args.v4_run_dir)
    q_model, q_metadata = q_evaluation._load_model(
        SimpleNamespace(checkpoint=args.iterative_q_checkpoint),
        torch.device(args.device),
    )
    if q_metadata.get("scenario_protocol") != "unified_window_v1":
        raise ValueError("Iterative Q checkpoint does not use unified_window_v1")
    v4_model = MaskablePPO.load(
        args.v4_run_dir / "maskable_residual_v4_best_validation",
        device=args.device,
    )

    greedy_rows = [
        _run_hourly(config, seed, "greedy") for seed in args.seeds
    ]
    greedy_by_seed = {
        int(row["seed"]): {
            key: row[key]
            for key in (
                "total_cost_eur",
                "operating_cost_eur",
                "vent_penalty_eur",
                "vented_t",
                "stored_t",
                "unit_cost_eur_per_t",
            )
        }
        for row in greedy_rows
    }
    rows = list(greedy_rows)
    rows.extend(
        _q_rows(args, config, q_model, q_metadata, greedy_by_seed)
    )
    rows.extend(_v4_rows(config, v4_model, tuple(args.seeds)))
    rows.extend(
        _run_hourly(config, seed, "native_mpc")
        for seed in args.seeds
    )

    with (args.out_dir / "per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "scenario_protocol": "unified_window_v1",
        "override_windows_h": [list(window) for window in WINDOWS_H],
        "seeds": [int(seed) for seed in args.seeds],
        "summary": _summary(rows),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
