"""Zero-shot cross-scenario evaluation for Iterative Q and residual PPO v4."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from experiments import evaluate_iterative_action_q as q_evaluation
from sim.control.baselines import greedy_shuttle_policy
from sim.control.event_based.residual_rl_v2.env import MaskedResidualEnvConfig
from sim.control.event_based.residual_rl_v2.evaluation import evaluate_seeds
from sim.control.event_based.residual_rl_v3.env import (
    RiskGatedResidualDispatchEnv,
    RiskGatedResidualEnvConfig,
)
from sim.control.event_based.residual_rl_v3.risk_gate import (
    AdaptiveRiskGateConfig,
)
from sim.control.event_based.residual_rl_v4.factory import (
    make_tail_robust_native_env,
)
from sim.control.event_based.rl.reward import HighLevelRewardConfig
from sim.environment.event_residual_gym import EventJointResidualGymEnv
from sim.network_scenarios import NORTHERN_LIGHTS_PHASE1_CAPTURE_PROFILE_PATH


TASK_SEEDS = {
    "q_on_v4_normal": tuple(range(6_000_001, 6_000_021)),
    "q_on_v4_hard": tuple(range(7_000_001, 7_000_021)),
    "v4_on_q_original": tuple(range(7_000, 7_030)),
}

Q_GATE = {
    "name": "unified_p4",
    "required_heads": 4,
    "margin": 0.4,
    "max_overrides": 8,
    "min_hour": None,
    "max_hour": None,
    "windows": [
        [108.0, 179.0],
        [180.0, 251.0],
        [252.0, 323.0],
        [324.0, 395.0],
        [396.0, 467.0],
        [468.0, 539.0],
        [540.0, 611.0],
        [612.0, 680.0],
    ],
    "uncertainty_beta": 0.0,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterative-q-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASK_SEEDS),
        default=list(TASK_SEEDS),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _q_original_physical(variant: str):
    args = SimpleNamespace(
        episode_hours=720,
        reward_scale=1e-5,
        device="cpu",
    )
    return q_evaluation._make_native(args, variant)


def _v4_physical(config: dict, hard_probability: float):
    return make_tail_robust_native_env(
        scenario=str(config["scenario"]),
        episode_hours=int(config["episode_hours"]),
        forecast_context_hours=int(config["forecast_context_hours"]),
        decision_interval_h=float(config["decision_interval_h"]),
        event_triggered=bool(config["event_triggered"]),
        weather_mode=str(config["weather_mode"]),
        hard_scenario_probability=float(hard_probability),
        reward=HighLevelRewardConfig(**config["high_level_reward"]),
        gate=AdaptiveRiskGateConfig(**config["risk_gate"]),
        gate_mode=str(config["risk_gate_mode"]),
        outside_risk_intervention_penalty=float(
            config["outside_risk_intervention_penalty"]
        ),
    ).env


def _v4_on_q_original_env(config: dict, variant: str):
    physical = _q_original_physical(variant)
    return RiskGatedResidualDispatchEnv(
        physical,
        config=RiskGatedResidualEnvConfig(
            residual=MaskedResidualEnvConfig(
                decision_interval_h=float(config["decision_interval_h"]),
                event_triggered=bool(config["event_triggered"]),
                reward=HighLevelRewardConfig(**config["high_level_reward"]),
            ),
            adaptive_gate=AdaptiveRiskGateConfig(**config["risk_gate"]),
            gate_mode=str(config["risk_gate_mode"]),
            outside_risk_intervention_penalty=float(
                config["outside_risk_intervention_penalty"]
            ),
        ),
    )


def _physical_metrics(env):
    stored_t = float(env.ledger.stored_t)
    total_cost = float(env.ledger.total_cost)
    return {
        "total_cost_eur": total_cost,
        "operating_cost_eur": float(env.ledger.operating_cost),
        "vent_penalty_eur": float(env.ledger.vent_penalty),
        "vented_t": float(env.ledger.vented_t),
        "stored_t": stored_t,
        "unit_cost_eur_per_t": (
            total_cost / stored_t if stored_t > 1e-9 else float("nan")
        ),
    }


def _greedy_records(physical_factory, seeds):
    records = {}
    for seed in seeds:
        env = physical_factory()
        env.reset(seed=int(seed))
        while env.t < env.n_steps:
            env.step(greedy_shuttle_policy(env))
        records[int(seed)] = _physical_metrics(env)
    return records


def _q_rows(args, model, metadata, config, *, hard_probability, seeds):
    variant = str(metadata["observation_variant"])
    physical_factory = lambda: _v4_physical(config, hard_probability)
    baselines = _greedy_records(physical_factory, seeds)
    q_args = SimpleNamespace(
        eval_seeds=list(seeds),
        episode_hours=int(config["episode_hours"]),
        reward_scale=1e-5,
        device=args.device,
    )

    def event_factory():
        return EventJointResidualGymEnv(
            physical_factory(),
            variant,
            include_episode_progress=True,
            greedy_control_variate=True,
            hourly_gamma=1.0,
        )

    return q_evaluation.evaluate_gate(
        q_args,
        model,
        metadata,
        Q_GATE,
        baselines,
        torch.device(args.device),
        event_env_factory=event_factory,
    )


def _v4_rows(model, config, variant, seeds):
    physical_factory = lambda: _q_original_physical(variant)
    baselines = _greedy_records(physical_factory, seeds)
    env = _v4_on_q_original_env(config, variant)
    records = evaluate_seeds(model, env, seeds)
    rows = []
    for record in records:
        baseline = baselines[int(record["seed"])]
        row = {
            "seed": int(record["seed"]),
            "decisions": int(record["decisions"]),
            "selected_interventions": int(record["selected_interventions"]),
            "changed_decisions": int(record["changed_decisions"]),
            "total_cost_eur": float(record["total_cost_eur"]),
            "operating_cost_eur": float(record["operating_cost_eur"]),
            "vent_penalty_eur": float(
                record["total_cost_eur"] - record["operating_cost_eur"]
            ),
            "vented_t": float(record["vented_t"]),
            "stored_t": float(record["stored_t"]),
            "unit_cost_eur_per_t": float(
                record["unit_total_cost_eur_per_t"]
            ),
        }
        for key, value in baseline.items():
            row[f"greedy_{key}"] = value
            row[f"delta_{key}"] = row[key] - value
        rows.append(row)
    return rows


def _summary(rows):
    deltas = np.asarray(
        [row["delta_total_cost_eur"] for row in rows], dtype=np.float64
    )
    rng = np.random.default_rng(0)
    boot = deltas[
        rng.integers(0, len(deltas), size=(10_000, len(deltas)))
    ].mean(axis=1)
    return {
        "episodes": len(rows),
        "mean_total_cost_eur": float(
            np.mean([row["total_cost_eur"] for row in rows])
        ),
        "mean_greedy_total_cost_eur": float(
            np.mean([row["greedy_total_cost_eur"] for row in rows])
        ),
        "mean_delta_total_cost_eur": float(deltas.mean()),
        "mean_delta_95pct_ci_eur": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "wins": int((deltas < -1e-6).sum()),
        "ties": int((np.abs(deltas) <= 1e-6).sum()),
        "losses": int((deltas > 1e-6).sum()),
        "mean_vented_t": float(np.mean([row["vented_t"] for row in rows])),
        "mean_stored_t": float(np.mean([row["stored_t"] for row in rows])),
        "mean_unit_cost_eur_per_t": float(
            np.mean([row["unit_cost_eur_per_t"] for row in rows])
        ),
    }


def _write_task(out_dir: Path, task: str, rows, metadata):
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=False)
    with (task_dir / "evaluation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {**metadata, "summary": _summary(rows)}
    (task_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({task: payload["summary"]}, indent=2), flush=True)
    return payload


def run(args):
    if not NORTHERN_LIGHTS_PHASE1_CAPTURE_PROFILE_PATH.is_file():
        raise FileNotFoundError(
            "Real hourly capture-rate CSV is required: "
            f"{NORTHERN_LIGHTS_PHASE1_CAPTURE_PROFILE_PATH}"
        )
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output directory: {args.out_dir}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(
        (args.v4_run_dir / "config.json").read_text(encoding="utf-8")
    )
    if config.get("algorithm") != "maskable_residual_ppo_v4":
        raise ValueError("v4 run directory has an incompatible algorithm")

    q_model, q_metadata = q_evaluation._load_model(
        SimpleNamespace(checkpoint=args.iterative_q_checkpoint),
        torch.device(args.device),
    )
    v4_model = MaskablePPO.load(
        args.v4_run_dir / "maskable_residual_v4_best_validation",
        device=args.device,
    )
    variant = str(q_metadata["observation_variant"])
    all_summaries = {}
    for task in args.tasks:
        seeds = TASK_SEEDS[task]
        if task == "q_on_v4_normal":
            rows = _q_rows(
                args,
                q_model,
                q_metadata,
                config,
                hard_probability=0.0,
                seeds=seeds,
            )
            controller = "iterative_action_q_p4"
            scenario = "v4_normal"
        elif task == "q_on_v4_hard":
            rows = _q_rows(
                args,
                q_model,
                q_metadata,
                config,
                hard_probability=1.0,
                seeds=seeds,
            )
            controller = "iterative_action_q_p4"
            scenario = "v4_hard"
        else:
            rows = _v4_rows(v4_model, config, variant, seeds)
            controller = "residual_ppo_v4_best_validation"
            scenario = "iterative_q_original"
        all_summaries[task] = _write_task(
            args.out_dir,
            task,
            rows,
            {
                "task": task,
                "controller": controller,
                "scenario": scenario,
                "seeds": list(seeds),
                "hourly_capture_csv": str(
                    NORTHERN_LIGHTS_PHASE1_CAPTURE_PROFILE_PATH
                ),
                "training_performed": False,
            },
        )
    return all_summaries


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
