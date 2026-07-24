"""Evaluate the pure-RL v3 risk ensemble on fixed scenarios.

在固定场景上评估纯 RL v3 风险 ensemble。
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from algorithms.rl.reward import (
    HARD_VIOLATION_CODES,
    HighLevelRewardConfig,
)
from algorithms.residual_rl_v2.evaluation import validation_metrics

from .ensemble_executor import (
    EnsembleRiskConfig,
    V3RiskEnsemble,
    load_v3_ensemble,
)
from .factory import make_risk_gated_native_env
from .risk_gate import AdaptiveRiskGateConfig


def evaluate_ensemble_seed(
    ensemble: V3RiskEnsemble,
    env,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one scenario with transparent switching diagnostics.

    使用透明的切换诊断评估一个场景。
    """
    started_at = perf_counter()
    observation = env.reset(seed=int(seed))
    episode_reward = 0.0
    decisions = 0
    disagreement_decisions = 0
    high_risk_decisions = 0
    high_risk_disagreements = 0
    seed1_switches = 0
    rule_fallbacks = 0
    changed_decisions = 0
    changed_native_steps = 0
    avoided_vent_t = 0.0
    incremental_stored_t = 0.0
    total_cost_saving_eur = 0.0
    risk_score_sum = 0.0
    actions: Counter[str] = Counter()
    selected_policies: Counter[str] = Counter()
    policy_action_triplets: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    done = False
    while not done:
        action_mask = env.action_masks()
        decision = ensemble.select_action(
            env,
            observation,
            action_mask,
        )
        observation, reward, terminated, truncated, info = env.step(
            decision.action
        )
        decisions += 1
        episode_reward += float(reward)
        disagreement_decisions += int(decision.policy_disagreement)
        high_risk_decisions += int(decision.high_risk)
        high_risk_disagreements += int(
            decision.high_risk
            and decision.policy_disagreement
        )
        seed1_switches += int(
            decision.selected_policy == "risk_policy_seed1"
        )
        rule_fallbacks += int(decision.rule_fallback)
        changed_decisions += int(info["native_action_changed"])
        changed_native_steps += int(info["changed_native_steps"])
        avoided_vent_t += float(info["avoided_vent_t"])
        incremental_stored_t += float(info["incremental_stored_t"])
        total_cost_saving_eur += float(
            info["total_cost_saving_eur"]
        )
        risk_score_sum += float(decision.risk_score)
        actions[str(info["action_label"])] += 1
        selected_policies[decision.selected_policy] += 1
        policy_action_triplets[
            json.dumps(
                decision.policy_actions,
                sort_keys=True,
            )
        ] += 1
        violations.update(info["violation_counts"])
        done = bool(terminated or truncated)

    physical = env.env
    captured_t = float(physical.cumulative_captured_t)
    stored_t = float(physical.cumulative_stored_t)
    total_cost = float(physical.ledger.total_cost)
    hard_violations = sum(
        int(count)
        for code, count in violations.items()
        if code in HARD_VIOLATION_CODES
    )
    return {
        "seed": int(seed),
        "decisions": decisions,
        "episode_reward": episode_reward,
        "disagreement_decisions": disagreement_decisions,
        "disagreement_rate": (
            disagreement_decisions / max(1, decisions)
        ),
        "high_risk_decisions": high_risk_decisions,
        "high_risk_rate": high_risk_decisions / max(1, decisions),
        "high_risk_disagreements": high_risk_disagreements,
        "seed1_switches": seed1_switches,
        "seed1_switch_rate": seed1_switches / max(1, decisions),
        "rule_fallbacks": rule_fallbacks,
        "mean_risk_score": risk_score_sum / max(1, decisions),
        "changed_decisions": changed_decisions,
        "effective_intervention_rate": (
            changed_decisions / max(1, decisions)
        ),
        "changed_native_steps": changed_native_steps,
        "local_avoided_vent_t": avoided_vent_t,
        "local_incremental_stored_t": incremental_stored_t,
        "local_total_cost_saving_eur": total_cost_saving_eur,
        "captured_t": captured_t,
        "stored_t": stored_t,
        "vented_t": float(physical.ledger.vented_t),
        "storage_rate": stored_t / captured_t if captured_t > 1e-9 else 0.0,
        "operating_cost_eur": float(physical.ledger.operating_cost),
        "total_cost_eur": total_cost,
        "unit_total_cost_eur_per_t": (
            total_cost / stored_t if stored_t > 1e-9 else float("nan")
        ),
        "hard_violations": hard_violations,
        "wall_clock_seconds": perf_counter() - started_at,
        "actions": dict(actions),
        "selected_policies": dict(selected_policies),
        "policy_action_triplets": dict(policy_action_triplets),
    }


def evaluate_ensemble(
    *,
    seed0_run: Path,
    seed1_run: Path,
    seed2_run: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
    hard_scenario_probability: float,
    model_choice: str = "best",
    risk_config: EnsembleRiskConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate and save one ensemble configuration.

    评估并保存一个 ensemble 配置。
    """
    if output_dir.exists():
        raise FileExistsError(output_dir)
    ensemble, config = load_v3_ensemble(
        seed0_run,
        seed1_run,
        seed2_run,
        model_choice=model_choice,
        config=risk_config,
    )
    env = make_risk_gated_native_env(
        scenario=str(config["scenario"]),
        episode_hours=int(config["episode_hours"]),
        forecast_context_hours=int(
            config["forecast_context_hours"]
        ),
        decision_interval_h=float(config["decision_interval_h"]),
        event_triggered=bool(config["event_triggered"]),
        weather_mode=str(config["weather_mode"]),
        hard_scenario_probability=float(hard_scenario_probability),
        reward=HighLevelRewardConfig(**config["high_level_reward"]),
        gate=AdaptiveRiskGateConfig(**config["risk_gate"]),
        gate_mode=str(config["risk_gate_mode"]),
        outside_risk_intervention_penalty=float(
            config["outside_risk_intervention_penalty"]
        ),
    )
    records = [
        evaluate_ensemble_seed(ensemble, env, seed)
        for seed in seeds
    ]
    summary = validation_metrics(
        records,
        cvar_tail_fraction=float(config["cvar_tail_fraction"]),
        tail_vent_penalty_eur_per_t=float(
            config["tail_vent_penalty_eur_per_t"]
        ),
    )
    numeric_keys = (
        "disagreement_rate",
        "high_risk_rate",
        "seed1_switch_rate",
        "rule_fallbacks",
        "mean_risk_score",
        "effective_intervention_rate",
        "stored_t",
        "vented_t",
        "storage_rate",
        "total_cost_eur",
        "unit_total_cost_eur_per_t",
        "hard_violations",
        "wall_clock_seconds",
    )
    summary.update(
        {
            f"{key}_mean": sum(
                float(record[key]) for record in records
            )
            / len(records)
            for key in numeric_keys
        }
    )
    output_dir.mkdir(parents=True)
    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "seed0_run": str(seed0_run),
                "seed1_run": str(seed1_run),
                "seed2_run": str(seed2_run),
                "model_choice": model_choice,
                "hard_scenario_probability": (
                    hard_scenario_probability
                ),
                "seeds": list(seeds),
                "risk_config": asdict(
                    risk_config or EnsembleRiskConfig()
                ),
                "summary": summary,
                "per_seed": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    scalar_columns = [
        key
        for key, value in records[0].items()
        if not isinstance(value, dict)
    ]
    with (output_dir / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=scalar_columns,
        )
        writer.writeheader()
        writer.writerows(
            {
                key: value
                for key, value in record.items()
                if key in scalar_columns
            }
            for record in records
        )
    return records, summary


def main() -> None:
    """Evaluate an ensemble from a terminal.

    从终端评估 ensemble。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-run", type=Path, required=True)
    parser.add_argument("--seed1-run", type=Path, required=True)
    parser.add_argument("--seed2-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--hard-scenario-probability",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=("best", "final"),
        default="best",
    )
    parser.add_argument("--risk-hours-h", type=float, default=96.0)
    parser.add_argument("--risk-fill-ratio", type=float, default=0.80)
    parser.add_argument("--risk-speed-min", type=float, default=0.65)
    parser.add_argument("--high-risk-score", type=int, default=2)
    args = parser.parse_args()
    risk = EnsembleRiskConfig(
        hours_to_overflow_h=args.risk_hours_h,
        fill_ratio=args.risk_fill_ratio,
        forecast_speed_min=args.risk_speed_min,
        high_risk_score=args.high_risk_score,
    )
    records, summary = evaluate_ensemble(
        seed0_run=args.seed0_run,
        seed1_run=args.seed1_run,
        seed2_run=args.seed2_run,
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        hard_scenario_probability=args.hard_scenario_probability,
        model_choice=args.model,
        risk_config=risk,
    )
    for row in records:
        print(
            "V3 ensemble | "
            f"seed={row['seed']} | "
            f"stored={row['stored_t']:,.1f} t | "
            f"vented={row['vented_t']:,.1f} t | "
            f"cost=EUR {row['total_cost_eur']:,.0f} | "
            f"switch={100.0 * row['seed1_switch_rate']:.1f}%",
            flush=True,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
