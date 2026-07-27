"""Evaluate the hourly greedy-shuttle baseline on v4 test scenarios.

在 v4 测试场景上评估逐小时 greedy-shuttle 基线。
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any

from sim.control.baselines import greedy_shuttle_policy

from sim.control.event_based.residual_rl_v3.risk_gate import (
    AdaptiveRiskGateConfig,
)
from sim.control.event_based.rl.reward import (
    HARD_VIOLATION_CODES,
    HighLevelRewardConfig,
)

from .factory import make_tail_robust_native_env


def evaluate_greedy(
    reference_run_dir: Path,
    *,
    seeds: tuple[int, ...],
    hard_scenario_probability: float,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate greedy on the exact physical configuration used by v4.

    在 v4 使用的完全相同物理配置上评估 greedy。
    """
    if not seeds:
        raise ValueError("At least one seed is required.")
    config = json.loads(
        (reference_run_dir / "config.json").read_text(encoding="utf-8")
    )
    if config.get("algorithm") != "maskable_residual_ppo_v4":
        raise ValueError("reference_run_dir must contain a v4 run.")
    env = make_tail_robust_native_env(
        scenario=str(config["scenario"]),
        episode_hours=int(config["episode_hours"]),
        forecast_context_hours=int(
            config["forecast_context_hours"]
        ),
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
        scenario_protocol=str(
            config.get("scenario_protocol", "v4_mixed_window")
        ),
        hard_scenario_probability=hard_scenario_probability,
        reward=HighLevelRewardConfig(
            **config["high_level_reward"]
        ),
        gate=AdaptiveRiskGateConfig(**config["risk_gate"]),
        gate_mode=str(config["risk_gate_mode"]),
        outside_risk_intervention_penalty=float(
            config["outside_risk_intervention_penalty"]
        ),
        override_windows_h=tuple(
            tuple(float(value) for value in window)
            for window in config.get("override_windows_h", ())
        ),
    ).env
    records = [
        _evaluate_seed(env, seed)
        for seed in seeds
    ]
    summary = _summarize(
        records,
        cvar_tail_fraction=float(config["cvar_tail_fraction"]),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "controller": "greedy_shuttle",
                "reference_run_dir": str(reference_run_dir),
                "hard_scenario_probability": (
                    hard_scenario_probability
                ),
                "seeds": list(seeds),
                "summary": summary,
                "per_seed": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(records[0]),
        )
        writer.writeheader()
        writer.writerows(records)
    return records, summary


def _evaluate_seed(env, seed: int) -> dict[str, Any]:
    """Run one deterministic hourly greedy episode.

    运行一个确定性的逐小时 greedy episode。
    """
    started_at = perf_counter()
    env.reset(seed=int(seed))
    violations: Counter[str] = Counter()
    done = False
    steps = 0
    while not done:
        action = greedy_shuttle_policy(env)
        _observation, _reward, terminated, truncated, info = env.step(
            action
        )
        violations.update(info.get("violations", ()))
        steps += 1
        done = bool(terminated or truncated)

    captured_t = float(env.cumulative_captured_t)
    stored_t = float(env.cumulative_stored_t)
    total_cost = float(env.ledger.total_cost)
    hard_violations = sum(
        int(count)
        for code, count in violations.items()
        if code in HARD_VIOLATION_CODES
    )
    return {
        "seed": int(seed),
        "physical_steps": steps,
        "captured_t": captured_t,
        "stored_t": stored_t,
        "vented_t": float(env.ledger.vented_t),
        "storage_rate": (
            stored_t / captured_t if captured_t > 1e-9 else 0.0
        ),
        "operating_cost_eur": float(env.ledger.operating_cost),
        "total_cost_eur": total_cost,
        "unit_total_cost_eur_per_t": (
            total_cost / stored_t if stored_t > 1e-9 else float("nan")
        ),
        "hard_violations": hard_violations,
        "wall_clock_seconds": perf_counter() - started_at,
    }


def _summarize(
    records: list[dict[str, Any]],
    *,
    cvar_tail_fraction: float,
) -> dict[str, float]:
    """Aggregate greedy metrics and venting tail risk.

    汇总 greedy 指标和放空尾部风险。
    """
    vents = sorted(
        (float(record["vented_t"]) for record in records),
        reverse=True,
    )
    tail_count = max(1, ceil(len(vents) * cvar_tail_fraction))
    keys = (
        "captured_t",
        "stored_t",
        "vented_t",
        "storage_rate",
        "operating_cost_eur",
        "total_cost_eur",
        "unit_total_cost_eur_per_t",
        "hard_violations",
        "wall_clock_seconds",
    )
    summary = {
        f"mean_{key}": sum(
            float(record[key]) for record in records
        )
        / len(records)
        for key in keys
    }
    summary.update(
        {
            "cvar_vented_t": sum(vents[:tail_count]) / tail_count,
            "worst_vented_t": max(vents),
            "hard_violations": sum(
                int(record["hard_violations"])
                for record in records
            ),
        }
    )
    return summary


def main() -> None:
    """Evaluate greedy from a terminal.

    从终端评估 greedy。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-run-dir",
        type=Path,
        required=True,
    )
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records, summary = evaluate_greedy(
        args.reference_run_dir,
        seeds=tuple(args.seeds),
        hard_scenario_probability=args.hard_scenario_probability,
        output_dir=args.output_dir,
    )
    for row in records:
        print(
            "Greedy shuttle | "
            f"seed={row['seed']} | "
            f"stored={row['stored_t']:,.1f} t | "
            f"vented={row['vented_t']:,.1f} t | "
            f"cost=EUR {row['total_cost_eur']:,.0f}",
            flush=True,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
