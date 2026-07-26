"""Shared deterministic evaluation utilities for residual PPO.

残差 PPO 的共享确定性评估工具。
"""

from __future__ import annotations

from collections import Counter
from math import ceil
from time import perf_counter
from typing import Any, Iterable

from sim.control.event_based.rl.reward import HARD_VIOLATION_CODES

from .env import ResidualDispatchEnv


def evaluate_seed(model, env: ResidualDispatchEnv, seed: int) -> dict[str, Any]:
    """Evaluate one seed and return physical, action, and event metrics.

    评估一个 seed，并返回物理、动作与事件指标。
    """
    started_at = perf_counter()
    observation = env.reset(seed=int(seed))
    total_reward = 0.0
    decisions = 0
    elapsed_hours = 0.0
    interventions_applied = 0
    actions: Counter[str] = Counter()
    triggers: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    done = False
    while not done:
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        decisions += 1
        elapsed_hours += float(info["elapsed_hours"])
        interventions_applied += int(bool(info["intervention_applied"]))
        actions[str(info["action_label"])] += 1
        triggers[str(info["decision_trigger"])] += 1
        difficulties[str(info["scenario_difficulty"])] += 1
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
        "mean_decision_interval_h": elapsed_hours / max(1, decisions),
        "interventions_applied": interventions_applied,
        "intervention_rate": interventions_applied / max(1, decisions),
        "episode_reward": total_reward,
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
        "triggers": dict(triggers),
        "scenario_difficulty": next(iter(difficulties), "unknown"),
    }


def evaluate_seeds(
    model,
    env: ResidualDispatchEnv,
    seeds: Iterable[int],
) -> list[dict[str, Any]]:
    """Evaluate a deterministic policy over fixed seeds.

    在固定 seed 集合上评估确定性策略。
    """
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("At least one evaluation seed is required.")
    return [evaluate_seed(model, env, seed) for seed in seed_values]


def validation_metrics(
    records: list[dict[str, Any]],
    *,
    cvar_tail_fraction: float = 0.25,
    tail_vent_penalty_eur_per_t: float = 500.0,
    hard_violation_penalty_eur: float = 1_000_000.0,
) -> dict[str, float]:
    """Compute mean, worst-seed, CVaR, and model-selection loss.

    计算均值、最差 seed、CVaR 与模型选择损失。

    Lower selection loss is better. CVaR is the mean vented mass in the worst
    ``cvar_tail_fraction`` of validation seeds.

    模型选择损失越低越好。CVaR 是验证集中最差
    ``cvar_tail_fraction`` 比例场景的平均放空量。
    """
    if not records:
        raise ValueError("validation_metrics requires at least one record.")
    if not 0.0 < cvar_tail_fraction <= 1.0:
        raise ValueError("cvar_tail_fraction must be inside (0, 1].")
    costs = [float(record["total_cost_eur"]) for record in records]
    vents = sorted(
        (float(record["vented_t"]) for record in records),
        reverse=True,
    )
    stored = [float(record["stored_t"]) for record in records]
    hard = sum(int(record["hard_violations"]) for record in records)
    tail_count = max(1, ceil(len(vents) * cvar_tail_fraction))
    cvar_vent = sum(vents[:tail_count]) / tail_count
    mean_cost = sum(costs) / len(costs)
    selection_loss = (
        mean_cost
        + tail_vent_penalty_eur_per_t * cvar_vent
        + hard_violation_penalty_eur * hard
    )
    return {
        "mean_total_cost_eur": mean_cost,
        "mean_stored_t": sum(stored) / len(stored),
        "mean_vented_t": sum(vents) / len(vents),
        "worst_vented_t": max(vents),
        "cvar_vented_t": cvar_vent,
        "hard_violations": float(hard),
        "selection_loss": selection_loss,
    }

