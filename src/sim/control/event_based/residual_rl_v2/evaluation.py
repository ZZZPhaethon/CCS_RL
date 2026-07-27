"""Deterministic masked-policy evaluation and tail-risk metrics.

确定性掩码策略评估与尾部风险指标。
"""

from __future__ import annotations

from collections import Counter
from math import ceil
from time import perf_counter
from typing import Any, Iterable

from sim.control.event_based.rl.reward import HARD_VIOLATION_CODES

from .env import MaskedResidualDispatchEnv


def evaluate_seed(
    model,
    env: MaskedResidualDispatchEnv,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one seed using a valid action mask at every decision.

    在每个决策点使用合法动作掩码评估一个 seed。
    """
    started_at = perf_counter()
    observation = env.reset(seed=int(seed))
    episode_reward = 0.0
    decisions = 0
    selected_interventions = 0
    feasible_interventions = 0
    changed_decisions = 0
    changed_native_steps = 0
    avoided_vent_t = 0.0
    incremental_stored_t = 0.0
    total_cost_saving_eur = 0.0
    actions: Counter[str] = Counter()
    triggers: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    done = False
    elapsed_hours = 0.0
    while not done:
        mask = env.action_masks()
        action, _state = model.predict(
            observation,
            deterministic=True,
            action_masks=mask,
        )
        observation, reward, terminated, truncated, info = env.step(
            int(action)
        )
        episode_reward += float(reward)
        decisions += 1
        elapsed_hours += float(info["elapsed_hours"])
        selected_interventions += int(info["intervention_selected"])
        feasible_interventions += int(
            info["intervention_feasible_at_decision"]
        )
        changed_decisions += int(info["native_action_changed"])
        changed_native_steps += int(info["changed_native_steps"])
        avoided_vent_t += float(info["avoided_vent_t"])
        incremental_stored_t += float(info["incremental_stored_t"])
        total_cost_saving_eur += float(info["total_cost_saving_eur"])
        actions[str(info["action_label"])] += 1
        triggers[str(info["decision_trigger"])] += 1
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
        "selected_interventions": selected_interventions,
        "selected_intervention_rate": (
            selected_interventions / max(1, decisions)
        ),
        "feasible_intervention_decisions": feasible_interventions,
        "changed_decisions": changed_decisions,
        "effective_intervention_rate": (
            changed_decisions / max(1, decisions)
        ),
        "changed_native_steps": changed_native_steps,
        "local_avoided_vent_t": avoided_vent_t,
        "local_incremental_stored_t": incremental_stored_t,
        "local_total_cost_saving_eur": total_cost_saving_eur,
        "episode_reward": episode_reward,
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
    }


def evaluate_seeds(
    model,
    env: MaskedResidualDispatchEnv,
    seeds: Iterable[int],
) -> list[dict[str, Any]]:
    """Evaluate fixed seeds.

    评估固定 seed 集合。
    """
    values = tuple(int(seed) for seed in seeds)
    if not values:
        raise ValueError("At least one evaluation seed is required.")
    return [evaluate_seed(model, env, seed) for seed in values]


def validation_metrics(
    records: list[dict[str, Any]],
    *,
    cvar_tail_fraction: float = 0.25,
    tail_vent_penalty_eur_per_t: float = 500.0,
    hard_violation_penalty_eur: float = 1_000_000.0,
) -> dict[str, float]:
    """Compute physical model-selection metrics; lower loss is better.

    计算物理模型选择指标；损失越低越好。
    """
    if not records:
        raise ValueError("validation_metrics requires records.")
    if not 0.0 < cvar_tail_fraction <= 1.0:
        raise ValueError("cvar_tail_fraction must be inside (0, 1].")
    costs = [float(row["total_cost_eur"]) for row in records]
    vents = sorted(
        (float(row["vented_t"]) for row in records),
        reverse=True,
    )
    stored = [float(row["stored_t"]) for row in records]
    hard = sum(int(row["hard_violations"]) for row in records)
    tail_count = max(1, ceil(len(vents) * cvar_tail_fraction))
    cvar_vent = sum(vents[:tail_count]) / tail_count
    mean_cost = sum(costs) / len(costs)
    return {
        "mean_total_cost_eur": mean_cost,
        "mean_stored_t": sum(stored) / len(stored),
        "mean_vented_t": sum(vents) / len(vents),
        "worst_vented_t": max(vents),
        "cvar_vented_t": cvar_vent,
        "mean_effective_intervention_rate": sum(
            float(row["effective_intervention_rate"])
            for row in records
        )
        / len(records),
        "hard_violations": float(hard),
        "selection_loss": (
            mean_cost
            + tail_vent_penalty_eur_per_t * cvar_vent
            + hard_violation_penalty_eur * hard
        ),
    }

