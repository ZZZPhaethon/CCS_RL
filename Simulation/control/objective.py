"""Construct objective weights and evaluate control-policy costs and rewards.

构建目标函数权重，并评估控制策略的成本与收益。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..economics import EconomicParameters


@dataclass(frozen=True)
class ControlObjectiveWeights:
    mode: str
    vent_eur_per_t: float
    operating_cost_weight: float
    storage_reward_eur_per_t: float
    overflow_risk_eur_per_t: float
    overflow_risk_lookahead_h: float


def control_objective_value(
    weights: ControlObjectiveWeights,
    *,
    operating_cost: float,
    vented_t: float,
    stored_t: float,
    overflow_risk_t: float = 0.0,
) -> float:
    return (
        weights.operating_cost_weight * float(operating_cost)
        + weights.vent_eur_per_t * float(vented_t)
        + weights.overflow_risk_eur_per_t * float(overflow_risk_t)
        - weights.storage_reward_eur_per_t * float(stored_t)
    )


def control_objective_weights(
    env,
    economics: EconomicParameters,
    *,
    storage_reward_eur_per_t: float | None = None,
) -> ControlObjectiveWeights:
    """Translate the RL reward configuration into minimization weights."""

    config = env.config
    mode = str(getattr(config, "reward_mode", "economic"))
    operating_cost_weight = float(getattr(config, "operating_cost_weight", 1.0))
    if mode == "vent_first":
        return ControlObjectiveWeights(
            mode=mode,
            vent_eur_per_t=float(getattr(config, "vent_first_vent_eur_per_t", 10_000.0)),
            operating_cost_weight=operating_cost_weight,
            storage_reward_eur_per_t=0.0,
            overflow_risk_eur_per_t=float(getattr(config, "overflow_risk_eur_per_t", 100.0)),
            overflow_risk_lookahead_h=float(getattr(config, "overflow_risk_lookahead_h", 24.0)),
        )
    if mode != "economic":
        raise ValueError(f"Unknown reward_mode: {mode}")

    store_reward = storage_reward_eur_per_t
    if store_reward is None:
        store_reward = getattr(config, "store_reward_eur_per_t", None)
    if store_reward is None:
        store_reward = getattr(config, "injection_reward_eur_per_t", 0.0)
    return ControlObjectiveWeights(
        mode=mode,
        vent_eur_per_t=(
            float(getattr(config, "vent_penalty_weight", 1.0))
            * economics.carbon_price_eur_per_t
        ),
        operating_cost_weight=operating_cost_weight,
        storage_reward_eur_per_t=float(store_reward),
        overflow_risk_eur_per_t=0.0,
        overflow_risk_lookahead_h=0.0,
    )
