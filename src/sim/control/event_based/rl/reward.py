"""Compute stable, realised rewards for high-level CCS RL.

计算稳定的、基于实际结果的高层 CCS RL 奖励。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


HARD_VIOLATION_CODES = frozenset(
    {
        "mass_balance_error",
        "inventory_capacity_exceeded",
        "reservoir_pressure_exceeded",
        "negative_inventory",
    }
)


@dataclass(frozen=True)
class HighLevelRewardConfig:
    """Economic and safety weights applied to one aggregated RL transition.

    应用于一个聚合 RL 转移的经济与安全权重。
    """

    stored_credit_eur_per_t: float = 80.0
    # Base vent cost matches the scenario carbon price.  Only venting above the
    # small tolerance receives the additional policy penalty, preventing the
    # learner from sacrificing large stored volumes to remove the final tonne.
    # 基础放空成本与场景碳价一致；只有超过小容许比例的放空才承担额外政策惩罚，
    # 避免策略为消除最后少量放空而牺牲大量封存。
    vent_penalty_eur_per_t: float = 80.0
    excess_vent_penalty_eur_per_t: float = 420.0
    vent_tolerance_fraction: float = 0.005
    operating_cost_weight: float = 1.0
    # Overflow risk is accumulated once per physical hour, so this weight is
    # EUR per tonne-hour rather than EUR per tonne.
    # 溢出风险每个物理小时累计一次，因此该权重单位为欧元/吨小时。
    overflow_risk_eur_per_t_hour: float = 10.0
    hard_violation_penalty_eur: float = 1_000_000.0
    reward_scale: float = 1e-6
    objective: str = "shaped_economic"

    @classmethod
    def objective_aligned(
        cls,
        *,
        reward_scale: float = 1e-6,
    ) -> "HighLevelRewardConfig":
        """Return a reward equal to scaled negative realised total cost."""

        return cls(
            stored_credit_eur_per_t=0.0,
            vent_penalty_eur_per_t=80.0,
            excess_vent_penalty_eur_per_t=0.0,
            vent_tolerance_fraction=0.0,
            operating_cost_weight=1.0,
            overflow_risk_eur_per_t_hour=0.0,
            hard_violation_penalty_eur=0.0,
            reward_scale=reward_scale,
            objective="realised_total_cost",
        )

    def __post_init__(self) -> None:
        """Reject negative weights and invalid vent-tolerance fractions.

        拒绝负权重以及无效的放空容许比例。
        """
        non_negative = {
            "stored_credit_eur_per_t": self.stored_credit_eur_per_t,
            "vent_penalty_eur_per_t": self.vent_penalty_eur_per_t,
            "excess_vent_penalty_eur_per_t": self.excess_vent_penalty_eur_per_t,
            "operating_cost_weight": self.operating_cost_weight,
            "overflow_risk_eur_per_t_hour": self.overflow_risk_eur_per_t_hour,
            "hard_violation_penalty_eur": self.hard_violation_penalty_eur,
            "reward_scale": self.reward_scale,
        }
        invalid = [name for name, value in non_negative.items() if value < 0.0]
        if invalid:
            raise ValueError(f"Reward weights must be non-negative: {invalid}.")
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive.")
        if not 0.0 <= self.vent_tolerance_fraction <= 1.0:
            raise ValueError("vent_tolerance_fraction must be inside [0, 1].")
        if self.objective not in {
            "shaped_economic",
            "realised_total_cost",
        }:
            raise ValueError(
                "objective must be 'shaped_economic' or "
                "'realised_total_cost'."
            )


def high_level_reward(
    *,
    stored_t: float,
    captured_t: float,
    vented_t: float,
    operating_cost: float,
    overflow_risk_t_hours: float,
    violation_counts: Mapping[str, int],
    config: HighLevelRewardConfig,
    total_cost_eur: float | None = None,
) -> tuple[float, dict[str, float | str]]:
    """Return a scaled reward and its unscaled component breakdown.

    返回缩放后奖励及其未缩放分解。

    Unit-cost ratios are intentionally excluded from the training reward because
    their denominator is unstable at the beginning of an episode. They remain
    evaluation metrics only.

    单位封存成本比值被有意排除在训练奖励之外，因为回合初期分母不稳定。它只作为评估
    指标使用。
    """
    if config.objective == "realised_total_cost":
        if total_cost_eur is None:
            raise ValueError(
                "realised_total_cost requires total_cost_eur."
            )
        realised_total_cost = max(0.0, float(total_cost_eur))
        return -config.reward_scale * realised_total_cost, {
            "objective": config.objective,
            "realised_total_cost_eur": realised_total_cost,
            "unscaled_reward_eur": -realised_total_cost,
        }

    hard_violations = sum(
        int(count)
        for code, count in violation_counts.items()
        if code in HARD_VIOLATION_CODES
    )
    stored_credit = config.stored_credit_eur_per_t * max(0.0, stored_t)
    vent_t = max(0.0, vented_t)
    vent_allowance_t = config.vent_tolerance_fraction * max(0.0, captured_t)
    excess_vent_t = max(0.0, vent_t - vent_allowance_t)
    base_vent_penalty = config.vent_penalty_eur_per_t * vent_t
    excess_vent_penalty = config.excess_vent_penalty_eur_per_t * excess_vent_t
    vent_penalty = base_vent_penalty + excess_vent_penalty
    operating_cost_term = config.operating_cost_weight * max(0.0, operating_cost)
    risk_penalty = config.overflow_risk_eur_per_t_hour * max(
        0.0,
        overflow_risk_t_hours,
    )
    violation_penalty = config.hard_violation_penalty_eur * hard_violations
    unscaled = (
        stored_credit
        - vent_penalty
        - operating_cost_term
        - risk_penalty
        - violation_penalty
    )
    return config.reward_scale * unscaled, {
        "objective": config.objective,
        "stored_credit_eur": stored_credit,
        "vent_allowance_t": vent_allowance_t,
        "excess_vent_t": excess_vent_t,
        "base_vent_penalty_eur": base_vent_penalty,
        "excess_vent_penalty_eur": excess_vent_penalty,
        "vent_penalty_eur": vent_penalty,
        "operating_cost_eur": operating_cost_term,
        "overflow_risk_t_hours": max(0.0, overflow_risk_t_hours),
        "overflow_risk_penalty_eur": risk_penalty,
        "hard_violation_penalty_eur": violation_penalty,
        "unscaled_reward_eur": unscaled,
    }
