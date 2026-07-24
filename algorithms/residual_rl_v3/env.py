"""Risk-gated extension of masked residual dispatch v2.

掩码残差调度 v2 的风险门控扩展。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from Simulation.environment import CCSEnv

from algorithms.residual_rl_v2.env import (
    MaskedResidualDispatchEnv,
    MaskedResidualEnvConfig,
)

from .risk_gate import AdaptiveRiskGateConfig, adaptive_risk_snapshot


@dataclass(frozen=True)
class RiskGatedResidualEnvConfig:
    """Combine v2 semi-MDP settings with an adaptive risk gate.

    组合 v2 半 MDP 设置和 adaptive 风险门控。
    """

    residual: MaskedResidualEnvConfig = MaskedResidualEnvConfig()
    adaptive_gate: AdaptiveRiskGateConfig = AdaptiveRiskGateConfig()
    gate_mode: str = "hard"
    outside_risk_intervention_penalty: float = 0.0

    def __post_init__(self) -> None:
        """Validate gate enforcement and reward shaping.

        校验门控执行方式和奖励塑形参数。
        """
        if self.gate_mode not in {"off", "soft", "hard"}:
            raise ValueError("gate_mode must be 'off', 'soft', or 'hard'.")
        if self.outside_risk_intervention_penalty < 0.0:
            raise ValueError(
                "outside_risk_intervention_penalty must be non-negative."
            )


class RiskGatedResidualDispatchEnv(MaskedResidualDispatchEnv):
    """Mask adaptive greedy outside observable risk states.

    在不存在可观测风险时掩码 adaptive greedy。
    """

    def __init__(
        self,
        env: CCSEnv,
        *,
        config: RiskGatedResidualEnvConfig | None = None,
    ) -> None:
        """Wrap the physical environment without changing v2 dynamics.

        包装物理环境，但不改变 v2 的动力学。
        """
        self.v3_config = config or RiskGatedResidualEnvConfig()
        self.last_gate_snapshot: dict[str, Any] = {}
        super().__init__(env, config=self.v3_config.residual)

    def action_masks(self) -> np.ndarray:
        """Apply physical v2 masks followed by the adaptive risk gate.

        先应用 v2 物理掩码，再应用 adaptive 风险门控。
        """
        mask = super().action_masks()
        snapshot = adaptive_risk_snapshot(
            self.env,
            self.v3_config.adaptive_gate,
        )
        gate_allowed = bool(
            snapshot["adaptive_risk_gate_allowed"]
        )
        if self.v3_config.gate_mode == "off":
            pass
        elif (
            self.v3_config.gate_mode == "hard"
            and self.v3_config.adaptive_gate
            .mask_all_interventions_outside_risk
            and not gate_allowed
        ):
            mask[1:] = False
        elif self.v3_config.gate_mode == "hard":
            mask[1] = bool(mask[1] and gate_allowed)
        self.last_gate_snapshot = snapshot
        return mask

    def step(self, action_index: int):
        """Execute a masked decision and preserve gate diagnostics.

        执行带掩码的决策，并保留门控诊断信息。
        """
        self.action_masks()
        snapshot = dict(self.last_gate_snapshot)
        result = super().step(action_index)
        observation, reward, terminated, truncated, info = result
        outside_risk = not bool(
            snapshot["adaptive_risk_gate_allowed"]
        )
        penalty = 0.0
        if (
            self.v3_config.gate_mode == "soft"
            and int(action_index) != 0
            and outside_risk
        ):
            penalty = float(
                self.v3_config.outside_risk_intervention_penalty
            )
            reward -= penalty
        info.update(snapshot)
        info["adaptive_action_unmasked"] = bool(
            info["action_mask"][1]
        )
        info["risk_gate_mode"] = self.v3_config.gate_mode
        info["outside_risk_intervention"] = bool(
            int(action_index) != 0 and outside_risk
        )
        info["outside_risk_intervention_penalty"] = penalty
        info["incremental_reward_before_risk_penalty"] = float(
            info["incremental_reward"]
        )
        info["incremental_reward"] = float(reward)
        return observation, reward, terminated, truncated, info
