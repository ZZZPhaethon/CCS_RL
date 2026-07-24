"""Risk-aware action selection across three residual PPO v3 policies.

在三个 residual PPO v3 策略之间进行风险感知动作选择。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO

from .env import RiskGatedResidualDispatchEnv
from .risk_gate import AdaptiveRiskGateConfig, adaptive_risk_snapshot


@dataclass(frozen=True)
class EnsembleRiskConfig:
    """Configure the switch from seed 0 to the high-risk seed 1 policy.

    配置从 seed 0 切换到高风险 seed 1 策略的条件。
    """

    hours_to_overflow_h: float = 96.0
    fill_ratio: float = 0.80
    forecast_speed_min: float = 0.65
    high_risk_score: int = 2
    overflow_risk_weight: int = 2
    hours_risk_weight: int = 1
    fill_risk_weight: int = 1
    weather_risk_weight: int = 1

    def __post_init__(self) -> None:
        """Validate switching thresholds and weights.

        校验切换阈值和权重。
        """
        if self.hours_to_overflow_h < 0.0:
            raise ValueError(
                "hours_to_overflow_h must be non-negative."
            )
        for name in ("fill_ratio", "forecast_speed_min"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be inside [0, 1].")
        if self.high_risk_score <= 0:
            raise ValueError("high_risk_score must be positive.")
        weights = (
            self.overflow_risk_weight,
            self.hours_risk_weight,
            self.fill_risk_weight,
            self.weather_risk_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("Risk weights must be non-negative.")


@dataclass(frozen=True)
class EnsembleDecision:
    """Store one transparent ensemble decision.

    保存一次可解释的 ensemble 决策。
    """

    action: int
    selected_policy: str
    policy_actions: dict[str, int]
    policy_disagreement: bool
    unique_action_count: int
    risk_score: int
    high_risk: bool
    rule_fallback: bool
    risk_snapshot: dict[str, Any]


class V3RiskEnsemble:
    """Use seed 0 normally and seed 1 only under risk plus disagreement.

    正常使用 seed 0，仅在风险和策略分歧同时存在时使用 seed 1。
    """

    def __init__(
        self,
        seed0_model,
        seed1_model,
        seed2_model,
        *,
        config: EnsembleRiskConfig | None = None,
    ) -> None:
        """Store three deterministic MaskablePPO policies.

        保存三个确定性的 MaskablePPO 策略。
        """
        self.models = {
            "seed0": seed0_model,
            "seed1": seed1_model,
            "seed2": seed2_model,
        }
        self.config = config or EnsembleRiskConfig()

    def select_action(
        self,
        env: RiskGatedResidualDispatchEnv,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> EnsembleDecision:
        """Select a legal action and return full switching diagnostics.

        选择合法动作并返回完整切换诊断。
        """
        mask = np.asarray(action_mask, dtype=bool)
        policy_actions = {
            name: _predict_action(model, observation, mask)
            for name, model in self.models.items()
        }
        unique_actions = set(policy_actions.values())
        disagreement = len(unique_actions) > 1
        snapshot = adaptive_risk_snapshot(
            env.env,
            AdaptiveRiskGateConfig(
                hours_to_overflow_threshold_h=(
                    self.config.hours_to_overflow_h
                ),
                fill_ratio_threshold=self.config.fill_ratio,
                weather_fill_ratio_threshold=0.0,
                weather_speed_threshold=(
                    self.config.forecast_speed_min
                ),
            ),
        )
        risk_score = self._risk_score(snapshot)
        high_risk = risk_score >= self.config.high_risk_score
        if not disagreement:
            selected_policy = "consensus"
            action = policy_actions["seed0"]
        elif high_risk:
            selected_policy = "risk_policy_seed1"
            action = policy_actions["seed1"]
        else:
            selected_policy = "default_policy_seed0"
            action = policy_actions["seed0"]

        fallback = not (
            0 <= int(action) < len(mask)
            and bool(mask[int(action)])
        )
        if fallback:
            action = 0
            selected_policy = "rule_fallback"
        return EnsembleDecision(
            action=int(action),
            selected_policy=selected_policy,
            policy_actions=policy_actions,
            policy_disagreement=disagreement,
            unique_action_count=len(unique_actions),
            risk_score=risk_score,
            high_risk=high_risk,
            rule_fallback=fallback,
            risk_snapshot=snapshot,
        )

    def _risk_score(self, snapshot: dict[str, Any]) -> int:
        """Convert physical signals into an integer switching score.

        将物理信号转换为整数切换分数。
        """
        score = 0
        if bool(snapshot["overflow_risk_active"]):
            score += self.config.overflow_risk_weight
        if (
            float(snapshot["min_hours_to_overflow"])
            <= self.config.hours_to_overflow_h
        ):
            score += self.config.hours_risk_weight
        if (
            float(snapshot["max_emitter_fill_ratio"])
            >= self.config.fill_ratio
        ):
            score += self.config.fill_risk_weight
        if (
            float(snapshot["forecast_fleet_speed_min"])
            <= self.config.forecast_speed_min
        ):
            score += self.config.weather_risk_weight
        return int(score)


def load_v3_ensemble(
    seed0_run: Path,
    seed1_run: Path,
    seed2_run: Path,
    *,
    model_choice: str = "best",
    config: EnsembleRiskConfig | None = None,
) -> tuple[V3RiskEnsemble, dict[str, Any]]:
    """Load three compatible v3 models and their shared environment config.

    加载三个兼容的 v3 模型及其共享环境配置。
    """
    if model_choice not in {"best", "final"}:
        raise ValueError("model_choice must be 'best' or 'final'.")
    runs = {
        "seed0": Path(seed0_run),
        "seed1": Path(seed1_run),
        "seed2": Path(seed2_run),
    }
    configs = {
        name: json.loads(
            (run / "config.json").read_text(encoding="utf-8")
        )
        for name, run in runs.items()
    }
    reference = configs["seed0"]
    compatibility_keys = (
        "algorithm",
        "scenario",
        "episode_hours",
        "forecast_context_hours",
        "decision_interval_h",
        "event_triggered",
        "weather_mode",
        "action_count",
        "observation_size",
    )
    for name, candidate in configs.items():
        mismatches = {
            key: (reference.get(key), candidate.get(key))
            for key in compatibility_keys
            if reference.get(key) != candidate.get(key)
        }
        if mismatches:
            raise ValueError(
                f"Incompatible ensemble run {name}: {mismatches}"
            )
    if reference.get("algorithm") != "maskable_residual_ppo_v3":
        raise ValueError("All ensemble models must be residual PPO v3.")
    model_name = (
        "maskable_residual_v3_best_validation"
        if model_choice == "best"
        else "maskable_residual_v3_final"
    )
    models = {
        name: MaskablePPO.load(run / model_name, device="cpu")
        for name, run in runs.items()
    }
    ensemble = V3RiskEnsemble(
        models["seed0"],
        models["seed1"],
        models["seed2"],
        config=config,
    )
    return ensemble, reference


def _predict_action(
    model,
    observation: np.ndarray,
    action_mask: np.ndarray,
) -> int:
    """Return one deterministic masked action.

    返回一个确定性的带掩码动作。
    """
    action, _state = model.predict(
        observation,
        deterministic=True,
        action_masks=action_mask,
    )
    return int(action)
