"""Constrained tail-risk checkpoint scoring for residual PPO v4.

用于 residual PPO v4 的受约束尾部风险 checkpoint 评分。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TailRiskSelectionConfig:
    """Configure physical and tail-risk checkpoint selection.

    配置基于物理约束和尾部风险的 checkpoint 选择。
    """

    normal_cvar_weight_eur_per_t: float = 37.5
    hard_cvar_weight_eur_per_t: float = 112.5
    hard_worst_weight_eur_per_t: float = 100.0
    hard_violation_penalty_eur: float = 10_000_000.0
    normal_vent_degradation_limit: float = 0.10
    hard_worst_improvement_fraction: float = 0.0


@dataclass(frozen=True)
class ReferenceValidationMetrics:
    """Store v3 metrics used to reject v4 regressions.

    保存用于拒绝 v4 退化的 v3 验证指标。
    """

    run_dir: str
    normal_mean_vented_t: float
    hard_worst_vented_t: float


def score_validation_checkpoint(
    normal: dict[str, float],
    hard: dict[str, float],
    *,
    config: TailRiskSelectionConfig,
    reference: ReferenceValidationMetrics | None,
) -> dict[str, Any]:
    """Return a lexicographic constraint rank and a tail-risk loss.

    返回按约束优先的字典序排名和尾部风险损失。
    """
    hard_violations = float(
        normal["hard_violations"] + hard["hard_violations"]
    )
    mean_cost = 0.5 * (
        float(normal["mean_total_cost_eur"])
        + float(hard["mean_total_cost_eur"])
    )
    robust_loss = (
        mean_cost
        + config.normal_cvar_weight_eur_per_t
        * float(normal["cvar_vented_t"])
        + config.hard_cvar_weight_eur_per_t
        * float(hard["cvar_vented_t"])
        + config.hard_worst_weight_eur_per_t
        * float(hard["worst_vented_t"])
        + config.hard_violation_penalty_eur * hard_violations
    )

    normal_limit = None
    hard_limit = None
    normal_constraint_passed = True
    hard_constraint_passed = True
    if reference is not None:
        normal_limit = reference.normal_mean_vented_t * (
            1.0 + config.normal_vent_degradation_limit
        )
        hard_limit = reference.hard_worst_vented_t * (
            1.0 - config.hard_worst_improvement_fraction
        )
        normal_constraint_passed = (
            float(normal["mean_vented_t"]) <= normal_limit
        )
        hard_constraint_passed = (
            float(hard["worst_vented_t"]) <= hard_limit
        )

    physical_constraint_passed = hard_violations <= 0.0
    failed_performance_constraints = sum(
        not passed
        for passed in (
            normal_constraint_passed,
            hard_constraint_passed,
        )
    )
    failed_constraints = (
        int(not physical_constraint_passed)
        + failed_performance_constraints
    )
    return {
        "robust_selection_loss": robust_loss,
        "failed_constraints": int(failed_constraints),
        "failed_performance_constraints": int(
            failed_performance_constraints
        ),
        "qualified": failed_constraints == 0,
        "physical_constraint_passed": physical_constraint_passed,
        "normal_constraint_passed": normal_constraint_passed,
        "hard_constraint_passed": hard_constraint_passed,
        "normal_mean_vent_limit_t": normal_limit,
        "hard_worst_vent_limit_t": hard_limit,
        "rank": (
            int(not physical_constraint_passed),
            int(failed_performance_constraints),
            robust_loss,
        ),
    }


def load_reference_validation(
    run_dir: Path,
    *,
    normal_validation_seeds: tuple[int, ...],
    hard_validation_seeds: tuple[int, ...],
) -> ReferenceValidationMetrics:
    """Load a compatible v3 best-validation result.

    加载与当前验证 seed 完全一致的 v3 最佳验证结果。
    """
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    expected_normal = tuple(config["normal_validation_seeds"])
    expected_hard = tuple(config["hard_validation_seeds"])
    if expected_normal != normal_validation_seeds:
        raise ValueError(
            "Reference normal validation seeds do not match v4."
        )
    if expected_hard != hard_validation_seeds:
        raise ValueError(
            "Reference hard validation seeds do not match v4."
        )
    best = json.loads(
        (run_dir / "validation" / "best.json").read_text(
            encoding="utf-8"
        )
    )
    return ReferenceValidationMetrics(
        run_dir=str(run_dir),
        normal_mean_vented_t=float(
            best["normal"]["mean_vented_t"]
        ),
        hard_worst_vented_t=float(
            best["hard"]["worst_vented_t"]
        ),
    )


def discover_reference_v3_run(
    *,
    scenario: str,
    normal_validation_seeds: tuple[int, ...],
    hard_validation_seeds: tuple[int, ...],
    root: Path = Path("logs") / "residual_rl_v3",
) -> Path | None:
    """Find the newest compatible completed v3 seed0 run.

    查找最新且验证集兼容的已完成 v3 seed0 训练。
    """
    candidates: list[Path] = []
    if not root.exists():
        return None
    for config_path in root.glob("*__seed0__*/config.json"):
        run_dir = config_path.parent
        if not (run_dir / "validation" / "best.json").exists():
            continue
        if not (run_dir / "training_complete.json").exists():
            continue
        try:
            config = json.loads(
                config_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("scenario") != scenario:
            continue
        if tuple(config.get("normal_validation_seeds", ())) != (
            normal_validation_seeds
        ):
            continue
        if tuple(config.get("hard_validation_seeds", ())) != (
            hard_validation_seeds
        ):
            continue
        candidates.append(run_dir)
    return max(candidates, key=lambda path: path.name) if candidates else None
