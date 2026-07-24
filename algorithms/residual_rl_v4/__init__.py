"""Tail-robust residual reinforcement learning v4.

面向尾部风险的第四版残差强化学习。
"""

from .model_selection import (
    ReferenceValidationMetrics,
    TailRiskSelectionConfig,
    score_validation_checkpoint,
)
from .replay_env import TailFailureReplayGymEnv
from .scenario import ReplayableDifficultyScenarioGenerator

__all__ = [
    "ReferenceValidationMetrics",
    "ReplayableDifficultyScenarioGenerator",
    "TailFailureReplayGymEnv",
    "TailRiskSelectionConfig",
    "score_validation_checkpoint",
]
