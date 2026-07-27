"""Masked residual RL with rule-counterfactual rewards.

带动态动作掩码和规则反事实奖励的残差强化学习。

This package is independent from both the original high-level PPO and the
first residual PPO implementation.

本包独立于原高层 PPO 和第一版 residual PPO，
不会覆盖它们的代码或模型。
"""

from .env import MaskedResidualDispatchEnv, MaskedResidualEnvConfig
from .executor import MaskedResidualRuleExecutor

__all__ = [
    "MaskedResidualDispatchEnv",
    "MaskedResidualEnvConfig",
    "MaskedResidualRuleExecutor",
]

