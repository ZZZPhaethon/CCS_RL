"""Residual reinforcement learning for event-triggered CCS dispatch.

用于事件触发 CCS 调度的残差强化学习。

The package is intentionally separate from :mod:`sim.control.event_based.rl`, so the
existing 192-action high-level PPO remains reproducible and unchanged.

该包与 :mod:`sim.control.event_based.rl` 完全分离，因此原有 192 动作高层 PPO 可继续复现，
且不会被修改。
"""

from .action_codec import ResidualActionCodec, ResidualIntervention
from .env import ResidualDispatchEnv, ResidualEnvConfig
from .executor import ResidualRuleExecutor

__all__ = [
    "ResidualActionCodec",
    "ResidualDispatchEnv",
    "ResidualEnvConfig",
    "ResidualIntervention",
    "ResidualRuleExecutor",
]
