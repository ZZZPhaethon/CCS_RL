"""Algorithms above the physical CCS simulation.

面向 CCS 物理仿真之上的算法层。

This package contains learning and hybrid-control contracts.  It deliberately
does not duplicate physical constraints: all proposed actions must be executed
and checked by :mod:`Simulation`.

本包放置学习与混合控制接口，不复制物理约束；所有动作均必须交由
``Simulation`` 执行并校验。
"""

from .contracts import (
    ActionExecutor,
    DispatchGoal,
    HighLevelPolicy,
    ReplanSchedule,
    should_replan,
)
from .hybrid import (
    GoalAwareNativeMpcExecutor,
    GoalAwareRuleExecutor,
    RollingMilpExecutor,
)
from .evaluation import RolloutMetrics, compare_executors, evaluate_executor

__all__ = [
    "ActionExecutor",
    "DispatchGoal",
    "GoalAwareNativeMpcExecutor",
    "GoalAwareRuleExecutor",
    "RollingMilpExecutor",
    "HighLevelPolicy",
    "RolloutMetrics",
    "compare_executors",
    "evaluate_executor",
    "ReplanSchedule",
    "should_replan",
]
