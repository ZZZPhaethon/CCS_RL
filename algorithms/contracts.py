"""Solver-independent contracts for hybrid CCS dispatch.

CCS 混合调度的求解器无关接口。

The high-level algorithm selects a robust operating goal at a sparse decision
time.  A rule-based controller, MPC, or MILP executor converts it into the
native one-step action accepted by ``Simulation.environment.CCSEnv``.  Keeping
this boundary explicit prevents an RL policy from bypassing physics or learning
an unnecessary low-level action for every simulation hour.

高层算法只在稀疏决策时刻选择稳健运行目标；规则、MPC 或 MILP 执行器再将
其转换为 ``Simulation.environment.CCSEnv`` 可执行的单步动作。显式划分边界可
避免 RL 绕过物理层，也避免其在每个仿真小时学习不必要的底层动作。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


NativeAction = dict[str, list[int]]
"""The native action shape: ``{'vessels': [...], 'wells': [...]}``.

原生动作格式：``{'vessels': [...], 'wells': [...]}``。
"""


@dataclass(frozen=True)
class DispatchGoal:
    """A high-level target that remains valid across several physics steps.

    跨越多个物理仿真步保持有效的高层运行目标。

    ``emitter_to_vessel`` follows the same direction as
    :meth:`CCSEnv.set_goal_assignment`: each emitter is assigned at most one
    preferred vessel. ``vessel_to_emitter`` supports independent vessel service
    preferences; ``None`` delegates that vessel to the adaptive executor. These
    are intentions, not permission to violate berth, weather, inventory, or
    pressure constraints. ``well_rate_indices`` use the environment's discrete
    injection-rate levels and can omit wells that the low-level executor should
    leave unchanged.

    ``emitter_to_vessel`` 与 :meth:`CCSEnv.set_goal_assignment` 方向一致：每个
    emitter 最多指定一艘优先服务船。``vessel_to_emitter`` 支持每艘船的独立服务
    偏好，其中 ``None`` 将该船交给自适应执行器。它们表示意图而非越过泊位、天气、
    库存或压力约束的许可。``well_rate_indices`` 使用环境的离散注入率等级；遗漏的井
    由底层执行器维持或自行处理。
    """

    emitter_to_vessel: Mapping[str, str] = field(default_factory=dict)
    vessel_to_emitter: Mapping[str, str | None] = field(default_factory=dict)
    well_rate_indices: Mapping[str, int] = field(default_factory=dict)
    replan_after_h: float = 24.0
    rationale: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Reject malformed goals before an executor receives them.

        在执行器接收目标前拒绝格式错误的目标。
        """
        if self.replan_after_h <= 0.0:
            raise ValueError("replan_after_h must be positive.")
        if any(not emitter_id for emitter_id in self.emitter_to_vessel):
            raise ValueError("Emitter identifiers must be non-empty.")
        if any(not vessel_id for vessel_id in self.emitter_to_vessel.values()):
            raise ValueError("Vessel identifiers must be non-empty.")
        if any(not vessel_id for vessel_id in self.vessel_to_emitter):
            raise ValueError("Vessel preference identifiers must be non-empty.")
        if any(
            emitter_id is not None and not emitter_id
            for emitter_id in self.vessel_to_emitter.values()
        ):
            raise ValueError("Preferred emitter identifiers must be non-empty or None.")
        if any(rate_index < 0 for rate_index in self.well_rate_indices.values()):
            raise ValueError("Well-rate indices must be non-negative.")


@dataclass(frozen=True)
class ReplanSchedule:
    """State needed to decide when the high-level policy acts again.

    判断高层策略何时再次决策所需的状态。
    """

    selected_at_h: float
    replan_after_h: float

    @property
    def due_at_h(self) -> float:
        """Return the next scheduled decision time in simulation hours.

        返回下一次计划决策时刻（仿真小时）。
        """
        return self.selected_at_h + self.replan_after_h


def should_replan(
    schedule: ReplanSchedule | None,
    now_h: float,
    *,
    event_requires_replan: bool = False,
) -> bool:
    """Return whether a new high-level goal should be selected.

    返回是否应选择新的高层目标。

    A vessel becoming available, a berth outage, an injection-rate loss, or a
    serious forecast revision should set ``event_requires_replan``.  The first
    decision always replans; otherwise time is measured in simulation hours.

    船舶可用、泊位故障、注入能力下降或显著预报修订等事件应令
    ``event_requires_replan`` 为真。首次决策始终重规划；其他情况按仿真小时判断。
    """
    if event_requires_replan or schedule is None:
        return True
    return now_h >= schedule.due_at_h - 1e-9


class HighLevelPolicy(Protocol):
    """Choose a dispatch goal from an observation and forecast context.

    根据观测和预测上下文选择调度目标。
    """

    def select_goal(
        self,
        observation: Sequence[float],
        context: Mapping[str, Any],
    ) -> DispatchGoal:
        """Return the goal to be held until the next replan event.

        返回保持到下一次重规划事件的目标。
        """


class ActionExecutor(Protocol):
    """Translate a high-level goal into one physically admissible action.

    将高层目标转化为一个物理可执行的动作。
    """

    def propose_action(
        self,
        goal: DispatchGoal,
        context: Mapping[str, Any],
    ) -> NativeAction:
        """Return an action that will still be checked by the environment.

        返回仍将由环境检查的动作。
        """
