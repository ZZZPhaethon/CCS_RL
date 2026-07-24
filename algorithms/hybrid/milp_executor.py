"""Adapter from the existing rolling MILP baseline to ``ActionExecutor``.

将现有滚动 MILP 基线适配为 ``ActionExecutor`` 的接口。

The MILP is a comparison baseline, not yet a goal-conditioned controller: it
optimises its existing physical/economic objective from the current state. A
``DispatchGoal`` supplies the replan cadence and is recorded in the environment,
but it is not silently turned into a hard MILP constraint.

该 MILP 是比较基线，尚不是一个目标条件化控制器：它从当前状态出发，优化现有的物理/经济目标。
``DispatchGoal`` 提供重规划周期并记录在环境中，但不会被惄然变成 MILP 的硬约束。
"""

from __future__ import annotations

from typing import Any, Mapping

from Simulation.control import RollingMilpController

from ..contracts import DispatchGoal, NativeAction
from .rule_executor import _environment_from_context, _validate_goal_membership


class RollingMilpExecutor:
    """Produce native actions from the replay-checked rolling MILP baseline.

    通过已回放校验的滚动 MILP 基线生成原生动作。

    The underlying controller validates the action slice that will execute
    before the next replan. It raises a clear error when the solver cannot
    produce an executable plan, rather than silently falling back to a rule.

    底层控制器会校验下一次重规划前将执行的动作片段。若求解器无法生成可执行
    计划，它会显式报错，而不会静默地退回为规则。
    """

    def __init__(
        self,
        *,
        planning_horizon_h: int = 168,
        time_limit_s: float = 30.0,
        solver: str = "cbc",
    ) -> None:
        """Configure MILP forecast horizon, solver time limit, and backend.

        配置 MILP 预见时域、求解时间上限和后端。
        """
        self.planning_horizon_h = max(1, int(planning_horizon_h))
        self.time_limit_s = max(0.1, float(time_limit_s))
        self.solver = solver
        self._controller: RollingMilpController | None = None
        self._simulator_identity: int | None = None
        self._replan_after_h: float | None = None
        self.last_plan_status = ""
        self.last_plan_valid = False

    def propose_action(
        self,
        goal: DispatchGoal,
        context: Mapping[str, Any],
    ) -> NativeAction:
        """Return the next replay-checked action from a fresh MILP plan/trace.

        返回来自新 MILP 计划或已验证轨迹的下一个动作。
        """
        goal.validate()
        env = _environment_from_context(context)
        _validate_goal_membership(goal, env)
        env.set_goal_assignment(dict(goal.emitter_to_vessel))

        simulator_identity = id(env.simulator)
        if (
            self._controller is None
            or simulator_identity != self._simulator_identity
            or goal.replan_after_h != self._replan_after_h
        ):
            self._controller = RollingMilpController(
                env,
                replan_every=max(1, int(round(goal.replan_after_h))),
                planning_horizon_h=self.planning_horizon_h,
                time_limit_s=self.time_limit_s,
                solver=self.solver,
            )
            self._simulator_identity = simulator_identity
            self._replan_after_h = goal.replan_after_h

        action = self._controller.policy(env)
        self.last_plan_status = self._controller.last_plan_status
        self.last_plan_valid = self._controller.last_plan_valid
        return {"vessels": list(action["vessels"]), "wells": list(action["wells"])}
