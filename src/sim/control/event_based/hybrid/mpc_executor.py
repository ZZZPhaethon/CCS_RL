"""Goal-aware adapter around the replay-validated native MPC controller.

面向高层目标的、基于回放验证的原生 MPC 接口。

The executor gives the native MPC a candidate induced by a high-level
``DispatchGoal``.  The MPC evaluates that candidate beside its safety-oriented
baselines in a copied physical environment, then executes only the first
replay-valid action.  Thus a goal is a preference, never a bypass of physics.

该执行器把高层 ``DispatchGoal`` 转化为原生 MPC 的一个候选方案。MPC 在复制的
物理环境中将它与安全导向基线共同评估，然后只执行通过回放验证的第一个
动作。因此高层目标只是偏好，不能绕过物理约束。
"""

from __future__ import annotations

from typing import Any, Mapping

from sim.control import RollingNativeMpcController
from sim.control.baselines import make_cluster_shuttle_policy
from sim.environment import CCSEnv

from ..contracts import DispatchGoal, NativeAction
from .rule_executor import _environment_from_context, _validate_goal_membership


class GoalAwareNativeMpcExecutor:
    """Use a dispatch-goal candidate inside rolling, replay-validated MPC.

    在滚动、回放验证的 MPC 中使用调度目标候选方案。

    A fresh controller is created for a new goal or a new reset episode.  For a
    continuing episode, its internally validated trace is reused until the
    goal's replan period expires.  The controller will then re-evaluate all
    candidates from the actual current state.

    新目标或新回合重置时会创建新控制器。在同一回合中，会复用内部已验证的
    动作轨迹，直到目标的重规划周期到期；届时 MPC 会基于实际当前状态重新评估所有
    候选方案。
    """

    def __init__(self, planning_horizon_h: int = 168) -> None:
        """Configure the MPC look-ahead horizon in simulation hours.

        配置 MPC 的预见时域（仿真小时）。
        """
        self.planning_horizon_h = max(1, int(planning_horizon_h))
        self._controller: RollingNativeMpcController | None = None
        self._goal_signature: tuple[object, ...] | None = None
        self._simulator_identity: int | None = None
        self.last_selected_candidate = ""

    def propose_action(
        self,
        goal: DispatchGoal,
        context: Mapping[str, Any],
    ) -> NativeAction:
        """Return the next replay-validated MPC action for ``goal``.

        返回面向 ``goal`` 的、通过回放验证的下一个 MPC 动作。
        """
        goal.validate()
        env = _environment_from_context(context)
        _validate_goal_membership(goal, env)
        env.set_goal_assignment(dict(goal.emitter_to_vessel))

        signature = _goal_signature(goal)
        simulator_identity = id(env.simulator)
        if (
            self._controller is None
            or signature != self._goal_signature
            or simulator_identity != self._simulator_identity
        ):
            self._controller = self._new_controller(env, goal)
            self._goal_signature = signature
            self._simulator_identity = simulator_identity

        action = self._controller.policy(env)
        self.last_selected_candidate = self._controller.last_candidate_name
        return {"vessels": list(action["vessels"]), "wells": list(action["wells"])}

    def _new_controller(
        self,
        env: CCSEnv,
        goal: DispatchGoal,
    ) -> RollingNativeMpcController:
        """Create a controller whose extra candidate represents ``goal``.

        创建一个以 ``goal`` 为额外候选方案的控制器。
        """
        replan_every_h = max(1, int(round(goal.replan_after_h)))
        return RollingNativeMpcController(
            env,
            replan_every=replan_every_h,
            planning_horizon_h=self.planning_horizon_h,
            preferred_policies={"goal_preference": _goal_policy(goal)},
        )


def _goal_policy(goal: DispatchGoal):
    """Build the MPC candidate policy implied by a high-level goal.

    构建由高层目标隐含的 MPC 候选策略。
    """
    assignment = dict(goal.emitter_to_vessel)

    def policy(env: CCSEnv) -> NativeAction:
        action = make_cluster_shuttle_policy(env, assignment)(env)
        well_mask = env.well_rate_action_mask()
        for well_id, target_index in goal.well_rate_indices.items():
            well_position = env.well_ids.index(well_id)
            if target_index < len(well_mask[well_position]) and well_mask[
                well_position
            ][target_index]:
                action["wells"][well_position] = target_index
        return {"vessels": list(action["vessels"]), "wells": list(action["wells"])}

    return policy


def _goal_signature(goal: DispatchGoal) -> tuple[object, ...]:
    """Return a stable value used to invalidate stale MPC traces.

    返回用于作废过期 MPC 轨迹的稳定标识。
    """
    return (
        tuple(sorted(goal.emitter_to_vessel.items())),
        tuple(sorted(goal.well_rate_indices.items())),
        goal.replan_after_h,
    )

