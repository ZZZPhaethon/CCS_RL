"""Execute a dispatch goal through the existing masked rule controller.

通过已有的带掩码规则控制器执行调度目标。

This is the first concrete ``ActionExecutor`` implementation.  It is designed
as a transparent baseline for the later RL+MPC comparison, not as a substitute
for a rolling optimiser.

这是第一个具体的 ``ActionExecutor`` 实现。它是后续 RL+MPC 比较中的透明基线，
而非滚动优化器的替代品。
"""

from __future__ import annotations

from typing import Any, Mapping

from Simulation.control.baselines import greedy_shuttle_policy, make_cluster_shuttle_policy
from Simulation.environment import CCSEnv, VESSEL_WAIT

from ..contracts import DispatchGoal, NativeAction


class GoalAwareRuleExecutor:
    """Convert an emitter-to-vessel goal into a masked native environment action.

    将 emitter 到船舶的目标转换为满足掩码的原生环境动作。

    The reusable cluster-shuttle rule controls vessels.  Explicit well targets
    override its well-rate choice only when the current environment mask allows
    them; otherwise the baseline's already-feasible rate is retained.  The
    environment and physical network still perform the authoritative checks.

    可复用的 cluster-shuttle 规则控制船舶。只有当前环境掩码允许时，显式注入井目标
    才会覆盖该规则的注入率选择；否则保留基线已可行的选择。环境与物理网络仍执行权威
    校验。
    """

    def propose_action(
        self,
        goal: DispatchGoal,
        context: Mapping[str, Any],
    ) -> NativeAction:
        """Return a safe candidate native action for a reset ``CCSEnv``.

        为已 reset 的 ``CCSEnv`` 返回安全的候选原生动作。

        Args:
            goal: High-level assignment and optional well-rate targets.
                / 高层分配和可选注入率目标。
            context: Must contain ``env`` with a reset ``CCSEnv`` instance.
                / 必须包含已 reset 的 ``CCSEnv`` 实例 ``env``。
        """
        goal.validate()
        env = _environment_from_context(context)
        _validate_goal_membership(goal, env)

        assignment = dict(goal.emitter_to_vessel)
        if goal.vessel_to_emitter:
            # The native goal observation still uses emitter->vessel.  When
            # several vessels prefer one emitter, retain the first preference;
            # actual dispatch below continues to honour every vessel profile.
            # 原生目标观测仍使用 emitter->vessel。多船偏好同一排放源时，观测保留
            # 第一条偏好；下方实际调度仍会处理每艘船的独立偏好。
            assignment = {}
            for vessel_id, emitter_id in goal.vessel_to_emitter.items():
                if emitter_id is not None:
                    assignment.setdefault(emitter_id, vessel_id)
            env.set_goal_assignment(assignment)
            action = _preference_shuttle_action(
                env,
                dict(goal.vessel_to_emitter),
            )
        else:
            env.set_goal_assignment(assignment)
            action = make_cluster_shuttle_policy(env, assignment)(env)
        well_mask = env.well_rate_action_mask()

        for well_id, target_index in goal.well_rate_indices.items():
            well_position = env.well_ids.index(well_id)
            if target_index < len(well_mask[well_position]) and well_mask[
                well_position
            ][target_index]:
                action["wells"][well_position] = target_index
        return {"vessels": list(action["vessels"]), "wells": list(action["wells"])}


def _environment_from_context(context: Mapping[str, Any]) -> CCSEnv:
    """Read and type-check the environment supplied by the orchestration loop.

    读取并检查编排循环提供的环境类型。
    """
    env = context.get("env")
    if not isinstance(env, CCSEnv):
        raise TypeError("context['env'] must be a reset Simulation.environment.CCSEnv.")
    if env.simulator is None:
        raise RuntimeError("Call env.reset() before requesting an action.")
    return env


def _validate_goal_membership(goal: DispatchGoal, env: CCSEnv) -> None:
    """Reject IDs outside the current scenario before executing a goal.

    在执行目标前拒绝不属于当前场景的实体标识符。
    """
    unknown_emitters = set(goal.emitter_to_vessel) - set(env.emitter_ids)
    unknown_vessels = set(goal.emitter_to_vessel.values()) - set(env.vessel_ids)
    unknown_preference_vessels = set(goal.vessel_to_emitter) - set(env.vessel_ids)
    unknown_preference_emitters = {
        emitter_id
        for emitter_id in goal.vessel_to_emitter.values()
        if emitter_id is not None and emitter_id not in env.emitter_ids
    }
    unknown_wells = set(goal.well_rate_indices) - set(env.well_ids)
    if (
        unknown_emitters
        or unknown_vessels
        or unknown_preference_vessels
        or unknown_preference_emitters
        or unknown_wells
    ):
        raise ValueError(
            "Goal contains IDs outside the environment: "
            f"emitters={sorted(unknown_emitters)}, "
            f"vessels={sorted(unknown_vessels)}, "
            f"preference_vessels={sorted(unknown_preference_vessels)}, "
            f"preference_emitters={sorted(unknown_preference_emitters)}, "
            f"wells={sorted(unknown_wells)}."
        )


def _preference_shuttle_action(
    env: CCSEnv,
    preferences: Mapping[str, str | None],
) -> NativeAction:
    """Apply feasible vessel-specific emitter preferences over a safe baseline.

    在安全基线之上应用每艘船各自的可行排放源偏好。

    ``None`` keeps the adaptive greedy decision.  A named emitter overrides
    that decision only while the native action mask permits departure.  Full
    vessels, unloading vessels, and vessels already sailing therefore retain
    the baseline's physically sensible terminal/wait behaviour.

    ``None`` 保留自适应贪心决策。指定排放源仅在原生动作掩码允许出发时覆盖该
    决策；满载、正在卸载或航行中的船仍保持基线合理的返港/等待行为。
    """
    action = greedy_shuttle_policy(env)
    masks = env.vessel_action_mask()
    for position, vessel_id in enumerate(env.vessel_ids):
        emitter_id = preferences.get(vessel_id)
        if emitter_id is None:
            continue
        vessel = env.network.entities[vessel_id]
        cargo_t = env.simulator.state.entity_inventory_t.get(vessel_id, 0.0)
        if cargo_t >= vessel.capacity_t - 1e-9:
            continue
        if env.simulator.state.vessel_berths.get(vessel_id) == emitter_id:
            action["vessels"][position] = VESSEL_WAIT
            continue
        preferred_action = env.vessel_go_emitter_action(emitter_id)
        if masks[position][preferred_action]:
            action["vessels"][position] = preferred_action
    return {
        "vessels": list(action["vessels"]),
        "wells": list(action["wells"]),
    }
