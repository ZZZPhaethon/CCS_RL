"""Encode factorised high-level vessel preferences for PPO.

为 PPO 编码小型的高层调度决策。

Each vessel independently selects either adaptive service or one preferred
emitter.  The Cartesian product is combined with three injection modes.  The
Phase 1 case therefore has ``(3 + 1) ** 3 * 3 = 192`` discrete intentions.
They guide a masked low-level executor and never bypass physical constraints.

每艘船独立选择自适应服务或一个偏好排放源，再与三种注入模式组合。Phase 1
场景因此包含 ``(3 + 1) ** 3 * 3 = 192`` 个离散意图；它们只指导带掩码的
底层执行器，不会绕过物理约束。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ..contracts import DispatchGoal


@dataclass(frozen=True)
class InjectionMode:
    """A named discrete injection preference for all available wells.

    适用于所有可用注入井的具名离散注入偏好。
    """

    name: str
    rate_index: int


DEFAULT_INJECTION_MODES = (
    InjectionMode("conservative", 1),
    InjectionMode("balanced", 3),
    InjectionMode("aggressive", 5),
)


class HighLevelActionCodec:
    """Map one discrete index to vessel preferences and an injection mode.

    将离散动作索引映射为船舶分配与注入偏好。
    """

    def __init__(
        self,
        emitter_ids: list[str],
        vessel_ids: list[str],
        well_ids: list[str],
        *,
        injection_modes: tuple[InjectionMode, ...] = DEFAULT_INJECTION_MODES,
    ) -> None:
        """Build a stable codec from the current scenario's entity identifiers.

        根据当前场景的实体标识符构建稳定编码器。
        """
        if not emitter_ids:
            raise ValueError("HighLevelActionCodec requires at least one emitter.")
        if not vessel_ids:
            raise ValueError("HighLevelActionCodec requires at least one vessel.")
        if not injection_modes:
            raise ValueError("HighLevelActionCodec requires at least one injection mode.")
        self.emitter_ids = tuple(emitter_ids)
        self.vessel_ids = tuple(vessel_ids)
        self.well_ids = tuple(well_ids)
        self.injection_modes = tuple(injection_modes)
        # ``None`` means that the low-level executor may select the best
        # currently feasible emitter for that vessel.
        # ``None`` 表示底层执行器可以为该船选择当前最佳的可行排放源。
        service_choices: tuple[str | None, ...] = (None, *self.emitter_ids)
        self._preference_profiles = tuple(
            dict(zip(self.vessel_ids, choices))
            for choices in product(service_choices, repeat=len(self.vessel_ids))
        )

    @property
    def action_count(self) -> int:
        """Return the total number of available high-level actions.

        返回高层动作的总数。
        """
        return len(self._preference_profiles) * len(self.injection_modes)

    def decode(self, action_index: int, *, replan_after_h: float) -> DispatchGoal:
        """Convert an action index into the goal held until the next decision.

        将动作索引转为保持到下一次决策的目标。
        """
        index = int(action_index)
        if not 0 <= index < self.action_count:
            raise ValueError(
                f"High-level action {action_index} is outside [0, {self.action_count})."
            )
        mode_count = len(self.injection_modes)
        preference_index, mode_index = divmod(index, mode_count)
        mode = self.injection_modes[mode_index]
        return DispatchGoal(
            vessel_to_emitter=dict(self._preference_profiles[preference_index]),
            well_rate_indices={well_id: mode.rate_index for well_id in self.well_ids},
            replan_after_h=replan_after_h,
            rationale=(
                f"PPO action: vessel_preferences={preference_index}; "
                f"mode={mode.name}."
            ),
            metadata={
                "action_index": index,
                "preference_index": preference_index,
                "injection_mode": mode.name,
            },
        )

    def label(self, action_index: int) -> str:
        """Return a concise human-readable label for logs and diagnostics.

        返回用于日志和诊断的精简人类可读标签。
        """
        goal = self.decode(action_index, replan_after_h=1.0)
        preferences = ",".join(
            f"{vessel}->{emitter or 'adaptive'}"
            for vessel, emitter in goal.vessel_to_emitter.items()
        )
        return f"{preferences}; {goal.metadata['injection_mode']}"

