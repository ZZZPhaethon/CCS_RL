"""Resolve the current timestep's effective values after disturbances.

This module does not generate disturbance scenarios. It is the lookup layer
used after a :class:`Scenario` has written current-step disturbance values into
:class:`PhysicalState`.

The physical entities (:class:`Emitter`, :class:`InjectionWell`, ...) store
nominal values. These helpers return the effective values for the current step:
use the value on :class:`PhysicalState` when a disturbance override exists,
otherwise fall back to the entity's nominal value.

解析扰动后的当前时间步有效值。
本模块不生成扰动场景，而是在 :class:`Scenario` 将当前时间步的扰动值写入
:class:`PhysicalState` 后提供查询层。物理实体保存名义参数；当状态中存在
扰动覆盖值时，这些辅助函数优先返回覆盖值，否则返回实体的名义值。
"""

from __future__ import annotations

from ..entities.emitter import Emitter
from ..entities.state import PhysicalState
from ..entities.storage import InjectionWell
from ..entities.terminal import Terminal


def emitter_availability(state: PhysicalState, emitter: Emitter) -> float:
    """Return an emitter's non-negative capture-availability multiplier.

    Values above 1 model output that exceeds the nominal profile.

    返回排放源的非负捕集可用率系数。
    大于 1 的值表示产出高于名义曲线。
    """
    value = state.emitter_availability.get(
        emitter.entity_id,
        emitter.availability,
    )
    return max(0.0, value)


def well_is_available(state: PhysicalState, well: InjectionWell) -> bool:
    """Return whether an injection well can accept flow in this step.

    返回注入井在当前时间步是否能够接受流量。
    """
    return bool(state.well_available.get(well.entity_id, well.available))


def well_injectivity_factor(
    state: PhysicalState,
    well: InjectionWell,
) -> float:
    """Return a non-negative multiplier for a well's nominal injection rate.

    Defaults to ``1.0`` (nominal). Values below ``1.0`` model injectivity
    decline or partial deratings; ``0.0`` means no injection capacity.

    返回施加于注入井名义最大注入速率的非负系数。
    默认值 ``1.0`` 表示名义能力；小于 ``1.0`` 表示注入能力衰减或降额，
    ``0.0`` 表示没有注入能力。
    """
    return max(0.0, float(state.injectivity_factor.get(well.entity_id, 1.0)))


def well_max_injection_tph(state: PhysicalState, well: InjectionWell) -> float:
    """Return the effective hourly injection limit after disturbances.

    返回考虑可用性和注入能力扰动后的有效每小时注入上限。
    """
    if not well_is_available(state, well):
        return 0.0
    return well.max_injection_tph * well_injectivity_factor(state, well)


def vessel_speed_factor(state: PhysicalState, vessel_id: str) -> float:
    """Return a non-negative multiplier for a vessel's nominal sailing speed.

    Defaults to ``1.0``. Values below ``1.0`` model weather-induced slowdowns.

    返回船舶名义航速的非负系数。
    默认值 ``1.0``；小于 ``1.0`` 的值表示天气造成的减速。
    """
    return max(0.0, float(state.vessel_speed_factor.get(vessel_id, 1.0)))


def leg_speed_factor(
    state: PhysicalState,
    origin_id: str,
    destination_id: str,
    *,
    fallback: float = 1.0,
) -> float:
    """Return a non-negative speed multiplier for one sailing leg.

    返回特定航段的非负航速系数。
    """
    leg_id = f"{origin_id}->{destination_id}"
    return max(0.0, float(state.leg_speed_factor.get(leg_id, fallback)))


def terminal_berth_count(state: PhysicalState, terminal: Terminal) -> int:
    """Return the usable berth count after applying a state override.

    返回应用状态覆盖值后的可用泊位数量。
    """
    value = state.berth_count_override.get(
        terminal.entity_id,
        terminal.berth_count,
    )
    return max(0, int(value))
