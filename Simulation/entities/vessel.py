"""Define vessel entities used to transport captured CO₂ by sea.

定义用于通过海运转移捕集二氧化碳的船舶实体。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vessel:
    """Represent an immutable CO₂ transport vessel and its operating limits.

    表示一艘不可变的二氧化碳运输船舶及其运行限制。

    Attributes:
        entity_id: Unique vessel identifier. / 船舶的唯一标识符。
        capacity_t: Maximum CO₂ cargo mass in tonnes.
            / 最大二氧化碳载货量，单位为吨。
        loading_rate_tph: Maximum loading rate in tonnes per hour.
            / 最大装载速率，单位为吨每小时。
        unloading_rate_tph: Maximum unloading rate in tonnes per hour.
            / 最大卸载速率，单位为吨每小时。
        volume_capacity_m3: Optional cargo volume capacity in cubic metres.
            / 可选的货舱体积容量，单位为立方米。
        speed_knots: Optional cruising speed in knots.
            / 可选的巡航速度，单位为节。
    """

    entity_id: str
    capacity_t: float
    loading_rate_tph: float
    unloading_rate_tph: float
    volume_capacity_m3: float | None = None
    speed_knots: float | None = None
