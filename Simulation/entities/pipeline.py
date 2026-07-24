"""Define pipeline entities and route metadata used by the simulation.

定义仿真中使用的管道实体及其路径元数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Store a route point as a two-dimensional coordinate pair.
# 将路径点表示为二维坐标对。
Coordinate = tuple[float, float]


@dataclass(frozen=True)
class Pipeline:
    """Represent an immutable CO₂ pipeline and its transport constraints.

    表示一条不可变的二氧化碳输送管道及其输送约束。

    Attributes:
        entity_id: Unique pipeline identifier. / 管道的唯一标识符。
        max_flow_tph: Maximum instantaneous flow rate in tonnes per hour.
            / 最大瞬时输送流量，单位为吨每小时。
        annual_capacity_tpy: Optional annual transport capacity,
            in tonnes per year.
            / 可选的年度输送能力，单位为吨每年。
        length_km: Optional pipeline length in kilometres.
            / 可选的管道长度，单位为千米。
        route_color: Optional display colour for visualising the route.
            / 可选的路径可视化显示颜色。
        route_coordinates: Ordered route points represented as
            coordinate pairs.
            / 按顺序排列的路径点，每个点以坐标对表示。
    """

    entity_id: str
    max_flow_tph: float
    annual_capacity_tpy: float | None = None
    length_km: float | None = None
    route_color: str | None = None
    route_coordinates: list[Coordinate] = field(default_factory=list)
