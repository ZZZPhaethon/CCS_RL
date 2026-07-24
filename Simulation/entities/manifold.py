"""Define subsea manifold-related components used by the simulation.

定义仿真中使用的海底管道（subsea manifold）相关组件。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubseaManifold:
    """Represent an immutable subsea manifold with a flow constraint.

    表示具有流量约束的不可变海底歧管。

    Attributes:
        entity_id: Unique manifold identifier. / 海底管道的唯一标识符。
        max_flow_tph: Maximum CO₂ flow rate in tonnes per hour.
            / 最大二氧化碳流量，单位为吨每小时。
        available: Whether the manifold is available for operation.
            / 海底管道当前是否可投入运行。
    """

    entity_id: str
    max_flow_tph: float
    available: bool = True
