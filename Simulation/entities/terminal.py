"""Define terminal entities used to buffer and transfer captured CO₂.

定义用于缓冲和转运捕集二氧化碳的终端实体。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Terminal:
    """Represent an immutable CO₂ transfer terminal and berth constraints.

    表示一个不可变的二氧化碳转运终端及其泊位约束。

    Attributes:
        entity_id: Unique terminal identifier. / 终端的唯一标识符。
        storage_capacity_t: Maximum on-site CO₂ inventory in tonnes.
            / 现场二氧化碳最大库存量，单位为吨。
        berth_count: Number of vessels that may berth simultaneously.
            / 可同时靠泊船舶的数量。
        site_name: Optional human-readable terminal name.
            / 可选的供人阅读的终端名称。
    """

    entity_id: str
    storage_capacity_t: float
    berth_count: int = 1
    site_name: str | None = None
