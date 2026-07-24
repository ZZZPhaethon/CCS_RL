"""Define the CO₂-emitter data model used by the simulation.

定义仿真中使用的二氧化碳排放源数据模型。
Notion: 需要注意的是emmitter在后续可以通过控制捕集率和时间来控制整个捕集的量,
避免那个emitter的捕集率过高导致buffer满了之后无法继续捕集,从而导致捕集量不够的问题(这里可能是那个算法表现不好的原因之一吧)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Emitter:
    """Represent an immutable CO₂ emitter and its operating constraints.

    表示一个不可变的二氧化碳排放源及其运行约束。

    Attributes:
        entity_id: Unique emitter identifier. / 排放源的唯一标识符。
        nominal_capture_tph: Nominal capture rate in tonnes per hour.
            / 名义捕集速率，单位为吨每小时。
        buffer_capacity_t: On-site buffer capacity in tonnes.
            / 现场缓冲储罐容量，单位为吨。
        min_utilization: Minimum permitted operating utilization, from 0 to 1.
            / 允许的最低运行利用率，取值范围为 0 到 1。
        default_utilization: Utilization used when no other value is supplied.
            / 未提供其他值时采用的默认利用率。
        availability: Fraction of time that the emitter is available,
            from 0 to 1.
            / 排放源可用时间占比，取值范围为 0 到 1。
        loading_rate_tph: Rate at which captured CO₂ can be loaded,
            in tonnes per hour.
            / 捕集的二氧化碳装载速率，单位为吨每小时。
        annual_target_export_tpy: Optional annual export target,
            in tonnes per year.
            / 可选的年度外运目标，单位为吨每年。
        max_production_tph: Optional maximum production rate,
            in tonnes per hour.
            / 可选的最大发电速率，单位为吨每小时。
        reference_name: Optional human-readable reference name.
            / 可选的供人阅读的参考名称。
        hourly_capture_profile_tph: Optional hourly capture-rate profile.
            / 可选的逐小时捕集速率曲线。
    """

    entity_id: str
    nominal_capture_tph: float
    buffer_capacity_t: float
    min_utilization: float = 0.0
    default_utilization: float = 1.0
    availability: float = 1.0
    loading_rate_tph: float = 800.0
    annual_target_export_tpy: float | None = None
    max_production_tph: float | None = None
    reference_name: str | None = None
    hourly_capture_profile_tph: tuple[float, ...] | None = None

    def capture_rate_tph_at(self, interval_start_h: float) -> float:
        """Return the capture rate at the start of a simulation interval.

        When an hourly profile is available, the interval start is converted to
        an hourly index and wrapped cyclically over the profile. Otherwise,
        the nominal capture rate is returned.

        当提供逐小时曲线时，将时间段起点转换为小时索引，并在曲线上
        循环取值；否则，返回名义捕集速率。

        Args:
            interval_start_h: Simulation-interval start time in hours.
                / 仿真时间段的起始时刻，单位为小时。

        Returns:
            Capture rate in tonnes per hour. / 捕集速率，单位为吨每小时。
        """
        if not self.hourly_capture_profile_tph:
            return self.nominal_capture_tph

        # Cycle the hourly profile so it can be reused across multiple days.
        # 循环使用逐小时曲线，使其可适用于跨越多天的仿真。
        hour_index = (
            int(interval_start_h) % len(self.hourly_capture_profile_tph)
        )
        return self.hourly_capture_profile_tph[hour_index]
