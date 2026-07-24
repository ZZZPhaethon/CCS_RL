"""Define injection-well and reservoir entities used for CO₂ storage.

定义二氧化碳封存所使用的注入井和储层实体。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .line_source import LineSourceParameters


@dataclass(frozen=True)
class InjectionWell:
    """Represent an immutable injection well and its operating limits.

    表示一个不可变的注入井及其运行限制。

    Attributes:
        entity_id: Unique injection-well identifier. / 注入井的唯一标识符。
        max_injection_tph: Maximum injection rate in tonnes per hour.
            / 最大注入速率，单位为吨每小时。
        min_stable_injection_tph: Minimum stable injection rate,
            in tonnes per hour.
            / 最低稳定注入速率，单位为吨每小时。
        injectivity_index_tph_per_bar: Optional injection rate
            per pressure unit.
            / 可选的注入能力指数，单位为吨每小时每巴。
        pressure_margin_bar: Optional allowable pressure margin in bar.
            / 可选的允许压力裕度，单位为巴。
        available: Whether the well is currently available.
            / 注入井当前是否可用。
    """

    entity_id: str
    max_injection_tph: float
    min_stable_injection_tph: float = 0.0
    injectivity_index_tph_per_bar: float | None = None
    pressure_margin_bar: float | None = None
    available: bool = True


@dataclass(frozen=True)
class Reservoir:
    """Represent an immutable CO₂ reservoir and pressure-based limits.

    表示一个不可变的二氧化碳储层及其基于压力的限制。

    Attributes:
        entity_id: Unique reservoir identifier. / 储层的唯一标识符。
        storage_capacity_t: Physical storage capacity in tonnes.
            / 物理封存容量，单位为吨。
        initial_pressure_bar: Reservoir pressure before injection, in bar.
            / 注入前的储层压力，单位为巴。
        pressure_at_capacity_bar: Reservoir pressure at full capacity, in bar.
            / 储层达到满容量时的压力，单位为巴。
        max_pressure_bar: Maximum permitted reservoir pressure, in bar.
            / 允许的最高储层压力，单位为巴。
        depth_m: Optional reservoir depth in metres.
            / 可选的储层深度，单位为米。
        reservoir_pressure_model: Optional pressure-model identifier.
            / 可选的储层压力模型标识符。
        seawater_depth_m: Optional seawater depth above the reservoir,
            in metres.
            / 可选的储层上方海水深度，单位为米。
        well_fracture_gradient_psi_per_ft: Optional fracture gradient
            at the well.
            / 可选的井筒破裂压力梯度，单位为磅每平方英寸每英尺。
        well_fracture_gradient_reference_depth_m: Reference depth
            for the gradient.
            / 破裂压力梯度对应的参考深度，单位为米。
        well_fracture_pressure_bar: Optional well fracture pressure, in bar.
            / 可选的井筒破裂压力，单位为巴。
        well_bottomhole_pressure_safety_factor: Optional pressure
            safety factor.
            / 可选的井底压力安全系数。
        well_bottomhole_pressure_limit_bar: Optional bottomhole pressure limit.
            / 可选的井底压力上限，单位为巴。
        line_source_parameters: Optional line-source model parameters.
            / 可选的线源模型参数。
        line_source_observation_radii_m: Radii used by the line-source model.
            / 线源模型使用的观测半径，单位为米。
        line_source_well_distances_m: Pairwise well distances by reservoir.
            / 按储层记录的注入井两两距离，单位为米。
        line_source_parameter_status: Availability status of line-source
            inputs.
            / 线源模型输入参数的可用状态。
    """

    entity_id: str
    storage_capacity_t: float
    initial_pressure_bar: float
    pressure_at_capacity_bar: float
    max_pressure_bar: float
    depth_m: float | None = None
    reservoir_pressure_model: str | None = None
    seawater_depth_m: float | None = None
    well_fracture_gradient_psi_per_ft: float | None = None
    well_fracture_gradient_reference_depth_m: float | None = None
    well_fracture_pressure_bar: float | None = None
    well_bottomhole_pressure_safety_factor: float | None = None
    well_bottomhole_pressure_limit_bar: float | None = None
    line_source_parameters: LineSourceParameters | None = None
    line_source_observation_radii_m: tuple[float, ...] = ()
    line_source_well_distances_m: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    line_source_parameter_status: dict[str, str] = field(default_factory=dict)

    def pressure_bar(self, stored_t: float) -> float:
        """Estimate reservoir pressure from the stored CO₂ quantity.

        The model linearly interpolates pressure between the initial and
        full-capacity pressures, while clamping the fill fraction to [0, 1].

        根据封存的二氧化碳量估算储层压力。
        该模型在初始压力和满容量压力之间线性插值，并将填充比例限制在 [0, 1]。

        Args:
            stored_t: Stored CO₂ quantity in tonnes. / 封存量，单位为吨。

        Returns:
            Estimated reservoir pressure in bar. / 估算的储层压力，单位为巴。
        """
        # Restrict the pressure calculation to the physical fill range.
        # 将压力计算限制在物理可行的填充范围内。
        fill_fraction = max(0.0, min(1.0, stored_t / self.storage_capacity_t))
        return self.initial_pressure_bar + fill_fraction * (
            self.pressure_at_capacity_bar - self.initial_pressure_bar
        )

    def pressure_margin_bar(self, stored_t: float) -> float:
        """Return the remaining pressure margin for a stored CO₂ quantity.

        返回给定二氧化碳封存量下剩余的压力裕度。

        Args:
            stored_t: Stored CO₂ quantity in tonnes. / 封存量，单位为吨。

        Returns:
            Remaining pressure margin in bar. / 剩余压力裕度，单位为巴。
        """
        return self.max_pressure_bar - self.pressure_bar(stored_t)

    def pressure_limited_capacity_t(self) -> float:
        """Return the maximum storage quantity allowed by the pressure limit.

        If pressure does not increase with stored quantity, the full physical
        capacity is available. Otherwise, the method linearly scales capacity
        to the permitted pressure range.

        返回压力限制下允许的最大封存量。
        若压力不随封存量增加，则可使用全部物理容量；否则按允许压力范围线性缩放。

        Returns:
            Pressure-limited storage capacity in tonnes.
            / 受压力限制的封存容量，单位为吨。
        """
        pressure_span = (
            self.pressure_at_capacity_bar - self.initial_pressure_bar
        )
        if pressure_span <= 0:
            return self.storage_capacity_t

        # Clamp the pressure-derived fraction to the physical capacity range.
        # 将由压力计算得到的比例限制在物理容量范围内。
        pressure_fraction = (
            self.max_pressure_bar - self.initial_pressure_bar
        ) / pressure_span
        return self.storage_capacity_t * max(0.0, min(1.0, pressure_fraction))
