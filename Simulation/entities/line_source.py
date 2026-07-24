"""Implement line-source pressure calculations for CO₂ injection wells.

实现用于二氧化碳注入井的线源压力计算。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Unit conversion and numerical constants used by the line-source equation.
# 线源方程中使用的单位换算和数值常量。
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
MILLIDARCY_TO_M2 = 9.869233e-16
PA_PER_BAR = 100_000.0
EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class LineSourceParameters:
    """Store the formation and fluid properties for a line-source model.

    存储线源模型所需的地层和流体属性。

    Attributes:
        initial_pressure_bar: Initial reservoir pressure, in bar.
            / 初始储层压力，单位为巴。
        permeability_md: Formation permeability, in millidarcies.
            / 地层渗透率，单位为毫达西。
        thickness_m: Effective formation thickness, in metres.
            / 有效地层厚度，单位为米。
        porosity_fraction: Dimensionless formation porosity.
            / 无量纲地层孔隙度。
        total_compressibility_1_pa: Total compressibility, in inverse pascals.
            / 总压缩系数，单位为帕斯卡的倒数。
        viscosity_pa_s: CO₂ viscosity, in pascal-seconds.
            / 二氧化碳黏度，单位为帕斯卡秒。
        co2_density_kg_m3: CO₂ density, in kilograms per cubic metre.
            / 二氧化碳密度，单位为千克每立方米。
        well_radius_m: Injection-well radius, in metres.
            / 注入井半径，单位为米。
        skin: Dimensionless well skin factor. / 无量纲井筒表皮系数。
    """

    initial_pressure_bar: float
    permeability_md: float
    thickness_m: float
    porosity_fraction: float
    total_compressibility_1_pa: float
    viscosity_pa_s: float
    co2_density_kg_m3: float
    well_radius_m: float
    skin: float = 0.0


def annual_mt_to_kg_s(rate_mtpa: float) -> float:
    """Convert an annual injection rate from Mt/year to kg/s.

    将年度注入速率从百万吨每年转换为千克每秒。
    """
    return rate_mtpa * 1_000_000_000.0 / SECONDS_PER_YEAR


def pressure_at_radius_bar(
    parameters: LineSourceParameters,
    injection_rate_mtpa: float,
    *,
    elapsed_days: float,
    radius_m: float,
) -> float:
    """Return reservoir pressure at an observation radius for a constant rate.

    返回恒定注入速率下观测半径处的储层压力。

    Args:
        parameters: Formation and fluid properties. / 地层和流体属性。
        injection_rate_mtpa: Injection rate in Mt/year. / 注入速率，单位为百万吨每年。
        elapsed_days: Time since injection started, in days.
            / 注入开始后的时间，单位为天。
        radius_m: Observation radius from the well, in metres.
            / 距井筒的观测半径，单位为米。

    Returns:
        Reservoir pressure in bar. / 储层压力，单位为巴。
    """
    pressure_change_bar = _pressure_change_bar(
        parameters,
        injection_rate_mtpa,
        elapsed_days=elapsed_days,
        radius_m=radius_m,
        skin=0.0,
    )
    return parameters.initial_pressure_bar + pressure_change_bar


def bottomhole_pressure_bar(
    parameters: LineSourceParameters,
    injection_rate_mtpa: float,
    *,
    elapsed_days: float,
) -> float:
    """Return bottomhole pressure for a constant injection rate.

    返回恒定注入速率下的井底压力。
    """
    pressure_change_bar = _pressure_change_bar(
        parameters,
        injection_rate_mtpa,
        elapsed_days=elapsed_days,
        radius_m=parameters.well_radius_m,
        skin=parameters.skin,
    )
    return parameters.initial_pressure_bar + pressure_change_bar


def variable_rate_pressure_at_radius_bar(
    parameters: LineSourceParameters,
    rate_history_mtpa: list[tuple[float, float]],
    *,
    elapsed_days: float,
    radius_m: float,
) -> float:
    """Return pressure at an observation radius for a rate history.

    返回给定注入速率历史下观测半径处的压力。
    """
    pressure_change_bar = _variable_rate_pressure_change_bar(
        parameters,
        rate_history_mtpa,
        elapsed_days=elapsed_days,
        radius_m=radius_m,
        skin=0.0,
    )
    return parameters.initial_pressure_bar + pressure_change_bar


def variable_rate_bottomhole_pressure_bar(
    parameters: LineSourceParameters,
    rate_history_mtpa: list[tuple[float, float]],
    *,
    elapsed_days: float,
) -> float:
    """Return bottomhole pressure for a piecewise-constant rate history.

    返回分段恒定注入速率历史下的井底压力。
    """
    pressure_change_bar = _variable_rate_pressure_change_bar(
        parameters,
        rate_history_mtpa,
        elapsed_days=elapsed_days,
        radius_m=parameters.well_radius_m,
        skin=parameters.skin,
    )
    return parameters.initial_pressure_bar + pressure_change_bar


def multiwell_bottomhole_pressures_bar(
    parameters: LineSourceParameters,
    injection_rates_mtpa_by_well: dict[str, float],
    *,
    elapsed_days: float,
    well_distances_m: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Return multiwell bottomhole pressures with interference effects.

    Each well's pressure includes its own bottomhole response and the pressure
    changes induced at that well by every other active well.

    返回考虑井间干扰的多井井底压力。
    每口井的压力包括自身的井底响应以及其他活动井在该井处引起的压力变化。
    """
    pressures: dict[str, float] = {}
    for well_id, injection_rate_mtpa in injection_rates_mtpa_by_well.items():
        pressure_bar = bottomhole_pressure_bar(
            parameters,
            injection_rate_mtpa,
            elapsed_days=elapsed_days,
        )
        for source_well_id, source_rate_mtpa in (
            injection_rates_mtpa_by_well.items()
        ):
            if source_well_id == well_id or source_rate_mtpa == 0.0:
                continue
            distance_m = _well_distance_m(
                well_distances_m,
                well_id,
                source_well_id,
            )
            pressure_bar += (
                pressure_at_radius_bar(
                    parameters,
                    source_rate_mtpa,
                    elapsed_days=elapsed_days,
                    radius_m=distance_m,
                )
                - parameters.initial_pressure_bar
            )
        pressures[well_id] = pressure_bar
    return pressures


def multiwell_variable_rate_bottomhole_pressures_bar(
    parameters: LineSourceParameters,
    injection_rate_history_mtpa_by_well: dict[str, list[tuple[float, float]]],
    *,
    elapsed_days: float,
    well_distances_m: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Return multiwell pressures for piecewise-constant rate histories.

    返回分段恒定注入速率历史下的多井压力。
    """
    pressures: dict[str, float] = {}
    for well_id, rate_history in injection_rate_history_mtpa_by_well.items():
        pressure_bar = variable_rate_bottomhole_pressure_bar(
            parameters,
            rate_history,
            elapsed_days=elapsed_days,
        )
        for source_well_id, source_rate_history in (
            injection_rate_history_mtpa_by_well.items()
        ):
            if source_well_id == well_id or not source_rate_history:
                continue
            distance_m = _well_distance_m(
                well_distances_m,
                well_id,
                source_well_id,
            )
            pressure_bar += (
                variable_rate_pressure_at_radius_bar(
                    parameters,
                    source_rate_history,
                    elapsed_days=elapsed_days,
                    radius_m=distance_m,
                )
                - parameters.initial_pressure_bar
            )
        pressures[well_id] = pressure_bar
    return pressures


def load_line_source_parameters(path: str | Path) -> LineSourceParameters:
    """Load line-source parameters from a JSON configuration file.

    The JSON file must contain a ``line_source_inputs`` mapping.

    从 JSON 配置文件加载线源模型参数。
    该文件必须包含 ``line_source_inputs`` 映射。
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _parameters_from_mapping(payload["line_source_inputs"])


def _parameters_from_mapping(payload: dict[str, Any]) -> LineSourceParameters:
    """Build parameters from the recognised keys of a configuration mapping.

    根据配置映射中已识别的键构建参数对象。
    """
    fields = LineSourceParameters.__dataclass_fields__
    values = {key: payload[key] for key in fields if key in payload}
    return LineSourceParameters(**values)


def _well_distance_m(
    well_distances_m: dict[str, dict[str, float]],
    observer_well_id: str,
    source_well_id: str,
) -> float:
    """Return the positive distance between an observation and source well.

    返回观测井与源井之间的正距离。

    Raises:
        ValueError: If no distance is configured or the distance is
            non-positive.
            / 未配置距离或距离非正时引发异常。
    """
    try:
        distance_m = well_distances_m[observer_well_id][source_well_id]
    except KeyError as exc:
        message = (
            f"Missing distance between {observer_well_id} "
            f"and {source_well_id}"
        )
        raise ValueError(message) from exc
    if distance_m <= 0.0:
        raise ValueError("well distance must be positive")
    return distance_m


def _pressure_change_bar(
    parameters: LineSourceParameters,
    injection_rate_mtpa: float,
    *,
    elapsed_days: float,
    radius_m: float,
    skin: float,
) -> float:
    """Return the pressure change for a validated non-negative flow rate.

    返回经过校验的非负流量对应的压力变化。
    """
    if elapsed_days <= 0.0:
        raise ValueError("elapsed_days must be positive")
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if injection_rate_mtpa < 0.0:
        raise ValueError("injection_rate_mtpa must be non-negative")

    return _signed_pressure_change_bar(
        parameters,
        injection_rate_mtpa,
        elapsed_days=elapsed_days,
        radius_m=radius_m,
        skin=skin,
    )


def _variable_rate_pressure_change_bar(
    parameters: LineSourceParameters,
    rate_history_mtpa: list[tuple[float, float]],
    *,
    elapsed_days: float,
    radius_m: float,
    skin: float,
) -> float:
    """Superpose pressure changes caused by a piecewise-constant rate history.

    按叠加原理计算分段恒定注入速率历史引起的压力变化。
    """
    if elapsed_days <= 0.0:
        raise ValueError("elapsed_days must be positive")
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")

    pressure_change_bar = 0.0
    previous_rate_mtpa = 0.0
    previous_start_day = -math.inf
    for start_day, rate_mtpa in rate_history_mtpa:
        if start_day < 0.0:
            raise ValueError("rate history start time must be non-negative")
        if start_day < previous_start_day:
            raise ValueError("rate history must be sorted by start time")
        if rate_mtpa < 0.0:
            raise ValueError("rate history rates must be non-negative")
        previous_start_day = start_day
        if start_day >= elapsed_days:
            break
        delta_rate_mtpa = rate_mtpa - previous_rate_mtpa
        if delta_rate_mtpa != 0.0:
            pressure_change_bar += _signed_pressure_change_bar(
                parameters,
                delta_rate_mtpa,
                elapsed_days=elapsed_days - start_day,
                radius_m=radius_m,
                skin=skin,
            )
        previous_rate_mtpa = rate_mtpa
    return pressure_change_bar


def _signed_pressure_change_bar(
    parameters: LineSourceParameters,
    injection_rate_mtpa: float,
    *,
    elapsed_days: float,
    radius_m: float,
    skin: float,
) -> float:
    """Return signed line-source pressure change without rate validation.

    Signed rates are required here to superpose rate changes in a history.

    返回未校验速率时的带符号线源压力变化。
    此处需要带符号的速率，以叠加注入速率历史中的变化量。
    """
    # Convert mass rate and permeability to the units required by the equation.
    # 将质量流量和渗透率转换为方程所需的单位。
    q_m3_s = (
        annual_mt_to_kg_s(injection_rate_mtpa) / parameters.co2_density_kg_m3
    )
    permeability_m2 = parameters.permeability_md * MILLIDARCY_TO_M2
    elapsed_s = elapsed_days * 24.0 * 3600.0
    diffusivity_argument = (
        parameters.porosity_fraction
        * parameters.viscosity_pa_s
        * parameters.total_compressibility_1_pa
        * radius_m
        * radius_m
        / (4.0 * permeability_m2 * elapsed_s)
    )
    # Include the well skin factor in the dimensionless pressure response.
    # 在无量纲压力响应中加入井筒表皮系数。
    response = _exponential_integral_e1(diffusivity_argument) + 2.0 * skin
    pressure_change_pa = (
        q_m3_s
        * parameters.viscosity_pa_s
        * response
        / (4.0 * math.pi * permeability_m2 * parameters.thickness_m)
    )
    return pressure_change_pa / PA_PER_BAR


def _exponential_integral_e1(x: float) -> float:
    """Evaluate the exponential integral E₁(x) for positive ``x``.

    A power series is used for small arguments and a continued fraction for
    larger arguments to retain numerical accuracy.

    对正数 ``x`` 计算指数积分 E₁(x)。
    对较小自变量使用幂级数，对较大自变量使用连分式以保持数值精度。
    """
    if x <= 0.0:
        raise ValueError("x must be positive")
    if x <= 1.0:
        total = -EULER_GAMMA - math.log(x)
        term_power = 1.0
        factorial = 1.0
        sign = 1.0
        for n in range(1, 200):
            term_power *= x
            factorial *= n
            term = sign * term_power / (n * factorial)
            total += term
            if abs(term) < 1e-15:
                return total
            sign *= -1.0
        return total

    tiny = 1e-300
    b = x + 1.0
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        a = -float(i * i)
        b += 2.0
        d_denominator = a * d + b
        if abs(d_denominator) < tiny:
            d_denominator = tiny
        d = 1.0 / d_denominator
        c = b + a / c
        if abs(c) < tiny:
            c = tiny
        delta = c * d
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h * math.exp(-x)
