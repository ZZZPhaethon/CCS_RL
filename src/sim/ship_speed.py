"""按 Bjerketvedt et al. (2020) 的 STAwave-1 公式计算浪高影响下的船速。"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

KNOT_TO_MPS = 0.514444
DEFAULT_SEAWATER_DENSITY_KG_M3 = 1025.0
GRAVITY_MPS2 = 9.81


@dataclass(frozen=True)
class ShipSpeedParameters:
    """保存船速修正公式需要的船舶参数和物理常数。"""

    power_kw: float
    beam_m: float
    waterline_length_m: float
    design_speed_knots: float = 12.0
    seawater_density_kg_m3: float = DEFAULT_SEAWATER_DENSITY_KG_M3
    gravity_mps2: float = GRAVITY_MPS2


BJERKETVEDT_2020_SHIPS: dict[int, ShipSpeedParameters] = {
    3750: ShipSpeedParameters(power_kw=1500.0, beam_m=15.0, waterline_length_m=90.0),
    5000: ShipSpeedParameters(power_kw=2000.0, beam_m=16.0, waterline_length_m=95.0),
    7500: ShipSpeedParameters(power_kw=2500.0, beam_m=17.0, waterline_length_m=100.0),
}

# Northern Lights 真实 CO2 运输船参数，可代表 Northern Pioneer / Northern Pathfinder 同型船。
# STAwave-1 公式需要水线长 L_BWL；这里暂用已披露的全长 LOA 约 130 m 作为近似长度。
NORTHERN_LIGHTS_SHIP = ShipSpeedParameters(
    power_kw=5500.0,
    beam_m=21.2,
    waterline_length_m=130.0,
    design_speed_knots=14.0,
)


def knots_to_mps(speed_knots: float) -> float:
    """将节转换为米每秒。"""
    return speed_knots * KNOT_TO_MPS


def mps_to_knots(speed_mps: float) -> float:
    """将米每秒转换为节。"""
    return speed_mps / KNOT_TO_MPS


def calm_water_resistance_n(parameters: ShipSpeedParameters) -> float:
    """根据设计功率和设计船速反推静水阻力。"""
    power_w = parameters.power_kw * 1000.0
    design_speed_mps = knots_to_mps(parameters.design_speed_knots)
    return power_w / design_speed_mps


def added_wave_resistance_n(significant_wave_height_m: float, parameters: ShipSpeedParameters) -> float:
    """根据 STAwave-1 公式计算显著波高造成的附加阻力。"""
    if significant_wave_height_m < 0.0:
        raise ValueError("significant_wave_height_m must be non-negative")

    return (
        (1.0 / 16.0)
        * parameters.seawater_density_kg_m3
        * parameters.gravity_mps2
        * significant_wave_height_m**2
        * parameters.beam_m
        * math.sqrt(parameters.beam_m / parameters.waterline_length_m)
    )


def speed_mps(significant_wave_height_m: float, parameters: ShipSpeedParameters) -> float:
    """计算给定显著波高下的船速，结果单位为米每秒。"""
    power_w = parameters.power_kw * 1000.0
    total_resistance_n = calm_water_resistance_n(parameters) + added_wave_resistance_n(
        significant_wave_height_m,
        parameters,
    )
    return power_w / total_resistance_n


def speed_knots(significant_wave_height_m: float, parameters: ShipSpeedParameters) -> float:
    """计算给定显著波高下的船速，结果单位为节。"""
    return mps_to_knots(speed_mps(significant_wave_height_m, parameters))


def speed_factor(
    significant_wave_height_m: float,
    parameters: ShipSpeedParameters,
    *,
    nominal_speed_knots: float | None = None,
) -> float:
    """Return the simulator speed multiplier implied by significant wave height.

    ``PhysicalSimulator`` already consumes weather as a multiplier on the route's
    nominal speed. This helper bridges STAwave-1 absolute speed to that existing
    ``vessel_speed_factor`` interface.
    """
    reference_speed_knots = parameters.design_speed_knots if nominal_speed_knots is None else nominal_speed_knots
    if reference_speed_knots <= 0.0:
        raise ValueError("reference speed must be positive")
    return max(0.0, speed_knots(significant_wave_height_m, parameters) / reference_speed_knots)


def speed_factor_series(
    significant_wave_heights_m: Iterable[float],
    parameters: ShipSpeedParameters,
    *,
    nominal_speed_knots: float | None = None,
) -> list[float]:
    """Convert an hourly significant-wave-height trajectory to speed factors."""
    return [
        speed_factor(height_m, parameters, nominal_speed_knots=nominal_speed_knots)
        for height_m in significant_wave_heights_m
    ]
