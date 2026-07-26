"""Risk diagnostics and gating for adaptive-greedy interventions.

用于 adaptive-greedy 干预的风险诊断与门控。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any

from sim.entities.emitter import Emitter
from sim.environment import CCSEnv


_EPS = 1e-9


@dataclass(frozen=True)
class AdaptiveRiskGateConfig:
    """Configure when adaptive greedy may override the rule.

    配置 adaptive greedy 何时可以覆盖规则动作。
    """

    hours_to_overflow_threshold_h: float = 48.0
    fill_ratio_threshold: float = 0.80
    weather_fill_ratio_threshold: float = 0.60
    weather_speed_threshold: float = 0.65
    weather_lookahead_h: int = 72
    require_dispatchable_empty_vessel: bool = True
    mask_all_interventions_outside_risk: bool = True

    def __post_init__(self) -> None:
        """Validate risk thresholds.

        校验风险阈值。
        """
        if self.hours_to_overflow_threshold_h < 0.0:
            raise ValueError(
                "hours_to_overflow_threshold_h must be non-negative."
            )
        for name in (
            "fill_ratio_threshold",
            "weather_fill_ratio_threshold",
            "weather_speed_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be inside [0, 1].")
        if self.weather_lookahead_h <= 0:
            raise ValueError("weather_lookahead_h must be positive.")


def adaptive_risk_snapshot(
    env: CCSEnv,
    config: AdaptiveRiskGateConfig,
) -> dict[str, Any]:
    """Return risk signals and the resulting adaptive gate decision.

    返回风险信号以及最终 adaptive 门控判断。
    """
    if env.simulator is None or env.scenario is None:
        raise RuntimeError("Call env.reset() before checking adaptive risk.")
    fill_ratios: dict[str, float] = {}
    overflow_hours: dict[str, float] = {}
    state = env.simulator.state
    for emitter_id in env.emitter_ids:
        emitter = env.network.entities[emitter_id]
        if not isinstance(emitter, Emitter):
            raise TypeError(f"{emitter_id} is not an Emitter.")
        inventory_t = float(
            state.entity_inventory_t.get(emitter_id, 0.0)
        )
        fill_ratios[emitter_id] = (
            inventory_t / max(_EPS, float(emitter.buffer_capacity_t))
        )
        availability = float(
            state.emitter_availability.get(
                emitter_id,
                emitter.availability,
            )
        )
        capture_tph = (
            float(emitter.nominal_capture_tph)
            * max(0.0, availability)
        )
        headroom_t = max(
            0.0,
            float(emitter.buffer_capacity_t) - inventory_t,
        )
        overflow_hours[emitter_id] = (
            inf if capture_tph <= _EPS else headroom_t / capture_tph
        )

    max_fill_ratio = max(fill_ratios.values(), default=0.0)
    min_hours_to_overflow = min(overflow_hours.values(), default=inf)
    forecast_speed_min = _forecast_fleet_speed_min(
        env,
        config.weather_lookahead_h,
    )
    dispatchable_empty_count = _dispatchable_empty_count(env)
    overflow_risk_active = float(env._overflow_risk_t()) > _EPS
    hours_risk = (
        min_hours_to_overflow
        <= config.hours_to_overflow_threshold_h
    )
    fill_risk = max_fill_ratio >= config.fill_ratio_threshold
    weather_risk = (
        max_fill_ratio >= config.weather_fill_ratio_threshold
        and forecast_speed_min <= config.weather_speed_threshold
    )
    physical_risk = (
        overflow_risk_active
        or hours_risk
        or fill_risk
        or weather_risk
    )
    vessel_requirement_met = (
        dispatchable_empty_count > 0
        or not config.require_dispatchable_empty_vessel
    )
    allowed = physical_risk and vessel_requirement_met
    reasons = [
        name
        for name, active in (
            ("overflow_risk_active", overflow_risk_active),
            ("hours_to_overflow", hours_risk),
            ("high_fill", fill_risk),
            ("weather_risk", weather_risk),
        )
        if active
    ]
    if not vessel_requirement_met:
        reasons.append("no_dispatchable_empty_vessel")
    return {
        "adaptive_risk_gate_allowed": bool(allowed),
        "adaptive_risk_gate_reasons": reasons,
        "overflow_risk_active": bool(overflow_risk_active),
        "hours_risk_active": bool(hours_risk),
        "fill_risk_active": bool(fill_risk),
        "weather_risk_active": bool(weather_risk),
        "min_hours_to_overflow": float(min_hours_to_overflow),
        "max_emitter_fill_ratio": float(max_fill_ratio),
        "forecast_fleet_speed_min": float(forecast_speed_min),
        "dispatchable_empty_vessels": int(dispatchable_empty_count),
        "emitter_fill_ratios": fill_ratios,
        "emitter_hours_to_overflow": overflow_hours,
    }


def _dispatchable_empty_count(env: CCSEnv) -> int:
    """Count empty berthed vessels with a legal emitter departure.

    统计可以合法驶向排放源的空载靠泊船舶。
    """
    assert env.simulator is not None
    state = env.simulator.state
    masks = env.vessel_action_mask()
    emitter_actions = [
        env.vessel_go_emitter_action(emitter_id)
        for emitter_id in env.emitter_ids
    ]
    count = 0
    for position, vessel_id in enumerate(env.vessel_ids):
        vessel_state = env.simulator.vessel_states[vessel_id]
        cargo_t = float(
            state.entity_inventory_t.get(vessel_id, 0.0)
        )
        if (
            vessel_state["mode"] == "berthed"
            and cargo_t <= _EPS
            and any(
                bool(masks[position][action])
                for action in emitter_actions
            )
        ):
            count += 1
    return count


def _forecast_fleet_speed_min(
    env: CCSEnv,
    window_h: int,
) -> float:
    """Return minimum fleet speed factor in the available forecast.

    返回可用预测时域内的最小船速系数。
    """
    assert env.simulator is not None
    assert env.scenario is not None
    scenario = env.scenario
    start = scenario.step_index(env.simulator.state.time_h)
    steps = max(
        1,
        int(round(window_h / scenario.time_step_hours)),
    )
    end = min(scenario.n_steps, start + steps)
    values: list[float] = []
    for vessel_id in env.vessel_ids:
        series = scenario.vessel_speed_factor.get(vessel_id, ())
        for index in range(start, max(start + 1, end)):
            values.append(
                float(series[index])
                if 0 <= index < len(series)
                else 1.0
            )
    return min(values, default=1.0)

