"""Evaluation metrics - one common scorecard for every method.

A controller (heuristic, MILP or RL policy) is just a callable
``policy(env) -> {"vessels": [...], "wells": [...]}``. :func:`run_episode` rolls it out against a
:class:`~sim.environment.CCSEnv` and returns an :class:`EpisodeMetrics` with the physical
and economic KPIs of section 13; :func:`evaluate` repeats this across seeds and
aggregates mean/std so baselines and policies can be compared on equal footing.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Callable

from .environment import CCSEnv

Policy = Callable[[CCSEnv], dict[str, list]]

_PRESSURE_RISK_MARGIN_FRACTION = 0.10
_EPS = 1e-9


@dataclass
class EpisodeMetrics:
    """Physical + economic KPIs for one episode."""

    horizon_hours: float = 0.0
    storage_target_rate: float = 0.0
    elapsed_hours: float = 0.0      # wall-clock simulated hours actually run

    # Mass-flow KPIs (tonnes).
    captured_t: float = 0.0
    stored_t: float = 0.0
    vented_t: float = 0.0
    in_transit_t: float = 0.0          # captured CO2 not yet stored at episode end
    in_transit_growth_t: float = 0.0   # end minus start captured-but-not-yet-stored CO2
    loss_rate: float = 0.0             # vented / captured: short-horizon truth
    storage_rate: float = 0.0          # stored / captured: only meaningful over a long horizon
    annual_storage_gap_t: float = 0.0  # target*captured - stored (long-horizon obligation only)

    # Economic KPIs (EUR).
    vessel_fuel: float = 0.0
    conditioning: float = 0.0
    reconditioning: float = 0.0
    loading: float = 0.0
    unloading: float = 0.0
    operating_cost: float = 0.0
    vent_penalty: float = 0.0
    storage_shortfall_penalty: float = 0.0
    total_cost: float = 0.0
    net: float = 0.0
    cost_per_stored_t: float | None = None
    total_cost_per_stored_t: float | None = None

    # Operational quality / resilience KPIs.
    throttle_hours: int = 0
    well_switch_count: int = 0
    berth_wait_vessel_hours: int = 0
    pressure_risk_hours: int = 0
    min_pressure_margin_fraction: float = 1.0
    longest_venting_streak_hours: int = 0

    # Reward bookkeeping.
    total_reward: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def report(self) -> str:
        lines = [
            f"Episode ({self.horizon_hours:.0f} h cap, target {self.storage_target_rate:.0%})",
            f"  ran                  : {self.elapsed_hours:.0f} h",
            f"  stored               : {self.stored_t:,.0f} t  (of {self.captured_t:,.0f} t captured)",
            f"  vented (lost)        : {self.vented_t:,.0f} t   loss rate {self.loss_rate:.2%}",
            f"  in-transit inventory : {self.in_transit_t:,.0f} t   "
            f"(changed {self.in_transit_growth_t:+,.0f} t this episode)",
            f"  storage rate         : {self.storage_rate:.1%}  [long-horizon KPI]",
            f"  annual storage gap   : {self.annual_storage_gap_t:,.0f} t  [long-horizon KPI]",
            f"  operating cost       : EUR {self.operating_cost:,.0f}",
            f"  vent penalty         : EUR {self.vent_penalty:,.0f}",
            f"  shortfall penalty    : EUR {self.storage_shortfall_penalty:,.0f}",
            f"  total cost           : EUR {self.total_cost:,.0f}",
            f"  net                  : EUR {self.net:,.0f}",
            f"  cost / stored t      : "
            + ("n/a" if self.cost_per_stored_t is None else f"EUR {self.cost_per_stored_t:,.1f}"),
            f"  throttle hours       : {self.throttle_hours}",
            f"  well start/stops     : {self.well_switch_count}",
            f"  berth-wait ship-hours: {self.berth_wait_vessel_hours}",
            f"  pressure-risk hours  : {self.pressure_risk_hours} "
            f"(min margin {self.min_pressure_margin_fraction:.1%})",
            f"  longest vent streak  : {self.longest_venting_streak_hours} h",
            f"  total reward         : {self.total_reward:,.2f}",
        ]
        return "\n".join(lines)


class _MetricsRecorder:
    """Accumulates KPIs by observing the env after each step."""

    def __init__(self, env: CCSEnv) -> None:
        self.env = env
        self.total_reward = 0.0
        self.throttle_hours = 0
        self.well_switch_count = 0
        self.berth_wait_vessel_hours = 0
        self.pressure_risk_hours = 0
        self.min_pressure_margin_fraction = 1.0
        self.longest_venting_streak_hours = 0
        self._venting_streak = 0
        self._well_active: dict[str, bool] = {wid: False for wid in env.well_ids}
        self._prev_cargo: dict[str, float] = {vid: 0.0 for vid in env.vessel_ids}

    def record_step(self, reward: float, info: dict) -> None:
        env = self.env
        state = env.simulator.state
        self.total_reward += reward

        if "flow_clipped" in info.get("violations", []):
            self.throttle_hours += 1

        for well_id in env.well_ids:
            active = state.last_injection_flow_tph.get(well_id, 0.0) > _EPS
            if active != self._well_active[well_id]:
                self.well_switch_count += 1
                self._well_active[well_id] = active

        for vessel_id in env.vessel_ids:
            cargo = state.entity_inventory_t.get(vessel_id, 0.0)
            berth = state.vessel_berths.get(vessel_id)
            at_terminal = berth in env.terminal_ids
            did_not_unload = cargo >= self._prev_cargo[vessel_id] - _EPS
            if at_terminal and cargo > _EPS and did_not_unload:
                self.berth_wait_vessel_hours += 1
            self._prev_cargo[vessel_id] = cargo

        margin_fraction = self._min_reservoir_margin_fraction()
        self.min_pressure_margin_fraction = min(self.min_pressure_margin_fraction, margin_fraction)
        if margin_fraction < _PRESSURE_RISK_MARGIN_FRACTION:
            self.pressure_risk_hours += 1

        vented = info.get("economics", {}).get("vented_t", 0.0)
        if vented > _EPS:
            self._venting_streak += 1
            self.longest_venting_streak_hours = max(self.longest_venting_streak_hours, self._venting_streak)
        else:
            self._venting_streak = 0

    def _min_reservoir_margin_fraction(self) -> float:
        env = self.env
        state = env.simulator.state
        fractions = []
        for reservoir_id in env.reservoir_ids:
            reservoir = env.network.entities[reservoir_id]
            span = reservoir.max_pressure_bar - reservoir.initial_pressure_bar
            if span <= 0:
                continue
            inv = state.entity_inventory_t.get(reservoir_id, 0.0)
            fractions.append(max(0.0, reservoir.pressure_margin_bar(inv) / span))
        return min(fractions) if fractions else 1.0

    def result(self) -> EpisodeMetrics:
        env = self.env
        ledger = env.ledger
        captured = env.cumulative_captured_t
        stored = env.cumulative_stored_t
        in_transit = env._in_transit_inventory()
        annual_gap_t = max(0.0, env.config.storage_target_rate * captured - stored)
        cost_per_stored = ledger.operating_cost / stored if stored > _EPS else None
        total_cost_per_stored = ledger.total_cost / stored if stored > _EPS else None
        return EpisodeMetrics(
            horizon_hours=env.n_steps * env.network.time_step_hours,
            storage_target_rate=env.config.storage_target_rate,
            elapsed_hours=env.t * env.network.time_step_hours,
            captured_t=captured,
            stored_t=stored,
            vented_t=ledger.vented_t,
            in_transit_t=in_transit,
            in_transit_growth_t=in_transit - env.initial_in_transit_t,
            loss_rate=env.loss_rate(),
            storage_rate=env.storage_rate(),
            annual_storage_gap_t=annual_gap_t,
            vessel_fuel=ledger.vessel_fuel,
            conditioning=ledger.conditioning,
            reconditioning=ledger.reconditioning,
            loading=ledger.loading,
            unloading=ledger.unloading,
            operating_cost=ledger.operating_cost,
            vent_penalty=ledger.vent_penalty,
            storage_shortfall_penalty=ledger.storage_shortfall_penalty,
            total_cost=ledger.total_cost,
            net=ledger.net,
            cost_per_stored_t=cost_per_stored,
            total_cost_per_stored_t=total_cost_per_stored,
            throttle_hours=self.throttle_hours,
            well_switch_count=self.well_switch_count,
            berth_wait_vessel_hours=self.berth_wait_vessel_hours,
            pressure_risk_hours=self.pressure_risk_hours,
            min_pressure_margin_fraction=self.min_pressure_margin_fraction,
            longest_venting_streak_hours=self.longest_venting_streak_hours,
            total_reward=self.total_reward,
        )


def run_episode(env: CCSEnv, policy: Policy, seed: int | None = None) -> EpisodeMetrics:
    """Roll out ``policy`` for one episode and return its KPIs."""
    env.reset(seed=seed)
    recorder = _MetricsRecorder(env)
    done = False
    while not done:
        action = policy(env)
        _obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        recorder.record_step(reward, info)
    return recorder.result()


def evaluate(
    env: CCSEnv,
    policy: Policy,
    seeds: list[int],
) -> tuple[list[EpisodeMetrics], dict[str, dict[str, float]]]:
    """Run a policy across seeds and return per-episode metrics plus mean/std."""
    episodes = [run_episode(env, policy, seed=seed) for seed in seeds]
    return episodes, aggregate_metrics(episodes)


def format_fixed_horizon(name_to_summary: dict[str, dict[str, dict[str, float]]]) -> str:
    """Render a fixed-horizon comparison: stored tonnes, loss, and cost."""
    header = f"{'policy':16} {'hours':>8} {'stored t':>10} {'total EUR':>14} {'op EUR':>12} {'vented t':>10}"
    lines = [header, "-" * len(header)]
    for name, s in name_to_summary.items():
        total_summary = s["total_cost"] if "total_cost" in s else s["operating_cost"]
        total_cost = total_summary["mean"]
        lines.append(
            f"{name:16} {s['elapsed_hours']['mean']:8.0f} "
            f"{s['stored_t']['mean']:10,.0f} "
            f"{total_cost:14,.0f} {s['operating_cost']['mean']:12,.0f} "
            f"{s['vented_t']['mean']:10,.0f}"
        )
    return "\n".join(lines)


def aggregate_metrics(episodes: list[EpisodeMetrics]) -> dict[str, dict[str, float]]:
    """Mean/std of every numeric KPI across a list of episodes."""
    if not episodes:
        return {}
    summary: dict[str, dict[str, float]] = {}
    sample = episodes[0].as_dict()
    for key, value in sample.items():
        # Booleans aggregate as fractions when a future KPI needs that treatment.
        if not isinstance(value, (int, float, bool)):
            continue
        values = [float(getattr(ep, key)) for ep in episodes if getattr(ep, key) is not None]
        if not values:
            continue
        summary[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary

