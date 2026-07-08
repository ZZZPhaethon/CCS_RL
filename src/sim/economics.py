"""Operational cost model for the ship-based CCS network.

The current research model keeps only the variable costs that are directly tied
to short-horizon operating decisions: sailing fuel, source-side CO2
conditioning, terminal-side reconditioning, loading/unloading hoteling fuel,
and the carbon value of vented CO2. Storage shortfall is retained as the
storage-obligation signal rather than a market price.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .entities.emitter import Emitter
from .entities.pipeline import Pipeline
from .entities.state import StepResult
from .entities.terminal import Terminal
from .entities.vessel import Vessel


@dataclass(frozen=True)
class EconomicParameters:
    """Variable operational rates in EUR (capex and bundled tariffs excluded)."""

    currency: str = "EUR"

    # Carbon value used to price vented/lost CO2.
    carbon_price_eur_per_t: float = 80.0

    # Ship fuel assumptions from the trip-cost fuel calculation. The legacy
    # field name is kept for compatibility; defaults now use the Northern Lights
    # Wartsila 31DF LNG/gas-mode baseline.
    ship_fuel_cost_hfo_eur_per_t: float = 600.0
    main_engine_fuel_use_kg_per_kwh: float = 0.148
    main_engine_power_kw: float = 5500.0
    cruise_power_fraction: float = 0.85
    hoteling_power_fraction: float = 0.05

    # Source-side conditioning before ship export. Default is the 15 bar case;
    # use 8.45 for the 7 bar case when running that scenario.
    conditioning_eur_per_t: float = 7.82

    # Terminal-side adjustment to pipeline/injection conditions.
    reconditioning_eur_per_t: float = 0.41

    # Diagnostic storage-target shortfall weight. Default is zero so shortfall is
    # reported as a KPI instead of being treated as an operating cost.
    storage_shortfall_eur_per_t: float = 0.0

    @property
    def vessel_fuel_eur_per_h_sailing(self) -> float:
        return self.main_engine_fuel_cost_eur(self.cruise_power_fraction, hours=1.0)

    @property
    def hoteling_fuel_eur_per_h(self) -> float:
        return self.main_engine_fuel_cost_eur(self.hoteling_power_fraction, hours=1.0)

    def main_engine_fuel_cost_eur(self, power_fraction: float, hours: float) -> float:
        fuel_t = (
            self.main_engine_power_kw
            * max(0.0, power_fraction)
            * max(0.0, hours)
            * self.main_engine_fuel_use_kg_per_kwh
            / 1000.0
        )
        return fuel_t * self.ship_fuel_cost_hfo_eur_per_t

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["vessel_fuel_eur_per_h_sailing"] = self.vessel_fuel_eur_per_h_sailing
        data["hoteling_fuel_eur_per_h"] = self.hoteling_fuel_eur_per_h
        return data


@dataclass
class StepEconomics:
    """Per-step economic breakdown in EUR (costs positive)."""

    vessel_fuel: float = 0.0
    conditioning: float = 0.0
    reconditioning: float = 0.0
    loading: float = 0.0
    unloading: float = 0.0
    vent_penalty: float = 0.0

    # Tonnage bookkeeping for downstream KPIs.
    stored_t: float = 0.0
    vented_t: float = 0.0
    conditioned_t: float = 0.0
    reconditioned_t: float = 0.0
    loaded_t: float = 0.0
    unloaded_t: float = 0.0
    handled_t: float = 0.0

    @property
    def operating_cost(self) -> float:
        """Cash operating cost excluding penalties."""
        return self.vessel_fuel + self.conditioning + self.reconditioning + self.loading + self.unloading

    @property
    def total_cost(self) -> float:
        """Operating cost plus penalties."""
        return self.operating_cost + self.vent_penalty

    @property
    def net(self) -> float:
        """Negative total cost, used as the economic reward term."""
        return -self.total_cost

    def as_dict(self) -> dict[str, object]:
        return {
            "vessel_fuel": self.vessel_fuel,
            "conditioning": self.conditioning,
            "reconditioning": self.reconditioning,
            "loading": self.loading,
            "unloading": self.unloading,
            "vent_penalty": self.vent_penalty,
            "operating_cost": self.operating_cost,
            "total_cost": self.total_cost,
            "net": self.net,
            "stored_t": self.stored_t,
            "vented_t": self.vented_t,
            "conditioned_t": self.conditioned_t,
            "reconditioned_t": self.reconditioned_t,
            "loaded_t": self.loaded_t,
            "unloaded_t": self.unloaded_t,
            "handled_t": self.handled_t,
        }


class CostModel:
    """Prices a single simulation step from its :class:`StepResult`."""

    def __init__(self, parameters: EconomicParameters | None = None) -> None:
        self.parameters = parameters or EconomicParameters()

    def evaluate_step(self, network, step_result: StepResult) -> StepEconomics:
        params = self.parameters
        state = step_result.state
        hours = network.time_step_hours

        vessel_ids = list(network._entities_of_type(Vessel))
        sailing_count = sum(1 for vid in vessel_ids if vid not in state.vessel_berths)

        stored_t = sum(state.last_injection_flow_tph.values()) * hours
        vented_t = sum(state.last_vent_tph.values()) * hours
        loaded_t, unloaded_t, loading_h, unloading_h = self._loading_unloading_activity(network, step_result.flows_t)
        conditioned_t = loaded_t
        reconditioned_t = self._pipeline_transfer_tonnes(network, step_result.flows_t) or stored_t
        handled_t = loaded_t + unloaded_t

        return StepEconomics(
            vessel_fuel=sailing_count * params.vessel_fuel_eur_per_h_sailing * hours,
            conditioning=conditioned_t * params.conditioning_eur_per_t,
            reconditioning=reconditioned_t * params.reconditioning_eur_per_t,
            loading=loading_h * params.hoteling_fuel_eur_per_h,
            unloading=unloading_h * params.hoteling_fuel_eur_per_h,
            vent_penalty=vented_t * params.carbon_price_eur_per_t,
            stored_t=stored_t,
            vented_t=vented_t,
            conditioned_t=conditioned_t,
            reconditioned_t=reconditioned_t,
            loaded_t=loaded_t,
            unloaded_t=unloaded_t,
            handled_t=handled_t,
        )

    def storage_shortfall_penalty(
        self,
        cumulative_captured_t: float,
        cumulative_stored_t: float,
        target_rate: float,
    ) -> float:
        """Penalty for falling below a required storage-rate obligation.

        ``target_rate`` is the contracted fraction of captured CO2 that must be
        safely stored (e.g. 0.9). The shortfall is the tonnage gap to that
        obligation, priced above the venting penalty so the constraint dominates
        pure cost minimisation.
        """
        required_t = max(0.0, target_rate) * max(0.0, cumulative_captured_t)
        shortfall_t = max(0.0, required_t - cumulative_stored_t)
        return shortfall_t * self.parameters.storage_shortfall_eur_per_t

    def _loading_unloading_activity(self, network, flows_t: dict[tuple[str, str], float]) -> tuple[float, float, float, float]:
        loaded_t = 0.0
        unloaded_t = 0.0
        loading_h = 0.0
        unloading_h = 0.0
        for (source_id, target_id), amount_t in flows_t.items():
            source = network.entities.get(source_id)
            target = network.entities.get(target_id)
            if isinstance(source, Emitter) and isinstance(target, Vessel):
                loaded_t += amount_t
                loading_rate_tph = min(source.loading_rate_tph, target.loading_rate_tph)
                loading_h += self._transfer_hours(amount_t, loading_rate_tph, network.time_step_hours)
            elif isinstance(source, Vessel) and isinstance(target, Terminal):
                unloaded_t += amount_t
                unloading_h += self._transfer_hours(amount_t, source.unloading_rate_tph, network.time_step_hours)
        return loaded_t, unloaded_t, loading_h, unloading_h

    def _transfer_hours(self, amount_t: float, rate_tph: float, max_hours: float) -> float:
        if amount_t <= 0.0 or rate_tph <= 0.0 or max_hours <= 0.0:
            return 0.0
        return min(max_hours, amount_t / rate_tph)

    def _pipeline_transfer_tonnes(self, network, flows_t: dict[tuple[str, str], float]) -> float:
        transferred_t = 0.0
        for (source_id, target_id), amount_t in flows_t.items():
            source = network.entities.get(source_id)
            target = network.entities.get(target_id)
            if isinstance(source, Terminal) and isinstance(target, Pipeline):
                transferred_t += amount_t
        return transferred_t


@dataclass
class EconomicLedger:
    """Accumulates per-step economics across an episode for reporting."""

    vessel_fuel: float = 0.0
    conditioning: float = 0.0
    reconditioning: float = 0.0
    loading: float = 0.0
    unloading: float = 0.0
    vent_penalty: float = 0.0
    storage_shortfall_penalty: float = 0.0
    stored_t: float = 0.0
    vented_t: float = 0.0
    conditioned_t: float = 0.0
    reconditioned_t: float = 0.0
    loaded_t: float = 0.0
    unloaded_t: float = 0.0
    handled_t: float = 0.0

    def add(self, step: StepEconomics) -> None:
        self.vessel_fuel += step.vessel_fuel
        self.conditioning += step.conditioning
        self.reconditioning += step.reconditioning
        self.loading += step.loading
        self.unloading += step.unloading
        self.vent_penalty += step.vent_penalty
        self.stored_t += step.stored_t
        self.vented_t += step.vented_t
        self.conditioned_t += step.conditioned_t
        self.reconditioned_t += step.reconditioned_t
        self.loaded_t += step.loaded_t
        self.unloaded_t += step.unloaded_t
        self.handled_t += step.handled_t

    @property
    def operating_cost(self) -> float:
        return self.vessel_fuel + self.conditioning + self.reconditioning + self.loading + self.unloading

    @property
    def total_cost(self) -> float:
        return self.operating_cost + self.vent_penalty + self.storage_shortfall_penalty

    @property
    def net(self) -> float:
        return -self.total_cost

    def as_dict(self) -> dict[str, object]:
        return {
            "vessel_fuel": self.vessel_fuel,
            "conditioning": self.conditioning,
            "reconditioning": self.reconditioning,
            "loading": self.loading,
            "unloading": self.unloading,
            "vent_penalty": self.vent_penalty,
            "storage_shortfall_penalty": self.storage_shortfall_penalty,
            "operating_cost": self.operating_cost,
            "total_cost": self.total_cost,
            "net": self.net,
            "stored_t": self.stored_t,
            "vented_t": self.vented_t,
            "conditioned_t": self.conditioned_t,
            "reconditioned_t": self.reconditioned_t,
            "loaded_t": self.loaded_t,
            "unloaded_t": self.unloaded_t,
            "handled_t": self.handled_t,
        }
