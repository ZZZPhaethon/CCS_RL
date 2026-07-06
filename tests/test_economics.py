import unittest

from sim.economics import CostModel, EconomicLedger, EconomicParameters, StepEconomics
from sim.entities import (
    Emitter,
    InjectionWell,
    PhysicalState,
    Reservoir,
    Terminal,
    Vessel,
)
from sim.entities.state import StepResult
from sim.network import PhysicalNetwork


def _priced_network() -> PhysicalNetwork:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("brevik", nominal_capture_tph=100.0, buffer_capacity_t=1_000.0))
    network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
    network.add_entity(Vessel("ship_2", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
    network.add_entity(Terminal("oygarden", storage_capacity_t=2_000.0, berth_count=1))
    network.add_entity(InjectionWell("well_1", max_injection_tph=200.0))
    network.add_entity(
        Reservoir(
            "aurora",
            storage_capacity_t=1_000_000.0,
            initial_pressure_bar=100.0,
            pressure_at_capacity_bar=200.0,
            max_pressure_bar=200.0,
        )
    )
    return network


class EconomicParameterTests(unittest.TestCase):
    def test_defaults_match_simplified_cost_boundary(self):
        params = EconomicParameters()
        self.assertAlmostEqual(params.carbon_price_eur_per_t, 80.0)
        self.assertAlmostEqual(params.ship_fuel_cost_hfo_eur_per_t, 600.0)
        self.assertAlmostEqual(params.main_engine_fuel_use_kg_per_kwh, 0.148)
        self.assertAlmostEqual(params.main_engine_power_kw, 5500.0)
        self.assertAlmostEqual(params.cruise_power_fraction, 0.85)
        self.assertAlmostEqual(params.hoteling_power_fraction, 0.05)
        self.assertAlmostEqual(params.conditioning_eur_per_t, 7.82)
        self.assertAlmostEqual(params.reconditioning_eur_per_t, 0.41)
        self.assertAlmostEqual(params.storage_shortfall_eur_per_t, 0.0)
        self.assertFalse(hasattr(params, "backlog_penalty_eur_per_t"))

    def test_fuel_hourly_rates_are_derived_from_ship_energy_inputs(self):
        params = EconomicParameters()
        expected_sailing = 5500.0 * 0.85 * 0.148 / 1000.0 * 600.0
        expected_hoteling = 5500.0 * 0.05 * 0.148 / 1000.0 * 600.0
        self.assertAlmostEqual(params.vessel_fuel_eur_per_h_sailing, expected_sailing)
        self.assertAlmostEqual(params.hoteling_fuel_eur_per_h, expected_hoteling)

    def test_as_dict_exposes_simplified_rates(self):
        data = EconomicParameters().as_dict()
        self.assertIn("ship_fuel_cost_hfo_eur_per_t", data)
        self.assertIn("vessel_fuel_eur_per_h_sailing", data)
        self.assertIn("hoteling_fuel_eur_per_h", data)
        self.assertIn("conditioning_eur_per_t", data)
        self.assertNotIn("backlog_penalty_eur_per_t", data)
        self.assertNotIn("injection_cost_eur_per_t", data)
        self.assertNotIn("loading_eur_per_t", data)
        self.assertNotIn("unloading_eur_per_t", data)


class CostModelStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = _priced_network()
        self.model = CostModel()

    def _step_result(self) -> StepResult:
        # ship_1 berthed at the terminal, ship_2 at sea; one well injecting 100 t/h;
        # the emitter venting 10 t/h; 50 t loaded and 30 t unloaded this hour.
        state = PhysicalState(
            vessel_berths={"ship_1": "oygarden"},
            last_injection_flow_tph={"well_1": 100.0},
            last_vent_tph={"brevik": 10.0},
        )
        flows_t = {
            ("brevik", "ship_1"): 50.0,   # loading (emitter -> vessel)
            ("ship_1", "oygarden"): 30.0,  # unloading (vessel -> terminal)
            ("oygarden", "pipeline"): 100.0,  # not handling, ignored
        }
        return StepResult(state=state, flows_t=flows_t, violations=[], mass_balance_error_t=0.0)

    def test_step_breakdown_matches_calibrated_rates(self):
        econ = self.model.evaluate_step(self.network, self._step_result())
        params = EconomicParameters()

        self.assertAlmostEqual(econ.vessel_fuel, params.vessel_fuel_eur_per_h_sailing)      # only ship_2 sailing
        self.assertAlmostEqual(econ.conditioning, 50.0 * 7.82)
        self.assertAlmostEqual(econ.reconditioning, 100.0 * 0.41)
        self.assertAlmostEqual(econ.loading, (50.0 / 800.0) * params.hoteling_fuel_eur_per_h)
        self.assertAlmostEqual(econ.unloading, (30.0 / 800.0) * params.hoteling_fuel_eur_per_h)
        self.assertAlmostEqual(econ.vent_penalty, 10.0 * 80.0)
        self.assertNotIn("revenue_storage", econ.as_dict())
        self.assertAlmostEqual(econ.stored_t, 100.0)
        self.assertAlmostEqual(econ.vented_t, 10.0)
        self.assertAlmostEqual(econ.conditioned_t, 50.0)
        self.assertAlmostEqual(econ.reconditioned_t, 100.0)
        self.assertAlmostEqual(econ.loaded_t, 50.0)
        self.assertAlmostEqual(econ.unloaded_t, 30.0)
        self.assertAlmostEqual(econ.handled_t, 80.0)

    def test_net_is_negative_costs_and_penalties(self):
        econ = self.model.evaluate_step(self.network, self._step_result())
        params = EconomicParameters()
        expected_operating = (
            params.vessel_fuel_eur_per_h_sailing
            + 391.0
            + 41.0
            + (50.0 / 800.0) * params.hoteling_fuel_eur_per_h
            + (30.0 / 800.0) * params.hoteling_fuel_eur_per_h
        )
        self.assertAlmostEqual(econ.operating_cost, expected_operating)
        self.assertAlmostEqual(econ.total_cost, expected_operating + 800.0)
        self.assertAlmostEqual(econ.net, -(expected_operating + 800.0))

    def test_time_step_scales_time_based_costs(self):
        network = _priced_network()
        network.time_step_hours = 2.0
        econ = CostModel().evaluate_step(network, self._step_result())
        # Fuel and rate-derived storage tonnage double.
        self.assertAlmostEqual(econ.vessel_fuel, 1 * EconomicParameters().vessel_fuel_eur_per_h_sailing * 2.0)
        self.assertAlmostEqual(econ.stored_t, 200.0)
        self.assertAlmostEqual(econ.reconditioning, 200.0 * 0.41)


class ShortfallPenaltyTests(unittest.TestCase):
    def test_no_penalty_when_target_met(self):
        model = CostModel()
        penalty = model.storage_shortfall_penalty(
            cumulative_captured_t=1_000.0,
            cumulative_stored_t=950.0,
            target_rate=0.9,
        )
        self.assertEqual(penalty, 0.0)

    def test_penalty_prices_the_tonnage_gap(self):
        model = CostModel(EconomicParameters(storage_shortfall_eur_per_t=100.0))
        # Need 900 stored, only 800 -> 100 t short * 100 EUR/t.
        penalty = model.storage_shortfall_penalty(
            cumulative_captured_t=1_000.0,
            cumulative_stored_t=800.0,
            target_rate=0.9,
        )
        self.assertAlmostEqual(penalty, 10_000.0)


class EconomicLedgerTests(unittest.TestCase):
    def test_ledger_accumulates_steps(self):
        ledger = EconomicLedger()
        ledger.add(StepEconomics(conditioning=7.82, reconditioning=0.41, loading=1.0, unloading=2.0, stored_t=1.0))
        ledger.add(
            StepEconomics(
                conditioning=7.82,
                reconditioning=0.41,
                loading=1.0,
                unloading=2.0,
                stored_t=1.0,
                vent_penalty=80.0,
                vented_t=1.0,
            )
        )
        ledger.storage_shortfall_penalty = 500.0

        self.assertAlmostEqual(ledger.stored_t, 2.0)
        self.assertAlmostEqual(ledger.vented_t, 1.0)
        self.assertNotIn("revenue_storage", ledger.as_dict())
        self.assertAlmostEqual(ledger.operating_cost, 22.46)
        self.assertAlmostEqual(ledger.total_cost, 22.46 + 80.0 + 500.0)
        self.assertAlmostEqual(ledger.net, -(22.46 + 80.0 + 500.0))


if __name__ == "__main__":
    unittest.main()
