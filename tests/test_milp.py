import unittest

try:
    import pulp  # noqa: F401

    HAVE_PULP = True
except ImportError:
    HAVE_PULP = False

from sim import control
from sim.control import milp as milp_module
from sim.control.milp import extract_params
from sim.economics import EconomicParameters
from sim.environment import CCSEnv, CCSEnvConfig
from sim.entities import Emitter, InjectionWell, Pipeline, Reservoir, SubseaManifold, Terminal, Vessel
from sim.network import PhysicalNetwork
from sim.scenario_generation import Scenario, ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network


def _env(cap_hours: int = 400) -> CCSEnv:
    return CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=cap_hours)),
        config=CCSEnvConfig(episode_hours=cap_hours),
    )


def _unbalanced_source_env() -> CCSEnv:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("slow_source", nominal_capture_tph=5.0, buffer_capacity_t=2_000.0))
    network.add_entity(Emitter("fast_source", nominal_capture_tph=100.0, buffer_capacity_t=2_000.0))
    network.add_entity(Vessel("short_ship", capacity_t=1_000.0, loading_rate_tph=1_000.0, unloading_rate_tph=1_000.0, speed_knots=1.0))
    network.add_entity(Vessel("long_ship", capacity_t=1_000.0, loading_rate_tph=1_000.0, unloading_rate_tph=1_000.0, speed_knots=1.0))
    network.add_entity(Terminal("terminal", storage_capacity_t=5_000.0, berth_count=1))
    network.add_entity(Pipeline("pipeline", max_flow_tph=1_000.0))
    network.add_entity(SubseaManifold("manifold", max_flow_tph=1_000.0))
    network.add_entity(InjectionWell("well", max_injection_tph=1_000.0))
    network.add_entity(Reservoir("reservoir", storage_capacity_t=1e7, initial_pressure_bar=100.0, pressure_at_capacity_bar=200.0, max_pressure_bar=200.0))
    network.connect("slow_source", "short_ship")
    network.connect("fast_source", "long_ship")
    network.connect("short_ship", "terminal")
    network.connect("long_ship", "terminal")
    network.connect("terminal", "pipeline")
    network.connect("pipeline", "manifold")
    network.connect("manifold", "well")
    network.connect("well", "reservoir")
    return CCSEnv(
        network,
        {"slow_source": (0.0, 0.0), "fast_source": (0.0, 1.0), "terminal": (0.0, 2.0)},
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=300, randomize_initial_inventory=False)),
        config=CCSEnvConfig(episode_hours=300),
        routes={
            "short_ship": {"origin": "slow_source", "destination": "terminal", "distance_km": 1.852, "speed_knots": 1.0},
            "long_ship": {"origin": "fast_source", "destination": "terminal", "distance_km": 92.6, "speed_knots": 1.0},
        },
    )


def _streaming_unload_env() -> CCSEnv:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("source", nominal_capture_tph=0.0, buffer_capacity_t=2_000.0))
    network.add_entity(Vessel("ship", capacity_t=1_000.0, loading_rate_tph=1_000.0, unloading_rate_tph=500.0, speed_knots=1.0))
    network.add_entity(Terminal("terminal", storage_capacity_t=100.0, berth_count=1))
    network.add_entity(Pipeline("pipeline", max_flow_tph=500.0))
    network.add_entity(SubseaManifold("manifold", max_flow_tph=500.0))
    network.add_entity(InjectionWell("well", max_injection_tph=500.0))
    network.add_entity(Reservoir("reservoir", storage_capacity_t=1e7, initial_pressure_bar=100.0, pressure_at_capacity_bar=200.0, max_pressure_bar=200.0))
    network.connect("source", "ship")
    network.connect("ship", "terminal")
    network.connect("terminal", "pipeline")
    network.connect("pipeline", "manifold")
    network.connect("manifold", "well")
    network.connect("well", "reservoir")
    return CCSEnv(
        network,
        {"source": (0.0, 0.0), "terminal": (0.0, 1.0)},
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=8, randomize_initial_inventory=False)),
        config=CCSEnvConfig(episode_hours=8),
        routes={
            "ship": {"origin": "source", "destination": "terminal", "distance_km": 1.852, "speed_knots": 1.0},
        },
    )


def _tiny_buffer_no_delivery_env() -> CCSEnv:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("source", nominal_capture_tph=100.0, buffer_capacity_t=100.0))
    network.add_entity(Vessel("ship", capacity_t=1_000.0, loading_rate_tph=1_000.0, unloading_rate_tph=1_000.0, speed_knots=1.0))
    network.add_entity(Terminal("terminal", storage_capacity_t=1_000.0, berth_count=1))
    network.add_entity(Pipeline("pipeline", max_flow_tph=500.0))
    network.add_entity(SubseaManifold("manifold", max_flow_tph=500.0))
    network.add_entity(InjectionWell("well", max_injection_tph=500.0))
    network.add_entity(Reservoir("reservoir", storage_capacity_t=1e7, initial_pressure_bar=100.0, pressure_at_capacity_bar=200.0, max_pressure_bar=200.0))
    network.connect("source", "ship")
    network.connect("ship", "terminal")
    network.connect("terminal", "pipeline")
    network.connect("pipeline", "manifold")
    network.connect("manifold", "well")
    network.connect("well", "reservoir")
    return CCSEnv(
        network,
        {"source": (0.0, 0.0), "terminal": (0.0, 1.0)},
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=3, randomize_initial_inventory=False)),
        config=CCSEnvConfig(episode_hours=3),
        routes={
            "ship": {"origin": "source", "destination": "terminal", "distance_km": 1_852.0, "speed_knots": 1.0},
        },
    )


def _two_berth_parallel_env() -> CCSEnv:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("source", nominal_capture_tph=0.0, buffer_capacity_t=3_000.0))
    network.add_entity(Vessel("ship_a", capacity_t=1_000.0, loading_rate_tph=1_000.0, unloading_rate_tph=1_000.0, speed_knots=1.0))
    network.add_entity(Vessel("ship_b", capacity_t=1_000.0, loading_rate_tph=1_000.0, unloading_rate_tph=1_000.0, speed_knots=1.0))
    network.add_entity(Terminal("terminal", storage_capacity_t=3_000.0, berth_count=2))
    network.add_entity(Pipeline("pipeline", max_flow_tph=2_000.0))
    network.add_entity(SubseaManifold("manifold", max_flow_tph=2_000.0))
    network.add_entity(InjectionWell("well", max_injection_tph=2_000.0))
    network.add_entity(Reservoir("reservoir", storage_capacity_t=1e7, initial_pressure_bar=100.0, pressure_at_capacity_bar=200.0, max_pressure_bar=200.0))
    network.connect("source", "ship_a")
    network.connect("source", "ship_b")
    network.connect("ship_a", "terminal")
    network.connect("ship_b", "terminal")
    network.connect("terminal", "pipeline")
    network.connect("pipeline", "manifold")
    network.connect("manifold", "well")
    network.connect("well", "reservoir")
    return CCSEnv(
        network,
        {"source": (0.0, 0.0), "terminal": (0.0, 1.0)},
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=3, randomize_initial_inventory=False)),
        config=CCSEnvConfig(episode_hours=3),
        routes={
            "ship_a": {"origin": "source", "destination": "terminal", "distance_km": 1.852, "speed_knots": 1.0},
            "ship_b": {"origin": "source", "destination": "terminal", "distance_km": 1.852, "speed_knots": 1.0},
        },
    )


def _scenario_for_env(env: CCSEnv, horizon_h: int, *, capture: float = 1.0, wells_available: bool = True) -> Scenario:
    return Scenario(
        time_step_hours=1.0,
        n_steps=horizon_h,
        emitter_availability={emitter_id: [capture] * horizon_h for emitter_id in env.emitter_ids},
        vessel_speed_factor={vessel_id: [1.0] * horizon_h for vessel_id in env.vessel_ids},
        well_available={well_id: [wells_available] * horizon_h for well_id in env.well_ids},
        injectivity_factor={well_id: [1.0] * horizon_h for well_id in env.well_ids},
        seed=17,
    )


class MilpExportTests(unittest.TestCase):
    def test_fixed_horizon_solver_is_exported(self):
        self.assertTrue(hasattr(milp_module, "solve_max_storage_fixed_horizon"))
        self.assertTrue(hasattr(control, "solve_max_storage_fixed_horizon"))

    def test_min_makespan_solver_is_not_exported(self):
        self.assertFalse(hasattr(milp_module, "solve_min_makespan"))
        self.assertFalse(hasattr(control, "solve_min_makespan"))


@unittest.skipUnless(HAVE_PULP, "pulp/CBC not installed")
class MilpTests(unittest.TestCase):
    def test_extract_params_from_network(self):
        env = _env()
        vessels, inj_cap, capture_rate, term_cap = extract_params(env)
        self.assertEqual(len(vessels), 2)
        self.assertGreater(inj_cap, 0.0)
        self.assertEqual(term_cap, 6_000.0)
        for v in vessels:
            self.assertGreater(v.round_trip_h, v.startup_h)

    def test_fixed_horizon_cost_objective_stores_co2(self):
        result = milp_module.solve_max_storage_fixed_horizon(
            _env(),
            horizon_h=168,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
        )
        self.assertEqual(result.status, "Optimal")
        self.assertEqual(result.horizon_h, 168)
        self.assertGreater(result.stored_t, 0.0)
        self.assertGreater(result.deliveries, 0)
        self.assertGreater(result.operating_cost, 0.0)
        self.assertAlmostEqual(
            result.operating_cost,
            result.vessel_fuel + result.conditioning + result.reconditioning + result.loading + result.unloading,
        )
        self.assertAlmostEqual(result.total_cost_per_stored_t, result.total_cost / result.stored_t)

    def test_fixed_horizon_can_pool_capture_under_flexible_emitter_actions(self):
        result = milp_module.solve_max_storage_fixed_horizon(
            _unbalanced_source_env(),
            horizon_h=120,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
        )
        self.assertEqual(result.status, "Optimal")
        self.assertGreater(len(result.schedule["short_ship"]), 0)
        self.assertGreater(result.stored_t, 1_000.0)

    def test_fixed_horizon_scenario_capture_outage_limits_storage(self):
        env = _env(cap_hours=80)
        scenario = _scenario_for_env(env, horizon_h=80, capture=0.0)

        result = milp_module.solve_max_storage_fixed_horizon(env, horizon_h=80, scenario=scenario)

        self.assertEqual(result.status, "Optimal")
        self.assertAlmostEqual(result.stored_t, 0.0)
        self.assertEqual(result.deliveries, 0)

    def test_fixed_horizon_scenario_well_outage_blocks_storage(self):
        env = _env(cap_hours=80)
        scenario = _scenario_for_env(env, horizon_h=80, wells_available=False)
        scenario.initial_inventory_t = {"terminal": 1_000.0}

        result = milp_module.solve_max_storage_fixed_horizon(env, horizon_h=80, scenario=scenario)

        self.assertEqual(result.status, "Optimal")
        self.assertAlmostEqual(result.stored_t, 0.0)

    def test_fixed_horizon_scenario_unloads_over_time_into_small_terminal_to_meet_target(self):
        env = _streaming_unload_env()
        scenario = _scenario_for_env(env, horizon_h=8, capture=0.0)
        scenario.initial_inventory_t = {"source": 1_000.0}

        result = milp_module.solve_max_storage_fixed_horizon(
            env,
            horizon_h=8,
            scenario=scenario,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
        )

        self.assertEqual(result.status, "Optimal")
        self.assertAlmostEqual(result.stored_t, 900.0)
        self.assertAlmostEqual(result.shortfall_t, 0.0)

    def test_fixed_horizon_accounts_for_unserved_capture_as_vent_and_shortfall(self):
        env = _tiny_buffer_no_delivery_env()

        result = milp_module.solve_max_storage_fixed_horizon(env, horizon_h=3)

        self.assertEqual(result.status, "Optimal")
        self.assertAlmostEqual(result.stored_t, 0.0)
        self.assertAlmostEqual(result.vented_t, 200.0)
        self.assertAlmostEqual(result.in_transit_t, 100.0)
        self.assertAlmostEqual(result.in_transit_growth_t, 100.0)
        self.assertAlmostEqual(result.shortfall_t, 270.0)
        self.assertGreater(result.total_cost, result.operating_cost)

    def test_fixed_horizon_respects_terminal_berth_count(self):
        result = milp_module.solve_max_storage_fixed_horizon(
            _two_berth_parallel_env(),
            horizon_h=3,
            initial_buffer_t=2_000.0,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
        )

        self.assertEqual(result.status, "Optimal")
        self.assertEqual(result.deliveries, 2)
        self.assertGreaterEqual(result.stored_t, 1_800.0)
        self.assertAlmostEqual(result.in_transit_t, 2_000.0 - result.stored_t - result.vented_t)
        self.assertAlmostEqual(result.in_transit_growth_t, -result.stored_t - result.vented_t)

    def test_static_solution_validation_rejects_nonoptimal_fractional_or_unbalanced_results(self):
        nonoptimal = milp_module._validate_static_solution(
            status="Not Solved",
            binary_values=[0.0, 0.5],
            stored_t=353_734.9,
            vented_t=0.0,
            in_transit_t=49_465.1,
            captured_from_operations_t=403_200.0,
            initial_in_transit_t=0.0,
            max_storable_from_deliveries_t=60_000.0,
        )
        self.assertFalse(nonoptimal.is_valid)
        self.assertIn("status", nonoptimal.validation_error)

        fractional = milp_module._validate_static_solution(
            status="Optimal",
            binary_values=[0.0, 0.25, 1.0],
            stored_t=100.0,
            vented_t=0.0,
            in_transit_t=0.0,
            captured_from_operations_t=100.0,
            initial_in_transit_t=0.0,
            max_storable_from_deliveries_t=100.0,
        )
        self.assertFalse(fractional.is_valid)
        self.assertIn("integrality", fractional.validation_error)

        unbalanced = milp_module._validate_static_solution(
            status="Optimal",
            binary_values=[1.0],
            stored_t=200.0,
            vented_t=0.0,
            in_transit_t=0.0,
            captured_from_operations_t=100.0,
            initial_in_transit_t=0.0,
            max_storable_from_deliveries_t=100.0,
        )
        self.assertFalse(unbalanced.is_valid)
        self.assertIn("mass balance", unbalanced.validation_error)


if __name__ == "__main__":
    unittest.main()
