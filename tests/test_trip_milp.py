import unittest

from sim.control import trip_milp
from sim.economics import EconomicParameters
from sim.environment import VESSEL_GO_TERMINAL
from sim.scenario_generation import Scenario
from tests.test_cplex_milp import _two_ship_high_rate_source_env
from tests.test_rolling_milp import _no_capture_env, _two_source_one_ship_fast_env


def _add_route_coordinates(env) -> None:
    route = env._routes["ship"]
    route["coordinates"] = [(0.0, 0.0), (0.0, 1.0)]
    route["return_coordinates"] = [(0.0, 1.0), (0.0, 0.0)]


@unittest.skipIf(
    trip_milp.pulp is None or not trip_milp.pulp.CPLEX_CMD(msg=0).available(),
    "external CPLEX executable not available",
)
class TripMilpTests(unittest.TestCase):
    def test_relaxed_trip_milp_returns_aggregate_oracle_without_native_actions(self):
        env = _no_capture_env(cap_hours=4)
        _add_route_coordinates(env)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 500.0
        env.cumulative_captured_t = 500.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=4,
            emitter_availability={"source": [0.0] * 4},
            vessel_speed_factor={"ship": [1.0] * 4},
            well_available={"well": [True] * 4},
            injectivity_factor={"well": [1.0] * 4},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        result = trip_milp.solve_relaxed_trip_milp_with_cplex(
            env,
            horizon_h=4,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
        )

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertEqual(result.level, "relaxed_trip")
        self.assertGreater(result.stored_t, 0.0)
        self.assertGreaterEqual(len(result.trips), 1)
        self.assertEqual(result.native_actions_by_hour, [])
        self.assertGreater(result.binary_count, 0)

    def test_relaxed_trip_milp_respects_one_unload_per_terminal_hour(self):
        env = _two_ship_high_rate_source_env()
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 2_000.0
        env.cumulative_captured_t = 2_000.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=3,
            emitter_availability={"source": [0.0] * 3},
            vessel_speed_factor={vessel_id: [1.0] * 3 for vessel_id in env.vessel_ids},
            well_available={"well": [True] * 3},
            injectivity_factor={"well": [1.0] * 3},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        result = trip_milp.solve_relaxed_trip_milp_with_cplex(
            env,
            horizon_h=3,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
        )

        arrivals_by_hour: dict[int, int] = {}
        for trip in result.trips:
            arrivals_by_hour[trip.arrival_h] = arrivals_by_hour.get(trip.arrival_h, 0) + 1
        self.assertTrue(result.is_valid, result.validation_error)
        self.assertLessEqual(max(arrivals_by_hour.values(), default=0), 1)

    def test_relaxed_trip_milp_excludes_shortfall_from_total_cost(self):
        env = _no_capture_env(cap_hours=1)
        _add_route_coordinates(env)
        env.reset(seed=1)
        env.cumulative_captured_t = 1_000.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=1,
            emitter_availability={"source": [0.0]},
            vessel_speed_factor={"ship": [1.0]},
            well_available={"well": [True]},
            injectivity_factor={"well": [1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        result = trip_milp.solve_relaxed_trip_milp_with_cplex(
            env,
            horizon_h=1,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
        )

        self.assertGreater(result.shortfall_t, 0.0)
        self.assertAlmostEqual(
            result.total_cost,
            result.operating_cost + result.vented_t * 80.0,
        )
        self.assertEqual(result.storage_reward_eur_per_t, 1_000.0)
        self.assertAlmostEqual(result.net_reward, 1_000.0 * result.stored_t - result.total_cost)

    def test_executable_trip_milp_expands_to_replayable_rl_actions(self):
        env = _no_capture_env(cap_hours=4)
        _add_route_coordinates(env)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 500.0
        env.cumulative_captured_t = 500.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=4,
            emitter_availability={"source": [0.0] * 4},
            vessel_speed_factor={"ship": [1.0] * 4},
            well_available={"well": [True] * 4},
            injectivity_factor={"well": [1.0] * 4},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        result = trip_milp.solve_executable_trip_milp_with_cplex(
            env,
            horizon_h=4,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
        )
        replay = trip_milp.replay_trip_milp_plan(env, result)

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertEqual(result.level, "executable_trip")
        self.assertEqual(len(result.native_actions_by_hour), 4)
        self.assertIn(VESSEL_GO_TERMINAL, result.vessel_actions_by_hour["ship"])
        self.assertTrue(replay.is_executable, replay.violations)
        self.assertAlmostEqual(replay.stored_t, result.stored_t, delta=1e-6)

    def test_executable_trip_milp_excludes_shortfall_from_total_cost(self):
        env = _no_capture_env(cap_hours=1)
        _add_route_coordinates(env)
        env.reset(seed=1)
        env.cumulative_captured_t = 1_000.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=1,
            emitter_availability={"source": [0.0]},
            vessel_speed_factor={"ship": [1.0]},
            well_available={"well": [True]},
            injectivity_factor={"well": [1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        result = trip_milp.solve_executable_trip_milp_with_cplex(
            env,
            horizon_h=1,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
        )

        self.assertGreater(result.shortfall_t, 0.0)
        self.assertAlmostEqual(
            result.total_cost,
            result.operating_cost + result.vented_t * 80.0,
        )
        self.assertEqual(result.storage_reward_eur_per_t, 1_000.0)
        self.assertAlmostEqual(result.net_reward, 1_000.0 * result.stored_t - result.total_cost)

    def test_executable_trip_milp_can_serve_non_home_emitter(self):
        env = _two_source_one_ship_fast_env()
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source_a"] = 0.0
        env.simulator.state.entity_inventory_t["source_b"] = 500.0
        env.cumulative_captured_t = 500.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=5,
            emitter_availability={emitter_id: [0.0] * 5 for emitter_id in env.emitter_ids},
            vessel_speed_factor={"ship": [1.0] * 5},
            well_available={"well": [True] * 5},
            injectivity_factor={"well": [1.0] * 5},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        result = trip_milp.solve_executable_trip_milp_with_cplex(
            env,
            horizon_h=5,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
        )
        replay = trip_milp.replay_trip_milp_plan(env, result)

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertIn("source_b", [trip.emitter_id for trip in result.trips])
        self.assertIn(env.vessel_go_emitter_action("source_b"), result.vessel_actions_by_hour["ship"])
        self.assertTrue(replay.is_executable, replay.violations)
        self.assertAlmostEqual(replay.stored_t, result.stored_t, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
