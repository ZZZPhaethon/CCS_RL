import unittest

from sim.control import trip_milp
from sim.economics import EconomicParameters
from sim.environment import VESSEL_GO_TERMINAL, VESSEL_WAIT
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

    def test_relaxed_trip_milp_can_be_materialized_to_replayable_actions(self):
        env = _no_capture_env(cap_hours=5)
        _add_route_coordinates(env)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 500.0
        env.cumulative_captured_t = 500.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=5,
            emitter_availability={"source": [0.0] * 5},
            vessel_speed_factor={"ship": [1.0] * 5},
            well_available={"well": [True] * 5},
            injectivity_factor={"well": [1.0] * 5},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        relaxed = trip_milp.solve_relaxed_trip_milp_with_cplex(
            env,
            horizon_h=5,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
        )
        materialized = trip_milp.materialize_relaxed_trip_plan(env, relaxed)
        replay = trip_milp.replay_trip_milp_plan(env, materialized, stored_tol_t=1e9)

        self.assertEqual(materialized.level, "relaxed_trip_materialized")
        self.assertEqual(len(materialized.native_actions_by_hour), 5)
        self.assertIn(VESSEL_GO_TERMINAL, materialized.vessel_actions_by_hour["ship"])
        self.assertTrue(replay.is_executable, replay.violations)
        self.assertAlmostEqual(replay.stored_gap_t, 0.0)
        self.assertAlmostEqual(materialized.stored_t, replay.stored_t)
        self.assertGreater(replay.stored_t, 0.0)

    def test_executable_trip_mip_start_selects_matching_native_action_trip(self):
        env = _no_capture_env(cap_hours=5)
        _add_route_coordinates(env)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 500.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=5,
            emitter_availability={"source": [0.0] * 5},
            vessel_speed_factor={"ship": [1.0] * 5},
            well_available={"well": [True] * 5},
            injectivity_factor={"well": [1.0] * 5},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)
        options = trip_milp._executable_trip_options(env, env.scenario, start_step=0, horizon_h=5)
        choose = {i: trip_milp.pulp.LpVariable(f"trip_{i}", cat="Binary") for i in range(len(options))}
        well_rate_options = trip_milp._well_rate_options_by_hour(env, env.scenario, horizon_h=5)
        well_choice = {
            (well_id, t, rate_index): trip_milp.pulp.LpVariable(
                f"well_{well_id}_{t}_{rate_index}",
                cat="Binary",
            )
            for well_id in env.well_ids
            for t in range(5)
            for rate_index in well_rate_options[(well_id, t)]
        }
        native_actions = [
            {"vessels": [VESSEL_WAIT], "wells": [0]},
            {"vessels": [VESSEL_GO_TERMINAL], "wells": [1]},
            {"vessels": [VESSEL_WAIT], "wells": [1]},
            {"vessels": [VESSEL_WAIT], "wells": [1]},
            {"vessels": [VESSEL_WAIT], "wells": [0]},
        ]

        selected = trip_milp._apply_executable_trip_mip_start(
            env,
            env.scenario,
            start_step=0,
            horizon_h=5,
            options=options,
            choose=choose,
            well_rate_options=well_rate_options,
            well_choice=well_choice,
            native_actions_by_hour=native_actions,
        )

        self.assertGreaterEqual(selected, 1)
        matched = [
            option
            for i, option in enumerate(options)
            if choose[i].varValue == 1
        ]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].emitter_id, "source")
        self.assertEqual(matched[0].load_start_h, 0)
        self.assertEqual(matched[0].depart_h, 1)

    def test_native_warm_start_options_are_added_when_pruning_removed_them(self):
        env = _no_capture_env(cap_hours=5)
        _add_route_coordinates(env)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 500.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=5,
            emitter_availability={"source": [0.0] * 5},
            vessel_speed_factor={"ship": [1.0] * 5},
            well_available={"well": [True] * 5},
            injectivity_factor={"well": [1.0] * 5},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)
        options = [
            option
            for option in trip_milp._executable_trip_options(env, env.scenario, start_step=0, horizon_h=5)
            if option.depart_h != 1
        ]
        native_actions = [
            {"vessels": [VESSEL_WAIT], "wells": [0]},
            {"vessels": [VESSEL_GO_TERMINAL], "wells": [1]},
            {"vessels": [VESSEL_WAIT], "wells": [1]},
            {"vessels": [VESSEL_WAIT], "wells": [1]},
            {"vessels": [VESSEL_WAIT], "wells": [0]},
        ]

        augmented = trip_milp._options_with_native_warm_start(
            env,
            env.scenario,
            start_step=0,
            horizon_h=5,
            options=options,
            native_actions_by_hour=native_actions,
        )

        self.assertTrue(
            any(
                option.emitter_id == "source"
                and option.load_start_h == 0
                and option.depart_h == 1
                for option in augmented
            )
        )

    def test_native_warm_start_option_uses_replayed_load_amount(self):
        env = _no_capture_env(cap_hours=5)
        _add_route_coordinates(env)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 200.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=5,
            emitter_availability={"source": [0.0] * 5},
            vessel_speed_factor={"ship": [1.0] * 5},
            well_available={"well": [True] * 5},
            injectivity_factor={"well": [1.0] * 5},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)
        native_actions = [
            {"vessels": [VESSEL_WAIT], "wells": [0]},
            {"vessels": [VESSEL_GO_TERMINAL], "wells": [1]},
            {"vessels": [VESSEL_WAIT], "wells": [1]},
            {"vessels": [VESSEL_WAIT], "wells": [1]},
            {"vessels": [VESSEL_WAIT], "wells": [0]},
        ]

        augmented = trip_milp._options_with_native_warm_start(
            env,
            env.scenario,
            start_step=0,
            horizon_h=5,
            options=[],
            native_actions_by_hour=native_actions,
        )

        self.assertEqual(len(augmented), 1)
        self.assertAlmostEqual(augmented[0].amount_t, 200.0)

    def test_native_action_trace_materialization_uses_replay_metrics(self):
        env = _no_capture_env(cap_hours=5)
        _add_route_coordinates(env)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 200.0
        env.cumulative_captured_t = 200.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=5,
            emitter_availability={"source": [0.0] * 5},
            vessel_speed_factor={"ship": [1.0] * 5},
            well_available={"well": [True] * 5},
            injectivity_factor={"well": [1.0] * 5},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)
        native_actions = [
            {"vessels": [VESSEL_WAIT], "wells": [0]},
            {"vessels": [VESSEL_GO_TERMINAL], "wells": [1]},
            {"vessels": [VESSEL_WAIT], "wells": [1]},
            {"vessels": [VESSEL_WAIT], "wells": [1]},
            {"vessels": [VESSEL_WAIT], "wells": [0]},
        ]

        materialized = trip_milp.materialize_native_action_trace(
            env,
            native_actions,
            horizon_h=5,
            economics=EconomicParameters(carbon_price_eur_per_t=1_000.0),
        )
        replay = trip_milp.replay_trip_milp_plan(env, materialized)

        self.assertEqual(materialized.level, "native_action_trace")
        self.assertEqual(len(materialized.native_actions_by_hour), 5)
        self.assertTrue(replay.is_executable, replay.violations)
        self.assertAlmostEqual(replay.stored_gap_t, 0.0)
        self.assertAlmostEqual(materialized.stored_t, replay.stored_t)
        self.assertAlmostEqual(materialized.vented_t, replay.vented_t)

    def test_warm_start_does_not_replace_solver_trace(self):
        env = _no_capture_env(cap_hours=5)
        _add_route_coordinates(env)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 200.0
        env.cumulative_captured_t = 200.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=5,
            emitter_availability={"source": [0.0] * 5},
            vessel_speed_factor={"ship": [1.0] * 5},
            well_available={"well": [True] * 5},
            injectivity_factor={"well": [1.0] * 5},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)
        solver_actions = [{"vessels": [VESSEL_WAIT], "wells": [0]} for _ in range(5)]
        economics = EconomicParameters(carbon_price_eur_per_t=1_000.0)
        solver_result = trip_milp.materialize_native_action_trace(
            env,
            solver_actions,
            horizon_h=5,
            economics=economics,
            storage_reward_eur_per_t=1_000.0,
        )

        selected = trip_milp._replay_native_solver_trace(
            env,
            solver_result,
            horizon_h=5,
            economics=economics,
            storage_reward_eur_per_t=1_000.0,
        )

        self.assertEqual(selected.level, "executable_trip_replayed")
        self.assertEqual(selected.native_actions_by_hour, solver_result.native_actions_by_hour)

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
