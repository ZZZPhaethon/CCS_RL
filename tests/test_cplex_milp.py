import unittest
import inspect
from unittest.mock import patch

from sim.control import cplex_milp
from sim.economics import EconomicParameters
from sim.entities import Emitter, InjectionWell, Pipeline, Reservoir, SubseaManifold, Terminal, Vessel
from sim.environment import CCSEnv, CCSEnvConfig, WELL_RATE_LEVELS_MTPA
from sim.environment.env import VESSEL_GO_TERMINAL, VESSEL_WAIT
from sim.line_source import LineSourceParameters
from sim.network import PhysicalNetwork
from sim.operations.pressure_limits import projected_bottomhole_pressure_bar
from sim.scenario_generation import Scenario, ScenarioConfig, ScenarioGenerator
from tests.test_rolling_milp import _no_capture_env, _two_berth_parallel_env, _two_source_one_ship_fast_env


def _two_ship_high_rate_source_env() -> CCSEnv:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("source", nominal_capture_tph=0.0, buffer_capacity_t=3_000.0, loading_rate_tph=2_000.0))
    network.add_entity(Vessel("ship_a", capacity_t=1_000.0, loading_rate_tph=1_000.0, unloading_rate_tph=1_000.0, speed_knots=1.0))
    network.add_entity(Vessel("ship_b", capacity_t=1_000.0, loading_rate_tph=1_000.0, unloading_rate_tph=1_000.0, speed_knots=1.0))
    network.add_entity(Terminal("terminal", storage_capacity_t=3_000.0, berth_count=1))
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
        scenario_generator=ScenarioGenerator(
            config=ScenarioConfig(episode_hours=1, randomize_initial_inventory=False)
        ),
        config=CCSEnvConfig(episode_hours=1),
        routes={
            "ship_a": {"origin": "source", "destination": "terminal", "distance_km": 1.852, "speed_knots": 1.0},
            "ship_b": {"origin": "source", "destination": "terminal", "distance_km": 1.852, "speed_knots": 1.0},
        },
    )


def _two_ship_priority_env() -> CCSEnv:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("source", nominal_capture_tph=0.0, buffer_capacity_t=3_000.0, loading_rate_tph=2_000.0))
    network.add_entity(Vessel("ship_a", capacity_t=500.0, loading_rate_tph=500.0, unloading_rate_tph=500.0, speed_knots=1.0))
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
        scenario_generator=ScenarioGenerator(
            config=ScenarioConfig(episode_hours=1, randomize_initial_inventory=False)
        ),
        config=CCSEnvConfig(episode_hours=1),
        routes={
            "ship_a": {"origin": "source", "destination": "terminal", "distance_km": 1.852, "speed_knots": 1.0},
            "ship_b": {"origin": "source", "destination": "terminal", "distance_km": 1.852, "speed_knots": 1.0},
        },
    )


class CplexMilpInterfaceTests(unittest.TestCase):
    def test_module_exposes_full_scenario_solver(self):
        self.assertTrue(hasattr(cplex_milp, "solve_full_scenario_with_cplex"))

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_solver_command_uses_external_cplex_executable(self):
        cmd = cplex_milp._make_cplex_cmd(
            cplex_path="C:/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.exe",
            time_limit_s=12.5,
            mip_gap_rel=0.01,
            mip_gap_abs=3.0,
            warm_start=True,
            msg=True,
        )

        self.assertEqual(
            cmd.path,
            "C:/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.exe",
        )
        self.assertEqual(cmd.timeLimit, 12.5)
        self.assertEqual(cmd.optionsDict["gapRel"], 0.01)
        self.assertEqual(cmd.optionsDict["gapAbs"], 3.0)
        self.assertTrue(cmd.optionsDict["warmStart"])
        self.assertTrue(cmd.msg)

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_cplex_solver_writes_mip_solution_without_fixed_lp_postsolve(self):
        cmd = cplex_milp._make_cplex_cmd(msg=False)

        self.assertIsInstance(cmd, cplex_milp._CplexMipDirectSolutionCmd)
        actual_solve_source = inspect.getsource(cmd.actualSolve)
        self.assertIn("mipopt", actual_solve_source)
        self.assertNotIn("change problem fixed", actual_solve_source)

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_status_label_distinguishes_time_limit_feasible_from_optimal(self):
        label = cplex_milp._solution_status_label(
            cplex_milp.pulp.constants.LpStatusOptimal,
            cplex_milp.pulp.constants.LpSolutionIntegerFeasible,
        )

        self.assertEqual(label, "Integer Feasible")

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_native_actions_seed_arc_and_well_mip_start_values(self):
        env = _no_capture_env(cap_hours=3)
        env.reset(seed=1)
        scenario = Scenario(
            time_step_hours=1.0,
            n_steps=3,
            emitter_availability={"source": [1.0] * 3},
            vessel_speed_factor={"ship": [1.0] * 3},
            well_available={"well": [True] * 3},
            injectivity_factor={"well": [1.0] * 3},
        )
        arcs, starts = cplex_milp._build_action_arcs(env, scenario, start_step=0, horizon_h=3)
        arc_vars = {
            index: cplex_milp.pulp.LpVariable(f"test_arc_{index}", cat="Binary")
            for index in range(len(arcs))
        }
        well_rate_options = cplex_milp._well_rate_options_by_hour(env, scenario, horizon_h=3)
        well_choice = {
            (well_id, t, rate_index): cplex_milp.pulp.LpVariable(
                f"test_well_{well_id}_{t}_{rate_index}",
                cat="Binary",
            )
            for well_id in env.well_ids
            for t in range(3)
            for rate_index in well_rate_options[(well_id, t)]
        }
        native_actions = [
            {"vessels": [VESSEL_WAIT], "wells": [0]},
            {"vessels": [VESSEL_GO_TERMINAL], "wells": [2]},
            {"vessels": [VESSEL_WAIT], "wells": [1]},
        ]

        cplex_milp._apply_native_action_mip_start(
            env,
            arcs,
            starts,
            arc_vars,
            well_rate_options,
            well_choice,
            native_actions,
            horizon_h=3,
        )

        selected_wait = next(
            index
            for index, arc in enumerate(arcs)
            if not arc.is_sailing and arc.vessel_id == "ship" and arc.origin_id == "source" and arc.start_h == 0
        )
        selected_sail = next(
            index
            for index, arc in enumerate(arcs)
            if arc.is_sailing
            and arc.vessel_id == "ship"
            and arc.origin_id == "source"
            and arc.destination_id == "terminal"
            and arc.start_h == 1
        )

        self.assertEqual(arc_vars[selected_wait].value(), 1)
        self.assertEqual(arc_vars[selected_sail].value(), 1)
        self.assertEqual(well_choice[("well", 1, 2)].value(), 1)
        self.assertEqual(well_choice[("well", 1, 0)].value(), 0)

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_native_action_mip_start_can_seed_loading_state_values(self):
        env = _no_capture_env(cap_hours=2)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 500.0
        scenario = Scenario(
            time_step_hours=1.0,
            n_steps=2,
            emitter_availability={"source": [1.0] * 2},
            vessel_speed_factor={"ship": [1.0] * 2},
            well_available={"well": [True] * 2},
            injectivity_factor={"well": [1.0] * 2},
        )
        arcs, starts = cplex_milp._build_action_arcs(env, scenario, start_step=0, horizon_h=2)
        arc_vars = {
            index: cplex_milp.pulp.LpVariable(f"state_arc_{index}", cat="Binary")
            for index in range(len(arcs))
        }
        well_rate_options = cplex_milp._well_rate_options_by_hour(env, scenario, horizon_h=2)
        well_choice = {
            (well_id, t, rate_index): cplex_milp.pulp.LpVariable(
                f"state_well_{well_id}_{t}_{rate_index}",
                cat="Binary",
            )
            for well_id in env.well_ids
            for t in range(2)
            for rate_index in well_rate_options[(well_id, t)]
        }
        cargo = {
            ("ship", t): cplex_milp.pulp.LpVariable(f"state_cargo_{t}", lowBound=0.0)
            for t in range(3)
        }
        source_stock = {
            ("source", t): cplex_milp.pulp.LpVariable(f"state_source_{t}", lowBound=0.0)
            for t in range(3)
        }
        load = {
            ("ship", "source", t): cplex_milp.pulp.LpVariable(f"state_load_{t}", lowBound=0.0)
            for t in range(2)
        }
        load_active = {
            ("ship", "source", t): cplex_milp.pulp.LpVariable(f"state_load_active_{t}", cat="Binary")
            for t in range(2)
        }
        native_actions = [
            {"vessels": [VESSEL_WAIT], "wells": [0]},
            {"vessels": [VESSEL_GO_TERMINAL], "wells": [0]},
        ]

        cplex_milp._apply_native_action_mip_start(
            env,
            arcs,
            starts,
            arc_vars,
            well_rate_options,
            well_choice,
            native_actions,
            horizon_h=2,
            scenario=scenario,
            start_step=0,
            cargo=cargo,
            source_stock=source_stock,
            load=load,
            load_active=load_active,
        )

        self.assertEqual(cargo[("ship", 0)].value(), 0.0)
        self.assertEqual(cargo[("ship", 1)].value(), 500.0)
        self.assertEqual(source_stock[("source", 0)].value(), 500.0)
        self.assertEqual(source_stock[("source", 1)].value(), 0.0)
        self.assertEqual(load[("ship", "source", 0)].value(), 500.0)
        self.assertEqual(load_active[("ship", "source", 0)].value(), 1)

    def test_cost_breakdown_matches_physical_economic_parameters(self):
        vessels = [
            cplex_milp.CplexVesselParams(
                vessel_id="ship",
                source_id="source",
                capacity_t=1_000.0,
                load_rate_tph=500.0,
                unload_rate_tph=250.0,
                speed_knots=10.0,
            )
        ]
        result = cplex_milp._schedule_cost_breakdown(
            vessels,
            departures={"ship": [0, 20]},
            sail_hours={"ship": [3, 4]},
            stored_t=1_500.0,
            params=EconomicParameters(),
        )

        self.assertAlmostEqual(
            result.operating_cost,
            result.vessel_fuel
            + result.conditioning
            + result.reconditioning
            + result.loading
            + result.unloading,
        )
        self.assertGreater(result.vessel_fuel, 0.0)
        self.assertGreater(result.conditioning, 0.0)

    def test_sailing_fuel_hours_match_rl_arrival_hour_accounting(self):
        completed = cplex_milp._ActionArc(
            vessel_id="ship",
            start_h=0,
            end_h=5,
            origin_id="source",
            destination_id="terminal",
            action=VESSEL_GO_TERMINAL,
            is_sailing=True,
            arrives_within_horizon=True,
        )
        truncated = cplex_milp._ActionArc(
            vessel_id="ship",
            start_h=0,
            end_h=5,
            origin_id="source",
            destination_id="terminal",
            action=VESSEL_GO_TERMINAL,
            is_sailing=True,
            arrives_within_horizon=False,
        )

        self.assertEqual(cplex_milp._sailing_fuel_hours(completed), 4)
        self.assertEqual(cplex_milp._sailing_fuel_hours(truncated), 5)

    def test_sailing_hours_prefers_leg_speed_factor_over_vessel_factor(self):
        env = _no_capture_env(cap_hours=4)
        env.reset(seed=1)
        scenario = Scenario(
            time_step_hours=1.0,
            n_steps=4,
            emitter_availability={"source": [1.0] * 4},
            vessel_speed_factor={"ship": [1.0] * 4},
            leg_speed_factor={"source->terminal": [1.0, 0.5, 0.5, 0.5]},
            well_available={"well": [True] * 4},
            injectivity_factor={"well": [1.0] * 4},
        )

        hours = cplex_milp._sail_hours_between(
            env,
            "source",
            "terminal",
            "ship",
            scenario=scenario,
            start_step=1,
            max_horizon_h=4,
        )

        self.assertEqual(hours, 2)

    def test_well_rate_options_use_pressure_limited_rl_mask(self):
        env = _no_capture_env(cap_hours=2)
        env.reset(seed=1)
        scenario = Scenario(
            time_step_hours=1.0,
            n_steps=2,
            emitter_availability={"source": [1.0] * 2},
            vessel_speed_factor={"ship": [1.0] * 2},
            well_available={"well": [True] * 2},
            injectivity_factor={"well": [1.0] * 2},
        )

        only_off = tuple(index == 0 for index, _level in enumerate(WELL_RATE_LEVELS_MTPA))
        with patch.object(cplex_milp, "pressure_limited_rate_level_mask", return_value=only_off):
            options = cplex_milp._well_rate_options_by_hour(env, scenario, horizon_h=2)

        self.assertEqual(options[("well", 0)], [0])
        self.assertEqual(options[("well", 1)], [0])

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_full_scenario_solution_returns_rl_native_actions(self):
        env = _two_source_one_ship_fast_env()
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source_a"] = 0.0
        env.simulator.state.entity_inventory_t["source_b"] = 500.0
        env.cumulative_captured_t = 500.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=5,
            emitter_availability={"source_a": [1.0] * 5, "source_b": [1.0] * 5},
            vessel_speed_factor={"ship": [1.0] * 5},
            well_available={"well": [True] * 5},
            injectivity_factor={"well": [1.0] * 5},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        result = cplex_milp.solve_full_scenario_with_cplex(
            env,
            horizon_h=5,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
        )

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertEqual(len(result.native_actions_by_hour), 5)
        self.assertEqual(len(result.vessel_actions_by_hour["ship"]), 5)
        self.assertEqual(len(result.well_rate_indices_by_hour["well"]), 5)
        self.assertTrue(
            all(0 <= action < env.vessel_action_count for action in result.vessel_actions_by_hour["ship"])
        )
        self.assertTrue(
            all(0 <= index < len(WELL_RATE_LEVELS_MTPA) for index in result.well_rate_indices_by_hour["well"])
        )
        self.assertIn(env.vessel_go_emitter_action("source_b"), result.vessel_actions_by_hour["ship"])

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_full_scenario_excludes_shortfall_from_total_cost(self):
        env = _no_capture_env(cap_hours=1)
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

        result = cplex_milp.solve_full_scenario_with_cplex(
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

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_relaxed_full_scenario_allows_partial_cargo_terminal_sailing(self):
        env = _no_capture_env(cap_hours=2)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 0.0
        env.simulator.state.entity_inventory_t["ship"] = 250.0
        env.simulator.state.entity_inventory_t["terminal"] = 0.0
        env.cumulative_captured_t = 250.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=2,
            emitter_availability={"source": [0.0, 0.0]},
            vessel_speed_factor={"ship": [1.0, 1.0]},
            well_available={"well": [True, True]},
            injectivity_factor={"well": [1.0, 1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        result = cplex_milp.solve_full_scenario_with_cplex(
            env,
            horizon_h=2,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
        )

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertEqual(result.vessel_actions_by_hour["ship"][0], VESSEL_GO_TERMINAL)

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_full_scenario_allows_partial_vessel_terminal_action(self):
        env = _no_capture_env(cap_hours=2)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 0.0
        env.simulator.state.entity_inventory_t["ship"] = 250.0
        env.simulator.state.entity_inventory_t["terminal"] = 0.0
        env.cumulative_captured_t = 500.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=2,
            emitter_availability={"source": [0.0, 0.0]},
            vessel_speed_factor={"ship": [1.0, 1.0]},
            well_available={"well": [True, True]},
            injectivity_factor={"well": [1.0, 1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        result = cplex_milp.solve_full_scenario_with_cplex(
            env,
            horizon_h=2,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
        )

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertEqual(result.vessel_actions_by_hour["ship"][0], VESSEL_GO_TERMINAL)

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_terminal_unloading_matches_rl_one_vessel_per_hour(self):
        env = _two_berth_parallel_env()
        env.reset(seed=1)
        for vessel_id in env.vessel_ids:
            env.simulator.vessel_states[vessel_id] = {
                "mode": "berthed",
                "berth": "terminal",
                "origin": "terminal",
                "destination": "terminal",
                "progress": 0.0,
            }
            env.simulator.state.vessel_berths[vessel_id] = "terminal"
            env.simulator.state.entity_inventory_t[vessel_id] = 1_000.0
        env.simulator.state.entity_inventory_t["source"] = 0.0
        env.simulator.state.entity_inventory_t["terminal"] = 0.0
        env.cumulative_captured_t = 2_000.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=1,
            emitter_availability={"source": [0.0]},
            vessel_speed_factor={vessel_id: [1.0] for vessel_id in env.vessel_ids},
            well_available={"well": [True]},
            injectivity_factor={"well": [1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        with patch.object(cplex_milp, "WELL_RATE_LEVELS_MTPA", (0.0, 17.0)):
            result = cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
                storage_reward_eur_per_t=1_000.0,
                time_limit_s=10.0,
            )

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertLessEqual(result.stored_t, 1_000.0 + 1e-6)

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_terminal_berth_count_limits_unloading_not_waiting_occupancy(self):
        env = _two_ship_high_rate_source_env()
        env.reset(seed=1)
        for vessel_id in env.vessel_ids:
            env.simulator.vessel_states[vessel_id] = {
                "mode": "berthed",
                "berth": "terminal",
                "origin": "terminal",
                "destination": "terminal",
                "progress": 0.0,
            }
            env.simulator.state.vessel_berths[vessel_id] = "terminal"
            env.simulator.state.entity_inventory_t[vessel_id] = 1_000.0
        env.simulator.state.entity_inventory_t["source"] = 0.0
        env.simulator.state.entity_inventory_t["terminal"] = 0.0
        env.cumulative_captured_t = 2_000.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=1,
            emitter_availability={"source": [0.0]},
            vessel_speed_factor={vessel_id: [1.0] for vessel_id in env.vessel_ids},
            well_available={"well": [True]},
            injectivity_factor={"well": [1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        with patch.object(cplex_milp, "WELL_RATE_LEVELS_MTPA", (0.0, 17.0)):
            result = cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
                storage_reward_eur_per_t=1_000.0,
                time_limit_s=10.0,
            )

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertAlmostEqual(result.stored_t, 1_000.0, delta=1e-6)

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_emitter_loading_matches_rl_one_vessel_per_hour(self):
        env = _two_ship_high_rate_source_env()
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 2_000.0
        env.cumulative_captured_t = 2_000.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=1,
            emitter_availability={"source": [0.0]},
            vessel_speed_factor={vessel_id: [1.0] for vessel_id in env.vessel_ids},
            well_available={"well": [True]},
            injectivity_factor={"well": [1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)
        economics = EconomicParameters(conditioning_eur_per_t=-1_000.0, storage_shortfall_eur_per_t=0.0)

        result = cplex_milp.solve_full_scenario_with_cplex(
            env,
            horizon_h=1,
            economics=economics,
            time_limit_s=10.0,
        )

        loaded_t = result.conditioning / economics.conditioning_eur_per_t
        self.assertTrue(result.is_valid, result.validation_error)
        self.assertLessEqual(loaded_t, 1_000.0 + 1e-6)

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_relaxed_emitter_loading_can_choose_non_priority_vessel(self):
        env = _two_ship_priority_env()
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 2_000.0
        env.cumulative_captured_t = 2_000.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=1,
            emitter_availability={"source": [0.0]},
            vessel_speed_factor={vessel_id: [1.0] for vessel_id in env.vessel_ids},
            well_available={"well": [True]},
            injectivity_factor={"well": [1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)
        economics = EconomicParameters(conditioning_eur_per_t=-1_000.0, storage_shortfall_eur_per_t=0.0)

        result = cplex_milp.solve_full_scenario_with_cplex(
            env,
            horizon_h=1,
            economics=economics,
            time_limit_s=10.0,
        )

        loaded_t = result.conditioning / economics.conditioning_eur_per_t
        self.assertTrue(result.is_valid, result.validation_error)
        self.assertGreater(loaded_t, 500.0 - 1e-6)
        self.assertLessEqual(loaded_t, 1_000.0 + 1e-6)

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_relaxed_unloading_can_stop_before_automatic_max_unload(self):
        env = _no_capture_env(cap_hours=1)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 0.0
        env.simulator.state.entity_inventory_t["ship"] = 500.0
        env.simulator.state.entity_inventory_t["terminal"] = 0.0
        env.simulator.state.vessel_berths["ship"] = "terminal"
        env.simulator.vessel_states["ship"] = {
            "mode": "berthed",
            "berth": "terminal",
            "origin": "terminal",
            "destination": "terminal",
            "progress": 0.0,
        }
        env.cumulative_captured_t = 500.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=1,
            emitter_availability={"source": [0.0]},
            vessel_speed_factor={"ship": [1.0]},
            well_available={"well": [True]},
            injectivity_factor={"well": [1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)
        economics = EconomicParameters(storage_shortfall_eur_per_t=1_000.0)

        with patch.object(cplex_milp, "WELL_RATE_LEVELS_MTPA", (0.0, 2.0)):
            result = cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                economics=economics,
                storage_reward_eur_per_t=1_000.0,
                time_limit_s=10.0,
            )

        unloaded_t = result.unloading * 500.0 / economics.hoteling_fuel_eur_per_h
        self.assertTrue(result.is_valid, result.validation_error)
        self.assertGreater(result.stored_t, 0.0)
        self.assertLess(unloaded_t, 500.0)

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_relaxed_unloading_can_choose_non_priority_vessel(self):
        env = _two_ship_priority_env()
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 0.0
        env.simulator.state.entity_inventory_t["terminal"] = 0.0
        env.cumulative_captured_t = 1_500.0
        for vessel_id in env.vessel_ids:
            env.simulator.state.vessel_berths[vessel_id] = "terminal"
            env.simulator.vessel_states[vessel_id] = {
                "mode": "berthed",
                "berth": "terminal",
                "origin": "terminal",
                "destination": "terminal",
                "progress": 0.0,
            }
            env.simulator.state.entity_inventory_t[vessel_id] = env.network.entities[vessel_id].capacity_t
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=1,
            emitter_availability={"source": [0.0]},
            vessel_speed_factor={vessel_id: [1.0] for vessel_id in env.vessel_ids},
            well_available={"well": [True]},
            injectivity_factor={"well": [1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        with patch.object(cplex_milp, "WELL_RATE_LEVELS_MTPA", (0.0, 17.0)):
            result = cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
                storage_reward_eur_per_t=1_000.0,
                time_limit_s=10.0,
            )

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertGreater(result.stored_t, 500.0 - 1e-6)
        self.assertLessEqual(result.stored_t, 1_000.0 + 1e-6)

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_replay_full_scenario_cplex_plan_matches_simple_solution(self):
        env = _no_capture_env(cap_hours=1)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 0.0
        env.simulator.state.entity_inventory_t["ship"] = 500.0
        env.simulator.state.entity_inventory_t["terminal"] = 0.0
        env.simulator.state.vessel_berths["ship"] = "terminal"
        env.simulator.vessel_states["ship"] = {
            "mode": "berthed",
            "berth": "terminal",
            "origin": "terminal",
            "destination": "terminal",
            "progress": 0.0,
        }
        env.cumulative_captured_t = 500.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=1,
            emitter_availability={"source": [0.0]},
            vessel_speed_factor={"ship": [1.0]},
            well_available={"well": [True]},
            injectivity_factor={"well": [1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        result = cplex_milp.solve_full_scenario_with_cplex(
            env,
            horizon_h=1,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
        )
        replay = cplex_milp.replay_full_scenario_cplex_plan(env, result)

        self.assertTrue(replay.is_executable, replay.violations)
        self.assertAlmostEqual(replay.stored_t, result.stored_t, delta=1e-6)

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_injection_request_can_be_supply_clipped_like_rl(self):
        env = _no_capture_env(cap_hours=1)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 0.0
        env.simulator.state.entity_inventory_t["ship"] = 0.0
        env.simulator.state.entity_inventory_t["terminal"] = 250.0
        env.cumulative_captured_t = 500.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=1,
            emitter_availability={"source": [0.0]},
            vessel_speed_factor={"ship": [1.0]},
            well_available={"well": [True]},
            injectivity_factor={"well": [1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        result = cplex_milp.solve_full_scenario_with_cplex(
            env,
            horizon_h=1,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
        )
        replay = cplex_milp.replay_full_scenario_cplex_plan(env, result)

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertAlmostEqual(result.stored_t, 250.0, delta=1e-6)
        self.assertAlmostEqual(replay.stored_t, result.stored_t, delta=1e-6)

    def test_single_well_dynamic_bhp_expression_matches_rl_projection(self):
        env = _no_capture_env(cap_hours=2)
        env.reset(seed=1)
        reservoir = env.network.entities["reservoir"]
        object.__setattr__(
            reservoir,
            "line_source_parameters",
            LineSourceParameters(
                initial_pressure_bar=100.0,
                permeability_md=100.0,
                thickness_m=50.0,
                porosity_fraction=0.2,
                total_compressibility_1_pa=1e-9,
                viscosity_pa_s=5e-5,
                co2_density_kg_m3=700.0,
                well_radius_m=0.1,
                skin=0.0,
            ),
        )
        q0 = cplex_milp.mtpa_to_tph(2.5)
        q1 = cplex_milp.mtpa_to_tph(2.5)

        response_const, coeffs = cplex_milp._single_well_line_source_response_terms(
            env,
            "well",
            horizon_index=1,
            evaluation_h=2.0,
        )
        alpha = cplex_milp._reservoir_pressure_bar_per_tonne(reservoir)
        milp_bhp = (
            reservoir.initial_pressure_bar
            + alpha * (q0 + q1)
            + response_const
            + coeffs[0] * q0
            + coeffs[1] * q1
        )

        projected_state = env.simulator.state.copy()
        projected_state.entity_inventory_t["reservoir"] = q0
        projected_state.injection_rate_history_tph = {"well": [(0.0, q0)]}
        rl_bhp = projected_bottomhole_pressure_bar(
            env.network,
            projected_state,
            "well",
            q1,
            evaluation_time_h=2.0,
            interval_start_h=1.0,
        )

        self.assertAlmostEqual(milp_bhp, rl_bhp)

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_single_well_dynamic_bhp_uses_milp_injection_history(self):
        env = _no_capture_env(cap_hours=2)
        env.reset(seed=1)
        reservoir = env.network.entities["reservoir"]
        object.__setattr__(
            reservoir,
            "line_source_parameters",
            LineSourceParameters(
                initial_pressure_bar=100.0,
                permeability_md=100.0,
                thickness_m=50.0,
                porosity_fraction=0.2,
                total_compressibility_1_pa=1e-9,
                viscosity_pa_s=5e-5,
                co2_density_kg_m3=700.0,
                well_radius_m=0.1,
                skin=0.0,
            ),
        )
        object.__setattr__(reservoir, "well_bottomhole_pressure_limit_bar", 115.0)
        env.simulator.state.entity_inventory_t["terminal"] = 1_000.0
        env.cumulative_captured_t = 1_000.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=2,
            emitter_availability={"source": [0.0, 0.0]},
            vessel_speed_factor={"ship": [1.0, 1.0]},
            well_available={"well": [True, True]},
            injectivity_factor={"well": [1.0, 1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        result = cplex_milp.solve_full_scenario_with_cplex(
            env,
            horizon_h=2,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
        )

        max_index = len(WELL_RATE_LEVELS_MTPA) - 1
        self.assertTrue(result.is_valid, result.validation_error)
        self.assertNotEqual(result.well_rate_indices_by_hour["well"], [max_index, max_index])
        self.assertLess(result.stored_t, 2 * cplex_milp.mtpa_to_tph(WELL_RATE_LEVELS_MTPA[max_index]))


if __name__ == "__main__":
    unittest.main()
