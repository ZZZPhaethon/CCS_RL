import copy
import unittest
import inspect
from dataclasses import replace
from unittest.mock import patch

from sim.control import cplex_milp
from sim.control import rolling_milp
from sim.economics import CostModel, EconomicParameters
from sim.entities import Emitter, InjectionWell, Pipeline, Reservoir, SubseaManifold, Terminal, Vessel
from sim.environment import CCSEnv, CCSEnvConfig, WELL_RATE_LEVELS_MTPA
from sim.environment.env import VESSEL_GO_TERMINAL, VESSEL_WAIT
from sim.line_source import LineSourceParameters
from sim.network import PhysicalNetwork
from sim.operations.pressure_limits import projected_bottomhole_pressure_bar
from sim.routes import route_distance_km, sea_route
from sim.scenario_generation import Scenario, ScenarioConfig, ScenarioGenerator
from tests.test_rolling_milp import (
    _cold_env,
    _no_capture_env,
    _two_berth_parallel_env,
    _two_source_one_ship_fast_env,
)


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
    def test_automatic_well_constraint_uses_continuous_static_maximum(self):
        env = _no_capture_env(
            cap_hours=1,
            well_control_mode="automatic_max",
        )
        env.reset(seed=1)
        scenario = env.scenario
        physical_max = cplex_milp._physical_well_max_by_hour(
            env,
            scenario,
            horizon_h=1,
        )
        problem = cplex_milp.pulp.LpProblem(
            "test_automatic_static_well",
            cplex_milp.pulp.LpMinimize,
        )
        well_request = {
            ("well", 0): cplex_milp.pulp.LpVariable(
                "static_request",
                lowBound=0.0,
                upBound=physical_max[("well", 0)],
            )
        }
        well_inj = {
            ("well", 0): cplex_milp.pulp.LpVariable(
                "static_well_inj",
                lowBound=0.0,
            )
        }
        cplex_milp._add_continuous_automatic_well_request_constraints(
            problem,
            env,
            scenario,
            well_request,
            {},
            well_inj,
            physical_max,
            horizon_h=1,
        )
        problem += well_request[("well", 0)]

        status = problem.solve(cplex_milp.pulp.PULP_CBC_CMD(msg=False))
        requested_tph = well_request[("well", 0)].value()

        self.assertEqual(
            cplex_milp.pulp.LpStatus[status],
            "Optimal",
        )
        self.assertAlmostEqual(
            requested_tph,
            env.automatic_well_rates_tph()[0],
            delta=1e-4,
        )

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_automatic_dynamic_well_constraint_matches_continuous_environment_rate(self):
        env = _no_capture_env(
            cap_hours=1,
            well_control_mode="automatic_max",
        )
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
        lower_index = 2
        upper_index = 3
        lower_pressure = projected_bottomhole_pressure_bar(
            env.network,
            env.simulator.state,
            "well",
            cplex_milp.mtpa_to_tph(
                WELL_RATE_LEVELS_MTPA[lower_index]
            ),
            evaluation_time_h=1.0,
            interval_start_h=0.0,
        )
        upper_pressure = projected_bottomhole_pressure_bar(
            env.network,
            env.simulator.state,
            "well",
            cplex_milp.mtpa_to_tph(
                WELL_RATE_LEVELS_MTPA[upper_index]
            ),
            evaluation_time_h=1.0,
            interval_start_h=0.0,
        )
        object.__setattr__(
            reservoir,
            "well_bottomhole_pressure_limit_bar",
            (lower_pressure + upper_pressure) / 2.0,
        )
        physical_max = cplex_milp._physical_well_max_by_hour(
            env,
            env.scenario,
            horizon_h=1,
        )
        problem = cplex_milp.pulp.LpProblem(
            "test_automatic_dynamic_well",
            cplex_milp.pulp.LpMinimize,
        )
        well_request = {
            ("well", 0): cplex_milp.pulp.LpVariable(
                "dynamic_request",
                lowBound=0.0,
                upBound=physical_max[("well", 0)],
            )
        }
        well_regime = {
            ("well", 0, regime): cplex_milp.pulp.LpVariable(
                f"dynamic_regime_{regime}",
                cat="Binary",
            )
            for regime in ("off", "physical", "pressure")
        }
        well_inj = {
            ("well", 0): cplex_milp.pulp.LpVariable(
                "dynamic_well_inj",
                lowBound=0.0,
            )
        }
        cplex_milp._add_continuous_automatic_well_request_constraints(
            problem,
            env,
            env.scenario,
            well_request,
            well_regime,
            well_inj,
            physical_max,
            horizon_h=1,
        )
        problem += well_request[("well", 0)]

        status = problem.solve(cplex_milp.pulp.PULP_CBC_CMD(msg=False))
        requested_tph = well_request[("well", 0)].value()

        self.assertEqual(
            cplex_milp.pulp.LpStatus[status],
            "Optimal",
        )
        self.assertAlmostEqual(
            requested_tph,
            env.automatic_well_rates_tph()[0],
            delta=1e-4,
        )
        self.assertGreater(
            requested_tph,
            cplex_milp.mtpa_to_tph(
                WELL_RATE_LEVELS_MTPA[lower_index]
            ),
        )
        self.assertLess(
            requested_tph,
            cplex_milp.mtpa_to_tph(
                WELL_RATE_LEVELS_MTPA[upper_index]
            ),
        )

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_automatic_continuous_request_replays_with_supply_clipping(self):
        env = _no_capture_env(
            cap_hours=1,
            well_control_mode="automatic_max",
        )
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["terminal"] = 250.0
        env.cumulative_captured_t = 250.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=1,
            emitter_availability={"source": [0.0]},
            vessel_speed_factor={"ship": [1.0]},
            well_available={"well": [True]},
            injectivity_factor={"well": [1.0]},
        )
        env.scenario.apply_to_state(
            env.simulator.state,
            time_h=0.0,
        )

        with patch.object(
            cplex_milp,
            "_make_cplex_cmd",
            return_value=cplex_milp.pulp.PULP_CBC_CMD(msg=False),
        ):
            result = cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                economics=EconomicParameters(
                    storage_shortfall_eur_per_t=1_000.0
                ),
                economic_objective=True,
                time_limit_s=10.0,
            )
        replay = cplex_milp.replay_full_scenario_cplex_plan(
            env,
            result,
        )

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertEqual(result.well_rate_indices_by_hour, {})
        self.assertNotIn("wells", result.native_actions_by_hour[0])
        self.assertAlmostEqual(
            result.well_request_tph_by_hour["well"][0],
            env.automatic_well_rates_tph()[0],
            delta=1e-6,
        )
        self.assertAlmostEqual(result.stored_t, 250.0, delta=1e-6)
        self.assertTrue(replay.is_exact, replay.mismatches)

    def test_cross_source_leg_distance_is_independent_of_vessel_home_route(self):
        env = _cold_env(cap_hours=24)
        env.reset(seed=1)
        terminal_id = env.terminal_ids[0]
        source_id = env.emitter_ids[0]
        origin = env.locations[source_id]
        destination = env.locations[terminal_id]
        maritime_route = sea_route(origin, destination)
        coordinates = list(maritime_route.coordinates)
        if not coordinates:
            coordinates = [origin, destination]
        else:
            if coordinates[0] != origin:
                coordinates.insert(0, origin)
            if coordinates[-1] != destination:
                coordinates.append(destination)
        expected = round(route_distance_km(coordinates), 2)

        for vessel_id in env.vessel_ids:
            route = env._routes[vessel_id]
            simulator_leg = env.simulator._dynamic_leg_route(
                route, source_id, terminal_id
            )
            self.assertAlmostEqual(simulator_leg["distance_km"], expected)
            self.assertAlmostEqual(
                cplex_milp._dynamic_leg_distance_km(
                    env, route, source_id, terminal_id
                ),
                expected,
            )
            self.assertAlmostEqual(
                rolling_milp._dynamic_leg_distance_km(
                    env, route, source_id, terminal_id
                ),
                expected,
            )

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
    def test_cplex_solver_ignores_parallel_default_log_cleanup_lock(self):
        cmd = cplex_milp._make_cplex_cmd(msg=False)

        with patch.object(cmd, "delete_tmp_files", side_effect=PermissionError("locked")):
            cmd._delete_default_log()

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_status_label_distinguishes_time_limit_feasible_from_optimal(self):
        label = cplex_milp._solution_status_label(
            cplex_milp.pulp.constants.LpStatusOptimal,
            cplex_milp.pulp.constants.LpSolutionIntegerFeasible,
        )

        self.assertEqual(label, "Integer Feasible")

    def test_parse_cplex_log_extracts_time_limit_diagnostics(self):
        parsed = cplex_milp._parse_cplex_log(
            """
Reduced MIP has 1,234 rows, 5,678 columns, and 90,123 nonzeros.
MIP start 'm1' defined initial solution with objective 1000.0000.
Found incumbent of value 9.5000000000e+02 after 1.25 sec. (10.00 ticks)
MIP - Time limit exceeded, integer feasible:  Objective =  3.9524708239e+02
Current MIP best bound =  3.1000000000e+02 (gap = 27.5%)
Solution time =  120.01 sec.  Iterations = 12,345  Nodes = 678
""",
            warm_start_requested=True,
        )

        self.assertEqual(parsed["termination_reason"], "Time limit exceeded, integer feasible")
        self.assertEqual(parsed["best_bound"], 310.0)
        self.assertAlmostEqual(parsed["relative_gap"], 0.275)
        self.assertEqual(parsed["iterations"], 12345)
        self.assertEqual(parsed["nodes"], 678)
        self.assertEqual(parsed["reduced_rows"], 1234)
        self.assertEqual(parsed["reduced_columns"], 5678)
        self.assertEqual(parsed["reduced_nonzeros"], 90123)
        self.assertEqual(parsed["first_incumbent_time_s"], 0.0)
        self.assertEqual(parsed["first_incumbent_objective"], 1000.0)
        self.assertTrue(parsed["warm_start_accepted"])
        self.assertIn("defined initial solution", parsed["warm_start_message"])

    def test_parse_cplex_log_extracts_time_limit_without_integer_solution(self):
        parsed = cplex_milp._parse_cplex_log(
            "MIP - Time limit exceeded, no integer solution.\n",
            warm_start_requested=False,
        )

        self.assertEqual(
            parsed["termination_reason"],
            "Time limit exceeded, no integer solution.",
        )
        self.assertIsNone(parsed["first_incumbent_time_s"])
        self.assertIsNone(parsed["first_incumbent_objective"])

    def test_parse_cplex_log_extracts_first_incumbent_without_warm_start(self):
        parsed = cplex_milp._parse_cplex_log(
            "Found incumbent of value 4.2500e+02 after 12.75 sec. (30 ticks)\n",
            warm_start_requested=False,
        )

        self.assertEqual(parsed["first_incumbent_time_s"], 12.75)
        self.assertEqual(parsed["first_incumbent_objective"], 425.0)

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_mip_start_audit_reports_violated_and_partial_constraints(self):
        problem = cplex_milp.pulp.LpProblem("mip_start_audit")
        x = cplex_milp.pulp.LpVariable("x", cat="Binary")
        y = cplex_milp.pulp.LpVariable("y", lowBound=0.0)
        problem += x <= 0.0
        problem += x + y <= 2.0
        x.setInitialValue(1.0)

        audit = cplex_milp._audit_mip_start(problem)

        self.assertEqual(audit.total_variables, 2)
        self.assertEqual(audit.initialized_variables, 1)
        self.assertEqual(audit.missing_variable_names, ("y",))
        self.assertEqual(audit.evaluated_constraints, 1)
        self.assertEqual(audit.partial_constraint_count, 1)
        self.assertEqual(audit.violated_constraint_count, 1)
        self.assertAlmostEqual(audit.max_constraint_violation, 1.0)

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_mip_start_value_clamps_floating_point_roundoff_to_bounds(self):
        terminal_stock = cplex_milp.pulp.LpVariable(
            "terminal_stock_159",
            lowBound=0.0,
            upBound=9150.0,
        )

        cplex_milp._set_start_value(
            {159: terminal_stock},
            159,
            9150.000000000002,
        )

        self.assertEqual(terminal_stock.varValue, 9150.0)

    def test_initial_sailing_fuel_hours_include_only_in_horizon_sailing_steps(self):
        starts = {
            "arrives": cplex_milp._PathStart(start_h=4, node_id="terminal"),
            "beyond_horizon": cplex_milp._PathStart(start_h=10, node_id=None),
            "berthed": cplex_milp._PathStart(start_h=0, node_id="source"),
        }

        hours = cplex_milp._initial_sailing_fuel_hours(starts, horizon_h=10)

        self.assertEqual(hours, 13)

    def test_reachable_action_arc_pruning_follows_zero_hour_closure(self):
        arcs = [
            cplex_milp._ActionArc("ship", 0, 0, "a", "b", 1, True),
            cplex_milp._ActionArc("ship", 0, 1, "b", "c", 2, True),
            cplex_milp._ActionArc("ship", 0, 1, "x", "c", 2, True),
            cplex_milp._ActionArc("ship", 1, 2, "c", "c", 0, False),
        ]

        reachable = cplex_milp._reachable_action_arcs(
            arcs,
            {"ship": cplex_milp._PathStart(0, "a")},
        )

        self.assertEqual(reachable, [arcs[0], arcs[1], arcs[3]])

    def test_truncated_arc_carries_remaining_cleanup_fuel(self):
        env = _no_capture_env(cap_hours=3)
        env._routes["ship"]["distance_km"] = 10.0 * 1.852
        env.reset(seed=1)
        scenario = Scenario(
            time_step_hours=1.0,
            n_steps=3,
            emitter_availability={"source": [1.0] * 3},
            vessel_speed_factor={"ship": [1.0] * 3},
            well_available={"well": [True] * 3},
            injectivity_factor={"well": [1.0] * 3},
        )

        arcs, _starts = cplex_milp._build_action_arcs(
            env, scenario, start_step=0, horizon_h=3
        )
        arc = next(
            item
            for item in arcs
            if item.is_sailing
            and item.origin_id == "source"
            and item.destination_id == "terminal"
            and item.start_h == 0
        )

        self.assertFalse(arc.arrives_within_horizon)
        self.assertEqual(arc.remaining_cleanup_fuel_h, 7)

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_fixed_state_cleanup_prices_remaining_sailing_and_ready_return(self):
        env = _no_capture_env(cap_hours=2)
        env._routes["ship"]["distance_km"] = 10.0 * 1.852
        env.reset(seed=1)
        env.simulator.state.vessel_berths["ship"] = None
        env.simulator.vessel_states["ship"].update(
            {
                "mode": "sailing",
                "berth": None,
                "origin": "source",
                "destination": "terminal",
                "progress": 0.4,
                "distance_km": 10.0 * 1.852,
            }
        )
        params = EconomicParameters()

        cost = cplex_milp._terminal_cleanup_cost_for_state(env, params)

        self.assertAlmostEqual(
            cost,
            (6.0 + 9.0) * params.vessel_fuel_eur_per_h_sailing,
            places=6,
        )

    def test_fifo_diagnostic_mode_rejects_unknown_value(self):
        env = _no_capture_env(cap_hours=1)
        env.reset(seed=1)

        with self.assertRaisesRegex(ValueError, "fifo_diagnostic_mode"):
            cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                fifo_diagnostic_mode="unknown",
            )
        with self.assertRaisesRegex(
            ValueError, "Unknown integrality relaxation groups"
        ):
            cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                integrality_relax_groups=("unknown",),
            )
        with self.assertRaisesRegex(ValueError, "vessel_visit_load_cut_stride_h"):
            cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                vessel_visit_load_cuts=True,
                vessel_visit_load_cut_stride_h=0,
            )
        with self.assertRaisesRegex(ValueError, "source_visit_vent_cut_stride_h"):
            cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                source_visit_vent_cuts=True,
                source_visit_vent_cut_stride_h=0,
            )
        with self.assertRaisesRegex(ValueError, "terminal_visit_cut_stride_h"):
            cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                terminal_visit_cuts=True,
                terminal_visit_cut_stride_h=0,
            )
        with self.assertRaisesRegex(
            ValueError, "service_reachability_cut_stride_h"
        ):
            cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                service_reachability_cuts=True,
                service_reachability_cut_stride_h=0,
            )
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                fixed_terminal_departures_by_vessel={"ship": -1},
            )
        with self.assertRaisesRegex(ValueError, "Unknown vessel/source pairs"):
            cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                fixed_terminal_departures_by_vessel_source={
                    ("ship", "unknown_source"): 0
                },
            )
        with self.assertRaisesRegex(ValueError, "fixed source reposition"):
            cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                fixed_source_reposition_departures_by_vessel={"ship": -1},
            )
        with self.assertRaisesRegex(ValueError, "minimum total source"):
            cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                min_total_source_reposition_departures=-1,
            )
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                fixed_terminal_departures_by_vessel_source={
                    ("ship", "source"): -1
                },
            )
        with self.assertRaisesRegex(ValueError, "Unknown vessel/source pairs"):
            cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                fixed_terminal_to_source_departures_by_vessel_source={
                    ("ship", "unknown_source"): 0
                },
            )

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_terminal_cleanup_value_reports_separate_future_cost(self):
        env = _no_capture_env(cap_hours=2)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 500.0

        result = cplex_milp.solve_full_scenario_with_cplex(
            env,
            horizon_h=1,
            economics=EconomicParameters(),
            economic_objective=True,
            terminal_cleanup_value=True,
            environment_aligned_service=True,
            cleanup_unary_trip_slots=True,
            time_limit_s=10.0,
        )

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertTrue(result.terminal_cleanup_value_enabled)
        self.assertGreater(result.terminal_cleanup_cost, 0.0)
        self.assertAlmostEqual(
            result.terminal_cleanup_cost,
            result.terminal_cleanup_vessel_fuel
            + result.terminal_cleanup_conditioning
            + result.terminal_cleanup_reconditioning
            + result.terminal_cleanup_loading
            + result.terminal_cleanup_unloading,
            places=6,
        )
        self.assertAlmostEqual(
            result.augmented_objective_value,
            result.objective_value + result.terminal_cleanup_cost,
            places=6,
        )
        terminal_env = copy.deepcopy(env)
        replay = cplex_milp.replay_native_actions(
            terminal_env,
            result.native_actions_by_hour,
            horizon_h=1,
            copy_env=False,
        )
        self.assertTrue(replay.is_executable)
        self.assertAlmostEqual(
            cplex_milp._terminal_cleanup_cost_for_state(
                terminal_env,
                EconomicParameters(),
            ),
            result.terminal_cleanup_cost,
            places=6,
        )

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_action_warm_start_completes_terminal_cleanup_variables(self):
        env = _no_capture_env(cap_hours=2)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 500.0
        actions = [{"vessels": [VESSEL_WAIT], "wells": [0]}]
        params = EconomicParameters()

        result = cplex_milp.solve_full_scenario_with_cplex(
            env,
            horizon_h=1,
            economics=params,
            warm_start_native_actions_by_hour=actions,
            economic_objective=True,
            terminal_cleanup_value=True,
            terminal_cleanup_mip_start_mode="complete",
            environment_aligned_service=True,
            cleanup_unary_trip_slots=True,
            time_limit_s=10.0,
        )

        self.assertIsNotNone(result.mip_start_audit)
        self.assertFalse(
            any(
                name.startswith("tail_")
                for name in result.mip_start_audit.missing_variable_names
            )
        )
        seeded_cleanup_cost = result.diagnostic_variable_values[
            "mip_start_terminal_cleanup_cost"
        ]
        terminal_env = copy.deepcopy(env)
        replay = cplex_milp.replay_native_actions(
            terminal_env,
            actions,
            horizon_h=1,
            copy_env=False,
        )
        self.assertTrue(replay.is_executable)
        self.assertAlmostEqual(
            seeded_cleanup_cost,
            cplex_milp._terminal_cleanup_cost_for_state(terminal_env, params),
            places=6,
        )

    def test_warm_start_unload_replay_preserves_terminal_fifo_head(self):
        env = _two_berth_parallel_env()
        env.reset(seed=1)
        state = env.simulator.state
        for vessel_id in env.vessel_ids:
            state.entity_inventory_t[vessel_id] = 1_000.0
            state.vessel_berths[vessel_id] = "terminal"
            env.simulator.vessel_states[vessel_id].update(
                {
                    "mode": "berthed",
                    "berth": "terminal",
                    "origin": "terminal",
                    "destination": "terminal",
                    "progress": 0.0,
                    "distance_km": 0.0,
                }
            )
        state.terminal_unload_queues["terminal"] = ["ship_b", "ship_a"]
        actions = [{"vessels": [VESSEL_WAIT, VESSEL_WAIT], "wells": [0]}]

        unloads = cplex_milp._replay_native_action_unloads(env, actions, horizon_h=1)

        self.assertEqual(unloads, {0: {"ship_b": 1_000.0}})

    def test_warm_start_unload_replay_keeps_blocked_fifo_head_active(self):
        env = _two_berth_parallel_env()
        env.reset(seed=1)
        state = env.simulator.state
        terminal = env.network.entities["terminal"]
        state.entity_inventory_t["terminal"] = terminal.storage_capacity_t
        for vessel_id in env.vessel_ids:
            state.entity_inventory_t[vessel_id] = 1_000.0
            state.vessel_berths[vessel_id] = "terminal"
            env.simulator.vessel_states[vessel_id].update(
                {
                    "mode": "berthed",
                    "berth": "terminal",
                    "origin": "terminal",
                    "destination": "terminal",
                    "progress": 0.0,
                    "distance_km": 0.0,
                }
            )
        state.terminal_unload_queues["terminal"] = ["ship_b", "ship_a"]
        actions = [{"vessels": [VESSEL_WAIT, VESSEL_WAIT], "wells": [0]}]

        unloads = cplex_milp._replay_native_action_unloads(
            env, actions, horizon_h=1
        )

        self.assertEqual(unloads, {0: {"ship_b": 0.0}})

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_lexicographic_solver_keeps_last_feasible_stage_if_next_stage_fails(self):
        env = _no_capture_env(cap_hours=1)
        env.reset(seed=1)
        original_solve = cplex_milp.pulp.LpProblem.solve
        calls = 0

        def fail_second_stage(problem, solver=None, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                for variable in problem.variables():
                    variable.varValue = None
                problem.status = cplex_milp.pulp.constants.LpStatusInfeasible
                problem.sol_status = cplex_milp.pulp.constants.LpSolutionNoSolutionFound
                return problem.status
            return original_solve(problem, solver, **kwargs)

        with patch.object(cplex_milp.pulp.LpProblem, "solve", new=fail_second_stage):
            result = cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                economics=EconomicParameters(),
                lexicographic_vent_first=True,
                time_limit_s=10.0,
            )

        self.assertEqual(calls, 2)
        self.assertTrue(result.is_valid, result.validation_error)
        self.assertEqual(result.status, "Optimal")
        self.assertEqual(
            [diagnostic.stage for diagnostic in result.stage_diagnostics],
            ["vent", "end_unstored"],
        )

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

    def test_weather_cleanup_leg_bound_uses_best_full_forecast_departure(self):
        env = _no_capture_env(cap_hours=4)
        env.reset(seed=1)
        route = env._routes["ship"]
        scenario = Scenario(
            time_step_hours=1.0,
            n_steps=4,
            emitter_availability={"source": [1.0] * 4},
            vessel_speed_factor={"ship": [0.5, 1.0, 1.0, 1.0]},
            well_available={"well": [True] * 4},
            injectivity_factor={"well": [1.0] * 4},
        )

        fuel_h = cplex_milp._best_future_weather_leg_fuel_h(
            env,
            scenario,
            "ship",
            origin_id="source",
            destination_id="terminal",
            distance_km=float(route["distance_km"]),
            earliest_start_step=0,
        )

        self.assertEqual(fuel_h, 0)

        scenario.vessel_speed_factor["ship"] = [0.5] * 4
        fuel_h = cplex_milp._best_future_weather_leg_fuel_h(
            env,
            scenario,
            "ship",
            origin_id="source",
            destination_id="terminal",
            distance_km=float(route["distance_km"]),
            earliest_start_step=0,
        )

        self.assertEqual(fuel_h, 1)

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_cleanup_source_mode_partition_cut_preserves_integer_tail_cost(self):
        env = _no_capture_env(cap_hours=4)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 500.0
        params = EconomicParameters()

        baseline = cplex_milp._terminal_cleanup_cost_for_state(env, params)
        partitioned = cplex_milp._terminal_cleanup_cost_for_state(
            env,
            params,
            source_mode_partition_cut=True,
        )

        self.assertAlmostEqual(partitioned, baseline, places=6)

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_cleanup_headroom_risk_prices_capture_before_first_response(self):
        env = _no_capture_env(cap_hours=4)
        env.reset(seed=1)
        emitter = replace(
            env.network.entities["source"], nominal_capture_tph=100.0
        )
        env.network.entities["source"] = emitter
        env.simulator.state.entity_inventory_t["source"] = 950.0
        env.simulator.state.entity_inventory_t["ship"] = 500.0
        env.simulator.state.vessel_berths["ship"] = "terminal"
        env.simulator.vessel_states["ship"].update(
            {
                "mode": "berthed",
                "berth": "terminal",
                "origin": "terminal",
                "destination": "terminal",
                "progress": 0.0,
            }
        )
        params = EconomicParameters()

        baseline = cplex_milp._terminal_cleanup_cost_for_state(env, params)
        risk_cost, values = cplex_milp._terminal_cleanup_solution_for_state(
            env,
            params,
            source_headroom_risk=True,
        )
        normal_capture_tph = (
            emitter.nominal_capture_tph
            * emitter.availability
            * emitter.default_utilization
        )
        expected_vent_t = max(
            0.0,
            950.0 + 2.0 * normal_capture_tph - emitter.buffer_capacity_t,
        )

        self.assertAlmostEqual(
            values["tail_headroom_vent_source"], expected_vent_t
        )
        self.assertAlmostEqual(
            risk_cost - baseline,
            expected_vent_t * params.carbon_price_eur_per_t,
            places=6,
        )

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_cleanup_response_time_includes_remaining_leg_and_actual_unload(self):
        env = _no_capture_env(cap_hours=4)
        env._routes["ship"]["distance_km"] = 10.0 * 1.852
        env.reset(seed=1)
        emitter = replace(
            env.network.entities["source"], nominal_capture_tph=10.0
        )
        env.network.entities["source"] = emitter
        env.simulator.state.entity_inventory_t["source"] = 900.0
        env.simulator.state.entity_inventory_t["ship"] = 500.0
        env.simulator.state.vessel_berths["ship"] = None
        env.simulator.vessel_states["ship"].update(
            {
                "mode": "sailing",
                "berth": None,
                "origin": "source",
                "destination": "terminal",
                "progress": 0.4,
                "distance_km": 10.0 * 1.852,
            }
        )
        params = EconomicParameters()

        baseline = cplex_milp._terminal_cleanup_cost_for_state(env, params)
        timed = cplex_milp._terminal_cleanup_cost_for_state(
            env,
            params,
            source_headroom_risk=True,
        )
        expected_response_h = 6.0 + 1.0 + 10.0
        expected_vent_t = max(
            0.0,
            900.0
            + emitter.nominal_capture_tph * expected_response_h
            - emitter.buffer_capacity_t,
        )

        self.assertAlmostEqual(
            timed - baseline,
            expected_vent_t * params.carbon_price_eur_per_t,
            places=6,
        )

    @unittest.skipIf(cplex_milp.pulp is None, "pulp not installed")
    def test_weather_cleanup_bound_raises_fixed_state_tail_under_slow_weather(self):
        env = _no_capture_env(cap_hours=4)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 500.0
        env.scenario.vessel_speed_factor["ship"] = [0.5] * 4
        params = EconomicParameters()

        nominal = cplex_milp._terminal_cleanup_cost_for_state(env, params)
        weather = cplex_milp._terminal_cleanup_cost_for_state(
            env,
            params,
            weather_aware_sailing_lower_bound=True,
        )

        self.assertGreater(weather, nominal)

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
    def test_environment_aligned_service_allows_two_vessels_at_one_terminal(self):
        env = _two_ship_priority_env()
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 0.0
        env.simulator.state.entity_inventory_t["terminal"] = 0.0
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
        env.cumulative_captured_t = sum(
            env.simulator.state.entity_inventory_t[vessel_id]
            for vessel_id in env.vessel_ids
        )

        result = cplex_milp.solve_full_scenario_with_cplex(
            env,
            horizon_h=1,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
            storage_reward_eur_per_t=1_000.0,
            time_limit_s=10.0,
            environment_aligned_service=True,
        )

        self.assertTrue(result.is_valid, result.validation_error)

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_environment_aligned_service_unloads_terminal_fifo_head(self):
        env = _two_ship_priority_env()
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 0.0
        env.simulator.state.entity_inventory_t["terminal"] = 0.0
        for vessel_id in env.vessel_ids:
            env.simulator.state.vessel_berths[vessel_id] = "terminal"
            env.simulator.vessel_states[vessel_id] = {
                "mode": "berthed",
                "berth": "terminal",
                "origin": "terminal",
                "destination": "terminal",
                "progress": 0.0,
                "distance_km": 0.0,
            }
            env.simulator.state.entity_inventory_t[vessel_id] = env.network.entities[vessel_id].capacity_t
        env.simulator.state.terminal_unload_queues["terminal"] = ["ship_b", "ship_a"]

        result = cplex_milp.solve_full_scenario_with_cplex(
            env,
            horizon_h=1,
            economics=EconomicParameters(storage_shortfall_eur_per_t=0.0),
            storage_reward_eur_per_t=0.0,
            time_limit_s=10.0,
            environment_aligned_service=True,
        )

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertAlmostEqual(result.unload_t_by_hour["ship_b"][0], 1_000.0)
        self.assertAlmostEqual(result.unload_t_by_hour["ship_a"][0], 0.0)

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_environment_aligned_service_uses_environment_loading_order(self):
        env = _two_ship_high_rate_source_env()
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 1_500.0
        env.simulator.state.entity_inventory_t["terminal"] = 0.0
        for vessel_id in env.vessel_ids:
            env.simulator.state.entity_inventory_t[vessel_id] = 0.0
            env._routes[vessel_id]["distance_km"] = 1_000.0

        result = cplex_milp.solve_full_scenario_with_cplex(
            env,
            horizon_h=1,
            economics=EconomicParameters(
                storage_shortfall_eur_per_t=0.0,
                ship_fuel_cost_hfo_eur_per_t=1e9,
            ),
            storage_reward_eur_per_t=0.0,
            time_limit_s=10.0,
            environment_aligned_service=True,
        )

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertAlmostEqual(result.load_t_by_hour["ship_a"][0], 1_000.0)
        self.assertAlmostEqual(result.load_t_by_hour["ship_b"][0], 0.0)

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_factored_load_min_matches_choice3_and_replay(self):
        base_env = _no_capture_env(cap_hours=1)
        base_env.reset(seed=1)
        base_env.simulator.state.entity_inventory_t["source"] = 400.0
        base_env.simulator.state.entity_inventory_t["ship"] = 200.0
        base_env.simulator.state.entity_inventory_t["terminal"] = 0.0
        base_env._routes["ship"]["distance_km"] = 1_000.0
        economics = EconomicParameters(ship_fuel_cost_hfo_eur_per_t=1e9)
        base_env.cost_model = CostModel(economics)

        results = {}
        replays = {}
        for formulation in ("choice3", "factored"):
            env = copy.deepcopy(base_env)
            result = cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=1,
                economics=economics,
                storage_reward_eur_per_t=0.0,
                time_limit_s=10.0,
                environment_aligned_service=True,
                load_min_formulation=formulation,
            )
            results[formulation] = result
            replays[formulation] = cplex_milp.replay_full_scenario_cplex_plan(
                copy.deepcopy(base_env),
                result,
            )

        for result in results.values():
            self.assertTrue(result.is_valid, result.validation_error)
        for replay in replays.values():
            self.assertTrue(replay.is_exact, replay.mismatches)
        for formulation in ("factored",):
            for vessel_id in base_env.vessel_ids:
                self.assertAlmostEqual(
                    results[formulation].load_t_by_hour[vessel_id][0],
                    results["choice3"].load_t_by_hour[vessel_id][0],
                )
            self.assertAlmostEqual(results[formulation].load_t_by_hour["ship"][0], 300.0)
            self.assertAlmostEqual(
                results[formulation].total_cost,
                results["choice3"].total_cost,
            )

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_valid_inequality_groups_preserve_optimal_solution_and_replay(self):
        base_env = _no_capture_env(cap_hours=2)
        base_env.reset(seed=1)
        base_env.simulator.state.entity_inventory_t["source"] = 400.0
        base_env.simulator.state.entity_inventory_t["ship"] = 200.0
        economics = EconomicParameters(ship_fuel_cost_hfo_eur_per_t=1e9)
        base_env.cost_model = CostModel(economics)

        variants = {
            "baseline": {},
            "vessel_visit_load": {"vessel_visit_load_cuts": True},
            "source_visit_vent": {"source_visit_vent_cuts": True},
            "terminal_visit": {"terminal_visit_cuts": True},
            "service_reachability": {"service_reachability_cuts": True},
            "route_cargo_flow": {"route_cargo_flow_linking": True},
            "prune_unreachable_route_arcs": {
                "prune_unreachable_route_arcs": True
            },
        }
        results = {}
        for variant, cut_kwargs in variants.items():
            env = copy.deepcopy(base_env)
            result = cplex_milp.solve_full_scenario_with_cplex(
                env,
                horizon_h=2,
                economics=economics,
                storage_reward_eur_per_t=0.0,
                time_limit_s=10.0,
                environment_aligned_service=True,
                **cut_kwargs,
            )
            replay = cplex_milp.replay_full_scenario_cplex_plan(
                copy.deepcopy(base_env),
                result,
            )
            self.assertTrue(result.is_valid, result.validation_error)
            self.assertTrue(replay.is_exact, replay.mismatches)
            results[variant] = result

        for variant in (
            "vessel_visit_load",
            "source_visit_vent",
            "terminal_visit",
            "service_reachability",
            "route_cargo_flow",
            "prune_unreachable_route_arcs",
        ):
            self.assertAlmostEqual(
                results[variant].total_cost,
                results["baseline"].total_cost,
            )
            self.assertEqual(
                results[variant].native_actions_by_hour,
                results["baseline"].native_actions_by_hour,
            )

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_route_cargo_flow_cleanup_link_preserves_optimum_and_replay(self):
        base_env = _no_capture_env(cap_hours=2)
        base_env.reset(seed=1)
        base_env.simulator.state.entity_inventory_t["source"] = 400.0
        base_env.simulator.state.entity_inventory_t["ship"] = 200.0
        economics = EconomicParameters(ship_fuel_cost_hfo_eur_per_t=1e9)
        base_env.cost_model = CostModel(economics)

        variants = {
            "baseline": {},
            "route_cargo_flow": {"route_cargo_flow_linking": True},
            "cleanup_unary_trips": {
                "route_cargo_flow_linking": True,
                "cleanup_unary_trip_slots": True,
            },
            "cleanup_aggregate_full_trip_dominance": {
                "route_cargo_flow_linking": True,
                "cleanup_unary_trip_slots": True,
                "cleanup_aggregate_full_trip_dominance": True,
            },
            "cleanup_return_partition": {
                "route_cargo_flow_linking": True,
                "cleanup_unary_trip_slots": True,
                "cleanup_return_partition_cut": True,
            },
        }
        results = {}
        for variant, kwargs in variants.items():
            result = cplex_milp.solve_full_scenario_with_cplex(
                copy.deepcopy(base_env),
                horizon_h=2,
                economics=economics,
                economic_objective=True,
                terminal_cleanup_value=True,
                environment_aligned_service=True,
                time_limit_s=10.0,
                **kwargs,
            )
            replay = cplex_milp.replay_full_scenario_cplex_plan(
                copy.deepcopy(base_env), result
            )
            self.assertTrue(result.is_valid, result.validation_error)
            self.assertTrue(replay.is_exact, replay.mismatches)
            results[variant] = result

        for variant in (
            "route_cargo_flow",
            "cleanup_unary_trips",
            "cleanup_aggregate_full_trip_dominance",
            "cleanup_return_partition",
        ):
            self.assertAlmostEqual(
                results[variant].augmented_objective_value,
                results["baseline"].augmented_objective_value,
            )

    @unittest.skipIf(
        cplex_milp.pulp is None or not cplex_milp.pulp.CPLEX_CMD(msg=0).available(),
        "external CPLEX executable not available",
    )
    def test_environment_aligned_service_forces_automatic_terminal_unloading(self):
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
            "distance_km": 0.0,
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
            economics=EconomicParameters(storage_shortfall_eur_per_t=0.0),
            storage_reward_eur_per_t=0.0,
            time_limit_s=10.0,
            environment_aligned_service=True,
        )

        self.assertTrue(result.is_valid, result.validation_error)
        self.assertAlmostEqual(result.unload_t_by_hour["ship"][0], 500.0)

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
