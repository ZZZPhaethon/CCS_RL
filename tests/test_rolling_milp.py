import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import pulp  # noqa: F401

    HAVE_PULP = True
    HAVE_CPLEX = bool(pulp.CPLEX_CMD(msg=0).available())
except ImportError:
    HAVE_PULP = False
    HAVE_CPLEX = False

from sim.control.rolling_milp import (
    RollingMilpController,
    _capture_tonnes,
    _initial_root_cplex_options,
    _materialize_cplex_actions,
    _native_warm_start_score,
    _native_mpc_warm_start,
    _plan_native_cplex_actions,
    _sail_hours_between,
    _shifted_milp_warm_start,
    _solve_native_cplex_result,
)
from sim.control.native_mpc import (
    RollingNativeMpcController,
    _NativeMpcCandidate,
    _select_native_mpc_candidate,
    native_mpc_candidate_names,
)
from sim.control.baselines import greedy_shuttle_policy
from sim.economics import EconomicParameters
from sim.entities import Emitter, InjectionWell, Pipeline, Reservoir, SubseaManifold, Terminal, Vessel
from sim.environment import (
    CCSEnv,
    CCSEnvConfig,
    VESSEL_GO_TERMINAL,
    VESSEL_WAIT,
)
from sim.metrics import run_episode
from sim.network import PhysicalNetwork
from sim.routes import route_distance_km, sea_route
from sim.scenario_generation import Scenario, ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network


def _cold_env(cap_hours: int = 600, **env_config) -> CCSEnv:
    # Cold start (no initial inventory) so the MILP bound and the controllers face
    # the same empty-system task.
    return CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(
            config=ScenarioConfig(episode_hours=cap_hours, randomize_initial_inventory=False)
        ),
        config=CCSEnvConfig(episode_hours=cap_hours, **env_config),
    )


def _no_capture_env(cap_hours: int = 24, **env_config) -> CCSEnv:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("source", nominal_capture_tph=0.0, buffer_capacity_t=1_000.0))
    network.add_entity(Vessel("ship", capacity_t=500.0, loading_rate_tph=500.0, unloading_rate_tph=500.0, speed_knots=100.0))
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
        scenario_generator=ScenarioGenerator(
            config=ScenarioConfig(episode_hours=cap_hours, randomize_initial_inventory=False)
        ),
        config=CCSEnvConfig(episode_hours=cap_hours, **env_config),
        routes={
            "ship": {
                "origin": "source",
                "destination": "terminal",
                "distance_km": 1.852,
                "speed_knots": 1.0,
                "coordinates": [(0.0, 0.0), (0.0, 1.0)],
                "return_coordinates": [(0.0, 1.0), (0.0, 0.0)],
            },
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
        scenario_generator=ScenarioGenerator(
            config=ScenarioConfig(episode_hours=3, randomize_initial_inventory=False)
        ),
        config=CCSEnvConfig(episode_hours=3),
        routes={
            "ship_a": {"origin": "source", "destination": "terminal", "distance_km": 1.852, "speed_knots": 1.0},
            "ship_b": {"origin": "source", "destination": "terminal", "distance_km": 1.852, "speed_knots": 1.0},
        },
    )


def _two_source_one_ship_fast_env() -> CCSEnv:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("source_a", nominal_capture_tph=0.0, buffer_capacity_t=2_000.0))
    network.add_entity(Emitter("source_b", nominal_capture_tph=0.0, buffer_capacity_t=2_000.0))
    network.add_entity(Vessel("ship", capacity_t=500.0, loading_rate_tph=500.0, unloading_rate_tph=500.0, speed_knots=100000.0))
    network.add_entity(Terminal("terminal", storage_capacity_t=2_000.0, berth_count=1))
    network.add_entity(Pipeline("pipeline", max_flow_tph=500.0))
    network.add_entity(SubseaManifold("manifold", max_flow_tph=500.0))
    network.add_entity(InjectionWell("well", max_injection_tph=500.0))
    network.add_entity(Reservoir("reservoir", storage_capacity_t=1e7, initial_pressure_bar=100.0, pressure_at_capacity_bar=200.0, max_pressure_bar=200.0))
    network.connect("source_a", "ship")
    network.connect("ship", "terminal")
    network.connect("terminal", "pipeline")
    network.connect("pipeline", "manifold")
    network.connect("manifold", "well")
    network.connect("well", "reservoir")
    return CCSEnv(
        network,
        {
            "source_a": (59.05, 9.70),
            "source_b": (59.86, 10.84),
            "terminal": (60.58, 4.84),
        },
        scenario_generator=ScenarioGenerator(
            config=ScenarioConfig(episode_hours=12, randomize_initial_inventory=False)
        ),
        config=CCSEnvConfig(episode_hours=12),
        routes={
            "ship": {"origin": "source_a", "destination": "terminal", "distance_km": 1.852, "speed_knots": 100000.0},
        },
    )


class RollingMilpInterfaceTests(unittest.TestCase):
    def test_automatic_well_mode_returns_vessel_control_only(self):
        env = _cold_env(24, well_control_mode="automatic_max")
        env.reset(seed=0)
        controller = RollingMilpController(env, replan_every=12)
        controller._plan_origin_h = 0.0
        controller._has_active_plan = True
        controller._native_actions_by_hour = [
            {
                "vessels": [VESSEL_WAIT] * len(env.vessel_ids),
                "wells": [0] * len(env.well_ids),
            }
        ]

        action = controller.policy(env)

        self.assertEqual(
            action,
            {"vessels": [VESSEL_WAIT] * len(env.vessel_ids)},
        )

    def test_warm_start_score_includes_terminal_cleanup_value(self):
        env = _no_capture_env(cap_hours=2)
        env.reset(seed=1)
        actions = [
            {"vessels": [VESSEL_WAIT], "wells": [0]},
            {"vessels": [VESSEL_WAIT], "wells": [0]},
        ]

        with patch(
            "sim.control.cplex_milp._terminal_cleanup_cost_for_state",
            return_value=123.0,
        ) as cleanup:
            score = _native_warm_start_score(
                env,
                actions,
                2,
                "economic",
                economics=EconomicParameters(),
                terminal_cleanup_value=True,
            )

        self.assertIsNotNone(score)
        self.assertAlmostEqual(score[0], 123.0)
        cleanup.assert_called_once()

    def test_initial_full_horizon_uses_barrier_without_affecting_later_windows(self):
        env = _no_capture_env(cap_hours=200)
        env.reset(seed=1)

        self.assertEqual(
            _initial_root_cplex_options(env, 168, enabled=True),
            ("barrier", ["set mip strategy startalgorithm 4"]),
        )
        self.assertEqual(
            _initial_root_cplex_options(env, 167, enabled=True),
            ("automatic", []),
        )
        self.assertEqual(
            _initial_root_cplex_options(env, 168, enabled=False),
            ("automatic", []),
        )
        env.step({"vessels": [VESSEL_WAIT], "wells": [0]})
        self.assertEqual(
            _initial_root_cplex_options(env, 168, enabled=True),
            ("automatic", []),
        )

    def test_controller_defaults_to_greedy_only_warm_start(self):
        controller = RollingMilpController(
            _cold_env(),
        )

        self.assertFalse(controller.shifted_milp_warm_start)
        self.assertEqual(controller.warm_start_mode, "greedy")
        self.assertTrue(controller.terminal_cleanup_value)
        self.assertIsNone(controller.mip_gap_rel)
        self.assertIsNone(controller.solver_threads)
        self.assertEqual(controller.terminal_cleanup_mip_start_mode, "partial")
        self.assertTrue(controller.vessel_visit_load_cuts)
        self.assertEqual(controller.vessel_visit_load_cut_stride_h, 12)
        self.assertTrue(controller.source_visit_vent_cuts)
        self.assertEqual(controller.source_visit_vent_cut_stride_h, 12)
        self.assertTrue(controller.terminal_visit_cuts)
        self.assertEqual(controller.terminal_visit_cut_stride_h, 12)
        self.assertTrue(controller.service_reachability_cuts)
        self.assertEqual(controller.service_reachability_cut_stride_h, 12)
        self.assertTrue(controller.route_cargo_flow_linking)
        self.assertTrue(controller.cleanup_unary_trip_slots)
        self.assertFalse(controller.cleanup_aggregate_full_trip_dominance)

    def test_controller_allows_no_warm_start(self):
        controller = RollingMilpController(
            _cold_env(),
            warm_start_mode="none",
        )

        self.assertEqual(controller.warm_start_mode, "none")
        self.assertFalse(controller.cleanup_return_partition_cut)
        self.assertFalse(controller.cleanup_source_mode_partition_cut)
        self.assertFalse(
            controller.weather_aware_cleanup_sailing_lower_bound
        )
        self.assertFalse(controller.cleanup_source_headroom_risk)
        self.assertFalse(controller.prune_unreachable_route_arcs)
        self.assertFalse(controller.warm_start_end_unstored_guard)
        self.assertTrue(controller.initial_barrier_root)

    def test_shifted_warm_start_drops_executed_prefix_and_uses_mpc_tail(self):
        env = _no_capture_env(cap_hours=8)
        env.reset(seed=1)
        previous = [
            {"vessels": [hour], "wells": [hour]}
            for hour in range(5)
        ]
        mpc = [
            {"vessels": [100 + hour], "wells": [100 + hour]}
            for hour in range(4)
        ]
        replay = SimpleNamespace(is_executable=True)

        with (
            patch(
                "sim.control.rolling_milp._materialize_cplex_actions",
                side_effect=lambda _env, actions: actions,
            ),
            patch(
                "sim.control.rolling_milp.replay_native_actions",
                return_value=replay,
            ),
        ):
            shifted = _shifted_milp_warm_start(
                env,
                previous,
                elapsed_h=2,
                mpc_actions=mpc,
                horizon_h=4,
            )

        self.assertEqual(shifted, [*previous[2:], mpc[3]])

    def test_controller_accepts_shifted_start_and_valid_cut_options(self):
        controller = RollingMilpController(
            _cold_env(),
            mip_gap_rel=0.05,
            solver_threads=4,
            terminal_cleanup_mip_start_mode="complete",
            shifted_milp_warm_start=True,
            vessel_visit_load_cuts=True,
            vessel_visit_load_cut_stride_h=12,
            source_visit_vent_cuts=True,
            source_visit_vent_cut_stride_h=12,
            terminal_visit_cuts=True,
            terminal_visit_cut_stride_h=12,
            service_reachability_cuts=True,
            service_reachability_cut_stride_h=6,
            route_cargo_flow_linking=True,
            cleanup_unary_trip_slots=True,
            cleanup_aggregate_full_trip_dominance=True,
            cleanup_return_partition_cut=True,
            cleanup_source_mode_partition_cut=True,
            weather_aware_cleanup_sailing_lower_bound=True,
            cleanup_source_headroom_risk=True,
            prune_unreachable_route_arcs=False,
            warm_start_end_unstored_guard=True,
            initial_barrier_root=False,
        )

        self.assertTrue(controller.shifted_milp_warm_start)
        self.assertEqual(controller.mip_gap_rel, 0.05)
        self.assertEqual(controller.solver_threads, 4)
        self.assertEqual(controller.terminal_cleanup_mip_start_mode, "complete")
        self.assertTrue(controller.vessel_visit_load_cuts)
        self.assertEqual(controller.vessel_visit_load_cut_stride_h, 12)
        self.assertTrue(controller.source_visit_vent_cuts)
        self.assertEqual(controller.source_visit_vent_cut_stride_h, 12)
        self.assertTrue(controller.terminal_visit_cuts)
        self.assertEqual(controller.terminal_visit_cut_stride_h, 12)
        self.assertTrue(controller.service_reachability_cuts)
        self.assertEqual(controller.service_reachability_cut_stride_h, 6)
        self.assertTrue(controller.route_cargo_flow_linking)
        self.assertTrue(controller.cleanup_unary_trip_slots)
        self.assertTrue(controller.cleanup_aggregate_full_trip_dominance)
        self.assertTrue(controller.cleanup_return_partition_cut)
        self.assertTrue(controller.cleanup_source_mode_partition_cut)
        self.assertTrue(
            controller.weather_aware_cleanup_sailing_lower_bound
        )
        self.assertTrue(controller.cleanup_source_headroom_risk)
        self.assertFalse(controller.prune_unreachable_route_arcs)
        self.assertTrue(controller.warm_start_end_unstored_guard)
        self.assertFalse(controller.initial_barrier_root)

    def test_controller_accepts_progress_and_lookahead_options(self):
        messages: list[str] = []
        progress = messages.append
        controller = RollingMilpController(
            _cold_env(),
            progress=progress,
            planning_horizon_h=96,
            time_limit_s=1.0,
        )
        self.assertEqual(controller.planning_horizon_h, 96)
        self.assertEqual(controller.time_limit_s, 1.0)
        self.assertIs(controller.progress, progress)

    def test_controller_defaults_use_week_horizon_with_longer_solver_budget(self):
        controller = RollingMilpController(_cold_env())

        self.assertEqual(controller.planning_horizon_h, 168)
        self.assertEqual(controller.replan_every, 24)
        self.assertEqual(controller.time_limit_s, 30.0)

    def test_invalid_plan_raises_instead_of_executing_it(self):
        env = _cold_env(cap_hours=24)
        env.reset(seed=1)
        messages: list[str] = []

        invalid_plan = SimpleNamespace(
            vessel_actions_by_hour={vessel_id: [VESSEL_WAIT] for vessel_id in env.vessel_ids},
            injection_tph=[999.0],
            vented_t=0.0,
            shortfall_t=0.0,
            total_cost=0.0,
            status="Not Solved",
            is_valid=False,
            validation_error="solver status Not Solved",
        )

        with patch("sim.control.rolling_milp._plan_native_cplex_actions", return_value=invalid_plan):
            controller = RollingMilpController(
                env,
                replan_every=12,
                progress=messages.append,
            )
            with self.assertRaisesRegex(RuntimeError, "solver status Not Solved"):
                controller.policy(env)

        self.assertEqual(controller.last_plan_status, "Not Solved")
        self.assertFalse(controller.last_plan_valid)
        self.assertTrue(any("invalid" in message and "Not Solved" in message for message in messages))

    def test_controller_executes_planned_hourly_action_directly(self):
        env = _cold_env(cap_hours=24)
        env.reset(seed=1)
        vessel_id = env.vessel_ids[0]
        home = str(env._routes[vessel_id]["origin"])
        other = next(eid for eid in env.emitter_ids if eid != home)
        terminal = str(env._routes[vessel_id]["destination"])
        planned_action = env.vessel_go_emitter_action(other)
        env.simulator.state.vessel_berths[vessel_id] = terminal
        env.simulator.vessel_states[vessel_id] = {
            "mode": "berthed",
            "berth": terminal,
            "destination": None,
            "progress": 0.0,
        }
        env.simulator.state.entity_inventory_t[vessel_id] = 0.0
        env.simulator.state.entity_inventory_t[home] = 10_000.0
        env.simulator.state.entity_inventory_t[other] = 0.0

        plan = SimpleNamespace(
            vessel_actions_by_hour={
                vid: [planned_action if vid == vessel_id else VESSEL_WAIT]
                for vid in env.vessel_ids
            },
            injection_tph=[0.0],
            native_actions_by_hour=[
                {
                    "vessels": [
                        planned_action if vid == vessel_id else VESSEL_WAIT
                        for vid in env.vessel_ids
                    ],
                    "wells": [0] * len(env.well_ids),
                }
            ],
            vented_t=0.0,
            shortfall_t=0.0,
            total_cost=0.0,
            status="Optimal",
            is_valid=True,
            validation_error="",
        )
        with patch("sim.control.rolling_milp._plan_native_cplex_actions", return_value=plan):
            action = RollingMilpController(env, replan_every=12).policy(env)

        self.assertEqual(action["vessels"][env.vessel_ids.index(vessel_id)], planned_action)

    def test_controller_uses_the_replayed_native_well_action(self):
        env = _cold_env(cap_hours=24)
        env.reset(seed=1)
        planned_wells = [env.highest_feasible_well_rate_index(well_id) for well_id in env.well_ids]
        plan = SimpleNamespace(
            vessel_actions_by_hour={vessel_id: [VESSEL_WAIT] for vessel_id in env.vessel_ids},
            injection_tph=[0.0],
            native_actions_by_hour=[
                {
                    "vessels": [VESSEL_WAIT] * len(env.vessel_ids),
                    "wells": planned_wells,
                }
            ],
            vented_t=0.0,
            shortfall_t=0.0,
            total_cost=0.0,
            status="Optimal",
            is_valid=True,
            validation_error="",
        )

        with patch("sim.control.rolling_milp._plan_native_cplex_actions", return_value=plan):
            action = RollingMilpController(env, replan_every=12).policy(env)

        self.assertEqual(action["wells"], planned_wells)

    def test_controller_only_requires_the_pre_replan_action_slice_to_be_executable(self):
        env = _no_capture_env(cap_hours=30)
        env.reset(seed=1)
        actions = [
            {"vessels": [VESSEL_WAIT], "wells": [0]}
            for _ in range(30)
        ]
        actions[24] = {"vessels": [99], "wells": [0]}
        plan = SimpleNamespace(
            vessel_actions_by_hour={"ship": [action["vessels"][0] for action in actions]},
            injection_tph=[0.0] * 30,
            native_actions_by_hour=actions,
            vented_t=0.0,
            shortfall_t=0.0,
            total_cost=0.0,
            status="Optimal",
            is_valid=False,
            solver_is_valid=True,
            replay_is_valid=False,
            replay_is_exact=False,
            replay_mismatches=("action[24] is not executable",),
            validation_error="action[24] is not executable",
            termination_reason="Integer optimal, tolerance (0.1/1e-06)",
            requested_mip_gap_rel=0.1,
        )

        with patch("sim.control.rolling_milp._plan_native_cplex_actions", return_value=plan):
            controller = RollingMilpController(env, replan_every=24, planning_horizon_h=30)
            action = controller.policy(env)

        self.assertEqual(action, {"vessels": [VESSEL_WAIT], "wells": [0]})
        self.assertTrue(controller.last_plan_valid)
        self.assertTrue(controller.last_execution_replay_is_valid)
        self.assertFalse(controller.last_model_replay_is_exact)
        diagnostic = controller.replan_diagnostics[-1]
        self.assertEqual(diagnostic["execution_replay_mismatches"], "")
        self.assertEqual(
            diagnostic["model_replay_mismatches"],
            "action[24] is not executable",
        )
        self.assertIn("tolerance", diagnostic["termination_reason"])
        self.assertEqual(diagnostic["requested_mip_gap_rel"], 0.1)

    def test_planner_sailing_hours_between_emitters_uses_maritime_route(self):
        env = _cold_env(cap_hours=24)
        env.reset(seed=1)
        vessel_id = "vessel_a"
        speed_knots = float(env._routes[vessel_id]["speed_knots"])
        source_a = TOY_TWO_SOURCE_LOCATIONS["source_a"]
        source_b = TOY_TWO_SOURCE_LOCATIONS["source_b"]
        expected_hours = max(1, math.ceil(sea_route(source_a, source_b).distance_km / (speed_knots * 1.852)))
        direct_hours = max(1, math.ceil(route_distance_km([source_a, source_b]) / (speed_knots * 1.852)))

        self.assertGreater(expected_hours, direct_hours)
        self.assertEqual(_sail_hours_between(env, "source_a", "source_b", vessel_id), expected_hours)

    def test_capture_forecast_uses_future_scenario_availability(self):
        env = _cold_env(cap_hours=6)
        env.reset(seed=1)
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=6,
            emitter_availability={
                "source_a": [1.0, 0.25, 0.0, 1.0, 1.0, 1.0],
                "source_b": [1.0] * 6,
            },
            vessel_speed_factor={vessel_id: [1.0] * 6 for vessel_id in env.vessel_ids},
            well_available={well_id: [True] * 6 for well_id in env.well_ids},
            injectivity_factor={well_id: [1.0] * 6 for well_id in env.well_ids},
        )
        env.simulator.state.time_h = 1.0
        env.scenario.apply_to_state(env.simulator.state, time_h=1.0)

        self.assertAlmostEqual(_capture_tonnes(env, "source_a", 0), 20.0)
        self.assertAlmostEqual(_capture_tonnes(env, "source_a", 1), 0.0)

class NativeMpcTests(unittest.TestCase):
    def test_native_mpc_objective_is_fixed_to_vent_inventory_then_operating_cost(self):
        low_inventory = _NativeMpcCandidate(
            name="low_inventory",
            native_actions_by_hour=[],
            vented_t=0.0,
            end_unstored_t=10.0,
            operating_cost=100.0,
            total_cost=120.0,
            is_valid=True,
        )
        low_cost = _NativeMpcCandidate(
            name="low_cost",
            native_actions_by_hour=[],
            vented_t=0.0,
            end_unstored_t=100.0,
            operating_cost=10.0,
            total_cost=20.0,
            is_valid=True,
        )

        best = min([low_inventory, low_cost], key=RollingNativeMpcController._candidate_key)

        self.assertEqual(best.name, "low_inventory")

    def test_native_mpc_constructor_accepts_economic_objective_mode(self):
        env = _cold_env(cap_hours=24)

        controller = RollingNativeMpcController(env, objective_mode="economic")

        self.assertEqual(controller.objective_mode, "economic")

    def test_native_mpc_economic_objective_includes_terminal_cleanup(self):
        cheap_episode_expensive_cleanup = _NativeMpcCandidate(
            name="cheap_episode_expensive_cleanup",
            native_actions_by_hour=[],
            vented_t=0.0,
            end_unstored_t=10.0,
            operating_cost=10.0,
            total_cost=10.0,
            is_valid=True,
            terminal_cleanup_operating_cost=100.0,
        )
        expensive_episode_cheap_cleanup = _NativeMpcCandidate(
            name="expensive_episode_cheap_cleanup",
            native_actions_by_hour=[],
            vented_t=0.0,
            end_unstored_t=10.0,
            operating_cost=20.0,
            total_cost=20.0,
            is_valid=True,
            terminal_cleanup_operating_cost=0.0,
        )

        best, _limits = _select_native_mpc_candidate(
            [
                cheap_episode_expensive_cleanup,
                expensive_episode_cheap_cleanup,
            ],
            objective_mode="economic",
            vent_eur_per_t=80.0,
        )

        self.assertEqual(best.name, "expensive_episode_cheap_cleanup")

    def test_native_mpc_accepts_a_goal_preference_candidate(self):
        env = _cold_env(cap_hours=24)
        controller = RollingNativeMpcController(
            env,
            planning_horizon_h=24,
            preferred_policies={"goal_preference": greedy_shuttle_policy},
        )

        run_episode(env, controller, seed=1)

        self.assertIn("goal_preference", controller.preferred_policies)
        self.assertGreater(
            controller.candidate_evaluations,
            len(native_mpc_candidate_names(env)),
        )

    def test_native_mpc_can_add_terminal_cleanup_to_candidate_values(self):
        env = _cold_env(cap_hours=24)
        controller = RollingNativeMpcController(
            env,
            planning_horizon_h=24,
            objective_mode="economic",
            terminal_cleanup_value=True,
        )

        with patch(
            "sim.control.native_mpc._terminal_cleanup_cost_for_state",
            return_value=123.0,
        ) as cleanup:
            run_episode(env, controller, seed=1)

        self.assertEqual(cleanup.call_count, controller.candidate_evaluations)
        self.assertEqual(
            controller.last_terminal_cleanup_operating_cost,
            123.0,
        )

    def test_native_mpc_economic_safe_rejects_boundary_exploiting_candidate(self):
        reference = _NativeMpcCandidate(
            name="forecast_urgency",
            native_actions_by_hour=[],
            vented_t=10.0,
            end_unstored_t=90.0,
            operating_cost=100.0,
            total_cost=900.0,
            is_valid=True,
        )
        boundary_exploit = _NativeMpcCandidate(
            name="boundary_exploit",
            native_actions_by_hour=[],
            vented_t=0.0,
            end_unstored_t=200.0,
            operating_cost=1.0,
            total_cost=1.0,
            is_valid=True,
        )

        best, limits = _select_native_mpc_candidate(
            [reference, boundary_exploit],
            objective_mode="economic_safe",
            vent_eur_per_t=80.0,
        )

        self.assertEqual(best.name, "forecast_urgency")
        self.assertEqual(limits.max_nonstored_t, 100.0)

    def test_native_mpc_economic_lex_guard_uses_separate_lexicographic_limits(self):
        lex_reference = _NativeMpcCandidate(
            name="lex_reference",
            native_actions_by_hour=[],
            vented_t=5.0,
            end_unstored_t=80.0,
            operating_cost=100.0,
            total_cost=500.0,
            is_valid=True,
        )
        cheaper_but_worse = _NativeMpcCandidate(
            name="cheaper_but_worse",
            native_actions_by_hour=[],
            vented_t=6.0,
            end_unstored_t=20.0,
            operating_cost=1.0,
            total_cost=1.0,
            is_valid=True,
        )

        best, limits = _select_native_mpc_candidate(
            [lex_reference, cheaper_but_worse],
            objective_mode="economic_lex_guard",
            vent_eur_per_t=80.0,
        )

        self.assertEqual(best.name, "lex_reference")
        self.assertEqual(limits.max_vented_t, 5.0)
        self.assertEqual(limits.max_end_unstored_t, 80.0)

    def test_native_mpc_economic_execution_guard_rejects_deferred_prefix(self):
        reference = _NativeMpcCandidate(
            name="reference",
            native_actions_by_hour=[],
            vented_t=0.0,
            end_unstored_t=10.0,
            operating_cost=100.0,
            total_cost=100.0,
            is_valid=True,
            execution_vented_t=0.0,
            execution_unstored_t=50.0,
        )
        deferred = _NativeMpcCandidate(
            name="deferred",
            native_actions_by_hour=[],
            vented_t=0.0,
            end_unstored_t=20.0,
            operating_cost=1.0,
            total_cost=1.0,
            is_valid=True,
            execution_vented_t=0.0,
            execution_unstored_t=60.0,
        )

        best, limits = _select_native_mpc_candidate(
            [reference, deferred],
            objective_mode="economic_execution_guard",
            vent_eur_per_t=80.0,
        )

        self.assertEqual(best.name, "reference")
        self.assertEqual(limits.max_execution_vented_t, 0.0)
        self.assertEqual(limits.max_execution_unstored_t, 50.0)

    def test_native_mpc_runs_a_replayable_episode_without_a_milp_solver(self):
        env = _cold_env(cap_hours=48)
        controller = RollingNativeMpcController(
            env,
            replan_every=24,
            planning_horizon_h=48,
        )

        metrics = run_episode(env, controller, seed=1)

        self.assertEqual(metrics.elapsed_hours, 48)
        self.assertTrue(controller.last_trace_replay_is_valid)
        self.assertTrue(controller.last_trace_replay_is_exact)
        self.assertGreaterEqual(controller.candidate_evaluations, 2)

    def test_native_mpc_supports_automatic_well_control(self):
        env = _cold_env(
            cap_hours=24,
            well_control_mode="automatic_max",
        )
        controller = RollingNativeMpcController(
            env,
            planning_horizon_h=24,
        )

        metrics = run_episode(env, controller, seed=1)

        self.assertEqual(metrics.elapsed_hours, 24)
        self.assertNotIn("wells", controller._native_actions_by_hour[0])
        self.assertTrue(controller.last_trace_replay_is_exact)


@unittest.skipUnless(HAVE_PULP, "pulp not installed")
class RollingMilpTests(unittest.TestCase):
    @unittest.skipUnless(
        HAVE_CPLEX,
        "CPLEX executable not installed",
    )
    def test_native_cplex_rolling_plan_uses_lexicographic_replayable_actions(self):
        env = _no_capture_env(cap_hours=2)
        env.reset(seed=1)

        plan = _plan_native_cplex_actions(
            env,
            planning_horizon_h=2,
            economics=EconomicParameters(),
            time_limit_s=10.0,
        )

        self.assertIn(plan.status, {"Optimal", "Integer Feasible"})
        self.assertTrue(plan.solver_is_valid, plan.validation_error)
        self.assertTrue(plan.replay_is_valid, plan.validation_error)
        self.assertEqual(len(plan.native_actions_by_hour), 2)

    def test_cplex_action_materialization_dispatches_as_soon_as_terminal_unload_finishes(self):
        env = _no_capture_env(cap_hours=2)
        env.reset(seed=1)
        env.simulator.state.vessel_berths["ship"] = "terminal"
        env.simulator.vessel_states["ship"] = {
            "mode": "berthed",
            "berth": "terminal",
            "origin": "terminal",
            "destination": "terminal",
            "progress": 0.0,
        }
        env.simulator.state.entity_inventory_t["ship"] = 500.0
        go_source = env.vessel_go_emitter_action("source")
        planned = [
            {"vessels": [go_source], "wells": [0]},
            {"vessels": [VESSEL_WAIT], "wells": [0]},
        ]

        materialized = _materialize_cplex_actions(env, planned)

        self.assertEqual(
            [action["vessels"][0] for action in materialized],
            [VESSEL_WAIT, go_source],
        )

    def test_native_cplex_warm_start_uses_replay_valid_mpc_actions(self):
        env = _no_capture_env(cap_hours=2)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 500.0

        actions = _native_mpc_warm_start(env, horizon_h=2)

        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[1]["vessels"], [VESSEL_GO_TERMINAL])

    def test_native_cplex_returns_strict_model_failure_without_retry(self):
        env = _no_capture_env(cap_hours=1)
        env.reset(seed=1)
        strict = SimpleNamespace(is_valid=False)

        with patch(
            "sim.control.cplex_milp.solve_full_scenario_with_cplex",
            return_value=strict,
        ) as solve:
            result = _solve_native_cplex_result(
                env,
                1,
                EconomicParameters(),
                [{"vessels": [VESSEL_WAIT], "wells": [0]}],
                10.0,
            )

        self.assertIs(result, strict)
        self.assertEqual(
            [call.kwargs["environment_aligned_service"] for call in solve.call_args_list],
            [True],
        )
        self.assertEqual(
            solve.call_args.kwargs["cplex_options"],
            [
                "set parallel 1",
                "set simplex tolerances feasibility 1e-7",
                "set mip limits cutpasses 1",
                "set mip strategy heuristicfreq 10",
                "set mip strategy search 1",
            ],
        )

    def test_native_cplex_economic_safe_uses_same_mpc_boundary_limit(self):
        env = _no_capture_env(cap_hours=1)
        env.reset(seed=1)
        result_stub = SimpleNamespace(is_valid=True)

        with patch(
            "sim.control.cplex_milp.solve_full_scenario_with_cplex",
            return_value=result_stub,
        ) as solve:
            result = _solve_native_cplex_result(
                env,
                1,
                EconomicParameters(),
                [{"vessels": [VESSEL_WAIT], "wells": [0]}],
                10.0,
                objective_mode="economic_safe",
                safe_progress_limit_t=123.0,
            )

        self.assertIs(result, result_stub)
        self.assertTrue(solve.call_args.kwargs["economic_objective"])
        self.assertFalse(solve.call_args.kwargs["lexicographic_vent_first"])
        self.assertEqual(solve.call_args.kwargs["max_nonstored_t"], 123.0)
        self.assertEqual(
            solve.call_args.kwargs["cplex_options"],
            [
                "set parallel 1",
                "set simplex tolerances feasibility 1e-7",
            ],
        )

    def test_native_cplex_economic_strict_uses_separate_mpc_boundary_limits(self):
        env = _no_capture_env(cap_hours=1)
        env.reset(seed=1)
        result_stub = SimpleNamespace(is_valid=True)

        with patch(
            "sim.control.cplex_milp.solve_full_scenario_with_cplex",
            return_value=result_stub,
        ) as solve:
            _solve_native_cplex_result(
                env,
                1,
                EconomicParameters(),
                [{"vessels": [VESSEL_WAIT], "wells": [0]}],
                10.0,
                objective_mode="economic_lex_guard",
                safe_vent_limit_t=12.0,
                safe_end_unstored_limit_t=34.0,
            )

        self.assertTrue(solve.call_args.kwargs["economic_objective"])
        self.assertEqual(solve.call_args.kwargs["max_vented_t"], 12.0)
        self.assertEqual(solve.call_args.kwargs["max_end_unstored_t"], 34.0)

    def test_native_cplex_can_enable_trip_cleanup_terminal_value(self):
        env = _no_capture_env(cap_hours=1)
        env.reset(seed=1)
        result_stub = SimpleNamespace(is_valid=True)

        with patch(
            "sim.control.cplex_milp.solve_full_scenario_with_cplex",
            return_value=result_stub,
        ) as solve:
            _solve_native_cplex_result(
                env,
                1,
                EconomicParameters(),
                [{"vessels": [VESSEL_WAIT], "wells": [0]}],
                10.0,
                objective_mode="economic",
                terminal_cleanup_value=True,
            )

        self.assertTrue(solve.call_args.kwargs["terminal_cleanup_value"])

    def test_native_cplex_can_enable_cleanup_source_headroom_risk(self):
        env = _no_capture_env(cap_hours=1)
        env.reset(seed=1)

        with patch(
            "sim.control.cplex_milp.solve_full_scenario_with_cplex",
            return_value=SimpleNamespace(is_valid=True),
        ) as solve:
            _solve_native_cplex_result(
                env,
                1,
                EconomicParameters(),
                [{"vessels": [VESSEL_WAIT], "wells": [0]}],
                10.0,
                terminal_cleanup_value=True,
                cleanup_source_headroom_risk=True,
            )

        self.assertTrue(
            solve.call_args.kwargs["cleanup_source_headroom_risk"]
        )

    def test_native_cplex_can_enable_route_cargo_flow_linking(self):
        env = _no_capture_env(cap_hours=1)
        env.reset(seed=1)
        result_stub = SimpleNamespace(is_valid=True)

        with patch(
            "sim.control.cplex_milp.solve_full_scenario_with_cplex",
            return_value=result_stub,
        ) as solve:
            _solve_native_cplex_result(
                env,
                1,
                EconomicParameters(),
                [{"vessels": [VESSEL_WAIT], "wells": [0]}],
                10.0,
                route_cargo_flow_linking=True,
            )

        self.assertTrue(solve.call_args.kwargs["route_cargo_flow_linking"])

    @unittest.skipUnless(HAVE_CPLEX, "CPLEX executable not installed")
    def test_controller_executes_an_exact_replay_valid_plan(self):
        env = _cold_env(cap_hours=96)
        controller = RollingMilpController(
            env,
            replan_every=48,
            planning_horizon_h=48,
            time_limit_s=1.0,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
        )
        metrics = run_episode(env, controller, seed=1)

        self.assertEqual(metrics.elapsed_hours, 96)
        self.assertTrue(controller.last_plan_valid)
        self.assertTrue(controller.last_model_replay_is_exact)
        self.assertFalse(controller.last_model_replay_mismatches)
        self.assertEqual(controller.model_inexact_replan_count, 0)

    @unittest.skipUnless(HAVE_CPLEX, "CPLEX executable not installed")
    def test_controller_consistently_executes_replay_valid_plans_after_reset(self):
        env = _cold_env(cap_hours=96)
        controller = RollingMilpController(env, replan_every=48, planning_horizon_h=48, time_limit_s=3.0)
        for _ in range(2):
            metrics = run_episode(env, controller, seed=1)
            self.assertEqual(metrics.elapsed_hours, 96)
            self.assertTrue(controller.last_plan_valid)
            self.assertTrue(controller.last_model_replay_is_exact)

    @unittest.skipUnless(HAVE_CPLEX, "CPLEX executable not installed")
    def test_controller_uses_an_exact_executable_trace(self):
        env = _cold_env(cap_hours=96)
        env.reset(seed=1)
        controller = RollingMilpController(env, replan_every=12, planning_horizon_h=48)
        action = controller.policy(env)

        self.assertEqual(len(action["vessels"]), len(env.vessel_ids))
        self.assertEqual(len(action["wells"]), len(env.well_ids))
        self.assertTrue(controller.last_plan_valid)
        self.assertTrue(controller.last_model_replay_is_exact)

    @unittest.skipUnless(HAVE_CPLEX, "CPLEX executable not installed")
    def test_controller_shortens_the_final_planning_window_to_the_episode(self):
        env = _no_capture_env(cap_hours=25)
        controller = RollingMilpController(
            env,
            replan_every=24,
            planning_horizon_h=168,
            time_limit_s=3.0,
        )

        metrics = run_episode(env, controller, seed=1)

        self.assertEqual(metrics.elapsed_hours, 25)
        self.assertEqual(controller.replan_count, 2)
        self.assertEqual(len(controller._native_actions_by_hour), 1)

    @unittest.skipUnless(HAVE_CPLEX, "CPLEX executable not installed")
    def test_no_capture_plan_keeps_empty_vessel_waiting(self):
        env = _no_capture_env(cap_hours=24)
        env.reset(seed=1)
        vessel_id = env.vessel_ids[0]
        terminal = str(env._routes[vessel_id]["destination"])
        env.simulator.state.vessel_berths[vessel_id] = terminal
        env.simulator.vessel_states[vessel_id] = {
            "mode": "berthed",
            "berth": terminal,
            "destination": None,
            "progress": 0.0,
        }
        env.simulator.state.entity_inventory_t[vessel_id] = 0.0

        action = RollingMilpController(env, replan_every=12, planning_horizon_h=12).policy(env)

        self.assertEqual(action["vessels"], [VESSEL_WAIT])

if __name__ == "__main__":
    unittest.main()
