import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import pulp  # noqa: F401

    HAVE_PULP = True
except ImportError:
    HAVE_PULP = False

from sim.control.baselines import greedy_shuttle_policy
from sim.control.rolling_milp import (
    RollingMilpController,
    _build_action_arcs,
    _capture_tonnes,
    _plan_explicit_actions,
    _sail_hours_between,
)
from sim.control.native_mpc import RollingNativeMpcController, _NativeMpcCandidate
from sim.control.trip_milp import materialize_native_action_trace
from sim.economics import EconomicParameters
from sim.entities import Emitter, InjectionWell, Pipeline, Reservoir, SubseaManifold, Terminal, Vessel
from sim.environment import (
    CCSEnv,
    CCSEnvConfig,
    MIN_WELL_RATE_INDEX,
    WELL_RATE_LEVELS_MTPA,
    VESSEL_GO_TERMINAL,
    VESSEL_WAIT,
)
from sim.metrics import run_episode
from sim.network import PhysicalNetwork
from sim.routes import route_distance_km, sea_route
from sim.scenario_generation import Scenario, ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network


def _cold_env(cap_hours: int = 600) -> CCSEnv:
    # Cold start (no initial inventory) so the MILP bound and the controllers face
    # the same empty-system task.
    return CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(
            config=ScenarioConfig(episode_hours=cap_hours, randomize_initial_inventory=False)
        ),
        config=CCSEnvConfig(episode_hours=cap_hours),
    )


def _no_capture_env(cap_hours: int = 24) -> CCSEnv:
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
        config=CCSEnvConfig(episode_hours=cap_hours),
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

    def test_explicit_planner_default_solver_budget_matches_controller(self):
        defaults = _plan_explicit_actions.__defaults__

        self.assertIsNotNone(defaults)
        self.assertEqual(defaults[-1], 30.0)

    def test_invalid_plan_raises_instead_of_executing_a_fallback_policy(self):
        env = _cold_env(cap_hours=24)
        env.reset(seed=1)
        messages: list[str] = []
        fallback_action = {
            "vessels": [VESSEL_WAIT] * len(env.vessel_ids),
            "wells": [MIN_WELL_RATE_INDEX] * len(env.well_ids),
        }

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

        with patch("sim.control.rolling_milp._plan_explicit_actions", return_value=invalid_plan):
            controller = RollingMilpController(
                env,
                replan_every=12,
                progress=messages.append,
                fallback_policy=lambda _env: fallback_action,
            )
            with self.assertRaisesRegex(RuntimeError, "solver status Not Solved"):
                controller.policy(env)

        self.assertEqual(controller.last_plan_status, "Not Solved")
        self.assertFalse(controller.last_plan_valid)
        self.assertEqual(controller.fallback_count, 0)
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
            vented_t=0.0,
            shortfall_t=0.0,
            total_cost=0.0,
            status="Optimal",
            is_valid=True,
            validation_error="",
        )
        with patch("sim.control.rolling_milp._plan_explicit_actions", return_value=plan):
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

        with patch("sim.control.rolling_milp._plan_explicit_actions", return_value=plan):
            action = RollingMilpController(env, replan_every=12).policy(env)

        self.assertEqual(action["wells"], planned_wells)

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

    def test_action_arcs_use_future_weather_speed_forecast(self):
        env = _no_capture_env(cap_hours=4)
        env.reset(seed=1)
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=4,
            emitter_availability={"source": [1.0] * 4},
            vessel_speed_factor={"ship": [1.0, 0.5, 0.5, 0.5]},
            well_available={"well": [True] * 4},
            injectivity_factor={"well": [1.0] * 4},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        arcs, _starts = _build_action_arcs(env, horizon_h=4)
        sail_at_t1 = next(
            arc
            for arc in arcs
            if arc.vessel_id == "ship"
            and arc.start_h == 1
            and arc.origin_id == "source"
            and arc.destination_id == "terminal"
        )

        self.assertEqual(sail_at_t1.duration_h, 2)


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

    def test_native_mpc_constructor_rejects_alternate_objective_mode(self):
        env = _cold_env(cap_hours=24)

        with self.assertRaises(TypeError):
            RollingNativeMpcController(env, objective_mode="vent_then_total_cost")

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


@unittest.skipUnless(HAVE_PULP, "pulp/CBC not installed")
class RollingMilpTests(unittest.TestCase):
    def test_controller_rejects_an_inexact_plan(self):
        env = _cold_env(cap_hours=96)
        controller = RollingMilpController(
            env,
            replan_every=48,
            planning_horizon_h=48,
            time_limit_s=1.0,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
        )
        with self.assertRaisesRegex(RuntimeError, "expected"):
            run_episode(env, controller, seed=1)
        self.assertFalse(controller.last_plan_valid)
        self.assertTrue(controller.last_validation_error)

    def test_controller_consistently_rejects_inexact_plans_after_reset(self):
        env = _cold_env(cap_hours=96)
        controller = RollingMilpController(env, replan_every=48, planning_horizon_h=48, time_limit_s=3.0)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "expected"):
                run_episode(env, controller, seed=1)
            self.assertFalse(controller.last_plan_valid)

    def test_controller_rejects_inexact_fixed_horizon_plan_without_storage_goal(self):
        env = _cold_env(cap_hours=96)
        env.reset(seed=1)
        controller = RollingMilpController(env, replan_every=12, planning_horizon_h=48)
        with self.assertRaisesRegex(RuntimeError, "expected"):
            controller.policy(env)
        self.assertFalse(controller.last_plan_valid)

    def test_unusual_state_does_not_silently_fallback_to_greedy(self):
        env = _cold_env(cap_hours=96)
        env.reset(seed=1)
        vessel_id = env.vessel_ids[0]
        home = str(env._routes[vessel_id]["origin"])
        other = next(eid for eid in env.emitter_ids if eid != home)
        terminal = str(env._routes[vessel_id]["destination"])
        env.simulator.state.vessel_berths[vessel_id] = terminal
        env.simulator.vessel_states[vessel_id] = {
            "mode": "berthed",
            "berth": terminal,
            "destination": None,
            "progress": 0.0,
        }
        env.simulator.state.entity_inventory_t[vessel_id] = 0.0
        env.simulator.state.entity_inventory_t[home] = 0.0
        env.simulator.state.entity_inventory_t[other] = 5_000.0
        other_vessel = next(vid for vid in env.vessel_ids if vid != vessel_id)
        env._routes[other_vessel]["speed_knots"] = 0.001

        controller = RollingMilpController(env, replan_every=12, planning_horizon_h=96, time_limit_s=1.0)

        with self.assertRaises(RuntimeError):
            controller.policy(env)
        self.assertFalse(controller.last_plan_valid)

    def test_no_capture_plan_does_not_fallback_to_unplanned_greedy_sailing(self):
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

    def test_plan_returns_hourly_actions_and_no_delivery_schedule(self):
        env = _two_berth_parallel_env()
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 2_000.0

        plan = _plan_explicit_actions(env, planning_horizon_h=3, economics=EconomicParameters())

        self.assertEqual(set(plan.vessel_actions_by_hour), set(env.vessel_ids))
        self.assertEqual([len(actions) for actions in plan.vessel_actions_by_hour.values()], [3, 3])
        self.assertEqual(len(plan.injection_tph), 3)
        self.assertFalse(hasattr(plan, "schedule"))
        for actions in plan.vessel_actions_by_hour.values():
            self.assertTrue(all(0 <= action < env.vessel_action_count for action in actions))

    def test_plan_exposes_an_exact_replay_of_its_native_action_trace(self):
        env = _two_berth_parallel_env()
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 2_000.0

        plan = _plan_explicit_actions(env, planning_horizon_h=3, economics=EconomicParameters())
        replay = materialize_native_action_trace(
            env,
            plan.native_actions_by_hour,
            horizon_h=3,
            economics=EconomicParameters(),
        )

        self.assertTrue(plan.replay_is_valid)
        self.assertTrue(replay.is_valid, replay.validation_error)
        self.assertAlmostEqual(plan.replay_vented_t, replay.vented_t)
        self.assertAlmostEqual(plan.replay_stored_t, replay.stored_t)
        self.assertAlmostEqual(plan.replay_total_cost, replay.total_cost)

    def test_explicit_plan_can_depart_one_emitter_for_another(self):
        env = _two_source_one_ship_fast_env()
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source_a"] = 0.0
        env.simulator.state.entity_inventory_t["source_b"] = 500.0
        env.cumulative_captured_t = 500.0

        plan = _plan_explicit_actions(
            env,
            planning_horizon_h=5,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
        )

        self.assertTrue(plan.replay_is_valid)
        self.assertFalse(plan.replay_is_exact)
        self.assertFalse(plan.is_valid)
        self.assertIn(env.vessel_go_emitter_action("source_b"), plan.vessel_actions_by_hour["ship"])

    def test_vent_first_plan_minimizes_vent_before_operating_cost(self):
        economics = EconomicParameters(carbon_price_eur_per_t=0.0)
        economic_env = _cold_env(cap_hours=24)
        economic_env.reset(seed=1)
        vent_first_env = _cold_env(cap_hours=24)
        vent_first_env.config.reward_mode = "vent_first"
        vent_first_env.reset(seed=1)
        for env in (economic_env, vent_first_env):
            for emitter_id in env.emitter_ids:
                emitter = env.network.entities[emitter_id]
                env.simulator.state.entity_inventory_t[emitter_id] = emitter.buffer_capacity_t

        economic = _plan_explicit_actions(economic_env, planning_horizon_h=24, economics=economics)
        vent_first = _plan_explicit_actions(vent_first_env, planning_horizon_h=24, economics=economics)

        self.assertTrue(vent_first.replay_is_valid)
        self.assertFalse(vent_first.replay_is_exact)
        self.assertFalse(vent_first.is_valid)
        self.assertLess(vent_first.vented_t, economic.vented_t)

    def test_vent_first_tie_breaker_clears_terminal_inventory_before_cost(self):
        env = _no_capture_env(cap_hours=3)
        env.reset(seed=1)
        env.config.reward_mode = "vent_first"
        env.simulator.state.entity_inventory_t["terminal"] = 1_000.0
        env.cumulative_captured_t = 1_000.0

        plan = _plan_explicit_actions(env, planning_horizon_h=3, economics=EconomicParameters())

        self.assertTrue(plan.replay_is_valid)
        self.assertFalse(plan.replay_is_exact)
        self.assertFalse(plan.is_valid)
        self.assertGreater(plan.replay_stored_t, 0.0)

    def test_explicit_plan_can_start_voyage_that_finishes_after_lookahead(self):
        env = _no_capture_env(cap_hours=2)
        env.reset(seed=1)
        env._routes["ship"]["distance_km"] = 18.52
        env.simulator.state.entity_inventory_t["source"] = 500.0
        env.cumulative_captured_t = 500.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=2,
            emitter_availability={"source": [1.0, 1.0]},
            vessel_speed_factor={"ship": [1.0, 1.0]},
            well_available={"well": [True, True]},
            injectivity_factor={"well": [1.0, 1.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        plan = _plan_explicit_actions(
            env,
            planning_horizon_h=2,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
        )

        self.assertTrue(plan.is_valid, plan.validation_error)
        self.assertEqual(plan.vessel_actions_by_hour["ship"], [VESSEL_WAIT, VESSEL_GO_TERMINAL])

    def test_explicit_plan_limits_future_injection_by_well_forecast(self):
        env = _no_capture_env(cap_hours=3)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 1_000.0
        env.simulator.state.entity_inventory_t["terminal"] = 1_000.0
        env.cumulative_captured_t = 1_000.0
        env.scenario = Scenario(
            time_step_hours=1.0,
            n_steps=3,
            emitter_availability={"source": [1.0, 1.0, 1.0]},
            vessel_speed_factor={"ship": [1.0, 1.0, 1.0]},
            well_available={"well": [True, True, True]},
            injectivity_factor={"well": [1.0, 0.2, 0.0]},
        )
        env.scenario.apply_to_state(env.simulator.state, time_h=0.0)

        plan = _plan_explicit_actions(
            env,
            planning_horizon_h=3,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
        )

        self.assertTrue(plan.replay_is_valid)
        self.assertFalse(plan.replay_is_exact)
        self.assertFalse(plan.is_valid)
        self.assertGreater(plan.injection_tph[0], 400.0)
        self.assertLessEqual(plan.injection_tph[1], 100.0 + 1e-6)
        self.assertAlmostEqual(plan.injection_tph[2], 0.0)

    def test_explicit_plan_counts_existing_cumulative_storage_gap(self):
        env = _no_capture_env(cap_hours=1)
        env.reset(seed=1)
        env.cumulative_captured_t = 1_000.0
        env.cumulative_stored_t = 0.0
        env.simulator.state.entity_inventory_t["terminal"] = 500.0

        plan = _plan_explicit_actions(
            env,
            planning_horizon_h=1,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
        )

        self.assertTrue(plan.replay_is_valid)
        self.assertFalse(plan.replay_is_exact)
        self.assertFalse(plan.is_valid)
        self.assertGreater(plan.injection_tph[0], 400.0)


if __name__ == "__main__":
    unittest.main()
