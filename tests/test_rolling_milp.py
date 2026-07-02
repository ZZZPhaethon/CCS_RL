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
from sim.economics import EconomicParameters
from sim.entities import Emitter, InjectionWell, Pipeline, Reservoir, SubseaManifold, Terminal, Vessel
from sim.environment import CCSEnv, CCSEnvConfig, MAX_WELL_RATE_MTPA, MIN_WELL_RATE_MTPA, VESSEL_WAIT
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
            "ship": {"origin": "source", "destination": "terminal", "distance_km": 1.852, "speed_knots": 1.0},
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

    def test_invalid_plan_uses_fallback_policy_and_logs_reason(self):
        env = _cold_env(cap_hours=24)
        env.reset(seed=1)
        messages: list[str] = []
        fallback_action = {
            "vessels": [VESSEL_WAIT] * len(env.vessel_ids),
            "wells": [MIN_WELL_RATE_MTPA] * len(env.well_ids),
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
            action = controller.policy(env)

        self.assertEqual(action, fallback_action)
        self.assertEqual(controller.last_plan_status, "Not Solved")
        self.assertFalse(controller.last_plan_valid)
        self.assertEqual(controller.fallback_count, 1)
        self.assertTrue(any("fallback" in message and "Not Solved" in message for message in messages))

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


@unittest.skipUnless(HAVE_PULP, "pulp/CBC not installed")
class RollingMilpTests(unittest.TestCase):
    def test_controller_runs_to_horizon_and_stores_co2(self):
        env = _cold_env(cap_hours=96)
        controller = RollingMilpController(env, replan_every=48, planning_horizon_h=48, time_limit_s=1.0)
        metrics = run_episode(env, controller, seed=1)
        self.assertEqual(metrics.elapsed_hours, 96)
        self.assertGreater(metrics.stored_t, 0.0)

    def test_controller_resets_between_episodes(self):
        env = _cold_env(cap_hours=96)
        controller = RollingMilpController(env, replan_every=48, planning_horizon_h=48, time_limit_s=3.0)
        a = run_episode(env, controller, seed=1).stored_t
        b = run_episode(env, controller, seed=1).stored_t  # reused controller
        self.assertEqual(a, b)  # stale plan would make the second run differ

    def test_controller_uses_fixed_horizon_plan_without_storage_goal(self):
        env = _cold_env(cap_hours=96)
        env.reset(seed=1)
        controller = RollingMilpController(env, replan_every=12, planning_horizon_h=48)
        action = controller.policy(env)
        self.assertEqual(len(action["vessels"]), len(env.vessel_ids))
        self.assertEqual(len(action["wells"]), len(env.well_ids))
        self.assertTrue(all(MIN_WELL_RATE_MTPA <= rate <= MAX_WELL_RATE_MTPA for rate in action["wells"]))

    def test_empty_vessel_returns_to_best_available_emitter_not_fixed_home(self):
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

        action = RollingMilpController(env, replan_every=12, planning_horizon_h=96).policy(env)

        self.assertEqual(action["vessels"][0], env.vessel_go_emitter_action(other))

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

    def test_explicit_plan_can_depart_one_emitter_for_another(self):
        env = _two_source_one_ship_fast_env()
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source_a"] = 0.0
        env.simulator.state.entity_inventory_t["source_b"] = 500.0

        plan = _plan_explicit_actions(
            env,
            planning_horizon_h=5,
            economics=EconomicParameters(storage_shortfall_eur_per_t=1_000.0),
        )

        self.assertTrue(plan.is_valid, plan.validation_error)
        self.assertEqual(plan.vessel_actions_by_hour["ship"][0], env.vessel_go_emitter_action("source_b"))

    def test_explicit_plan_limits_future_injection_by_well_forecast(self):
        env = _no_capture_env(cap_hours=3)
        env.reset(seed=1)
        env.simulator.state.entity_inventory_t["source"] = 1_000.0
        env.simulator.state.entity_inventory_t["terminal"] = 1_000.0
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

        self.assertTrue(plan.is_valid, plan.validation_error)
        self.assertLessEqual(plan.injection_tph[1], 100.0 + 1e-6)
        self.assertAlmostEqual(plan.injection_tph[2], 0.0)


if __name__ == "__main__":
    unittest.main()
