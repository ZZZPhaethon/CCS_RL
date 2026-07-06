import unittest

from sim.environment import (
    MAX_WELL_RATE_INDEX,
    MIN_WELL_RATE_INDEX,
    MIN_WELL_RATE_MTPA,
    VESSEL_GO_TERMINAL,
    VESSEL_WAIT,
    CCSEnv,
    CCSEnvConfig,
    WELL_RATE_LEVELS_MTPA,
)
from sim.economics import CostModel, EconomicParameters
from sim.entities import Emitter, InjectionWell, Pipeline, Reservoir, Terminal, Vessel
from sim.line_source import LineSourceParameters, bottomhole_pressure_bar
from sim.network import PhysicalNetwork
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network


def _env(cost_model=None, **config) -> CCSEnv:
    return CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=48)),
        cost_model=cost_model,
        config=CCSEnvConfig(episode_hours=48, **config),
    )


def _action(vessels=None, wells=None):
    return {
        "vessels": [VESSEL_WAIT, VESSEL_WAIT] if vessels is None else vessels,
        "wells": [MIN_WELL_RATE_INDEX, MIN_WELL_RATE_INDEX] if wells is None else wells,
    }


class EnvSpaceTests(unittest.TestCase):
    def test_action_and_observation_dimensions(self):
        env = _env()
        self.assertEqual(env.vessel_action_dims, [4, 4])  # 2 vessels, 2 emitters
        self.assertEqual(env.well_rate_action_dims, [len(WELL_RATE_LEVELS_MTPA)] * 2)
        self.assertEqual(env.well_rate_levels_mtpa(), list(WELL_RATE_LEVELS_MTPA))
        obs = env.reset(seed=0)
        self.assertEqual(len(obs), env.observation_size)
        self.assertEqual(len(obs), len(env.feature_names))

    def test_reset_returns_finite_normalized_observation(self):
        obs = _env().reset(seed=0)
        self.assertTrue(all(isinstance(x, float) for x in obs))
        self.assertTrue(all(-1e6 < x < 1e6 for x in obs))

    def test_vessel_action_mask_shape_matches_vessel_action_dims(self):
        env = _env()
        env.reset(seed=0)
        mask = env.vessel_action_mask()
        self.assertEqual([len(m) for m in mask], env.vessel_action_dims)


class EnvDynamicsTests(unittest.TestCase):
    def test_vessels_start_at_emitters_and_can_choose_terminal_or_other_emitters(self):
        env = _env()
        env.reset(seed=0)
        source_a_action = env.vessel_go_emitter_action("source_a")
        source_b_action = env.vessel_go_emitter_action("source_b")

        vessel_a_mask = env.vessel_action_mask()[env.vessel_ids.index("vessel_a")]
        vessel_b_mask = env.vessel_action_mask()[env.vessel_ids.index("vessel_b")]

        self.assertTrue(vessel_a_mask[VESSEL_WAIT])
        self.assertTrue(vessel_a_mask[VESSEL_GO_TERMINAL])
        self.assertFalse(vessel_a_mask[source_a_action])
        self.assertTrue(vessel_a_mask[source_b_action])
        self.assertTrue(vessel_b_mask[VESSEL_GO_TERMINAL])
        self.assertTrue(vessel_b_mask[source_a_action])
        self.assertFalse(vessel_b_mask[source_b_action])

    def test_sailing_vessel_can_only_wait(self):
        env = _env()
        env.reset(seed=0)
        action = _action(vessels=[VESSEL_GO_TERMINAL, VESSEL_GO_TERMINAL])
        env.step(action)
        # Both ships are now sailing -> mask permits WAIT only.
        for i in range(len(env.vessel_ids)):
            self.assertEqual(env.vessel_action_mask()[i], [True, False, False, False])

    def test_well_rate_action_mask_respects_well_capacity_and_maintenance(self):
        env = _env()
        env.reset(seed=0)
        self.assertEqual(env.well_rate_action_mask()[0], [True, True, True, True, False, False])

        env.simulator.state.well_available[env.well_ids[0]] = False
        self.assertEqual(env.well_rate_action_mask()[0], [True, False, False, False, False, False])

    def test_well_rate_action_mask_respects_bottomhole_pressure_limit(self):
        parameters = LineSourceParameters(
            initial_pressure_bar=300.0,
            permeability_md=10.0,
            thickness_m=100.0,
            porosity_fraction=0.22,
            total_compressibility_1_pa=7e-10,
            viscosity_pa_s=6e-5,
            co2_density_kg_m3=630.0,
            well_radius_m=0.10795,
            skin=0.0,
        )
        pressure_limit_bar = bottomhole_pressure_bar(
            parameters,
            1.0,
            elapsed_days=1.0 / 24.0,
        )
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Emitter("source", nominal_capture_tph=0.0, buffer_capacity_t=1_000.0))
        network.add_entity(
            Vessel(
                "ship",
                capacity_t=500.0,
                loading_rate_tph=500.0,
                unloading_rate_tph=500.0,
                speed_knots=1.0,
            )
        )
        network.add_entity(Terminal("terminal", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=500.0))
        network.add_entity(InjectionWell("well", max_injection_tph=500.0))
        network.add_entity(
            Reservoir(
                "reservoir",
                storage_capacity_t=1_000_000.0,
                initial_pressure_bar=300.0,
                pressure_at_capacity_bar=350.0,
                max_pressure_bar=350.0,
                well_bottomhole_pressure_limit_bar=pressure_limit_bar,
                line_source_parameters=parameters,
            )
        )
        network.connect("source", "ship")
        network.connect("ship", "terminal")
        network.connect("terminal", "pipeline")
        network.connect("pipeline", "well")
        network.connect("well", "reservoir")
        env = CCSEnv(
            network,
            {"source": (0.0, 0.0), "terminal": (0.0, 1.0)},
            scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=2)),
            config=CCSEnvConfig(episode_hours=2),
            routes={
                "ship": {
                    "origin": "source",
                    "destination": "terminal",
                    "distance_km": 1.0,
                    "speed_knots": 1.0,
                }
            },
        )
        env.reset(seed=0)

        self.assertEqual(env.well_rate_action_mask()[0], [True, True, True, False, False, False])

    def test_lowest_available_well_rate_injects_half_mtpa_per_year(self):
        env = _env()
        env.reset(seed=0)
        terminal_id = env.terminal_ids[0]
        reservoir_id = env.reservoir_ids[0]
        env.simulator.state.entity_inventory_t[terminal_id] = 1_000.0
        before = env.simulator.state.entity_inventory_t.get(reservoir_id, 0.0)

        env.step(_action(wells=[MIN_WELL_RATE_INDEX, MIN_WELL_RATE_INDEX]))

        expected_per_well_tph = 0.5 * 1_000_000.0 / (365.25 * 24.0)
        after = env.simulator.state.entity_inventory_t.get(reservoir_id, 0.0)
        self.assertAlmostEqual(after - before, 2 * expected_per_well_tph)

    def test_well_rate_indices_map_to_discrete_rate_levels(self):
        env = _env()
        env.reset(seed=0)
        terminal_id = env.terminal_ids[0]
        reservoir_id = env.reservoir_ids[0]
        env.simulator.state.entity_inventory_t[terminal_id] = 1_000.0
        before = env.simulator.state.entity_inventory_t.get(reservoir_id, 0.0)

        env.step(_action(wells=[0, 3]))

        expected_tph = 1.5 * 1_000_000.0 / (365.25 * 24.0)
        after = env.simulator.state.entity_inventory_t.get(reservoir_id, 0.0)
        self.assertAlmostEqual(after - before, expected_tph)

    def test_vessel_can_milk_run_between_emitters_before_terminal(self):
        env = _env()
        env.reset(seed=0)
        vessel_id = "vessel_a"
        vessel = env.network.entities[vessel_id]
        env.simulator.state.entity_inventory_t["source_a"] = 200.0
        env.simulator.state.entity_inventory_t["source_b"] = 1_000.0
        env.simulator.state.entity_inventory_t[vessel_id] = 0.0
        env.simulator.state.entity_inventory_t["vessel_b"] = env.network.entities["vessel_b"].capacity_t
        env._routes[vessel_id]["speed_knots"] = 10_000.0

        env.step(_action())
        first_load_t = env.simulator.state.entity_inventory_t[vessel_id]
        expected_first_load_t = 200.0 + env.simulator.state.last_capture_tph["source_a"]
        self.assertAlmostEqual(first_load_t, expected_first_load_t)

        env.step(_action(vessels=[env.vessel_go_emitter_action("source_b"), VESSEL_WAIT]))
        self.assertEqual(env.simulator.state.vessel_berths[vessel_id], "source_b")

        env.step(_action())
        final_load_t = env.simulator.state.entity_inventory_t[vessel_id]
        self.assertGreater(final_load_t, first_load_t)
        self.assertLessEqual(final_load_t, vessel.capacity_t)

    def test_episode_runs_to_done(self):
        env = _env()
        env.reset(seed=1)
        steps = 0
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(_action())
            done = terminated or truncated
            self.assertIsInstance(reward, float)
            steps += 1
        self.assertEqual(steps, env.n_steps)
        self.assertIn("storage_rate", info)

    def test_reward_excludes_shortfall_delta_penalty(self):
        env = _env(cost_model=CostModel(EconomicParameters(storage_shortfall_eur_per_t=100.0)))
        env.reset(seed=0)

        _obs, reward, _terminated, _truncated, info = env.step(_action())

        self.assertGreater(info["shortfall_delta_penalty"], 0.0)
        self.assertNotIn("backlog_penalty", info)
        self.assertIn("in_transit_t", info)
        self.assertIn("in_transit_growth_t", info)
        self.assertNotIn("backlog_t", info)
        self.assertNotIn("backlog_growth_t", info)
        self.assertAlmostEqual(env.ledger.storage_shortfall_penalty, info["shortfall_penalty"])
        self.assertAlmostEqual(
            reward,
            info["economics"]["net"] * env.config.reward_scale,
        )

    def test_horizon_end_is_truncation_not_termination(self):
        env = _env()
        env.reset(seed=0)
        terminated = truncated = False
        for _ in range(env.n_steps):
            _, _, terminated, truncated, _ = env.step(_action())
        # The operation never truly ends; the horizon is only a time limit, so the
        # trainer must bootstrap the value of the leftover state.
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_observation_has_no_horizon_relative_features(self):
        env = _env()
        self.assertIn("hour_of_week", env.feature_names)
        self.assertIn("in_transit_fill", env.feature_names)
        self.assertNotIn("backlog_fill", env.feature_names)
        self.assertNotIn("time_fraction", env.feature_names)

    def test_deterministic_for_seed_and_policy(self):
        def rollout():
            env = _env()
            env.reset(seed=123)
            rewards = []
            for _ in range(env.n_steps):
                _, r, _terminated, _truncated, _ = env.step(_action(wells=[3, 3]))
                rewards.append(r)
            return rewards

        self.assertEqual(rollout(), rollout())

    def test_shuttle_policy_actually_stores_co2(self):
        env = _env()
        env.reset(seed=2)
        done = False
        while not done:
            vessel_actions = []
            for i, _vid in enumerate(env.vessel_ids):
                mask = env.vessel_action_mask()[i]
                if mask[VESSEL_GO_TERMINAL]:
                    vessel_actions.append(VESSEL_GO_TERMINAL)
                else:
                    vessel_actions.append(VESSEL_WAIT)
            action = {
                "vessels": vessel_actions,
                "wells": [MAX_WELL_RATE_INDEX] * len(env.well_ids),
            }
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        self.assertGreater(env.cumulative_stored_t, 0.0)
        self.assertGreater(env.ledger.reconditioning, 0.0)
        self.assertTrue(0.0 <= info["storage_rate"] <= 1.0)


class EnvGuardTests(unittest.TestCase):
    def test_step_before_reset_raises(self):
        with self.assertRaises(RuntimeError):
            _env().step(_action())

    def test_wrong_action_length_raises(self):
        env = _env()
        env.reset(seed=0)
        with self.assertRaises(ValueError):
            env.step({"vessels": [VESSEL_WAIT], "wells": [MIN_WELL_RATE_INDEX]})

    def test_invalid_well_action_index_raises(self):
        env = _env()
        env.reset(seed=0)
        with self.assertRaises(ValueError):
            env.step(_action(wells=[len(WELL_RATE_LEVELS_MTPA), MIN_WELL_RATE_INDEX]))


if __name__ == "__main__":
    unittest.main()
