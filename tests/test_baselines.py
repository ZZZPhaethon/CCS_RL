import unittest

from sim.control.baselines import greedy_shuttle_policy
from sim.environment import CCSEnv, CCSEnvConfig, MIN_WELL_RATE_INDEX
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network


class GreedyBaselineTests(unittest.TestCase):
    def test_empty_vessel_goes_to_best_buffered_emitter(self):
        env = CCSEnv(
            make_toy_two_source_network(),
            TOY_TWO_SOURCE_LOCATIONS,
            scenario_generator=ScenarioGenerator(
                config=ScenarioConfig(episode_hours=24, randomize_initial_inventory=False)
            ),
            config=CCSEnvConfig(episode_hours=24),
        )
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

        action = greedy_shuttle_policy(env)

        self.assertEqual(action["vessels"][0], env.vessel_go_emitter_action(other))
        self.assertEqual(len(action["wells"]), len(env.well_ids))
        self.assertTrue(all(isinstance(index, int) for index in action["wells"]))
        self.assertTrue(all(index >= MIN_WELL_RATE_INDEX for index in action["wells"]))


if __name__ == "__main__":
    unittest.main()
