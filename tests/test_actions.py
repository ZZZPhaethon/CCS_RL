import unittest

from sim.actions import ActionFrame, ActionProposal, ActionResolver
from sim.network_scenarios import build_fixed_scenario_demo


class ActionInterfaceTests(unittest.TestCase):
    def _network(self):
        network, _ = build_fixed_scenario_demo("northern_lights_phase1_2well")
        return network

    def test_resolver_commits_heterogeneous_entity_actions_to_network_format(self):
        network = self._network()
        resolver = ActionResolver(network)
        frame = ActionFrame(
            time_h=0.0,
            proposals=[
                ActionProposal(
                    agent_id="emitter_agent",
                    entity_id="brevik",
                    verb="load_vessel",
                    params={"vessel_id": "northern_pioneer"},
                ),
                ActionProposal(
                    agent_id="ship_agent",
                    entity_id="northern_pioneer",
                    verb="sail_to",
                    params={"destination_id": "oygarden_terminal"},
                ),
                ActionProposal(
                    agent_id="pipeline_agent",
                    entity_id="oygarden_pipeline",
                    verb="set_flow",
                    params={"flow_tph": 200.0},
                ),
                ActionProposal(
                    agent_id="manifold_agent",
                    entity_id="aurora_subsea_manifold",
                    verb="set_well_split",
                    params={"well_splits": {"aurora_well_a7_ah": 0.6, "aurora_well_c1_h": 0.4}},
                ),
            ],
        )

        committed = resolver.resolve(frame)

        self.assertEqual(committed.time_h, 0.0)
        self.assertEqual(
            committed.actions,
            {
                "brevik": {"load_vessel": "northern_pioneer"},
                "northern_pioneer": {"sail_to": "oygarden_terminal"},
                "oygarden_pipeline": {"flow_tph": 200.0},
                "aurora_subsea_manifold": {"well_splits": {"aurora_well_a7_ah": 0.6, "aurora_well_c1_h": 0.4}},
            },
        )
        self.assertTrue(all(decision.accepted for decision in committed.decisions))

    def test_resolver_rejects_actions_not_supported_by_entity_type(self):
        network = self._network()
        resolver = ActionResolver(network)
        frame = ActionFrame(
            time_h=0.0,
            proposals=[
                ActionProposal(
                    agent_id="pipeline_agent",
                    entity_id="oygarden_pipeline",
                    verb="load_vessel",
                    params={"vessel_id": "northern_pioneer"},
                )
            ],
        )

        committed = resolver.resolve(frame)

        self.assertEqual(committed.actions, {})
        self.assertEqual(len(committed.decisions), 1)
        self.assertFalse(committed.decisions[0].accepted)
        self.assertIn("does not support", committed.decisions[0].reason)

    def test_resolver_rejects_invalid_numeric_action_parameters(self):
        network = self._network()
        resolver = ActionResolver(network)
        frame = ActionFrame(
            time_h=0.0,
            proposals=[
                ActionProposal(
                    agent_id="emitter_agent",
                    entity_id="brevik",
                    verb="set_capture_utilization",
                    params={"utilization": 1.5},
                ),
                ActionProposal(
                    agent_id="pipeline_agent",
                    entity_id="oygarden_pipeline",
                    verb="set_flow",
                    params={"flow_tph": -1.0},
                ),
            ],
        )

        committed = resolver.resolve(frame)

        self.assertEqual(committed.actions, {})
        self.assertEqual([decision.accepted for decision in committed.decisions], [False, False])
        self.assertIn("between 0 and 1", committed.decisions[0].reason)
        self.assertIn("non-negative", committed.decisions[1].reason)

    def test_resolver_rejects_invalid_boolean_action_parameters(self):
        network = self._network()
        resolver = ActionResolver(network)
        frame = ActionFrame(
            time_h=0.0,
            proposals=[
                ActionProposal(
                    agent_id="well_agent",
                    entity_id="aurora_well_a7_ah",
                    verb="set_available",
                    params={"available": "false"},
                )
            ],
        )

        committed = resolver.resolve(frame)

        self.assertEqual(committed.actions, {})
        self.assertFalse(committed.decisions[0].accepted)
        self.assertIn("must be boolean", committed.decisions[0].reason)

    def test_resolver_rejects_invalid_manifold_split_parameters(self):
        network = self._network()
        resolver = ActionResolver(network)
        frame = ActionFrame(
            time_h=0.0,
            proposals=[
                ActionProposal(
                    agent_id="manifold_agent",
                    entity_id="aurora_subsea_manifold",
                    verb="set_well_split",
                    params={"well_splits": {"aurora_well_a7_ah": 0.7, "aurora_well_c1_h": 0.7}},
                )
            ],
        )

        committed = resolver.resolve(frame)

        self.assertEqual(committed.actions, {})
        self.assertFalse(committed.decisions[0].accepted)
        self.assertIn("sum to 1", committed.decisions[0].reason)

    def test_resolver_merges_compatible_actions_for_same_entity(self):
        network = self._network()
        resolver = ActionResolver(network)
        frame = ActionFrame(
            time_h=0.0,
            proposals=[
                ActionProposal(
                    agent_id="emitter_agent",
                    entity_id="brevik",
                    verb="set_capture_utilization",
                    params={"utilization": 0.75},
                ),
                ActionProposal(
                    agent_id="emitter_agent",
                    entity_id="brevik",
                    verb="load_vessel",
                    params={"vessel_id": "northern_pioneer"},
                ),
            ],
        )

        committed = resolver.resolve(frame)

        self.assertEqual(committed.actions["brevik"], {"utilization": 0.75, "load_vessel": "northern_pioneer"})
        self.assertTrue(all(decision.accepted for decision in committed.decisions))

    def test_resolver_rejects_conflicting_actions_for_same_entity(self):
        network = self._network()
        resolver = ActionResolver(network)
        frame = ActionFrame(
            time_h=0.0,
            proposals=[
                ActionProposal(
                    agent_id="ship_agent",
                    entity_id="northern_pioneer",
                    verb="sail_to",
                    params={"destination_id": "oygarden_terminal"},
                ),
                ActionProposal(
                    agent_id="ship_agent",
                    entity_id="northern_pioneer",
                    verb="sail_to",
                    params={"destination_id": "brevik"},
                ),
            ],
        )

        committed = resolver.resolve(frame)

        self.assertEqual(committed.actions["northern_pioneer"], {"sail_to": "oygarden_terminal"})
        self.assertTrue(committed.decisions[0].accepted)
        self.assertFalse(committed.decisions[1].accepted)
        self.assertIn("conflicts", committed.decisions[1].reason)

    def test_supported_actions_are_reported_by_entity_type(self):
        network = self._network()
        resolver = ActionResolver(network)

        supported = resolver.supported_actions_by_entity()

        self.assertEqual(supported["brevik"], ["set_capture_utilization", "load_vessel", "hold"])
        self.assertEqual(supported["northern_pioneer"], ["sail_to", "hold"])
        self.assertEqual(supported["oygarden_terminal"], ["unload_vessel", "hold"])
        self.assertEqual(supported["oygarden_pipeline"], ["set_flow", "hold"])
        self.assertEqual(supported["aurora_subsea_manifold"], ["set_well_split", "hold"])
        self.assertEqual(supported["aurora_well_a7_ah"], ["set_available", "set_injection_limit", "hold"])
        self.assertEqual(supported["aurora_reservoir"], ["hold"])


if __name__ == "__main__":
    unittest.main()
