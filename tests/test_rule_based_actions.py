import unittest

from sim.entities import Emitter, InjectionWell, Pipeline, Terminal, Vessel
from sim.actions import ActionResolver
from sim.control.rule_based import RuleBasedActionGenerator
from sim.network_scenarios import build_northern_lights_phase1_demo


class RuleBasedActionGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.network, self.state = build_northern_lights_phase1_demo()
        self.routes = {
            "northern_pioneer": {"origin": "brevik", "destination": "oygarden_terminal"},
            "northern_pathfinder": {"origin": "celsio", "destination": "oygarden_terminal"},
            "northern_phoenix": {"origin": "yara_sluiskil", "destination": "oygarden_terminal"},
            "phase1_vessel_04": {"origin": "yara_sluiskil", "destination": "oygarden_terminal"},
        }
        self.generator = RuleBasedActionGenerator(self.network, self.routes)

    def _actions_by_entity(self):
        frame = self.generator.next_action_frame(self.state)
        actions = {}
        for proposal in frame.proposals:
            actions.setdefault(proposal.entity_id, {})[proposal.verb] = proposal.params
        return actions

    def test_emitters_are_kept_at_full_utilization(self):
        actions = self._actions_by_entity()

        emitter_ids = {
            entity_id
            for entity_id, entity in self.network.entities.items()
            if isinstance(entity, Emitter)
        }
        self.assertEqual(
            {
                entity_id
                for entity_id, entity_actions in actions.items()
                if entity_actions.get("set_capture_utilization") == {"utilization": 1.0}
            },
            emitter_ids,
        )

    def test_berthed_not_full_vessel_loads_at_home_emitter(self):
        self.state.vessel_berths["northern_pioneer"] = "brevik"
        self.state.entity_inventory_t["brevik"] = 1000.0
        self.state.entity_inventory_t["northern_pioneer"] = 2000.0

        actions = self._actions_by_entity()

        self.assertEqual(actions["brevik"]["load_vessel"], {"vessel_id": "northern_pioneer"})
        self.assertNotIn("sail_to", actions.get("northern_pioneer", {}))

    def test_full_vessel_sails_to_terminal(self):
        vessel = self.network.entities["northern_pioneer"]
        self.assertIsInstance(vessel, Vessel)
        self.state.vessel_berths["northern_pioneer"] = "brevik"
        self.state.entity_inventory_t["northern_pioneer"] = vessel.capacity_t

        actions = self._actions_by_entity()

        self.assertEqual(actions["northern_pioneer"]["sail_to"], {"destination_id": "oygarden_terminal"})
        self.assertNotIn("load_vessel", actions.get("brevik", {}))

    def test_empty_vessel_at_terminal_sails_to_best_buffered_emitter(self):
        self.state.vessel_berths["northern_pioneer"] = "oygarden_terminal"
        self.state.entity_inventory_t["northern_pioneer"] = 0.0
        self.state.entity_inventory_t["brevik"] = 0.0
        self.state.entity_inventory_t["celsio"] = 5_000.0

        actions = self._actions_by_entity()

        self.assertEqual(actions["northern_pioneer"]["sail_to"], {"destination_id": "celsio"})

    def test_loaded_vessel_at_terminal_unloads_and_pipeline_uses_one_well(self):
        self.state.vessel_berths["northern_pioneer"] = "oygarden_terminal"
        self.state.entity_inventory_t["northern_pioneer"] = 800.0

        actions = self._actions_by_entity()

        self.assertEqual(actions["oygarden_terminal"]["unload_vessel"], {"vessel_id": "northern_pioneer"})
        self.assertIn("set_flow", actions["oygarden_pipeline"])
        self.assertEqual(
            actions["aurora_subsea_manifold"]["set_well_split"],
            {"well_splits": {"aurora_well_a7_ah": 1.0}},
        )
        well_actions = {
            entity_id: entity_actions
            for entity_id, entity_actions in actions.items()
            if isinstance(self.network.entities.get(entity_id), InjectionWell)
        }
        self.assertEqual(well_actions, {})

    def test_terminal_unloading_is_first_in_first_out_for_rule_based_baseline(self):
        self.state.vessel_berths["northern_pathfinder"] = "oygarden_terminal"
        self.state.entity_inventory_t["northern_pathfinder"] = 7500.0
        self.generator.next_action_frame(self.state)

        self.state.vessel_berths["northern_pioneer"] = "oygarden_terminal"
        self.state.entity_inventory_t["northern_pioneer"] = 7500.0

        frame = self.generator.next_action_frame(self.state)
        committed = ActionResolver(self.network).resolve(frame)

        self.assertEqual(committed.actions["oygarden_terminal"]["unload_vessel"], "northern_pathfinder")

    def test_pipeline_flow_is_capped_by_selected_well_capacity(self):
        terminal = self.network.entities["oygarden_terminal"]
        pipeline = self.network.entities["oygarden_pipeline"]
        selected_well = self.network.entities["aurora_well_a7_ah"]
        self.assertIsInstance(terminal, Terminal)
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIsInstance(selected_well, InjectionWell)
        self.state.entity_inventory_t["oygarden_terminal"] = terminal.storage_capacity_t

        actions = self._actions_by_entity()

        self.assertEqual(
            actions["oygarden_pipeline"]["set_flow"],
            {"flow_tph": min(pipeline.max_flow_tph, selected_well.max_injection_tph)},
        )


if __name__ == "__main__":
    unittest.main()
