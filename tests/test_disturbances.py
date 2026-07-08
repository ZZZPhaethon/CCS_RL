import unittest

from sim.actions import ActionFrame, ActionProposal
from sim.scenario_generation.disturbance_resolver import (
    emitter_availability,
    leg_speed_factor,
    terminal_berth_count,
    vessel_speed_factor,
    well_injectivity_factor,
    well_is_available,
    well_max_injection_tph,
)
from sim.entities import (
    Emitter,
    InjectionWell,
    PhysicalState,
    Pipeline,
    Terminal,
    Vessel,
)
from sim.network import PhysicalNetwork
from sim.simulator import PhysicalSimulator


class DisturbanceResolverTests(unittest.TestCase):
    def test_overrides_fall_back_to_nominal_when_absent(self):
        state = PhysicalState()
        emitter = Emitter("e", nominal_capture_tph=100.0, buffer_capacity_t=1_000.0, availability=0.9)
        well = InjectionWell("w", max_injection_tph=200.0)
        terminal = Terminal("t", storage_capacity_t=1_000.0, berth_count=2)

        self.assertEqual(emitter_availability(state, emitter), 0.9)
        self.assertTrue(well_is_available(state, well))
        self.assertEqual(well_injectivity_factor(state, well), 1.0)
        self.assertEqual(well_max_injection_tph(state, well), 200.0)
        self.assertEqual(vessel_speed_factor(state, "ship"), 1.0)
        self.assertEqual(leg_speed_factor(state, "a", "b"), 1.0)
        self.assertEqual(terminal_berth_count(state, terminal), 2)

    def test_overrides_take_precedence_and_are_clamped(self):
        emitter = Emitter("e", nominal_capture_tph=100.0, buffer_capacity_t=1_000.0)
        well = InjectionWell("w", max_injection_tph=200.0)
        terminal = Terminal("t", storage_capacity_t=1_000.0, berth_count=2)
        state = PhysicalState(
            emitter_availability={"e": 1.5},  # capture multipliers can exceed 1.0
            well_available={"w": False},
            injectivity_factor={"w": -0.5},  # clamped to 0.0
            vessel_speed_factor={"ship": 0.25},
            leg_speed_factor={"a->b": 0.5},
            berth_count_override={"t": 0},
        )

        self.assertEqual(emitter_availability(state, emitter), 1.5)
        self.assertFalse(well_is_available(state, well))
        self.assertEqual(well_injectivity_factor(state, well), 0.0)
        self.assertEqual(well_max_injection_tph(state, well), 0.0)
        self.assertEqual(vessel_speed_factor(state, "ship"), 0.25)
        self.assertEqual(leg_speed_factor(state, "a", "b"), 0.5)
        self.assertEqual(terminal_berth_count(state, terminal), 0)


class DisturbancePhysicsTests(unittest.TestCase):
    def _capture_network(self) -> PhysicalNetwork:
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Emitter("brevik", nominal_capture_tph=100.0, buffer_capacity_t=10_000.0))
        return network

    def test_emitter_availability_override_derates_capture(self):
        network = self._capture_network()
        state = PhysicalState(emitter_availability={"brevik": 0.4})

        result = network.step(state)

        self.assertAlmostEqual(result.state.entity_inventory_t["brevik"], 40.0)
        self.assertAlmostEqual(result.state.last_capture_tph["brevik"], 40.0)

    def test_emitter_capture_multiplier_can_raise_capture_above_profile(self):
        network = self._capture_network()
        state = PhysicalState(emitter_availability={"brevik": 1.1})

        result = network.step(state)

        self.assertAlmostEqual(result.state.entity_inventory_t["brevik"], 110.0)
        self.assertAlmostEqual(result.state.last_capture_tph["brevik"], 110.0)

    def _injection_network(self) -> PhysicalNetwork:
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.add_entity(Terminal("oygarden", storage_capacity_t=2_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=500.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=200.0))
        network.connect("ship_1", "oygarden")
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")
        return network

    def test_well_maintenance_blocks_injection(self):
        network = self._injection_network()
        state = PhysicalState(
            vessel_berths={"ship_1": "oygarden"},
            entity_inventory_t={"ship_1": 800.0, "oygarden": 500.0},
            well_available={"well_1": False},
        )

        result = network.step(
            state,
            actions={"oygarden": {"unload_tph": 200.0}, "pipeline": {"flow_tph": 200.0}},
        )

        self.assertEqual(result.state.entity_inventory_t.get("well_1", 0.0), 0.0)
        self.assertEqual(result.state.last_injection_flow_tph["well_1"], 0.0)
        self.assertTrue(any(v.violation_type == "flow_clipped" for v in result.violations))

    def test_injectivity_factor_halves_injection_capacity(self):
        network = self._injection_network()
        state = PhysicalState(
            vessel_berths={"ship_1": "oygarden"},
            entity_inventory_t={"ship_1": 800.0, "oygarden": 500.0},
            injectivity_factor={"well_1": 0.5},
        )

        result = network.step(
            state,
            actions={"oygarden": {"unload_tph": 200.0}, "pipeline": {"flow_tph": 200.0}},
        )

        # Nominal well ceiling is 200 t/h; the 0.5 factor clips injection to 100 t.
        self.assertAlmostEqual(result.state.entity_inventory_t["well_1"], 100.0)
        self.assertAlmostEqual(result.state.last_injection_flow_tph["well_1"], 100.0)

    def test_manual_berth_override_limits_concurrent_unloads(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=300.0))
        network.add_entity(Vessel("ship_2", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=300.0))
        network.add_entity(Terminal("oygarden", storage_capacity_t=10_000.0, berth_count=2))
        network.add_entity(Pipeline("pipeline", max_flow_tph=1_000.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=1_000.0))
        network.connect("ship_1", "oygarden")
        network.connect("ship_2", "oygarden")
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")

        def cargo_after_unload(berth_count: int) -> dict[str, float]:
            state = PhysicalState(
                vessel_berths={"ship_1": "oygarden", "ship_2": "oygarden"},
                entity_inventory_t={"ship_1": 800.0, "ship_2": 800.0},
                berth_count_override={"oygarden": berth_count},
            )
            result = network.step(state, actions={"oygarden": {"unload_tph": 600.0}})
            return {ship: result.state.entity_inventory_t[ship] for ship in ("ship_1", "ship_2")}

        # With two berths both vessels unload (300 t/h cap each).
        both = cargo_after_unload(2)
        self.assertEqual(sum(c < 800.0 for c in both.values()), 2)

        # A one-berth override lets only a single vessel unload.
        one = cargo_after_unload(1)
        self.assertEqual(sum(c < 800.0 for c in one.values()), 1)


class DisturbanceVoyageTests(unittest.TestCase):
    def _make_simulator(self) -> PhysicalSimulator:
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(
            Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0, speed_knots=10.0)
        )
        state = PhysicalState(vessel_berths={"ship_1": "origin"})
        return PhysicalSimulator(
            network,
            state,
            routes={
                "ship_1": {
                    "origin": "origin",
                    "destination": "terminal",
                    "distance_km": 100.0,
                    "speed_knots": 10.0,
                    "coordinates": [(0.0, 0.0), (0.0, 1.0)],
                    "return_coordinates": [(0.0, 1.0), (0.0, 0.0)],
                }
            },
            locations={"origin": (0.0, 0.0), "terminal": (0.0, 1.0)},
        )

    def _sail_step(self, simulator: PhysicalSimulator):
        return simulator.step(
            ActionFrame(
                time_h=0.0,
                proposals=[
                    ActionProposal(
                        agent_id="ship_agent",
                        entity_id="ship_1",
                        verb="sail_to",
                        params={"destination_id": "terminal"},
                    )
                ],
            )
        )

    def test_weather_speed_factor_slows_voyage_progress(self):
        nominal = self._make_simulator()
        nominal_record = self._sail_step(nominal)
        nominal_lon = nominal_record.vessel_positions["ship_1"]["lon"]

        slowed = self._make_simulator()
        slowed.state.vessel_speed_factor["ship_1"] = 0.5
        slowed_record = self._sail_step(slowed)
        slowed_lon = slowed_record.vessel_positions["ship_1"]["lon"]

        self.assertGreater(nominal_lon, 0.0)
        self.assertAlmostEqual(slowed_lon, nominal_lon * 0.5, places=6)

    def test_leg_speed_factor_overrides_vessel_factor_for_current_leg(self):
        slowed = self._make_simulator()
        slowed.state.vessel_speed_factor["ship_1"] = 1.0
        slowed.state.leg_speed_factor["origin->terminal"] = 0.5
        slowed_record = self._sail_step(slowed)

        nominal = self._make_simulator()
        nominal_record = self._sail_step(nominal)

        self.assertAlmostEqual(
            slowed_record.vessel_positions["ship_1"]["lon"],
            nominal_record.vessel_positions["ship_1"]["lon"] * 0.5,
            places=6,
        )

    def test_zero_speed_factor_stalls_vessel_in_place(self):
        stalled = self._make_simulator()
        stalled.state.vessel_speed_factor["ship_1"] = 0.0
        record = self._sail_step(stalled)

        self.assertFalse(record.vessel_positions["ship_1"]["at_berth"])
        self.assertAlmostEqual(record.vessel_positions["ship_1"]["lon"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
