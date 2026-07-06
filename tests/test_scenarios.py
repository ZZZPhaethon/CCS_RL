import csv
import json
import unittest

from sim.entities import Emitter, InjectionWell, Pipeline, Reservoir, SubseaManifold, Terminal, Vessel
import sim.network_scenarios as scenarios
from sim.routes import route_distance_km
from sim.network_scenarios import (
    EOS_SUBSEA_TEMPLATE_LOCATION,
    NATURGASSPARKEN,
    NORTHERN_LIGHTS_PHASE1_CAPTURE_PROFILE_PATH,
    NORTHERN_LIGHTS_PHASE1_DATA_PATH,
    NORTHERN_LIGHTS_PHASE1_2WELL_DATA_PATH,
    NORTHERN_LIGHTS_PHASE2_DATA_PATH,
    available_fixed_scenario_choices,
    build_fixed_scenario_demo,
    build_northern_lights_phase2_demo,
    build_northern_lights_phase1_demo,
    fixed_scenario_locations,
)


class ScenarioTests(unittest.TestCase):
    def test_phase1_demo_has_one_hour_step_and_expected_chain(self):
        network, state = build_northern_lights_phase1_demo()

        self.assertEqual(network.time_step_hours, 1.0)
        self.assertIn("brevik", network.entities)
        self.assertIn("northern_pioneer", network.entities)
        self.assertIn("aurora_reservoir", network.entities)
        self.assertIn("aurora_subsea_manifold", network.entities)
        self.assertIn("yara_sluiskil", network.entities)
        self.assertEqual(network.downstream_of("oygarden_terminal"), ["oygarden_pipeline"])
        self.assertEqual(network.downstream_of("oygarden_pipeline"), ["aurora_subsea_manifold"])
        self.assertEqual(network.downstream_of("aurora_subsea_manifold"), ["aurora_well_a7_ah"])
        self.assertEqual(network.downstream_of("aurora_well_a7_ah"), ["aurora_reservoir"])
        self.assertEqual(state.entity_inventory_t["northern_pioneer"], 0.0)

    def test_phase1_demo_includes_reference_physical_parameters(self):
        network, _ = build_northern_lights_phase1_demo()
        brevik = network.entities["brevik"]
        celsio = network.entities["celsio"]
        vessel = network.entities["northern_pioneer"]
        terminal = network.entities["oygarden_terminal"]
        pipeline = network.entities["oygarden_pipeline"]
        manifold = network.entities["aurora_subsea_manifold"]
        reservoir = network.entities["aurora_reservoir"]

        self.assertIsInstance(brevik, Emitter)
        self.assertAlmostEqual(brevik.annual_target_export_tpy, 400_000.0)
        self.assertAlmostEqual(brevik.max_production_tph, 56.0)
        self.assertAlmostEqual(brevik.nominal_capture_tph, 400_000.0 / 8760.0)
        self.assertIsInstance(celsio, Emitter)
        self.assertAlmostEqual(celsio.annual_target_export_tpy, 350_000.0)
        self.assertAlmostEqual(celsio.max_production_tph, 48.0)
        self.assertAlmostEqual(celsio.nominal_capture_tph, 350_000.0 / 8760.0)
        self.assertIsInstance(vessel, Vessel)
        self.assertAlmostEqual(vessel.volume_capacity_m3, 7_500.0)
        self.assertAlmostEqual(vessel.speed_knots, 14.0)
        self.assertIsInstance(terminal, Terminal)
        self.assertEqual(terminal.berth_count, 1)
        self.assertEqual(terminal.site_name, "Naturgassparken / Oygarden receiving terminal")
        self.assertIsInstance(pipeline, Pipeline)
        self.assertAlmostEqual(pipeline.annual_capacity_tpy, 5_000_000.0)
        self.assertAlmostEqual(pipeline.max_flow_tph, 5_000_000.0 / 8760.0)
        self.assertAlmostEqual(pipeline.length_km, 100.4)
        self.assertEqual(pipeline.route_color, "#ff0000")
        self.assertGreaterEqual(len(pipeline.route_coordinates), 4)
        self.assertEqual(pipeline.route_coordinates[0], NATURGASSPARKEN)
        self.assertEqual(pipeline.route_coordinates[-1], EOS_SUBSEA_TEMPLATE_LOCATION)
        self.assertAlmostEqual(route_distance_km(pipeline.route_coordinates), 100.4, delta=0.5)
        self.assertIsInstance(manifold, SubseaManifold)
        self.assertAlmostEqual(manifold.max_flow_tph, 5_000_000.0 / 8760.0)
        self.assertIsInstance(reservoir, Reservoir)
        self.assertAlmostEqual(reservoir.depth_m, 2_600.0)
        self.assertAlmostEqual(reservoir.initial_pressure_bar, 300.0)
        self.assertAlmostEqual(reservoir.pressure_at_capacity_bar, 350.0)
        self.assertAlmostEqual(reservoir.max_pressure_bar, 350.0)
        self.assertAlmostEqual(reservoir.seawater_depth_m, 300.0)
        self.assertAlmostEqual(reservoir.well_fracture_gradient_psi_per_ft, 0.7)
        self.assertAlmostEqual(reservoir.well_fracture_gradient_reference_depth_m, 2900.0)
        self.assertAlmostEqual(reservoir.well_fracture_pressure_bar, 459.2)
        self.assertAlmostEqual(reservoir.well_bottomhole_pressure_safety_factor, 0.9)
        self.assertAlmostEqual(reservoir.well_bottomhole_pressure_limit_bar, 413.3)

    def test_phase1_demo_emitters_can_be_fully_curtailed(self):
        network, _ = build_northern_lights_phase1_demo()

        for entity in network.entities.values():
            if isinstance(entity, Emitter):
                self.assertEqual(entity.min_utilization, 0.0)

    def test_phase1_demo_marks_line_source_assumptions(self):
        network, _ = build_northern_lights_phase1_demo()
        reservoir = network.entities["aurora_reservoir"]

        self.assertIsInstance(reservoir, Reservoir)
        self.assertIsNotNone(reservoir.line_source_parameters)
        self.assertAlmostEqual(reservoir.line_source_parameters.initial_pressure_bar, 300.0)
        self.assertAlmostEqual(reservoir.line_source_parameters.permeability_md, 10.0)
        self.assertAlmostEqual(reservoir.line_source_parameters.thickness_m, 100.0)
        self.assertEqual(reservoir.line_source_parameter_status["well_radius_m"], "derived_from_concept_report")
        self.assertEqual(reservoir.line_source_parameter_status["viscosity_pa_s"], "assumed")
        self.assertEqual(reservoir.line_source_parameter_status["co2_density_kg_m3"], "assumed")
        self.assertEqual(reservoir.line_source_observation_radii_m, (100.0, 1000.0))

    def test_phase1_demo_loads_reference_values_from_external_data_file(self):
        network, _ = build_northern_lights_phase1_demo()
        with NORTHERN_LIGHTS_PHASE1_DATA_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        reservoir = network.entities["aurora_reservoir"]
        pipeline = network.entities["oygarden_pipeline"]
        reservoir_payload = payload["reservoir"]

        self.assertIsInstance(reservoir, Reservoir)
        self.assertIn("reservoir_average_pressure_at_capacity_bar", reservoir_payload)
        self.assertIn("reservoir_average_pressure_limit_bar", reservoir_payload)
        self.assertIn("well_bottomhole_pressure_limit_bar", reservoir_payload)
        self.assertNotIn("initial_pressure_bar", reservoir_payload)
        self.assertNotIn("pressure_at_capacity_bar", reservoir_payload)
        self.assertNotIn("max_pressure_bar", reservoir_payload)
        self.assertNotIn("depth_m", reservoir_payload)
        self.assertNotIn("pressure_at_capacity_bar_status", reservoir_payload)
        self.assertNotIn("max_pressure_bar_status", reservoir_payload)
        self.assertIn("initial_reservoir_pressure_bar", payload["line_source_parameters"])
        self.assertNotIn("initial_pressure_bar", payload["line_source_parameters"])
        self.assertAlmostEqual(
            reservoir.line_source_parameters.permeability_md,
            payload["line_source_parameters"]["permeability_md"],
        )
        self.assertAlmostEqual(pipeline.length_km, payload["pipeline"]["length_km"])

    def test_phase2_demo_loads_public_scenario_with_current_two_wells(self):
        network, state = build_northern_lights_phase2_demo()
        with NORTHERN_LIGHTS_PHASE2_DATA_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        emitters = [entity for entity in network.entities.values() if isinstance(entity, Emitter)]
        vessels = [entity for entity in network.entities.values() if isinstance(entity, Vessel)]
        wells = [entity for entity in network.entities.values() if isinstance(entity, InjectionWell)]

        self.assertEqual(len(emitters), 5)
        self.assertEqual(len(vessels), 8)
        self.assertEqual(len(wells), 2)
        self.assertNotIn("aurora_phase2_well_1", network.entities)
        self.assertAlmostEqual(
            sum(emitter.annual_target_export_tpy or 0.0 for emitter in emitters),
            payload["contracted_annual_target_export_tpy"],
        )
        self.assertEqual(network.downstream_of("aurora_subsea_manifold"), ["aurora_well_a7_ah", "aurora_well_c1_h"])
        self.assertEqual(network.downstream_of("aurora_well_a7_ah"), ["aurora_reservoir"])
        self.assertEqual(network.downstream_of("aurora_well_c1_h"), ["aurora_reservoir"])
        self.assertEqual(state.entity_inventory_t["stockholm_exergi"], 0.0)

    def test_phase1_demo_has_three_emitters_four_ships_and_one_well(self):
        network, state = build_northern_lights_phase1_demo()
        with NORTHERN_LIGHTS_PHASE1_DATA_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        emitters = [entity for entity in network.entities.values() if isinstance(entity, Emitter)]
        vessels = [entity for entity in network.entities.values() if isinstance(entity, Vessel)]
        wells = [entity for entity in network.entities.values() if isinstance(entity, InjectionWell)]

        self.assertEqual(len(emitters), 3)
        self.assertEqual(len(vessels), 4)
        self.assertEqual(len(wells), 1)
        self.assertIn("yara_sluiskil", network.entities)
        self.assertIn("aurora_well_a7_ah", network.entities)
        self.assertNotIn("aurora_well_c1_h", network.entities)
        celsio = network.entities["celsio"]
        self.assertIsInstance(celsio, Emitter)
        self.assertAlmostEqual(celsio.annual_target_export_tpy, 350_000.0)
        self.assertAlmostEqual(celsio.nominal_capture_tph, 350_000.0 / 8760.0)
        self.assertAlmostEqual(
            sum(emitter.annual_target_export_tpy or 0.0 for emitter in emitters),
            payload["contracted_annual_target_export_tpy"],
        )
        self.assertEqual(network.downstream_of("aurora_subsea_manifold"), ["aurora_well_a7_ah"])
        self.assertEqual(state.entity_inventory_t["yara_sluiskil"], 0.0)

    def test_phase1_2well_demo_preserves_current_two_well_case(self):
        network, state = build_fixed_scenario_demo("northern_lights_phase1_2well")
        with NORTHERN_LIGHTS_PHASE1_2WELL_DATA_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        wells = [entity for entity in network.entities.values() if isinstance(entity, InjectionWell)]

        self.assertEqual(payload["scenario_id"], "northern_lights_phase1_2well")
        self.assertEqual(len(wells), 2)
        self.assertEqual(network.downstream_of("aurora_subsea_manifold"), ["aurora_well_a7_ah", "aurora_well_c1_h"])
        self.assertEqual(network.downstream_of("aurora_well_a7_ah"), ["aurora_reservoir"])
        self.assertEqual(network.downstream_of("aurora_well_c1_h"), ["aurora_reservoir"])
        self.assertEqual(state.entity_inventory_t["aurora_well_c1_h"], 0.0)

    def test_phase1_demo_uses_hourly_emitter_profile_from_data(self):
        network, state = build_northern_lights_phase1_demo()
        with NORTHERN_LIGHTS_PHASE1_CAPTURE_PROFILE_PATH.open(encoding="utf-8", newline="") as handle:
            first_hour = next(csv.DictReader(handle))

        brevik = network.entities["brevik"]
        self.assertIsNotNone(getattr(brevik, "hourly_capture_profile_tph", None))

        result = network.step(state)

        expected_rate_tph = float(first_hour["brevik_capture_tph"])
        self.assertNotAlmostEqual(expected_rate_tph, brevik.nominal_capture_tph)
        self.assertAlmostEqual(result.state.last_capture_tph["brevik"], expected_rate_tph)

    def test_phase1_pipeline_and_well_use_requested_capacity(self):
        network, _ = build_northern_lights_phase1_demo()
        expected_pipeline_tph = 5_000_000.0 / 8760.0
        expected_well_tph = 2_500_000.0 / 8760.0

        pipeline = network.entities["oygarden_pipeline"]
        wells = [entity for entity in network.entities.values() if isinstance(entity, InjectionWell)]

        self.assertIsInstance(pipeline, Pipeline)
        self.assertAlmostEqual(pipeline.annual_capacity_tpy, 5_000_000.0)
        self.assertAlmostEqual(pipeline.max_flow_tph, expected_pipeline_tph)
        for well in wells:
            self.assertAlmostEqual(well.max_injection_tph, expected_well_tph)

    def test_toy_scenario_is_external_data_with_one_well(self):
        build_toy_demo = getattr(scenarios, "build_toy_demo", None)
        toy_locations = getattr(scenarios, "toy_locations", None)
        self.assertTrue(callable(build_toy_demo))
        self.assertTrue(callable(toy_locations))

        network, state = build_toy_demo()
        locations = toy_locations()
        with scenarios.TOY_DATA_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        wells = [entity for entity in network.entities.values() if isinstance(entity, InjectionWell)]
        pipeline = network.entities["pipeline"]

        self.assertEqual(payload["scenario_id"], "toy")
        self.assertEqual(len(wells), 1)
        self.assertIn("well_1", network.entities)
        self.assertNotIn("well_2", network.entities)
        self.assertEqual(network.downstream_of("manifold"), ["well_1"])
        self.assertEqual(network.downstream_of("well_1"), ["aurora"])
        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(len(pipeline.route_coordinates), len(payload["offshore_pipeline_route"]))
        self.assertEqual(locations["brevik"], (59.05, 9.7))
        self.assertEqual(state.entity_inventory_t["well_1"], 0.0)

    def test_fixed_scenario_selector_loads_scenarios_folder_cases(self):
        self.assertIn("toy", available_fixed_scenario_choices())
        self.assertIn("northern_lights_phase1", available_fixed_scenario_choices())
        self.assertIn("northern_lights_phase1_2well", available_fixed_scenario_choices())
        self.assertIn("northern_lights_phase2", available_fixed_scenario_choices())
        self.assertNotIn("northern_lights_phase2_scenario", available_fixed_scenario_choices())
        self.assertNotIn("northern_lights_phase2_public_2028", available_fixed_scenario_choices())

        network, state = build_fixed_scenario_demo("northern_lights_phase1")
        locations = fixed_scenario_locations("northern_lights_phase1")

        self.assertIn("yara_sluiskil", network.entities)
        self.assertIn("oygarden_terminal", locations)
        self.assertEqual(state.entity_inventory_t["yara_sluiskil"], 0.0)

    def test_scenario_ids_match_json_file_stems(self):
        for path in scenarios.SCENARIO_ROOT.glob("*.json"):
            with self.subTest(path=path.name):
                with path.open(encoding="utf-8") as handle:
                    payload = json.load(handle)

                self.assertEqual(payload["scenario_id"], path.stem)

    def test_fixed_scenarios_use_explicit_pressure_field_names(self):
        old_reservoir_fields = {
            "initial_pressure_bar",
            "pressure_at_capacity_bar",
            "max_pressure_bar",
            "depth_m",
        }
        required_reservoir_fields = {
            "reservoir_initial_pressure_bar",
            "reservoir_average_pressure_at_capacity_bar",
            "reservoir_average_pressure_limit_bar",
            "reservoir_pressure_model",
            "reservoir_depth_below_seabed_m",
            "seawater_depth_m",
            "well_fracture_gradient_psi_per_ft",
            "well_fracture_gradient_reference_depth_m",
            "well_fracture_pressure_bar",
            "well_bottomhole_pressure_safety_factor",
            "well_bottomhole_pressure_limit_bar",
        }

        for path in scenarios.SCENARIO_ROOT.glob("*.json"):
            with self.subTest(path=path.name):
                with path.open(encoding="utf-8") as handle:
                    payload = json.load(handle)

                reservoir_payload = payload["reservoir"]
                self.assertFalse(old_reservoir_fields & reservoir_payload.keys())
                self.assertLessEqual(required_reservoir_fields, reservoir_payload.keys())
                self.assertIn("initial_reservoir_pressure_bar", payload["line_source_parameters"])
                self.assertNotIn("initial_pressure_bar", payload["line_source_parameters"])

    def test_fixed_scenario_selector_accepts_file_stems(self):
        network, _state = build_fixed_scenario_demo("northern_lights_phase2")

        self.assertIn("stockholm_exergi", network.entities)


if __name__ == "__main__":
    unittest.main()
