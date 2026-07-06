import unittest

from sim.entities import (
    Emitter,
    InjectionWell,
    PhysicalState,
    Pipeline,
    Reservoir,
    SubseaManifold,
    Terminal,
    Vessel,
)
from sim.line_source import (
    LineSourceParameters,
    bottomhole_pressure_bar,
    pressure_at_radius_bar,
    variable_rate_pressure_at_radius_bar,
)
from sim.network import PhysicalNetwork


class PhysicalLayerTests(unittest.TestCase):
    def test_one_hour_step_preserves_mass_with_canonical_chain(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Emitter("brevik", nominal_capture_tph=100.0, buffer_capacity_t=1_000.0))
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.add_entity(Terminal("oygarden", storage_capacity_t=2_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=200.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=200.0))
        network.connect("brevik", "ship_1")
        network.connect("ship_1", "oygarden")
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")

        state = PhysicalState(vessel_berths={"ship_1": "oygarden"})
        state.entity_inventory_t["ship_1"] = 800.0

        result = network.step(state, actions={"oygarden": {"unload_tph": 200.0}, "pipeline": {"flow_tph": 200.0}})

        self.assertAlmostEqual(result.state.entity_inventory_t["brevik"], 100.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["ship_1"], 600.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["well_1"], 200.0)
        self.assertAlmostEqual(result.mass_balance_error_t, 0.0, places=9)
        self.assertEqual(result.violations, [])

    def test_injection_limit_backpressures_terminal_and_reports_clip(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=500.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=100.0))
        network.connect("ship_1", "oygarden")
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")

        state = PhysicalState(vessel_berths={"ship_1": "oygarden"})
        state.entity_inventory_t["ship_1"] = 800.0
        state.entity_inventory_t["oygarden"] = 950.0

        result = network.step(state, actions={"oygarden": {"unload_tph": 500.0}, "pipeline": {"flow_tph": 500.0}})

        self.assertAlmostEqual(result.state.entity_inventory_t["oygarden"], 1_000.0)
        self.assertAlmostEqual(result.flows_t[("ship_1", "oygarden")], 150.0)
        self.assertAlmostEqual(result.flows_t[("oygarden", "pipeline")], 100.0)
        self.assertTrue(any(v.violation_type == "flow_clipped" for v in result.violations))
        self.assertAlmostEqual(result.mass_balance_error_t, 0.0, places=9)

    def test_pipeline_cannot_pull_more_than_terminal_inventory_plus_unload(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=500.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=500.0))
        network.connect("ship_1", "oygarden")
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")

        state = PhysicalState()

        result = network.step(state, actions={"oygarden": {"unload_tph": 0.0}, "pipeline": {"flow_tph": 500.0}})

        self.assertAlmostEqual(result.state.entity_inventory_t.get("oygarden", 0.0), 0.0)
        self.assertAlmostEqual(result.flows_t.get(("oygarden", "pipeline"), 0.0), 0.0)
        self.assertTrue(any(v.entity_id == "pipeline" for v in result.violations))

    def test_pipeline_flow_is_not_ramp_limited(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=500.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=500.0))
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")
        state = PhysicalState(entity_inventory_t={"oygarden": 500.0})

        result = network.step(state, actions={"pipeline": {"flow_tph": 500.0}})

        self.assertAlmostEqual(result.flows_t[("oygarden", "pipeline")], 500.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["well_1"], 500.0)
        self.assertEqual(result.violations, [])

    def test_pipeline_does_not_flow_without_action(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=500.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=500.0))
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")
        state = PhysicalState(entity_inventory_t={"oygarden": 300.0})

        result = network.step(state, actions={})

        self.assertAlmostEqual(result.state.entity_inventory_t["oygarden"], 300.0)
        self.assertAlmostEqual(result.state.entity_inventory_t.get("well_1", 0.0), 0.0)
        self.assertEqual(result.flows_t, {})
        self.assertEqual(result.violations, [])

    def test_injection_well_updates_downstream_reservoir_pressure(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=200.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=200.0))
        network.add_entity(
            Reservoir(
                "reservoir_1",
                storage_capacity_t=1_000.0,
                initial_pressure_bar=250.0,
                pressure_at_capacity_bar=300.0,
                max_pressure_bar=310.0,
            )
        )
        network.connect("ship_1", "oygarden")
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")
        network.connect("well_1", "reservoir_1")
        state = PhysicalState(entity_inventory_t={"ship_1": 800.0}, vessel_berths={"ship_1": "oygarden"})

        result = network.step(state, actions={"oygarden": {"unload_tph": 200.0}, "pipeline": {"flow_tph": 200.0}})
        snapshot = network.snapshot(result.state)

        self.assertAlmostEqual(result.state.entity_inventory_t["reservoir_1"], 200.0)
        self.assertAlmostEqual(result.state.entity_inventory_t.get("well_1", 0.0), 0.0)
        self.assertAlmostEqual(result.flows_t[("pipeline", "well_1")], 200.0)
        self.assertAlmostEqual(result.flows_t[("well_1", "reservoir_1")], 200.0)
        self.assertGreater(snapshot["entities"]["reservoir_1"]["pressure_bar"], 250.0)
        self.assertAlmostEqual(result.mass_balance_error_t, 0.0, places=9)

    def test_line_source_pressures_are_reported_after_injection(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=100.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=100.0))
        network.add_entity(
            Reservoir(
                "reservoir_1",
                storage_capacity_t=1_000_000.0,
                initial_pressure_bar=260.0,
                pressure_at_capacity_bar=300.0,
                max_pressure_bar=315.0,
                reservoir_pressure_model="linear_average_reservoir_pressure_proxy",
                seawater_depth_m=300.0,
                well_fracture_gradient_psi_per_ft=0.65,
                well_fracture_gradient_reference_depth_m=2900.0,
                well_fracture_pressure_bar=426.4,
                well_bottomhole_pressure_safety_factor=0.9,
                well_bottomhole_pressure_limit_bar=383.8,
                line_source_parameters=LineSourceParameters(
                    initial_pressure_bar=260.0,
                    permeability_md=500.0,
                    thickness_m=173.0,
                    porosity_fraction=0.22,
                    total_compressibility_1_pa=7e-10,
                    viscosity_pa_s=6e-5,
                    co2_density_kg_m3=630.0,
                    well_radius_m=0.10795,
                    skin=0.0,
                ),
                line_source_observation_radii_m=(100.0, 1000.0),
            )
        )
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")
        network.connect("well_1", "reservoir_1")
        state = PhysicalState(entity_inventory_t={"oygarden": 100.0})

        result = network.step(state, actions={"pipeline": {"flow_tph": 100.0}})
        snapshot = network.snapshot(result.state)

        well_snapshot = snapshot["entities"]["well_1"]
        reservoir_snapshot = snapshot["entities"]["reservoir_1"]
        self.assertAlmostEqual(well_snapshot["line_source_rate_tph"], 100.0)
        self.assertGreater(well_snapshot["bottomhole_pressure_bar"], 260.0)
        self.assertAlmostEqual(well_snapshot["well_bottomhole_pressure_limit_bar"], 383.8)
        self.assertAlmostEqual(
            well_snapshot["well_bottomhole_pressure_margin_bar"],
            383.8 - well_snapshot["bottomhole_pressure_bar"],
        )
        self.assertAlmostEqual(reservoir_snapshot["reservoir_average_pressure_limit_bar"], 315.0)
        self.assertAlmostEqual(reservoir_snapshot["well_bottomhole_pressure_limit_bar"], 383.8)
        self.assertIn("100.0", reservoir_snapshot["line_source_pressure_bar_by_radius_m"])
        self.assertIn("1000.0", reservoir_snapshot["line_source_pressure_bar_by_radius_m"])
        self.assertGreater(reservoir_snapshot["line_source_pressure_bar_by_radius_m"]["100.0"], 260.0)
        self.assertGreater(
            well_snapshot["bottomhole_pressure_bar"],
            reservoir_snapshot["line_source_pressure_bar_by_radius_m"]["100.0"],
        )

    def test_bottomhole_pressure_limit_clips_injection_to_safe_rate(self):
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
        expected_rate_mtpa = 1.0
        requested_rate_mtpa = 2.5
        hours_per_year = 365.25 * 24.0
        expected_rate_tph = expected_rate_mtpa * 1_000_000.0 / hours_per_year
        requested_rate_tph = requested_rate_mtpa * 1_000_000.0 / hours_per_year
        pressure_limit_bar = bottomhole_pressure_bar(
            parameters,
            expected_rate_mtpa,
            elapsed_days=1.0 / 24.0,
        )

        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=500.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=500.0))
        network.add_entity(
            Reservoir(
                "reservoir_1",
                storage_capacity_t=1_000_000.0,
                initial_pressure_bar=300.0,
                pressure_at_capacity_bar=350.0,
                max_pressure_bar=350.0,
                well_bottomhole_pressure_limit_bar=pressure_limit_bar,
                line_source_parameters=parameters,
            )
        )
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")
        network.connect("well_1", "reservoir_1")
        state = PhysicalState(entity_inventory_t={"oygarden": 1_000.0})

        result = network.step(state, actions={"pipeline": {"flow_tph": requested_rate_tph}})
        snapshot = network.snapshot(result.state)

        actual_rate_tph = result.flows_t[("pipeline", "well_1")]
        self.assertLess(actual_rate_tph, requested_rate_tph)
        self.assertAlmostEqual(actual_rate_tph, expected_rate_tph, places=6)
        self.assertTrue(
            any(v.violation_type == "bottomhole_pressure_clipped" for v in result.violations)
        )
        self.assertLessEqual(
            snapshot["entities"]["well_1"]["bottomhole_pressure_bar"],
            pressure_limit_bar + 1e-9,
        )

    def test_snapshot_reports_next_step_pressure_feasible_rate_levels(self):
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
        network.add_entity(InjectionWell("well_1", max_injection_tph=500.0))
        network.add_entity(
            Reservoir(
                "reservoir_1",
                storage_capacity_t=1_000_000.0,
                initial_pressure_bar=300.0,
                pressure_at_capacity_bar=350.0,
                max_pressure_bar=350.0,
                well_bottomhole_pressure_limit_bar=pressure_limit_bar,
                line_source_parameters=parameters,
            )
        )
        network.connect("well_1", "reservoir_1")

        snapshot = network.snapshot(PhysicalState())
        well_snapshot = snapshot["entities"]["well_1"]

        self.assertEqual(well_snapshot["well_rate_levels_mtpa"], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        self.assertEqual(
            well_snapshot["next_step_well_rate_action_mask"],
            [True, True, True, False, False, False],
        )
        self.assertAlmostEqual(well_snapshot["next_step_pressure_limited_max_rate_mtpa"], 1.0)

    def test_line_source_pressure_uses_injection_history_after_rate_changes(self):
        parameters = LineSourceParameters(
            initial_pressure_bar=260.0,
            permeability_md=500.0,
            thickness_m=173.0,
            porosity_fraction=0.22,
            total_compressibility_1_pa=7e-10,
            viscosity_pa_s=6e-5,
            co2_density_kg_m3=630.0,
            well_radius_m=0.10795,
            skin=0.0,
        )
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=100.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=100.0))
        network.add_entity(
            Reservoir(
                "reservoir_1",
                storage_capacity_t=1_000_000.0,
                initial_pressure_bar=260.0,
                pressure_at_capacity_bar=300.0,
                max_pressure_bar=315.0,
                line_source_parameters=parameters,
                line_source_observation_radii_m=(100.0,),
            )
        )
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")
        network.connect("well_1", "reservoir_1")

        first = network.step(
            PhysicalState(entity_inventory_t={"oygarden": 100.0}),
            actions={"pipeline": {"flow_tph": 100.0}},
        )
        second = network.step(first.state, actions={})
        snapshot = network.snapshot(second.state)

        history = second.state.injection_rate_history_tph["well_1"]
        expected_pressure = variable_rate_pressure_at_radius_bar(
            parameters,
            [(start_h / 24.0, rate_tph * 365.25 * 24.0 / 1_000_000.0) for start_h, rate_tph in history],
            elapsed_days=second.state.time_h / 24.0,
            radius_m=100.0,
        )

        self.assertEqual(history, [(0.0, 100.0), (1.0, 0.0)])
        self.assertAlmostEqual(second.state.last_injection_flow_tph["well_1"], 0.0)
        self.assertGreater(snapshot["entities"]["well_1"]["bottomhole_pressure_bar"], parameters.initial_pressure_bar)
        self.assertAlmostEqual(
            snapshot["entities"]["reservoir_1"]["line_source_pressure_bar_by_radius_m"]["100.0"],
            expected_pressure,
        )

    def test_line_source_bottomhole_pressure_includes_multiwell_interference(self):
        parameters = LineSourceParameters(
            initial_pressure_bar=260.0,
            permeability_md=100.0,
            thickness_m=173.0,
            porosity_fraction=0.22,
            total_compressibility_1_pa=7e-10,
            viscosity_pa_s=6e-5,
            co2_density_kg_m3=630.0,
            well_radius_m=0.10795,
            skin=0.0,
        )
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=300.0))
        network.add_entity(SubseaManifold("manifold", max_flow_tph=300.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=300.0))
        network.add_entity(InjectionWell("well_2", max_injection_tph=300.0))
        network.add_entity(
            Reservoir(
                "reservoir_1",
                storage_capacity_t=1_000_000.0,
                initial_pressure_bar=260.0,
                pressure_at_capacity_bar=300.0,
                max_pressure_bar=315.0,
                line_source_parameters=parameters,
                line_source_observation_radii_m=(100.0,),
                line_source_well_distances_m={
                    "well_1": {"well_2": 500.0},
                    "well_2": {"well_1": 500.0},
                },
            )
        )
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "manifold")
        network.connect("manifold", "well_1")
        network.connect("manifold", "well_2")
        network.connect("well_1", "reservoir_1")
        network.connect("well_2", "reservoir_1")
        state = PhysicalState(entity_inventory_t={"oygarden": 300.0})

        result = network.step(
            state,
            actions={
                "pipeline": {"flow_tph": 300.0},
                "manifold": {"well_splits": {"well_1": 0.5, "well_2": 0.5}},
            },
        )
        snapshot = network.snapshot(result.state)

        single_well_bhp = snapshot["entities"]["well_1"]["bottomhole_pressure_bar"]
        interference_delta = snapshot["entities"]["well_1"]["line_source_interference_delta_bar"]
        elapsed_days = result.state.time_h / 24.0
        expected_bhp = bottomhole_pressure_bar(
            parameters,
            result.state.last_injection_flow_tph["well_1"] * 365.25 * 24.0 / 1_000_000.0,
            elapsed_days=elapsed_days,
        ) + (
            pressure_at_radius_bar(
                parameters,
                result.state.last_injection_flow_tph["well_2"] * 365.25 * 24.0 / 1_000_000.0,
                elapsed_days=elapsed_days,
                radius_m=500.0,
            )
            - parameters.initial_pressure_bar
        )
        self.assertGreater(interference_delta, 0.0)
        self.assertAlmostEqual(single_well_bhp, expected_bhp)
        self.assertAlmostEqual(
            single_well_bhp,
            snapshot["entities"]["well_2"]["bottomhole_pressure_bar"],
        )

    def test_pipeline_can_distribute_directly_to_multiple_wells(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=300.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=200.0))
        network.add_entity(InjectionWell("well_2", max_injection_tph=100.0))
        network.add_entity(
            Reservoir(
                "reservoir_1",
                storage_capacity_t=1_000.0,
                initial_pressure_bar=250.0,
                pressure_at_capacity_bar=300.0,
                max_pressure_bar=310.0,
            )
        )
        network.connect("ship_1", "oygarden")
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")
        network.connect("pipeline", "well_2")
        network.connect("well_1", "reservoir_1")
        network.connect("well_2", "reservoir_1")
        state = PhysicalState(entity_inventory_t={"ship_1": 800.0}, vessel_berths={"ship_1": "oygarden"})

        result = network.step(state, actions={"oygarden": {"unload_tph": 300.0}, "pipeline": {"flow_tph": 300.0}})

        self.assertAlmostEqual(result.flows_t[("oygarden", "pipeline")], 300.0)
        self.assertAlmostEqual(result.flows_t[("pipeline", "well_1")], 200.0)
        self.assertAlmostEqual(result.flows_t[("pipeline", "well_2")], 100.0)
        self.assertAlmostEqual(result.flows_t[("well_1", "reservoir_1")], 200.0)
        self.assertAlmostEqual(result.flows_t[("well_2", "reservoir_1")], 100.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["reservoir_1"], 300.0)
        self.assertAlmostEqual(result.mass_balance_error_t, 0.0, places=9)

    def test_manifold_split_action_controls_flow_to_multiple_wells(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=300.0))
        network.add_entity(SubseaManifold("manifold", max_flow_tph=300.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=300.0))
        network.add_entity(InjectionWell("well_2", max_injection_tph=300.0))
        network.add_entity(
            Reservoir(
                "reservoir_1",
                storage_capacity_t=1_000.0,
                initial_pressure_bar=250.0,
                pressure_at_capacity_bar=300.0,
                max_pressure_bar=310.0,
            )
        )
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "manifold")
        network.connect("manifold", "well_1")
        network.connect("manifold", "well_2")
        network.connect("well_1", "reservoir_1")
        network.connect("well_2", "reservoir_1")
        state = PhysicalState(entity_inventory_t={"oygarden": 300.0})

        result = network.step(
            state,
            actions={
                "pipeline": {"flow_tph": 300.0},
                "manifold": {"well_splits": {"well_1": 0.25, "well_2": 0.75}},
            },
        )

        self.assertAlmostEqual(result.flows_t[("pipeline", "manifold")], 300.0)
        self.assertAlmostEqual(result.flows_t[("manifold", "well_1")], 75.0)
        self.assertAlmostEqual(result.flows_t[("manifold", "well_2")], 225.0)
        self.assertAlmostEqual(result.flows_t[("well_1", "reservoir_1")], 75.0)
        self.assertAlmostEqual(result.flows_t[("well_2", "reservoir_1")], 225.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["reservoir_1"], 300.0)
        self.assertAlmostEqual(result.mass_balance_error_t, 0.0, places=9)

    def test_topology_can_be_rewired_without_changing_entities(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Emitter("emitter_a", nominal_capture_tph=50.0, buffer_capacity_t=500.0))
        network.add_entity(Emitter("emitter_b", nominal_capture_tph=70.0, buffer_capacity_t=500.0))
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))

        network.connect("emitter_a", "ship_1")
        self.assertEqual(network.downstream_of("emitter_a"), ["ship_1"])

        network.disconnect("emitter_a", "ship_1")
        network.connect("emitter_b", "ship_1")

        self.assertEqual(network.downstream_of("emitter_a"), [])
        self.assertEqual(network.downstream_of("emitter_b"), ["ship_1"])

    def test_loading_action_can_target_a_specific_vessel(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Emitter("brevik", nominal_capture_tph=0.0, buffer_capacity_t=1_000.0))
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.add_entity(Vessel("ship_2", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.connect("brevik", "ship_1")
        network.connect("brevik", "ship_2")
        state = PhysicalState(entity_inventory_t={"brevik": 300.0}, vessel_berths={"ship_2": "brevik"})

        result = network.step(state, actions={"brevik": {"load_tph": 200.0, "vessel_id": "ship_2"}})

        self.assertAlmostEqual(result.state.entity_inventory_t.get("ship_1", 0.0), 0.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["ship_2"], 200.0)

    def test_load_vessel_action_uses_max_feasible_loading_rate(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Emitter("brevik", nominal_capture_tph=0.0, buffer_capacity_t=1_000.0, loading_rate_tph=500.0))
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=300.0, unloading_rate_tph=800.0))
        network.connect("brevik", "ship_1")
        state = PhysicalState(entity_inventory_t={"brevik": 700.0}, vessel_berths={"ship_1": "brevik"})

        result = network.step(state, actions={"brevik": {"load_vessel": "ship_1"}})

        self.assertAlmostEqual(result.flows_t[("brevik", "ship_1")], 300.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["brevik"], 400.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["ship_1"], 300.0)

    def test_any_emitter_can_load_any_berthed_vessel_for_milk_runs(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Emitter("emitter_a", nominal_capture_tph=0.0, buffer_capacity_t=1_000.0, loading_rate_tph=500.0))
        network.add_entity(Emitter("emitter_b", nominal_capture_tph=0.0, buffer_capacity_t=1_000.0, loading_rate_tph=500.0))
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=300.0, unloading_rate_tph=800.0))
        network.connect("emitter_a", "ship_1")
        state = PhysicalState(entity_inventory_t={"emitter_b": 700.0}, vessel_berths={"ship_1": "emitter_b"})

        result = network.step(state, actions={"emitter_b": {"load_vessel": "ship_1"}})

        self.assertAlmostEqual(result.flows_t[("emitter_b", "ship_1")], 300.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["emitter_b"], 400.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["ship_1"], 300.0)

    def test_capture_and_loading_update_emitter_buffer_in_same_step(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(
            Emitter(
                "brevik",
                nominal_capture_tph=100.0,
                buffer_capacity_t=1_000.0,
                loading_rate_tph=500.0,
            )
        )
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=300.0, unloading_rate_tph=800.0))
        network.connect("brevik", "ship_1")
        state = PhysicalState(entity_inventory_t={"brevik": 200.0, "ship_1": 0.0}, vessel_berths={"ship_1": "brevik"})

        result = network.step(state, actions={"brevik": {"utilization": 1.0, "load_vessel": "ship_1"}})

        self.assertAlmostEqual(result.flows_t[("brevik", "ship_1")], 300.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["brevik"], 0.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["ship_1"], 300.0)
        self.assertAlmostEqual(result.mass_balance_error_t, 0.0, places=9)

    def test_emitter_vents_when_capture_exceeds_full_buffer_without_vessel(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Emitter("brevik", nominal_capture_tph=100.0, buffer_capacity_t=150.0))
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.connect("brevik", "ship_1")
        state = PhysicalState(entity_inventory_t={"brevik": 150.0}, vessel_berths={})

        result = network.step(state, actions={"brevik": {"utilization": 1.0}})

        self.assertAlmostEqual(result.state.entity_inventory_t["brevik"], 150.0)
        self.assertAlmostEqual(result.state.last_capture_tph["brevik"], 0.0)
        self.assertAlmostEqual(result.state.last_vent_tph["brevik"], 100.0)
        self.assertAlmostEqual(result.state.cumulative_vent_t["brevik"], 100.0)
        self.assertAlmostEqual(result.mass_balance_error_t, 0.0, places=9)
        self.assertTrue(any(v.violation_type == "vented_capture" and v.entity_id == "brevik" for v in result.violations))

    def test_loading_requires_vessel_at_emitter_berth(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Emitter("brevik", nominal_capture_tph=0.0, buffer_capacity_t=1_000.0))
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.connect("brevik", "ship_1")
        state = PhysicalState(entity_inventory_t={"brevik": 300.0}, vessel_berths={})

        result = network.step(state, actions={"brevik": {"load_tph": 200.0, "vessel_id": "ship_1"}})

        self.assertAlmostEqual(result.state.entity_inventory_t["brevik"], 300.0)
        self.assertAlmostEqual(result.state.entity_inventory_t.get("ship_1", 0.0), 0.0)
        self.assertTrue(any(v.violation_type == "berth_required" and v.entity_id == "ship_1" for v in result.violations))

    def test_unloading_action_can_target_a_specific_vessel_at_terminal(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.add_entity(Vessel("ship_2", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=500.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=500.0))
        network.connect("ship_1", "oygarden")
        network.connect("ship_2", "oygarden")
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")
        state = PhysicalState(
            entity_inventory_t={"ship_1": 300.0, "ship_2": 300.0},
            vessel_berths={"ship_2": "oygarden"},
        )

        result = network.step(
            state,
            actions={"oygarden": {"unload_tph": 200.0, "vessel_id": "ship_2"}, "pipeline": {"flow_tph": 200.0}},
        )

        self.assertAlmostEqual(result.state.entity_inventory_t["ship_1"], 300.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["ship_2"], 100.0)
        self.assertAlmostEqual(result.flows_t[("ship_2", "oygarden")], 200.0)

    def test_unload_vessel_action_uses_max_feasible_unloading_rate(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=300.0))
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=500.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=500.0))
        network.connect("ship_1", "oygarden")
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")
        state = PhysicalState(entity_inventory_t={"ship_1": 700.0}, vessel_berths={"ship_1": "oygarden"})

        result = network.step(state, actions={"oygarden": {"unload_vessel": "ship_1"}, "pipeline": {"flow_tph": 500.0}})

        self.assertAlmostEqual(result.flows_t[("ship_1", "oygarden")], 300.0)
        self.assertAlmostEqual(result.flows_t[("oygarden", "pipeline")], 300.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["ship_1"], 400.0)
        self.assertAlmostEqual(result.state.entity_inventory_t["well_1"], 300.0)

    def test_unloading_requires_vessel_at_terminal_berth(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.add_entity(Terminal("oygarden", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=500.0))
        network.add_entity(InjectionWell("well_1", max_injection_tph=500.0))
        network.connect("ship_1", "oygarden")
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")
        state = PhysicalState(entity_inventory_t={"ship_1": 300.0}, vessel_berths={})

        result = network.step(state, actions={"oygarden": {"unload_tph": 200.0}, "pipeline": {"flow_tph": 200.0}})

        self.assertAlmostEqual(result.state.entity_inventory_t["ship_1"], 300.0)
        self.assertAlmostEqual(result.state.entity_inventory_t.get("oygarden", 0.0), 0.0)
        self.assertAlmostEqual(result.flows_t.get(("ship_1", "oygarden"), 0.0), 0.0)
        self.assertTrue(any(v.violation_type == "berth_required" and v.entity_id == "ship_1" for v in result.violations))

    def test_step_result_and_snapshot_are_json_ready_for_rl_or_llm_clients(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Emitter("brevik", nominal_capture_tph=100.0, buffer_capacity_t=1_000.0))
        state = PhysicalState()

        result = network.step(state)
        snapshot = network.snapshot(result.state)
        payload = result.as_dict()

        self.assertEqual(snapshot["time_h"], 1.0)
        self.assertEqual(snapshot["entities"]["brevik"]["type"], "Emitter")
        self.assertEqual(payload["state"]["time_h"], 1.0)
        self.assertEqual(payload["violations"], [])


if __name__ == "__main__":
    unittest.main()
