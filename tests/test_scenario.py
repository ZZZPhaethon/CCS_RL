import unittest
import sim.scenario_generation as scenario_generation

from sim.entities import (
    Emitter,
    InjectionWell,
    PhysicalState,
    Reservoir,
    Terminal,
    Vessel,
)
from sim.network import PhysicalNetwork
from sim.scenario_generation import (
    Scenario,
    ScenarioConfig,
    ScenarioGenerator,
)
from sim.scenario_generation.generator import _capture_availability_series


def _network(with_reservoir: bool = False) -> PhysicalNetwork:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("brevik", nominal_capture_tph=100.0, buffer_capacity_t=1_000.0))
    network.add_entity(Emitter("oslo", nominal_capture_tph=80.0, buffer_capacity_t=800.0))
    network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
    network.add_entity(Terminal("oygarden", storage_capacity_t=2_000.0, berth_count=2))
    network.add_entity(InjectionWell("well_1", max_injection_tph=200.0))
    network.add_entity(InjectionWell("well_2", max_injection_tph=200.0))
    if with_reservoir:
        network.add_entity(
            Reservoir("aurora", storage_capacity_t=1_000_000.0, initial_pressure_bar=100.0,
                      pressure_at_capacity_bar=200.0, max_pressure_bar=200.0)
        )
    return network


def _quiet_config(**overrides) -> ScenarioConfig:
    base = dict(
        capture_noise_std=0.0,
        capture_outage_rate_per_week=0.0,
        weather_window_rate_per_week=0.0,
        well_maintenance_rate_per_week=0.0,
        injectivity_max_decline=0.0,
        injectivity_noise_std=0.0,
        randomize_initial_inventory=False,
    )
    base.update(overrides)
    return ScenarioConfig(**base)


class ScenarioGeneratorTests(unittest.TestCase):
    def test_top_level_scenario_generation_exports_only_core_scenario_types(self):
        self.assertEqual(
            set(scenario_generation.__all__),
            {"Scenario", "ScenarioConfig", "ScenarioGenerator"},
        )

    def test_sample_is_reproducible_for_a_seed(self):
        network = _network()
        gen = ScenarioGenerator(seed=7)
        a = gen.sample(network)
        b = gen.sample(network)
        self.assertEqual(a.emitter_availability, b.emitter_availability)
        self.assertEqual(a.vessel_speed_factor, b.vessel_speed_factor)
        self.assertEqual(a.injectivity_factor, b.injectivity_factor)
        self.assertEqual(a.well_available, b.well_available)

    def test_different_seeds_diverge(self):
        network = _network()
        a = ScenarioGenerator(seed=1).sample(network)
        b = ScenarioGenerator(seed=2).sample(network)
        self.assertNotEqual(a.emitter_availability, b.emitter_availability)

    def test_series_cover_every_entity_and_span_the_horizon(self):
        network = _network()
        scenario = ScenarioGenerator(seed=3).sample(network)
        self.assertEqual(scenario.n_steps, 168)
        self.assertEqual(set(scenario.emitter_availability), {"brevik", "oslo"})
        self.assertEqual(set(scenario.well_available), {"well_1", "well_2"})
        self.assertEqual(set(scenario.vessel_speed_factor), {"ship_1"})
        for series in scenario.emitter_availability.values():
            self.assertEqual(len(series), 168)

    def test_quiet_config_produces_nominal_world(self):
        network = _network()
        scenario = ScenarioGenerator(config=_quiet_config(), seed=5).sample(network)
        self.assertTrue(all(v == 1.0 for s in scenario.emitter_availability.values() for v in s))
        self.assertTrue(all(v == 1.0 for s in scenario.vessel_speed_factor.values() for v in s))
        self.assertTrue(all(v == 1.0 for s in scenario.injectivity_factor.values() for v in s))
        self.assertTrue(all(av for s in scenario.well_available.values() for av in s))
        self.assertEqual(scenario.initial_inventory_t, {})

    def test_default_weather_is_nominal_without_probability_windows(self):
        network = _network()
        config = _quiet_config(episode_hours=24)
        scenario = ScenarioGenerator(config=config, seed=0).sample(network)

        speeds = [
            value
            for series in scenario.vessel_speed_factor.values()
            for value in series
        ]
        self.assertEqual(set(speeds), {1.0})

    def test_disturbance_values_stay_in_physical_bounds(self):
        network = _network()
        scenario = ScenarioGenerator(seed=11).sample(network)
        for series in scenario.emitter_availability.values():
            self.assertTrue(all(v >= 0.0 for v in series))
        for series in scenario.injectivity_factor.values():
            self.assertTrue(all(0.3 <= v <= 1.0 for v in series))
        for series in scenario.vessel_speed_factor.values():
            self.assertTrue(all(0.0 < v <= 1.0 for v in series))

    def test_capture_noise_uses_gaussian_multiplier(self):
        class FakeRng:
            def __init__(self):
                self.gauss_args = None

            def gauss(self, mu, sigma):
                self.gauss_args = (mu, sigma)
                return 0.42

        rng = FakeRng()
        config = ScenarioConfig(capture_noise_std=0.30, capture_outage_rate_per_week=0.0)
        series = _capture_availability_series(rng, n_steps=1, dt=1.0, config=config)
        self.assertEqual(series, [0.42])
        self.assertEqual(rng.gauss_args, (1.0, 0.30))

    def test_capture_noise_default_std_is_thirty_percent(self):
        network = _network()
        config = _quiet_config(capture_noise_std=0.30)
        scenario = ScenarioGenerator(config=config, seed=11).sample(network)
        values = [
            value
            for series in scenario.emitter_availability.values()
            for value in series
        ]
        self.assertTrue(any(value > 1.0 for value in values))
        self.assertTrue(any(value < 1.0 for value in values))

    def test_high_output_window_raises_capture_availability(self):
        network = _network()
        config = _quiet_config(
            episode_hours=24,
            capture_high_output_rate_per_week=168.0,
            capture_high_output_mean_hours=1_000.0,
            capture_high_output_multiplier_range=(1.5, 1.5),
        )
        scenario = ScenarioGenerator(config=config, seed=12).sample(network)

        values = [
            value
            for series in scenario.emitter_availability.values()
            for value in series
        ]
        self.assertTrue(any(value > 1.0 for value in values))
        self.assertTrue(all(value <= 1.5 for value in values))

    def test_weather_window_slows_vessels_as_probability_window_weather(self):
        network = _network()
        config = _quiet_config(
            episode_hours=24,
            weather_window_rate_per_week=168.0,
            weather_window_mean_hours=1_000.0,
            weather_window_speed_factor_range=(0.6, 0.6),
        )
        scenario = ScenarioGenerator(config=config, seed=13).sample(network)

        speeds = [
            value
            for series in scenario.vessel_speed_factor.values()
            for value in series
        ]
        self.assertEqual(min(speeds), 0.6)
        self.assertTrue(all(0.6 <= value <= 1.0 for value in speeds))

    def test_randomized_initial_inventory_respects_capacity_fractions(self):
        network = _network()
        config = ScenarioConfig(
            randomize_initial_inventory=True,
            emitter_initial_fill_range=(0.1, 0.2),
            terminal_initial_fill_range=(0.05, 0.1),
        )
        scenario = ScenarioGenerator(config=config, seed=9).sample(network)
        self.assertTrue(100.0 <= scenario.initial_inventory_t["brevik"] <= 200.0)
        self.assertTrue(100.0 <= scenario.initial_inventory_t["oygarden"] <= 200.0)

    def test_default_initial_inventory_ranges_allow_half_full_starts(self):
        config = ScenarioConfig()
        self.assertEqual(config.emitter_initial_fill_range, (0.0, 0.5))
        self.assertEqual(config.terminal_initial_fill_range, (0.0, 0.5))

    def test_default_outage_and_maintenance_durations(self):
        config = ScenarioConfig()
        self.assertEqual(config.capture_noise_std, 0.30)
        self.assertEqual(config.capture_outage_mean_hours, 12.0)
        self.assertEqual(config.weather_window_rate_per_week, 0.3)
        self.assertEqual(config.well_maintenance_mean_hours, 24.0)

    def test_injectivity_disturbance_defaults_to_nominal(self):
        config = ScenarioConfig()
        self.assertEqual(config.injectivity_max_decline, 0.0)
        self.assertEqual(config.injectivity_noise_std, 0.0)
        self.assertEqual(config.injectivity_warmstart_min, 1.0)
        scenario = ScenarioGenerator(config=ScenarioConfig(warm_start=True), seed=1).sample(
            _network(with_reservoir=True)
        )
        self.assertTrue(all(value == 1.0 for value in scenario.injectivity_factor["well_1"]))


class WarmStartTests(unittest.TestCase):
    def test_warm_start_off_keeps_slow_vars_nominal(self):
        config = ScenarioConfig(warm_start=False, injectivity_noise_std=0.0, injectivity_max_decline=0.0)
        scenario = ScenarioGenerator(config=config, seed=1).sample(_network(with_reservoir=True))
        self.assertNotIn("aurora", scenario.initial_inventory_t)  # cold reservoir
        self.assertEqual(scenario.injectivity_factor["well_1"][0], 1.0)  # full injectivity

    def test_warm_start_prefills_reservoir_pressure(self):
        network = _network(with_reservoir=True)
        config = ScenarioConfig(warm_start=True, reservoir_initial_pressure_fill_range=(0.2, 0.6))
        scenario = ScenarioGenerator(config=config, seed=1).sample(network)
        reservoir = network.entities["aurora"]
        fill = scenario.initial_inventory_t["aurora"]
        self.assertGreater(fill, 0.0)
        self.assertLessEqual(fill, 0.6 * reservoir.pressure_limited_capacity_t())
        # A pre-filled reservoir means pressure no longer starts at full margin.
        self.assertLess(reservoir.pressure_margin_bar(fill), reservoir.pressure_margin_bar(0.0))

    def test_explicit_warm_start_can_vary_initial_injectivity_below_one(self):
        config = ScenarioConfig(warm_start=True, injectivity_warmstart_min=0.5)
        starts = {
            ScenarioGenerator(config=config, seed=s).sample(_network(with_reservoir=True))
            .injectivity_factor["well_1"][0]
            for s in range(8)
        }
        self.assertTrue(any(start < 1.0 for start in starts))
        self.assertTrue(all(0.5 <= start <= 1.0 for start in starts))


class ScenarioApplyTests(unittest.TestCase):
    def _scenario(self) -> Scenario:
        return Scenario(
            time_step_hours=1.0,
            n_steps=3,
            initial_inventory_t={"brevik": 250.0},
            emitter_availability={"brevik": [1.0, 0.0, 0.5]},
            vessel_speed_factor={"ship_1": [1.0, 0.6, 0.6]},
            leg_speed_factor={"brevik->oygarden": [1.0, 0.5, 0.4]},
            well_available={"well_1": [True, True, False]},
            injectivity_factor={"well_1": [1.0, 0.9, 0.8]},
        )

    def test_apply_initial_sets_starting_inventory(self):
        state = PhysicalState()
        self._scenario().apply_initial(state)
        self.assertEqual(state.entity_inventory_t["brevik"], 250.0)

    def test_apply_to_state_writes_the_right_step(self):
        scenario = self._scenario()
        state = PhysicalState()
        scenario.apply_to_state(state, time_h=1.0)
        self.assertEqual(state.emitter_availability["brevik"], 0.0)
        self.assertEqual(state.vessel_speed_factor["ship_1"], 0.6)
        self.assertEqual(state.leg_speed_factor["brevik->oygarden"], 0.5)
        self.assertTrue(state.well_available["well_1"])

    def test_step_index_clamps_past_the_horizon(self):
        scenario = self._scenario()
        self.assertEqual(scenario.step_index(-5.0), 0)
        self.assertEqual(scenario.step_index(100.0), 2)

    def test_scenario_drives_real_physics(self):
        # A forced well outage at step 2 must block injection in network.step.
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
        network.add_entity(Terminal("oygarden", storage_capacity_t=2_000.0, berth_count=1))
        network.add_entity(InjectionWell("well_1", max_injection_tph=200.0))
        from sim.entities import Pipeline

        network.add_entity(Pipeline("pipeline", max_flow_tph=200.0))
        network.connect("ship_1", "oygarden")
        network.connect("oygarden", "pipeline")
        network.connect("pipeline", "well_1")

        scenario = self._scenario()
        state = PhysicalState(
            vessel_berths={"ship_1": "oygarden"},
            entity_inventory_t={"ship_1": 800.0, "oygarden": 500.0},
        )
        scenario.apply_to_state(state, time_h=2.0)  # well_1 unavailable at index 2
        result = network.step(
            state, actions={"oygarden": {"unload_tph": 200.0}, "pipeline": {"flow_tph": 200.0}}
        )
        self.assertEqual(result.state.entity_inventory_t.get("well_1", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
