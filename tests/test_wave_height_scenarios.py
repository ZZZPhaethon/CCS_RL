import csv
import inspect
import tempfile
import unittest
from pathlib import Path

from sim.entities import Emitter, InjectionWell, PhysicalState, Terminal, Vessel
from sim.network import PhysicalNetwork
from sim.scenario_generation import ScenarioConfig
from sim.scenario_generation.wave_height import (
    LegWaveClimatologyScenarioGenerator,
    LSTMWaveHeightScenarioGenerator,
    WaveHeightScenarioGenerator,
    aggregate_wave_heights,
    build_candidate_leg_routes,
    densify_route,
)
from sim.ship_speed import BJERKETVEDT_2020_SHIPS, speed_factor_series
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network
from sim.environment import CCSEnv, CCSEnvConfig


class FakeWaveReader:
    def __init__(self, total_records=20):
        self.total_records = total_records
        self.calls = []

    def route_wave_height_series(self, route_coordinates, *, start_record=0, hours=None):
        self.calls.append((tuple(route_coordinates), start_record, hours))
        base = [0.0, 2.0, 4.0, 6.0, 3.0, 1.0]
        return [base[(start_record + i) % len(base)] for i in range(hours)]


def _network() -> PhysicalNetwork:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("source", nominal_capture_tph=100.0, buffer_capacity_t=1_000.0))
    network.add_entity(Vessel("ship", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
    network.add_entity(Terminal("terminal", storage_capacity_t=2_000.0, berth_count=1))
    network.add_entity(InjectionWell("well", max_injection_tph=200.0))
    network.connect("source", "ship")
    network.connect("ship", "terminal")
    return network


def _quiet_config(**overrides) -> ScenarioConfig:
    base = dict(
        episode_hours=3,
        capture_noise_std=0.0,
        capture_outage_rate_per_week=0.0,
        weather_window_rate_per_week=0.0,
        well_maintenance_rate_per_week=0.0,
        randomize_initial_inventory=False,
    )
    base.update(overrides)
    return ScenarioConfig(**base)


class RouteWaveHelpersTests(unittest.TestCase):
    def test_aggregate_wave_heights_supports_mean_max_and_percentile(self):
        values = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(aggregate_wave_heights(values, "mean"), 2.5)
        self.assertEqual(aggregate_wave_heights(values, "max"), 4.0)
        self.assertAlmostEqual(aggregate_wave_heights(values, "p75"), 3.25)

    def test_densify_route_adds_intermediate_points(self):
        route = densify_route([(0.0, 0.0), (0.0, 1.0)], spacing_km=25.0)

        self.assertEqual(route[0], (0.0, 0.0))
        self.assertEqual(route[-1], (0.0, 1.0))
        self.assertGreater(len(route), 2)

    def test_build_candidate_leg_routes_includes_emitter_to_emitter_legs(self):
        env = CCSEnv(
            make_toy_two_source_network(),
            TOY_TWO_SOURCE_LOCATIONS,
            config=CCSEnvConfig(episode_hours=3),
        )
        legs = build_candidate_leg_routes(env, default_speed_knots=12.0)

        self.assertIn("source_a->source_b", legs)
        self.assertIn("source_b->source_a", legs)
        self.assertIn("source_a->terminal", legs)
        self.assertIn("terminal->source_a", legs)
        self.assertEqual(legs["source_a->source_b"]["origin"], "source_a")
        self.assertEqual(legs["source_a->source_b"]["destination"], "source_b")
        self.assertGreater(legs["source_a->source_b"]["distance_km"], 0.0)


class WaveHeightScenarioGeneratorTests(unittest.TestCase):
    def test_wave_height_generator_replaces_vessel_speed_factor(self):
        reader = FakeWaveReader()
        routes = {
            "ship": {
                "coordinates": [(0.0, 0.0), (0.0, 1.0)],
                "speed_knots": 12.0,
            }
        }
        parameters = BJERKETVEDT_2020_SHIPS[5000]
        generator = WaveHeightScenarioGenerator(
            routes=routes,
            reader=reader,
            default_ship_parameters=parameters,
            config=_quiet_config(),
            seed=1,
        )

        scenario = generator.sample(_network(), seed=2)

        self.assertEqual(set(scenario.vessel_speed_factor), {"ship"})
        self.assertEqual(len(scenario.vessel_speed_factor["ship"]), 3)
        wave_heights = reader.route_wave_height_series(
            routes["ship"]["coordinates"],
            start_record=generator.last_start_record,
            hours=3,
        )
        expected = speed_factor_series(wave_heights, parameters, nominal_speed_knots=12.0)
        self.assertEqual(scenario.vessel_speed_factor["ship"], expected)

    def test_wave_height_scenario_drives_state_at_apply_time(self):
        reader = FakeWaveReader(total_records=3)
        generator = WaveHeightScenarioGenerator(
            routes={"ship": {"coordinates": [(0.0, 0.0), (0.0, 1.0)], "speed_knots": 12.0}},
            reader=reader,
            default_ship_parameters=BJERKETVEDT_2020_SHIPS[5000],
            config=_quiet_config(),
            seed=1,
        )
        scenario = generator.sample(_network(), seed=2)
        state = PhysicalState()

        scenario.apply_to_state(state, time_h=1.0)

        self.assertEqual(state.vessel_speed_factor["ship"], scenario.vessel_speed_factor["ship"][1])

    def test_same_seed_samples_same_weather_window(self):
        routes = {"ship": {"coordinates": [(0.0, 0.0), (0.0, 1.0)], "speed_knots": 12.0}}
        a = WaveHeightScenarioGenerator(routes=routes, reader=FakeWaveReader(), config=_quiet_config())
        b = WaveHeightScenarioGenerator(routes=routes, reader=FakeWaveReader(), config=_quiet_config())

        scenario_a = a.sample(_network(), seed=123)
        scenario_b = b.sample(_network(), seed=123)

        self.assertEqual(a.last_start_record, b.last_start_record)
        self.assertEqual(scenario_a.vessel_speed_factor, scenario_b.vessel_speed_factor)


class LSTMWaveHeightScenarioGeneratorTests(unittest.TestCase):
    def test_lstm_forecast_generator_uses_prediction_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_index",
                        "vessel_id",
                        "horizon_index",
                        "global_record",
                        "actual",
                        "predicted",
                        "error",
                    ],
                )
                writer.writeheader()
                for horizon_index, height in enumerate((0.0, 2.0, 4.0)):
                    writer.writerow(
                        {
                            "sample_index": 0,
                            "vessel_id": "ship",
                            "horizon_index": horizon_index,
                            "global_record": 100 + horizon_index,
                            "actual": 0.0,
                            "predicted": height,
                            "error": 0.0,
                        }
                    )

            routes = {
                "ship": {
                    "coordinates": [(0.0, 0.0), (0.0, 1.0)],
                    "speed_knots": 12.0,
                }
            }
            parameters = BJERKETVEDT_2020_SHIPS[5000]
            generator = LSTMWaveHeightScenarioGenerator(
                path,
                routes=routes,
                default_ship_parameters=parameters,
                fixed_start_global_record=100,
                config=_quiet_config(),
            )

            scenario = generator.sample(_network(), seed=2)

        expected = speed_factor_series((0.0, 2.0, 4.0), parameters, nominal_speed_knots=12.0)
        self.assertEqual(generator.last_start_global_record, 100)
        self.assertEqual(scenario.vessel_speed_factor["ship"], expected)


class LegWaveClimatologyScenarioGeneratorTests(unittest.TestCase):
    def test_leg_climatology_generator_has_no_base_weather_or_injectivity_passthrough_switches(self):
        parameters = inspect.signature(LegWaveClimatologyScenarioGenerator).parameters

        self.assertNotIn("keep_base_vessel_weather", parameters)
        self.assertNotIn("keep_base_injectivity", parameters)

    def test_leg_climatology_generator_uses_mean_by_hour_of_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leg_wave.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "global_record",
                        "source_file",
                        "source_record",
                        "leg_id",
                        "origin",
                        "destination",
                        "speed_factor_p75",
                    ],
                )
                writer.writeheader()
                for year, offset in ((2010, 0.0), (2011, 0.2)):
                    for hour, value in ((5, 0.6), (6, 0.8), (7, 1.0)):
                        writer.writerow(
                            {
                                "global_record": hour,
                                "source_file": f"wam10ei_{year}.nc",
                                "source_record": hour,
                                "leg_id": "source->terminal",
                                "origin": "source",
                                "destination": "terminal",
                                "speed_factor_p75": value + offset,
                            }
                        )

            generator = LegWaveClimatologyScenarioGenerator(
                path,
                config=_quiet_config(),
                fixed_start_hour_of_year=5,
            )
            scenario = generator.sample(_network(), seed=2)

        self.assertEqual(generator.last_start_hour_of_year, 5)
        self.assertEqual(
            scenario.leg_speed_factor["source->terminal"],
            [0.7, 0.9, 1.0],
        )

    def test_leg_climatology_can_amplify_data_driven_slowdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leg_wave.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "global_record",
                        "source_file",
                        "source_record",
                        "leg_id",
                        "origin",
                        "destination",
                        "speed_factor_p75",
                    ],
                )
                writer.writeheader()
                for hour, value in ((0, 1.0), (1, 0.8), (2, 0.4)):
                    writer.writerow(
                        {
                            "global_record": hour,
                            "source_file": "wam10ei_2010.nc",
                            "source_record": hour,
                            "leg_id": "source->terminal",
                            "origin": "source",
                            "destination": "terminal",
                            "speed_factor_p75": value,
                        }
                    )

            generator = LegWaveClimatologyScenarioGenerator(
                path,
                config=_quiet_config(
                    leg_wave_slowdown_multiplier=2.0,
                    leg_wave_speed_factor_floor=0.25,
                ),
                fixed_start_hour_of_year=0,
            )
            scenario = generator.sample(_network(), seed=2)

        self.assertEqual(len(scenario.leg_speed_factor["source->terminal"]), 3)
        for actual, expected in zip(
            scenario.leg_speed_factor["source->terminal"],
            [1.0, 0.6, 0.25],
        ):
            self.assertAlmostEqual(actual, expected)

    def test_leg_climatology_generator_disables_base_window_weather_and_injectivity_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leg_wave.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "global_record",
                        "source_file",
                        "source_record",
                        "leg_id",
                        "origin",
                        "destination",
                        "speed_factor_p75",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "global_record": 0,
                        "source_file": "wam10ei_2010.nc",
                        "source_record": 0,
                        "leg_id": "source->terminal",
                        "origin": "source",
                        "destination": "terminal",
                        "speed_factor_p75": 0.8,
                    }
                )

            generator = LegWaveClimatologyScenarioGenerator(
                path,
                config=ScenarioConfig(
                    episode_hours=3,
                    capture_noise_std=0.0,
                    capture_outage_rate_per_week=0.0,
                    weather_window_rate_per_week=168.0,
                    weather_window_mean_hours=1_000.0,
                    weather_window_speed_factor_range=(0.6, 0.6),
                    well_maintenance_rate_per_week=0.0,
                    randomize_initial_inventory=False,
                ),
                fixed_start_hour_of_year=0,
            )
            scenario = generator.sample(_network(), seed=5)

        self.assertTrue(all(value == 1.0 for value in scenario.vessel_speed_factor["ship"]))
        self.assertTrue(all(value == 1.0 for value in scenario.injectivity_factor["well"]))
        self.assertEqual(scenario.leg_speed_factor["source->terminal"], [0.8, 0.8, 0.8])


if __name__ == "__main__":
    unittest.main()
