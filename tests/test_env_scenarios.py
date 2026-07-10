import unittest
import csv
import tempfile
from pathlib import Path

from sim.control.baselines import greedy_shuttle_policy, idle_policy
from sim.environment import CCSEnvConfig, build_phase1_env
from sim.metrics import run_episode
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from sim.scenario_generation.wave_height import (
    LegWaveClimatologyScenarioGenerator,
    LSTMWaveHeightScenarioGenerator,
    WaveHeightScenarioGenerator,
)


class FakeWaveReader:
    total_records = 12

    def route_wave_height_series(self, route_coordinates, *, start_record=0, hours=None):
        return [0.0] * int(hours)


class Phase1EnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Building the env runs searoute once per vessel; do it a single time.
        cls.env = build_phase1_env(
            scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=72)),
            config=CCSEnvConfig(episode_hours=72, storage_target_rate=0.9),
        )

    def test_real_network_topology(self):
        env = self.env
        self.assertEqual(len(env.vessel_ids), 4)       # four Phase 1 ships
        self.assertEqual(len(env.emitter_ids), 3)      # Brevik, Celsio, Yara
        self.assertEqual(len(env.well_ids), 1)         # A7 AH Phase 1 injection well
        self.assertEqual(env.vessel_action_dims, [5, 5, 5, 5])
        self.assertEqual(len(env.well_rate_bounds()), 1)

    def test_phase1_well_action_bound_matches_requested_capacity(self):
        self.assertEqual(self.env.well_rate_bounds(), [(0.5, 2.5)])

    def test_routes_use_real_distances(self):
        # Yara (NL) -> Oygarden is far longer than Brevik (Norway) -> Oygarden.
        routes = self.env._routes
        yara_vessel = next(v for v, r in routes.items() if r["origin"] == "yara_sluiskil")
        brevik_vessel = next(v for v, r in routes.items() if r["origin"] == "brevik")
        self.assertGreater(routes[yara_vessel]["distance_km"], routes[brevik_vessel]["distance_km"])
        self.assertGreater(routes[brevik_vessel]["distance_km"], 300.0)

    def test_reset_returns_full_observation(self):
        obs = self.env.reset(seed=0)
        self.assertEqual(len(obs), self.env.observation_size)

    def test_idle_episode_only_runs_minimum_injection(self):
        metrics = run_episode(self.env, idle_policy, seed=1)
        self.assertGreater(metrics.stored_t, 0.0)
        self.assertEqual(metrics.berth_wait_vessel_hours, 0)

    def test_shuttle_stores_co2_without_overflow_in_a_week(self):
        metrics = run_episode(self.env, greedy_shuttle_policy, seed=2)
        self.assertGreater(metrics.stored_t, 0.0)
        # Real buffers give ~7 days of autonomy, so a short run should not vent.
        self.assertEqual(metrics.vented_t, 0.0)

    def test_phase1_default_weather_mode_uses_probability_window_generator(self):
        env = build_phase1_env(
            config=CCSEnvConfig(episode_hours=3),
            scenario_config=ScenarioConfig(episode_hours=3),
        )

        self.assertIsInstance(env.scenario_generator, ScenarioGenerator)
        self.assertNotIsInstance(env.scenario_generator, LegWaveClimatologyScenarioGenerator)

    def test_phase1_leg_wave_mode_uses_leg_wave_weather_when_csv_exists(self):
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
                        "leg_id": "brevik->oygarden_terminal",
                        "origin": "brevik",
                        "destination": "oygarden_terminal",
                        "speed_factor_p75": 0.8,
                    }
                )

            env = build_phase1_env(
                config=CCSEnvConfig(episode_hours=3),
                weather_mode="leg_wave_climatology",
                leg_wave_csv=path,
            )

        self.assertIsInstance(env.scenario_generator, LegWaveClimatologyScenarioGenerator)

    def test_phase1_three_vessel_scenario_uses_leg_wave_weather_when_csv_exists(self):
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
                        "leg_id": "brevik->oygarden_terminal",
                        "origin": "brevik",
                        "destination": "oygarden_terminal",
                        "speed_factor_p75": 0.8,
                    }
                )

            env = build_phase1_env(
                scenario="northern_lights_phase1_3vessels",
                config=CCSEnvConfig(episode_hours=3, include_weather_obs=True),
                weather_mode="leg_wave_climatology",
                leg_wave_csv=path,
            )

        self.assertIsInstance(env.scenario_generator, LegWaveClimatologyScenarioGenerator)
        self.assertEqual(env.config.weather_observation_layout, "leg")
        self.assertEqual(env.observation_size, 110)

    def test_phase1_window_weather_mode_uses_probability_window_generator(self):
        env = build_phase1_env(
            config=CCSEnvConfig(episode_hours=3),
            scenario_config=ScenarioConfig(
                episode_hours=3,
                capture_noise_std=0.0,
                capture_outage_rate_per_week=0.0,
                weather_window_rate_per_week=168.0,
                weather_window_mean_hours=1_000.0,
                weather_window_speed_factor_range=(0.6, 0.6),
                well_maintenance_rate_per_week=0.0,
                injectivity_max_decline=0.0,
                injectivity_noise_std=0.0,
                randomize_initial_inventory=False,
            ),
            weather_mode="window",
        )
        scenario = env.scenario_generator.sample(env.network, seed=1)

        self.assertIsInstance(env.scenario_generator, ScenarioGenerator)
        self.assertNotIsInstance(env.scenario_generator, LegWaveClimatologyScenarioGenerator)
        self.assertEqual(set(scenario.vessel_speed_factor["northern_pathfinder"]), {0.6})

    def test_phase1_block_weather_mode_uses_configured_update_interval(self):
        env = build_phase1_env(
            config=CCSEnvConfig(episode_hours=48),
            scenario_config=ScenarioConfig(
                episode_hours=48,
                weather_update_hours=24.0,
                weather_update_speed_factor_range=(0.6, 0.6),
            ),
            weather_mode="block",
        )

        scenario = env.scenario_generator.sample(env.network, seed=1)

        self.assertEqual(env.config.weather_observation_layout, "global")
        self.assertEqual(set(scenario.vessel_speed_factor["northern_pathfinder"]), {0.6})

    def test_weather_observation_uses_window_vessel_speed_when_leg_weather_is_absent(self):
        env = build_phase1_env(
            scenario="northern_lights_phase1_3vessels",
            config=CCSEnvConfig(episode_hours=3, include_weather_obs=True),
            scenario_config=ScenarioConfig(
                episode_hours=3,
                capture_noise_std=0.0,
                capture_outage_rate_per_week=0.0,
                weather_window_rate_per_week=168.0,
                weather_window_mean_hours=1_000.0,
                weather_window_speed_factor_range=(0.6, 0.6),
                well_maintenance_rate_per_week=0.0,
                injectivity_max_decline=0.0,
                injectivity_noise_std=0.0,
                randomize_initial_inventory=False,
            ),
            weather_mode="window",
        )

        obs = env.reset(seed=1)
        names = env.feature_names

        self.assertEqual(len(obs), env.observation_size)
        self.assertEqual(len(obs), len(env.feature_names))
        self.assertEqual(env.config.weather_observation_layout, "global")
        self.assertEqual(env.observation_size, 55)
        self.assertAlmostEqual(obs[names.index("weather.speed_now")], 0.6)

    def test_phase1_wave_height_mode_uses_netcdf_generator(self):
        env = build_phase1_env(
            config=CCSEnvConfig(episode_hours=3),
            scenario_config=ScenarioConfig(episode_hours=3),
            weather_mode="wave_height_netcdf",
            wave_height_reader=FakeWaveReader(),
        )

        self.assertIsInstance(env.scenario_generator, WaveHeightScenarioGenerator)

    def test_phase1_lstm_forecast_mode_uses_prediction_generator(self):
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
                for vessel_id in (
                    "northern_pathfinder",
                    "northern_phoenix",
                    "northern_pioneer",
                    "phase1_vessel_04",
                ):
                    for horizon_index in range(3):
                        writer.writerow(
                            {
                                "sample_index": 0,
                                "vessel_id": vessel_id,
                                "horizon_index": horizon_index,
                                "global_record": 100 + horizon_index,
                                "actual": 0.0,
                                "predicted": 0.0,
                                "error": 0.0,
                            }
                        )

            env = build_phase1_env(
                config=CCSEnvConfig(episode_hours=3),
                scenario_config=ScenarioConfig(episode_hours=3),
                weather_mode="lstm_forecast",
                lstm_prediction_csv=path,
            )

        self.assertIsInstance(env.scenario_generator, LSTMWaveHeightScenarioGenerator)

    def test_warm_start_reservoir_inventory_is_capped_at_half_pressure_capacity(self):
        env = build_phase1_env(
            scenario_generator=ScenarioGenerator(
                config=ScenarioConfig(
                    episode_hours=1,
                    warm_start=True,
                    randomize_initial_inventory=False,
                )
            ),
            config=CCSEnvConfig(episode_hours=1),
        )
        reservoir = env.network.entities["aurora_reservoir"]

        for seed in range(20):
            scenario = env.scenario_generator.sample(env.network, seed=seed)
            inventory_t = scenario.initial_inventory_t["aurora_reservoir"]
            self.assertLessEqual(
                inventory_t,
                0.5 * reservoir.pressure_limited_capacity_t(),
            )


if __name__ == "__main__":
    unittest.main()
