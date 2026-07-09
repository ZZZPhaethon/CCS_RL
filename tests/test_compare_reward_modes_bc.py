from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import compare_reward_modes_bc as compare
from sim.entities import Emitter


class _Metric:
    storage_rate = 0.5
    loss_rate = 0.1
    stored_t = 100.0
    vented_t = 10.0
    operating_cost = 20.0
    vent_penalty = 30.0


class CompareRewardModesTests(unittest.TestCase):
    def test_eval_baselines_includes_idle_and_greedy_teacher_rows(self):
        args = SimpleNamespace(eval_seeds=[1, 2])

        with (
            patch.object(compare, "make_env", return_value=object()),
            patch.object(compare, "run_episode", return_value=_Metric()),
        ):
            rows = compare.eval_baselines(args)

        self.assertEqual([row["policy"] for row in rows], ["idle", "greedy_teacher"])
        self.assertEqual(rows[0]["actual_total_cost"], 50.0)
        self.assertEqual(rows[0]["actual_cost_per_stored_t"], 0.5)

    def test_yara_buffer_override_replaces_frozen_emitter(self):
        emitter = Emitter("yara_sluiskil", nominal_capture_tph=100.0, buffer_capacity_t=15_000.0)
        env = SimpleNamespace(network=SimpleNamespace(entities={"yara_sluiskil": emitter}))
        args = SimpleNamespace(yara_buffer_capacity=7_500.0)

        compare.apply_yara_buffer_capacity_override(env, args)

        self.assertEqual(env.network.entities["yara_sluiskil"].buffer_capacity_t, 7_500.0)
        self.assertEqual(emitter.buffer_capacity_t, 15_000.0)

    def test_yara_buffer_tag_marks_output_artifacts(self):
        self.assertEqual(
            compare.yara_buffer_tag(SimpleNamespace(yara_buffer_capacity=7_500.0)),
            "_yara7500",
        )

    def test_bc_tag_marks_demonstration_config(self):
        args = SimpleNamespace(bc_episodes=100, nonwait_weight=20.0)

        self.assertEqual(compare.bc_tag(args), "_bc100w20")

    def test_disturbance_tag_marks_leg_wave_stress(self):
        args = SimpleNamespace(
            capture_noise_std=0.5,
            initial_inventory_fill_max=0.8,
            leg_wave_slowdown_multiplier=2.0,
            leg_wave_speed_factor_floor=0.25,
        )

        self.assertEqual(compare.disturbance_tag(args), "_cap0.5_inv0.8_wave2_floor0.25")

    def test_default_bc_budget_matches_partial_dispatch_diagnostics(self):
        with patch("sys.argv", ["compare_reward_modes_bc.py"]):
            args = compare.parse_args()

        self.assertEqual(args.bc_episodes, 30)
        self.assertEqual(args.bc_epochs, 20)
        self.assertEqual(args.reward_modes, ["economic", "vent_first"])
        self.assertEqual(args.weather_mode, "window")
        self.assertFalse(args.weather_obs)

    def test_reward_modes_can_limit_run_to_vent_first(self):
        with patch("sys.argv", ["compare_reward_modes_bc.py", "--reward-modes", "vent_first"]):
            args = compare.parse_args()

        self.assertEqual(args.reward_modes, ["vent_first"])

    def test_weather_obs_arg_is_parsed(self):
        with patch("sys.argv", ["compare_reward_modes_bc.py", "--weather-obs"]):
            args = compare.parse_args()

        self.assertTrue(args.weather_obs)

    def test_make_env_passes_weather_obs_to_native_env(self):
        args = SimpleNamespace(
            episode_hours=720,
            injection_reward_eur_per_t=80.0,
            store_reward=None,
            vent_weight=1.0,
            operating_cost_weight=1.0,
            reward_mode="economic",
            vent_first_vent_eur_per_t=10_000.0,
            overflow_risk_eur_per_t=100.0,
            overflow_risk_lookahead_h=24.0,
            carbon_price=80.0,
            enforce_full_load_dispatch=False,
            scenario="northern_lights_phase1_3vessels",
            weather_mode="window",
            weather_obs=True,
            wave_height_nc_paths=None,
            lstm_prediction_csv=None,
            capture_noise_std=0.30,
            initial_inventory_fill_max=0.5,
            leg_wave_slowdown_multiplier=1.0,
            leg_wave_speed_factor_floor=0.0,
            yara_buffer_capacity=None,
        )
        sentinel_env = object()

        with patch.object(compare, "make_native_env", return_value=sentinel_env) as make_native_env:
            env = compare.make_env(args, "economic")

        self.assertIs(env, sentinel_env)
        self.assertTrue(make_native_env.call_args.kwargs["include_weather_obs"])

    def test_stress_disturbance_args_are_parsed(self):
        with patch(
            "sys.argv",
            [
                "compare_reward_modes_bc.py",
                "--capture-noise-std", "0.5",
                "--initial-inventory-fill-max", "0.8",
                "--leg-wave-slowdown-multiplier", "2.0",
                "--leg-wave-speed-factor-floor", "0.25",
            ],
        ):
            args = compare.parse_args()

        self.assertEqual(args.capture_noise_std, 0.5)
        self.assertEqual(args.initial_inventory_fill_max, 0.8)
        self.assertEqual(args.leg_wave_slowdown_multiplier, 2.0)
        self.assertEqual(args.leg_wave_speed_factor_floor, 0.25)


if __name__ == "__main__":
    unittest.main()
