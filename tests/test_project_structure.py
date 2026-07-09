from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTests(unittest.TestCase):
    def test_visualization_is_split_into_package_modules(self):
        visualization_dir = ROOT / "src" / "sim" / "visualization"

        self.assertTrue(visualization_dir.is_dir())
        self.assertFalse((ROOT / "src" / "sim" / "visualization.py").exists())
        self.assertTrue((visualization_dir / "__init__.py").exists())
        self.assertTrue((visualization_dir / "core.py").exists())
        self.assertTrue((visualization_dir / "html.py").exists())
        self.assertTrue((visualization_dir / "writers.py").exists())

    def test_control_algorithms_live_in_control_package(self):
        control_dir = ROOT / "src" / "sim" / "control"

        self.assertTrue(control_dir.is_dir())
        self.assertTrue((control_dir / "__init__.py").exists())
        self.assertTrue((control_dir / "rule_based.py").exists())
        self.assertTrue((control_dir / "milp.py").exists())
        self.assertTrue((control_dir / "rolling_milp.py").exists())
        self.assertTrue((control_dir / "imitation.py").exists())
        self.assertTrue((control_dir / "baselines.py").exists())
        self.assertFalse((ROOT / "src" / "sim" / "rule_based.py").exists())
        self.assertFalse((ROOT / "src" / "sim" / "milp.py").exists())
        self.assertFalse((ROOT / "src" / "sim" / "rolling_milp.py").exists())
        self.assertFalse((ROOT / "src" / "sim" / "imitation.py").exists())
        metrics_source = (ROOT / "src" / "sim" / "metrics.py").read_text(encoding="utf-8")
        self.assertNotIn("def greedy_shuttle_policy", metrics_source)
        self.assertNotIn("def idle_policy", metrics_source)

    def test_scenario_generation_has_clear_file_names(self):
        scenario_generation_dir = ROOT / "src" / "sim" / "scenario_generation"

        self.assertTrue((ROOT / "src" / "sim" / "network_scenarios.py").exists())
        self.assertFalse((ROOT / "src" / "sim" / "scenarios.py").exists())
        self.assertTrue(scenario_generation_dir.is_dir())
        self.assertTrue((scenario_generation_dir / "__init__.py").exists())
        self.assertTrue((scenario_generation_dir / "generator.py").exists())
        self.assertTrue((scenario_generation_dir / "disturbance_resolver.py").exists())
        self.assertFalse((scenario_generation_dir / "load_shift.py").exists())
        self.assertFalse((ROOT / "src" / "sim" / "scenario.py").exists())
        self.assertFalse((ROOT / "src" / "sim" / "disturbances.py").exists())

    def test_rl_environment_lives_in_environment_package(self):
        environment_dir = ROOT / "src" / "sim" / "environment"

        self.assertTrue(environment_dir.is_dir())
        self.assertTrue((environment_dir / "__init__.py").exists())
        self.assertTrue((environment_dir / "env.py").exists())
        self.assertTrue((environment_dir / "factories.py").exists())
        self.assertTrue((environment_dir / "gym_adapter.py").exists())
        self.assertFalse((ROOT / "src" / "sim" / "env.py").exists())
        self.assertFalse((ROOT / "src" / "sim" / "env_scenarios.py").exists())
        self.assertFalse((ROOT / "src" / "sim" / "gym_env.py").exists())

    def test_reward_modes_hpc_script_defaults_to_probability_window_weather(self):
        script = (ROOT / "hpc" / "submit_reward_modes_bc.sh").read_text(encoding="utf-8")

        self.assertIn('WEATHER_MODE="${WEATHER_MODE:-window}"', script)

    def test_reward_modes_hpc_script_can_enable_weather_observations(self):
        script = (ROOT / "hpc" / "submit_reward_modes_bc.sh").read_text(encoding="utf-8")

        self.assertIn('WEATHER_OBS="${WEATHER_OBS:-1}"', script)
        self.assertIn('WEATHER_OBS_ARGS=(--weather-obs)', script)
        self.assertIn('"${WEATHER_OBS_ARGS[@]}"', script)

    def test_reward_modes_hpc_script_passes_weather_window_rate(self):
        script = (ROOT / "hpc" / "submit_reward_modes_bc.sh").read_text(encoding="utf-8")

        self.assertIn(
            'WEATHER_WINDOW_RATE_PER_WEEK="${WEATHER_WINDOW_RATE_PER_WEEK:-1.0}"',
            script,
        )
        self.assertIn(
            'echo "Weather window rate per week: $WEATHER_WINDOW_RATE_PER_WEEK"',
            script,
        )
        self.assertIn(
            '--weather-window-rate-per-week "$WEATHER_WINDOW_RATE_PER_WEEK"',
            script,
        )

    def test_hpc_shell_scripts_use_unix_line_endings(self):
        for path in (ROOT / "hpc").glob("*.sh"):
            with self.subTest(path=path.name):
                self.assertNotIn(b"\r\n", path.read_bytes())

    def test_action_protocol_lives_in_actions_package(self):
        actions_dir = ROOT / "src" / "sim" / "actions"

        self.assertTrue(actions_dir.is_dir())
        self.assertTrue((actions_dir / "__init__.py").exists())
        self.assertTrue((actions_dir / "action.py").exists())
        self.assertTrue((actions_dir / "resolver.py").exists())
        self.assertFalse((ROOT / "src" / "sim" / "actions.py").exists())
        self.assertFalse((ROOT / "src" / "sim" / "action_resolver.py").exists())


if __name__ == "__main__":
    unittest.main()
