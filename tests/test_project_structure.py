from pathlib import Path
import re
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

    def test_residual_rl_has_no_load_shift_entry_points(self):
        source = (ROOT / "scripts" / "train_residual_rl.py").read_text(encoding="utf-8")

        self.assertNotIn("scenario_generation.load_shift", source)
        self.assertNotIn("--load-shift", source)
        self.assertNotIn("args.load_shift", source)

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

    def test_forecast_encoder_comparison_runner_is_a_script(self):
        path = ROOT / "scripts" / "compare_forecast_encoders_rl.py"

        self.assertTrue(path.exists())
        source = path.read_text(encoding="utf-8")
        self.assertIn('if __name__ == "__main__":', source)
        self.assertIn("generate-demos", source)
        self.assertIn("train", source)
        self.assertIn("report", source)

    def test_scripts_and_experiments_have_distinct_entry_point_roles(self):
        script_names = {
            path.name
            for path in (ROOT / "scripts").glob("*.py")
            if path.name != "__init__.py"
        }
        experiment_names = {path.name for path in (ROOT / "experiments").glob("*.py")}

        self.assertTrue(
            all(
                name.startswith(("build_", "train_"))
                or name == "compare_forecast_encoders_rl.py"
                for name in script_names
            )
        )
        self.assertFalse(any(name.startswith("train_") for name in experiment_names))
        self.assertFalse(script_names & experiment_names)

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
