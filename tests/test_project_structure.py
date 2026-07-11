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

    def test_forecast_rl_hpc_scripts_have_borg_runtime_contract(self):
        paths = (
            ROOT / "hpc" / "submit_forecast_mpc_demos.sh",
            ROOT / "hpc" / "submit_forecast_encoder_rl.sh",
        )

        for path in paths:
            with self.subTest(path=path.name):
                self.assertTrue(path.exists())
                source = path.read_text(encoding="utf-8")
                self.assertTrue(source.startswith("#!/usr/bin/env bash\n"))
                self.assertIn("set -euo pipefail", source)
                self.assertIn("#SBATCH --partition=root", source)
                self.assertIn("#SBATCH --qos=long", source)
                self.assertIn("#SBATCH --nodes=1", source)
                self.assertIn("#SBATCH --ntasks=1", source)
                self.assertIn(
                    "source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh",
                    source,
                )
                self.assertIn("conda activate mas-ccus", source)
                self.assertIn(
                    'PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM}"',
                    source,
                )
                self.assertIn('export PYTHONPATH="src"', source)
                self.assertIn('if [[ -z "${GIT_COMMIT:-}" ]]', source)
                self.assertIn('GIT_COMMIT="$(git rev-parse HEAD', source)
                self.assertIn("cannot determine GIT_COMMIT", source)
                self.assertIn("export GIT_COMMIT", source)
                self.assertNotIn('GIT_COMMIT="unknown"', source)
                self.assertIn('echo "Git commit: $GIT_COMMIT"', source)
                self.assertIn("which python", source)
                self.assertIn("python --version", source)

    def test_forecast_mpc_demo_hpc_contract(self):
        path = ROOT / "hpc" / "submit_forecast_mpc_demos.sh"
        self.assertTrue(path.exists())
        source = path.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --job-name=ccs_mpc_demos", source)
        self.assertIn("#SBATCH --cpus-per-task=4", source)
        self.assertIn("#SBATCH --mem=32G", source)
        self.assertIn("#SBATCH --time=24:00:00", source)
        self.assertIn("#SBATCH -o logs/mpc_demos-%j.out", source)
        self.assertIn("#SBATCH -e logs/mpc_demos-%j.err", source)
        self.assertNotIn("#SBATCH --gres", source, msg="CPU demo must not request a GPU")
        self.assertIn(
            'DEMO_CACHE="${DEMO_CACHE:-output/rl_forecast/demos/mpc_720h_30eps.npz}"',
            source,
        )
        self.assertIn('EPISODE_HOURS="${EPISODE_HOURS:-720}"', source)
        self.assertIn(
            'DEMO_SEEDS="${DEMO_SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29}"',
            source,
        )
        self.assertIn("generate-demos", source)
        self.assertIn("forecast horizon 168h", source)
        self.assertIn("block weather", source)
        self.assertIn("vent-first reward", source)
        self.assertIn("partial dispatch", source)
        self.assertIn("889h demonstration environment", source)

    def test_forecast_encoder_rl_array_resources_and_mapping(self):
        path = ROOT / "hpc" / "submit_forecast_encoder_rl.sh"
        self.assertTrue(path.exists())
        source = path.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --job-name=ccs_forecast_rl", source)
        self.assertIn("#SBATCH --array=0-2", source)
        self.assertIn("#SBATCH --cpus-per-task=12", source)
        self.assertIn("#SBATCH --mem=100G", source)
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --time=24:00:00", source)
        self.assertIn("#SBATCH -o logs/forecast_rl-%A_%a.out", source)
        self.assertIn("#SBATCH -e logs/forecast_rl-%A_%a.err", source)
        self.assertIn("VARIANTS=(state flat tcn)", source)
        self.assertIn("MODEL_SEEDS=(0 1 2)", source)
        self.assertIn("VARIANT_INDEX=$((TASK_ID % 3))", source)
        self.assertIn("SEED_INDEX=$((TASK_ID / 3))", source)
        self.assertIn("out of range", source)
        self.assertIn("--array=0-8", source)

    def test_forecast_encoder_rl_uses_shared_cache_and_supported_runner_flags(self):
        demo_path = ROOT / "hpc" / "submit_forecast_mpc_demos.sh"
        train_path = ROOT / "hpc" / "submit_forecast_encoder_rl.sh"
        self.assertTrue(demo_path.exists())
        self.assertTrue(train_path.exists())
        demo_source = demo_path.read_text(encoding="utf-8")
        train_source = train_path.read_text(encoding="utf-8")
        cache_default = "output/rl_forecast/demos/mpc_720h_30eps.npz"

        self.assertIn(cache_default, demo_source)
        self.assertIn(cache_default, train_source)
        command = train_source.split(
            "python -u scripts/compare_forecast_encoders_rl.py train", 1
        )[1].split('echo "Job finished', 1)[0]
        flags = set(re.findall(r"--[a-z][a-z-]*", command))
        self.assertEqual(
            flags,
            {
                "--variant",
                "--demo-cache",
                "--timesteps",
                "--bc-epochs",
                "--n-steps",
                "--batch-size",
                "--bc-batch-size",
                "--model-seed",
                "--eval-seeds",
                "--device",
                "--out-dir",
                "--episode-hours",
            },
        )
        self.assertIn('TIMESTEPS="${TIMESTEPS:-100000}"', train_source)
        self.assertIn('BC_EPOCHS="${BC_EPOCHS:-20}"', train_source)
        self.assertIn('EPISODE_HOURS="${EPISODE_HOURS:-720}"', train_source)
        self.assertIn('EVAL_SEEDS="${EVAL_SEEDS:-101 102 103 104 105}"', train_source)
        self.assertIn('DEVICE="${DEVICE:-cuda}"', train_source)

    def test_forecast_encoder_rl_guards_and_diagnostics(self):
        path = ROOT / "hpc" / "submit_forecast_encoder_rl.sh"
        self.assertTrue(path.exists())
        source = path.read_text(encoding="utf-8")

        self.assertIn('if [[ ! -f "$DEMO_CACHE" ]]', source)
        self.assertIn("demonstration cache not found", source)
        self.assertIn('echo "CUDA_VISIBLE_DEVICES:', source)
        self.assertIn("torch.__version__", source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertIn("torch.cuda.device_count()", source)
        self.assertIn('echo "Variant: $VARIANT"', source)
        self.assertIn('echo "Model seed: $MODEL_SEED"', source)
        self.assertIn('echo "Demo cache: $DEMO_CACHE"', source)
        self.assertIn('echo "Output directory: $OUT_DIR"', source)
        self.assertIn('OUT_DIR="${OUT_DIR:-output/rl_forecast/pilot}"', source)
        self.assertIn("OUT_DIR=output/rl_forecast/smoke", source)
        self.assertIn("OUT_DIR=output/rl_forecast/formal", source)
        self.assertIn('export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"', source)
        self.assertIn('export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"', source)

    def test_forecast_hpc_submission_examples_create_logs_before_sbatch(self):
        expected = {
            "submit_forecast_mpc_demos.sh": "mkdir -p logs output/rl_forecast/demos",
            "submit_forecast_encoder_rl.sh": 'mkdir -p logs "$OUT_DIR"',
        }

        for name, in_body_mkdir in expected.items():
            with self.subTest(script=name):
                source = (ROOT / "hpc" / name).read_text(encoding="utf-8")
                self.assertIn(
                    "# LOGIN-NODE SUBMISSION PREREQUISITE (run from project root):",
                    source,
                )
                self.assertIn(in_body_mkdir, source)
                examples = [
                    line
                    for line in source.splitlines()
                    if re.match(r"^# .*\bsbatch\b", line)
                ]
                self.assertTrue(examples)
                for line in examples:
                    with self.subTest(script=name, example=line):
                        self.assertTrue(
                            line.startswith("# mkdir -p logs && sbatch "),
                            msg=f"submission must create logs first: {line}",
                        )

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
