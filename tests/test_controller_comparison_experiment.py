import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class ControllerComparisonExperimentTests(unittest.TestCase):
    def test_native_mpc_short_episode_handles_zero_greedy_vent(self):
        from experiments import rolling_native_mpc_headroom as experiment

        args = SimpleNamespace(
            hours=24,
            disturbance_profile="none",
            replan_every=24,
            planning_horizon_h=24,
            weather_update_hours=24.0,
            weather_speed_min=0.75,
            weather_speed_max=1.0,
            capture_noise_std=0.10,
            capture_high_output_rate_per_week=0.5,
            capture_high_output_mean_hours=48.0,
            capture_multiplier_min=1.25,
            capture_multiplier_max=1.75,
        )

        row, actions = experiment.run_seed(args, seed=1, economics=experiment.EconomicParameters())

        self.assertEqual(row["greedy_vented_t"], 0.0)
        self.assertEqual(row["vented_reduction_pct"], 0.0)
        self.assertTrue(row["replay_ok"])
        self.assertTrue(row["native_mpc_trace_replay_matches"])
        self.assertTrue(row["greedy_trace_replay_matches"])
        self.assertTrue(row["native_mpc_replay_is_exact"])
        self.assertEqual(row["native_mpc_replay_mismatches"], "")
        self.assertEqual(len(actions["native_mpc"]), args.hours)

    def test_native_mpc_cli_rejects_alternate_objective_mode(self):
        from experiments import rolling_native_mpc_headroom as experiment

        with self.assertRaises(SystemExit):
            experiment.parse_args(["--objective-mode", "vent_then_total_cost"])

    def test_native_mpc_experiment_can_include_replay_grounded_rolling_milp(self):
        from experiments import rolling_native_mpc_headroom as experiment

        args = SimpleNamespace(
            hours=2,
            disturbance_profile="none",
            replan_every=2,
            planning_horizon_h=2,
            weather_update_hours=24.0,
            weather_speed_min=0.75,
            weather_speed_max=1.0,
            capture_noise_std=0.10,
            capture_high_output_rate_per_week=0.5,
            capture_high_output_mean_hours=48.0,
            capture_multiplier_min=1.25,
            capture_multiplier_max=1.75,
            include_rolling_milp=True,
            rolling_milp_time_limit_s=5.0,
        )

        row, actions = experiment.run_seed(
            args,
            seed=1,
            economics=experiment.EconomicParameters(),
        )

        self.assertTrue(row["rolling_milp_replay_ok"])
        self.assertTrue(row["rolling_milp_replay_is_exact"])
        self.assertEqual(len(actions["rolling_milp"]), args.hours)

    def test_common_native_mpc_replay_detects_final_state_mismatch(self):
        from experiments import rolling_native_mpc_headroom as experiment
        from sim.control.baselines import greedy_shuttle_policy
        from sim.control.replay import replay_native_actions

        args = SimpleNamespace(
            hours=2,
            disturbance_profile="none",
            replan_every=2,
            planning_horizon_h=2,
            weather_update_hours=24.0,
            weather_speed_min=0.75,
            weather_speed_max=1.0,
            capture_noise_std=0.10,
            capture_high_output_rate_per_week=0.5,
            capture_high_output_mean_hours=48.0,
            capture_multiplier_min=1.25,
            capture_multiplier_max=1.75,
        )
        economics = experiment.EconomicParameters()
        actions, _metrics, expected = experiment.collect_actions(
            experiment.make_env(args, economics),
            greedy_shuttle_policy,
            seed=1,
        )
        replay_env = experiment.make_env(args, economics)
        replay_env.reset(seed=1)

        exact = replay_native_actions(
            replay_env,
            actions,
            horizon_h=args.hours,
            expected=expected,
        )
        inventories = dict(expected.entity_inventory_t)
        first_entity = next(iter(inventories))
        inventories[first_entity] += 1.0
        mismatched = replay_native_actions(
            replay_env,
            actions,
            horizon_h=args.hours,
            expected=replace(expected, entity_inventory_t=inventories),
        )

        self.assertTrue(exact.is_exact, exact.mismatches)
        self.assertFalse(mismatched.is_exact)
        self.assertTrue(any("entity_inventory_t" in value for value in mismatched.mismatches))

    def test_controller_comparison_imports_with_current_layout(self):
        from experiments import compare_controllers_same_scenarios as compare

        factories = compare.controller_factories(
            replan_every=12,
            progress=lambda _message: None,
            rolling_planning_horizon_h=96,
            rolling_time_limit_s=30.0,
            economics=compare.EconomicParameters(storage_shortfall_eur_per_t=0.0),
        )
        self.assertEqual(set(factories), {"idle", "greedy_shuttle", "rule_based", "rolling_milp"})

    def test_controller_comparison_uses_one_economic_parameter_set(self):
        from experiments import compare_controllers_same_scenarios as compare
        from sim.scenario_generation import ScenarioConfig

        economics = compare.EconomicParameters(storage_shortfall_eur_per_t=0.0)
        env = compare.make_env(
            cap_hours=2,
            scenario_seed_config=ScenarioConfig(episode_hours=2, randomize_initial_inventory=False),
            economics=economics,
        )
        factories = compare.controller_factories(
            replan_every=12,
            progress=lambda _message: None,
            rolling_planning_horizon_h=168,
            rolling_time_limit_s=30.0,
            economics=economics,
        )

        rolling = factories["rolling_milp"](env)

        self.assertEqual(env.cost_model.parameters.storage_shortfall_eur_per_t, 0.0)
        self.assertEqual(rolling.economics.storage_shortfall_eur_per_t, 0.0)
        self.assertEqual(rolling.time_limit_s, 30.0)

    def test_controller_comparison_can_build_selected_fixed_scenario(self):
        from experiments import compare_controllers_same_scenarios as compare
        from sim.scenario_generation import ScenarioConfig

        env = compare.make_env(
            cap_hours=2,
            scenario_seed_config=ScenarioConfig(episode_hours=2, randomize_initial_inventory=False),
            economics=compare.EconomicParameters(),
            fixed_scenario="northern_lights_phase1",
        )

        self.assertIn("yara_sluiskil", env.emitter_ids)
        self.assertIn("oygarden_terminal", env.terminal_ids)

    def test_greedy_cplex_phase1_comparison_can_build_three_vessel_scenario(self):
        from experiments import compare_greedy_cplex_phase1 as compare

        env = compare._make_env(2, compare.EconomicParameters(), "northern_lights_phase1_3vessels")

        self.assertEqual(len(env.vessel_ids), 3)
        self.assertIn("yara_sluiskil", env.emitter_ids)

    def test_rule_based_controller_factory_translates_to_hybrid_env_action(self):
        from experiments import compare_controllers_same_scenarios as compare
        from sim.environment import VESSEL_WAIT, WELL_RATE_LEVELS_MTPA
        from sim.scenario_generation import ScenarioConfig

        factories = compare.controller_factories(
            replan_every=12,
            progress=lambda _message: None,
            rolling_planning_horizon_h=48,
            rolling_time_limit_s=30.0,
            economics=compare.EconomicParameters(),
        )
        env = compare.make_env(
            cap_hours=2,
            scenario_seed_config=ScenarioConfig(episode_hours=2, randomize_initial_inventory=False),
            economics=compare.EconomicParameters(),
        )
        env.reset(seed=1)

        action = factories["rule_based"](env)(env)

        self.assertEqual(action["vessels"], [VESSEL_WAIT] * len(env.vessel_ids))
        self.assertEqual(action["wells"], [list(WELL_RATE_LEVELS_MTPA).index(1.5)])

    def test_metric_rows_and_summary_include_solve_time(self):
        from experiments import compare_controllers_same_scenarios as compare
        from sim.metrics import EpisodeMetrics

        metrics = EpisodeMetrics(
            elapsed_hours=2,
            stored_t=10.0,
            operating_cost=5.0,
            total_cost=15.0,
            total_cost_per_stored_t=1.5,
            vessel_fuel=1.0,
            conditioning=2.0,
            reconditioning=3.0,
            loading=4.0,
            unloading=5.0,
        )

        row = compare.metric_row(
            seed=1,
            controller="idle",
            metrics=metrics,
            signature="sig",
            solve_time_s=1.25,
        )
        summary = compare.summarize([row])

        self.assertEqual(row["solve_time_s"], 1.25)
        self.assertEqual(summary[0]["solve_time_s_mean"], 1.25)
        self.assertEqual(row["total_cost_per_stored_t"], 1.5)
        self.assertEqual(summary[0]["total_cost_per_stored_t_mean"], 1.5)
        self.assertEqual(summary[0]["vessel_fuel_mean"], 1.0)
        self.assertEqual(summary[0]["conditioning_mean"], 2.0)
        self.assertEqual(summary[0]["reconditioning_mean"], 3.0)
        self.assertEqual(summary[0]["loading_mean"], 4.0)
        self.assertEqual(summary[0]["unloading_mean"], 5.0)

    def test_summary_keeps_per_tonne_columns_when_controller_stores_nothing(self):
        from experiments import compare_controllers_same_scenarios as compare
        from sim.metrics import EpisodeMetrics

        row = compare.metric_row(
            seed=1,
            controller="idle",
            metrics=EpisodeMetrics(stored_t=0.0, cost_per_stored_t=None, total_cost_per_stored_t=None),
            signature="sig",
            solve_time_s=1.25,
        )

        summary = compare.summarize([row])

        self.assertEqual(summary[0]["cost_per_stored_t_mean"], "")
        self.assertEqual(summary[0]["cost_per_stored_t_std"], "")
        self.assertEqual(summary[0]["total_cost_per_stored_t_mean"], "")
        self.assertEqual(summary[0]["total_cost_per_stored_t_std"], "")

    def test_report_includes_solve_time(self):
        from experiments import compare_controllers_same_scenarios as compare

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            compare.write_report(
                report_path,
                summary_rows=[
                    {
                        "controller": "idle",
                        "elapsed_hours_mean": 2.0,
                        "stored_t_mean": 0.0,
                        "vented_t_mean": 1.0,
                        "storage_shortfall_penalty_mean": 0.0,
                        "total_cost_mean": 80.0,
                        "operating_cost_mean": 0.0,
                        "solve_time_s_mean": 1.25,
                    }
                ],
                benchmark={
                    "case": "static_milp_fixed_horizon_scenario_oracle",
                    "horizon_h": 2,
                    "status": "mocked",
                    "stored_t": 0.0,
                    "deliveries": 0,
                    "vented_t": 1.0,
                    "in_transit_t": 0.0,
                    "shortfall_t": 0.0,
                    "operating_cost": 0.0,
                    "total_cost": 80.0,
                    "cost_per_stored_t": 0.0,
                    "solve_time_s": 2.5,
                },
            )

            text = report_path.read_text(encoding="utf-8")

        self.assertIn("Solve s mean", text)
        self.assertIn("1.2", text)
        self.assertIn("solve time: 2.5s", text)

    def test_report_marks_invalid_static_milp_not_comparable(self):
        from experiments import compare_controllers_same_scenarios as compare

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            compare.write_report(
                report_path,
                summary_rows=[],
                benchmark={
                    "case": "static_milp_fixed_horizon_scenario_oracle",
                    "horizon_h": 2880,
                    "status": "Not Solved",
                    "solve_time_s": 442.4,
                    "is_valid": False,
                    "validation_error": "solver status Not Solved is not a validated integer solution",
                    "stored_t": "",
                    "deliveries": "",
                    "vented_t": "",
                    "in_transit_t": "",
                    "shortfall_t": "",
                    "operating_cost": "",
                    "total_cost": "",
                    "cost_per_stored_t": "",
                },
            )

            text = report_path.read_text(encoding="utf-8")

        self.assertIn("not comparable", text)
        self.assertIn("Not Solved", text)
        self.assertNotIn("stored:", text)

    def test_controller_comparison_cli_smoke_writes_outputs(self):
        from experiments import compare_controllers_same_scenarios as compare

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            argv = [
                "compare_controllers_same_scenarios.py",
                "--controllers",
                "idle",
                "--seeds",
                "1",
                "--cap-hours",
                "2",
                "--skip-static-milp",
                "--output-dir",
                str(output_dir),
            ]
            with patch.object(sys, "argv", argv):
                compare.main()

            self.assertTrue((output_dir / "controller_comparison_by_seed.csv").exists())
            self.assertTrue((output_dir / "controller_comparison_summary.csv").exists())
            self.assertTrue((output_dir / "static_milp_same_scenario_benchmark.csv").exists())
            self.assertTrue((output_dir / "controller_comparison_report.md").exists())

    def test_static_milp_time_limit_default_is_long_enough_for_fixed_horizon_runs(self):
        from experiments import compare_controllers_same_scenarios as compare

        with patch.object(sys, "argv", ["compare_controllers_same_scenarios.py"]):
            args = compare.parse_args()

        self.assertEqual(args.static_milp_time_limit_s, 600.0)

    def test_rolling_milp_defaults_use_week_horizon_and_daily_execution(self):
        from experiments import compare_controllers_same_scenarios as compare

        with patch.object(sys, "argv", ["compare_controllers_same_scenarios.py"]):
            args = compare.parse_args()

        self.assertEqual(args.rolling_planning_horizon_h, 168)
        self.assertEqual(args.rolling_replan_every, 24)
        self.assertEqual(args.rolling_time_limit_s, 30.0)
        self.assertEqual(args.scenario, "toy")
        self.assertEqual(args.storage_shortfall_penalty_eur_per_t, 0.0)

    def test_controller_comparison_cli_accepts_fixed_scenario_choice(self):
        from experiments import compare_controllers_same_scenarios as compare

        with patch.object(
            sys,
            "argv",
            ["compare_controllers_same_scenarios.py", "--scenario", "northern_lights_phase2"],
        ):
            args = compare.parse_args()

        self.assertEqual(args.scenario, "northern_lights_phase2")

    def test_controller_comparison_cli_rejects_internal_phase2_id(self):
        from experiments import compare_controllers_same_scenarios as compare

        with (
            patch.object(
                sys,
                "argv",
                ["compare_controllers_same_scenarios.py", "--scenario", "northern_lights_phase2_public_2028"],
            ),
            self.assertRaises(SystemExit),
        ):
            compare.parse_args()

    def test_fixed_horizon_cli_forwards_static_milp_solver_controls(self):
        from experiments import compare_controllers_same_scenarios as compare

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            argv = [
                "compare_controllers_same_scenarios.py",
                "--controllers",
                "idle",
                "--seeds",
                "1",
                "--cap-hours",
                "2",
                "--static-milp-time-limit-s",
                "123",
                "--static-milp-gap-rel",
                "0.05",
                "--static-milp-gap-abs",
                "10",
                "--storage-shortfall-penalty-eur-per-t",
                "0",
                "--output-dir",
                str(output_dir),
            ]
            benchmark = {
                "case": "static_milp_fixed_horizon_nominal",
                "scenario_signature": "sig",
                "horizon_h": 2,
                "status": "mocked",
                "is_valid": True,
                "validation_error": "",
                "max_binary_integrality_violation": 0.0,
                "stored_t": 0.0,
                "deliveries": 0,
                "vented_t": 0.0,
                "in_transit_t": 0.0,
                "in_transit_growth_t": 0.0,
                "shortfall_t": 0.0,
                "operating_cost": 0.0,
                "total_cost": 0.0,
                "cost_per_stored_t": 0.0,
                "time_limit_s": 123.0,
                "mip_gap_rel": 0.05,
                "mip_gap_abs": 10.0,
            }
            with (
                patch.object(sys, "argv", argv),
                patch.object(compare, "scenario_signature", return_value="sig"),
                patch.object(compare, "static_fixed_horizon_milp_benchmark", return_value=benchmark) as fixed,
            ):
                compare.main()

            fixed.assert_called_once()
            self.assertEqual(fixed.call_args.args, (2, 123.0, 0.05, 10.0))
            self.assertEqual(fixed.call_args.kwargs["seed"], 1)
            self.assertIsNotNone(fixed.call_args.kwargs["scenario_seed_config"])
            self.assertEqual(fixed.call_args.kwargs["fixed_scenario"], "toy")
            self.assertEqual(fixed.call_args.kwargs["economics"].storage_shortfall_eur_per_t, 0.0)

    def test_static_fixed_horizon_benchmark_passes_mip_gap_controls(self):
        from experiments import compare_controllers_same_scenarios as compare
        from sim.scenario_generation import ScenarioConfig

        result = SimpleNamespace(
            status="mocked",
            is_valid=True,
            validation_error="",
            max_binary_integrality_violation=0.0,
            stored_t=1.0,
            deliveries=1,
            operating_cost=2.0,
            vented_t=3.0,
            in_transit_t=6.0,
            in_transit_growth_t=7.0,
            shortfall_t=4.0,
            total_cost=5.0,
            cost_per_stored_t=2.0,
            total_cost_per_stored_t=5.0,
            vessel_fuel=0.5,
            conditioning=0.6,
            reconditioning=0.7,
            loading=0.8,
            unloading=0.9,
        )
        with patch.object(compare, "solve_max_storage_fixed_horizon", return_value=result) as solve:
            row = compare.static_fixed_horizon_milp_benchmark(
                720,
                600.0,
                0.01,
                100.0,
                scenario_seed_config=ScenarioConfig(episode_hours=720, randomize_initial_inventory=False),
                seed=7,
                economics=compare.EconomicParameters(storage_shortfall_eur_per_t=0.0),
            )

        self.assertEqual(row["case"], "static_milp_fixed_horizon_scenario_oracle")
        self.assertEqual(row["seed"], 7)
        self.assertEqual(row["scenario_signature"], compare.scenario_signature(solve.call_args.kwargs["scenario"]))
        self.assertTrue(row["is_valid"])
        self.assertEqual(row["vented_t"], 3.0)
        self.assertEqual(row["in_transit_t"], 6.0)
        self.assertEqual(row["in_transit_growth_t"], 7.0)
        self.assertEqual(row["shortfall_t"], 4.0)
        self.assertEqual(row["total_cost"], 5.0)
        self.assertEqual(row["total_cost_per_stored_t"], 5.0)
        self.assertEqual(row["vessel_fuel"], 0.5)
        self.assertEqual(row["conditioning"], 0.6)
        self.assertEqual(row["reconditioning"], 0.7)
        self.assertEqual(row["loading"], 0.8)
        self.assertEqual(row["unloading"], 0.9)
        self.assertEqual(row["mip_gap_rel"], 0.01)
        self.assertEqual(row["mip_gap_abs"], 100.0)
        solve.assert_called_once()
        self.assertEqual(solve.call_args.kwargs["horizon_h"], 720)
        self.assertEqual(solve.call_args.kwargs["time_limit_s"], 600.0)
        self.assertEqual(solve.call_args.kwargs["mip_gap_rel"], 0.01)
        self.assertEqual(solve.call_args.kwargs["mip_gap_abs"], 100.0)
        self.assertEqual(solve.call_args.kwargs["economics"].storage_shortfall_eur_per_t, 0.0)
        self.assertEqual(solve.call_args.kwargs["scenario"].seed, 7)

    def test_static_fixed_horizon_benchmark_blanks_invalid_solver_metrics(self):
        from experiments import compare_controllers_same_scenarios as compare
        from sim.scenario_generation import ScenarioConfig

        result = SimpleNamespace(
            status="Not Solved",
            is_valid=False,
            validation_error="solver status Not Solved is not a validated integer solution",
            max_binary_integrality_violation=0.5,
            stored_t=353_734.9,
            deliveries=8,
            operating_cost=824_048.0,
            vented_t=0.0,
            in_transit_t=49_465.1,
            in_transit_growth_t=49_465.1,
            shortfall_t=9_145.1,
            total_cost=824_048.0,
            cost_per_stored_t=2.33,
            total_cost_per_stored_t=2.33,
            vessel_fuel=1.0,
            conditioning=2.0,
            reconditioning=3.0,
            loading=4.0,
            unloading=5.0,
        )
        with patch.object(compare, "solve_max_storage_fixed_horizon", return_value=result):
            row = compare.static_fixed_horizon_milp_benchmark(
                2880,
                300.0,
                0.05,
                None,
                scenario_seed_config=ScenarioConfig(episode_hours=2880, randomize_initial_inventory=False),
                seed=1,
                economics=compare.EconomicParameters(storage_shortfall_eur_per_t=0.0),
            )

        self.assertFalse(row["is_valid"])
        self.assertEqual(row["stored_t"], "")
        self.assertEqual(row["deliveries"], "")
        self.assertEqual(row["total_cost"], "")
        self.assertEqual(row["total_cost_per_stored_t"], "")
        self.assertEqual(row["vessel_fuel"], "")
        self.assertEqual(row["conditioning"], "")
        self.assertEqual(row["reconditioning"], "")
        self.assertEqual(row["loading"], "")
        self.assertEqual(row["unloading"], "")
        self.assertIn("Not Solved", row["validation_error"])

    def test_fixed_target_cli_options_are_removed(self):
        from experiments import compare_controllers_same_scenarios as compare

        with (
            patch.object(sys, "argv", ["compare_controllers_same_scenarios.py", "--mode", "fixed-target"]),
            self.assertRaises(SystemExit),
        ):
            compare.parse_args()


if __name__ == "__main__":
    unittest.main()
