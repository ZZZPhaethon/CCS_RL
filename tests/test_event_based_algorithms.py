import unittest

import numpy as np

from sim.control.event_based.residual_rl.factory import make_residual_gym_env
from sim.control.event_based.residual_rl_v2.factory import (
    make_masked_residual_gym_env,
)
from sim.control.event_based.residual_rl_v3.factory import (
    make_risk_gated_gym_env,
)
from sim.control.event_based.residual_rl_v4.factory import (
    make_tail_robust_native_env,
    make_tail_replay_gym_env,
)
from sim.control.event_based.rl.gym_env import HighLevelDispatchGymEnv
from sim.control.event_based.rl.train_high_level_ppo import (
    make_high_level_native_env,
)
from sim.control.event_based.rl.reward import HighLevelRewardConfig
from sim.control.event_based.hybrid import GoalAwareNativeMpcExecutor
from sim.simulator import SimulatorStepCounter


class EventBasedAlgorithmCompatibilityTests(unittest.TestCase):
    def test_residual_versions_complete_short_unified_physics_rollout(self):
        factories = {
            "v1": lambda: make_residual_gym_env(
                episode_hours=24,
                forecast_context_hours=24,
                hard_scenario_probability=0.0,
            ),
            "v2": lambda: make_masked_residual_gym_env(
                episode_hours=24,
                forecast_context_hours=24,
                hard_scenario_probability=0.0,
            ),
            "v3": lambda: make_risk_gated_gym_env(
                episode_hours=24,
                forecast_context_hours=24,
                hard_scenario_probability=0.0,
            ),
            "v4": lambda: make_tail_replay_gym_env(
                episode_hours=24,
                forecast_context_hours=24,
                initial_hard_probability=0.0,
                replay_probability=0.0,
            ),
        }

        for name, factory in factories.items():
            with self.subTest(version=name):
                env = factory()
                observation, _info = env.reset(seed=123)
                self.assertEqual(observation.shape, env.observation_space.shape)
                if hasattr(env, "action_masks"):
                    self.assertTrue(env.action_masks()[0])
                for _decision in range(32):
                    _observation, _reward, terminated, truncated, _info = env.step(0)
                    if terminated or truncated:
                        break
                self.assertTrue(terminated or truncated)
                env.close()

    def test_high_level_ppo_environment_completes_short_unified_physics_rollout(self):
        env = HighLevelDispatchGymEnv(
            make_high_level_native_env(
                episode_hours=24,
                decision_interval_h=24.0,
                event_triggered=True,
            )
        )

        observation, _info = env.reset(seed=123)
        self.assertEqual(observation.shape, (79,))
        self.assertEqual(env.action_space.n, 64)
        self.assertEqual(env.action_masks().shape, (64,))
        self.assertTrue(env.action_masks().all())
        episode_reward = 0.0
        for _decision in range(32):
            _observation, reward, terminated, truncated, info = env.step(0)
            episode_reward += float(reward)
            self.assertEqual(info["dispatch_goal"]["well_rate_indices"], {})
            if terminated or truncated:
                break

        self.assertTrue(terminated or truncated)
        native = env.env
        self.assertTrue(native.env.automatic_well_control)
        self.assertEqual(native.env.well_rate_action_dims, [])
        self.assertEqual(
            native.config.reward.objective,
            "realised_total_cost",
        )
        self.assertAlmostEqual(
            episode_reward,
            -native.config.reward.reward_scale
            * native.env.ledger.total_cost,
            places=9,
        )
        env.close()

    def test_high_level_factory_defaults_to_local_formal_scenario_protocol(self):
        env = make_high_level_native_env(episode_hours=48)

        self.assertEqual(env.env.scenario_generator.config.episode_hours, 217)
        self.assertEqual(env.env.scenario_generator.config.weather_process, "block")
        self.assertTrue(env.env.scenario_generator.config.warm_start)

    def test_high_level_ppo_stops_exactly_at_simulator_hour_budget(self):
        counter = SimulatorStepCounter()
        env = make_high_level_native_env(
            episode_hours=24,
            decision_interval_h=24.0,
            event_triggered=False,
            simulator_step_counter=counter,
            max_simulator_hour_steps=5,
        )
        env.reset(seed=123)

        _observation, reward, terminated, truncated, info = env.step(0)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["decision_trigger"], "simulator_budget_exhausted")
        self.assertEqual(info["native_steps"], 5)
        self.assertEqual(info["simulator_step_calls"], 5)
        self.assertEqual(info["simulator_hour_steps"], 5.0)
        self.assertEqual(info["simulator_budget_fraction"], 1.0)
        self.assertTrue(info["simulator_budget_exhausted"])
        self.assertAlmostEqual(
            reward,
            -env.config.reward.reward_scale * info["total_cost"],
            places=12,
        )

        env.reset(seed=456)
        _observation, _reward, _terminated, truncated, info = env.step(0)
        self.assertTrue(truncated)
        self.assertEqual(info["native_steps"], 0)
        self.assertEqual(counter.snapshot().hour_steps, 5.0)

    def test_high_level_factory_supports_unified_window_protocol(self):
        env = make_high_level_native_env(
            episode_hours=48,
            forecast_context_hours=168,
            weather_mode="window",
            scenario_protocol="unified_window_v1",
        )
        config = env.env.scenario_generator.normal.config

        self.assertEqual(config.episode_hours, 216)
        self.assertEqual(config.weather_process, "window")
        self.assertEqual(config.capture_noise_std, 0.30)
        self.assertEqual(config.capture_high_output_rate_per_week, 0.5)
        self.assertEqual(config.capture_high_output_mean_hours, 48.0)
        self.assertEqual(
            config.capture_high_output_multiplier_range,
            (1.25, 1.75),
        )
        self.assertEqual(config.weather_window_rate_per_week, 0.5)
        self.assertEqual(config.weather_window_mean_hours, 48.0)
        self.assertEqual(
            config.weather_window_speed_factor_range,
            (0.50, 0.80),
        )
        self.assertEqual(config.well_maintenance_rate_per_week, 0.3)
        self.assertEqual(config.well_maintenance_mean_hours, 12.0)
        self.assertEqual(config.emitter_initial_fill_range, (0.0, 0.50))
        self.assertEqual(config.terminal_initial_fill_range, (0.0, 0.50))
        self.assertEqual(
            config.reservoir_initial_pressure_fill_range,
            (0.0, 0.50),
        )

    def test_high_level_ppo_uses_shared_configurable_future_summary(self):
        expected_sizes = {
            (): 65,
            (24, 72): 79,
            (168,): 72,
            (24, 72, 168): 86,
        }
        action_counts = set()

        for windows_h, expected_size in expected_sizes.items():
            with self.subTest(windows_h=windows_h):
                env = make_high_level_native_env(
                    episode_hours=24,
                    forecast_context_hours=168,
                    future_summary_windows_h=windows_h,
                    weather_mode="window",
                    scenario_protocol="unified_window_v1",
                )
                observation = env.reset(seed=123)

                self.assertEqual(env.observation_size, expected_size)
                self.assertEqual(observation.shape, (expected_size,))
                action_counts.add(env.action_count)

        self.assertEqual(action_counts, {64})

    def test_high_level_factory_can_use_hybrid_mpc_executor(self):
        env = make_high_level_native_env(
            episode_hours=24,
            executor="mpc",
        )

        env.reset(seed=123)
        self.assertIsInstance(env.executor, GoalAwareNativeMpcExecutor)

    def test_v4_limits_each_override_window_to_one_intervention(self):
        env = make_tail_robust_native_env(
            episode_hours=48,
            forecast_context_hours=168,
            scenario_protocol="unified_window_v1",
            gate_mode="off",
            override_windows_h=((0.0, 47.0),),
        )
        env.reset(seed=123)
        self.assertTrue(env.env.automatic_well_control)
        self.assertEqual(env.env.well_rate_action_dims, [])
        intervention = next(
            index
            for index, allowed in enumerate(env.action_masks())
            if index > 0 and allowed
        )

        _observation, _reward, terminated, truncated, info = env.step(
            intervention
        )

        self.assertFalse(terminated or truncated)
        self.assertEqual(info["used_override_windows"], 1)
        self.assertFalse(env.action_masks()[1:].any())

    def test_v4_supports_configurable_future_summary_windows(self):
        expected_sizes = {
            (): 89,
            (24, 72): 103,
            (168,): 96,
            (24, 72, 168): 110,
        }
        action_counts = set()
        action_masks = set()

        for windows_h, expected_size in expected_sizes.items():
            with self.subTest(windows_h=windows_h):
                env = make_tail_robust_native_env(
                    episode_hours=24,
                    forecast_context_hours=168,
                    future_summary_windows_h=windows_h,
                    scenario_protocol="unified_window_v1",
                    hard_scenario_probability=0.0,
                )
                observation = env.reset(seed=123)

                self.assertEqual(env.observation_size, expected_size)
                self.assertEqual(observation.shape, (expected_size,))
                action_counts.add(len(env.action_masks()))
                action_masks.add(tuple(env.action_masks()))

        self.assertEqual(len(action_counts), 1)
        self.assertEqual(len(action_masks), 1)

    def test_v4_default_behavior_is_unchanged_when_budget_is_omitted(self):
        common = {
            "episode_hours": 24,
            "forecast_context_hours": 24,
            "hard_scenario_probability": 0.0,
            "gate_mode": "off",
        }
        original = make_tail_robust_native_env(**common)
        explicit_none = make_tail_robust_native_env(
            **common,
            max_simulator_hour_steps=None,
        )

        np.testing.assert_array_equal(
            original.reset(seed=123),
            explicit_none.reset(seed=123),
        )
        np.testing.assert_array_equal(
            original.action_masks(),
            explicit_none.action_masks(),
        )
        original_result = original.step(0)
        explicit_none_result = explicit_none.step(0)

        np.testing.assert_array_equal(
            original_result[0],
            explicit_none_result[0],
        )
        self.assertEqual(original_result[1:4], explicit_none_result[1:4])
        self.assertEqual(
            original_result[4]["total_cost"],
            explicit_none_result[4]["total_cost"],
        )

    def test_v4_budget_counts_actual_and_counterfactual_simulator_steps(self):
        counter = SimulatorStepCounter()
        env = make_tail_robust_native_env(
            episode_hours=24,
            forecast_context_hours=24,
            decision_interval_h=24.0,
            event_triggered=False,
            hard_scenario_probability=0.0,
            gate_mode="off",
            simulator_step_counter=counter,
            max_simulator_hour_steps=6,
        )
        env.reset(seed=123)

        _observation, _reward, terminated, truncated, info = env.step(0)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["decision_trigger"], "simulator_budget_exhausted")
        self.assertEqual(info["native_steps"], 3)
        self.assertEqual(info["simulator_step_calls"], 6)
        self.assertEqual(info["simulator_hour_steps"], 6.0)
        self.assertEqual(info["simulator_budget_fraction"], 1.0)
        self.assertTrue(info["simulator_budget_exhausted"])

    def test_v4_budget_never_spends_an_unpaired_final_hour(self):
        counter = SimulatorStepCounter()
        env = make_tail_robust_native_env(
            episode_hours=24,
            forecast_context_hours=24,
            decision_interval_h=24.0,
            event_triggered=False,
            hard_scenario_probability=0.0,
            gate_mode="off",
            simulator_step_counter=counter,
            max_simulator_hour_steps=5,
        )
        env.reset(seed=123)

        _observation, _reward, _terminated, truncated, info = env.step(0)

        self.assertTrue(truncated)
        self.assertEqual(info["native_steps"], 2)
        self.assertEqual(counter.snapshot().hour_steps, 4.0)
        self.assertEqual(info["simulator_budget_fraction"], 0.8)
        self.assertTrue(info["simulator_budget_exhausted"])

    def test_v4_objective_aligned_reward_is_counterfactual_cost_saving(self):
        reward_config = HighLevelRewardConfig.objective_aligned()
        env = make_tail_robust_native_env(
            episode_hours=24,
            forecast_context_hours=24,
            hard_scenario_probability=0.0,
            gate_mode="off",
            reward=reward_config,
        )
        env.reset(seed=123)

        _observation, reward, _terminated, _truncated, info = env.step(0)

        self.assertEqual(
            info["actual_reward_breakdown"]["objective"],
            "realised_total_cost",
        )
        self.assertAlmostEqual(
            reward,
            reward_config.reward_scale
            * info["total_cost_saving_eur"],
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
