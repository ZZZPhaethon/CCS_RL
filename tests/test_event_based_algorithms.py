import unittest

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
from sim.control.event_based.hybrid import GoalAwareNativeMpcExecutor


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
        self.assertEqual(env.action_space.n, 192)
        for _decision in range(32):
            _observation, _reward, terminated, truncated, _info = env.step(0)
            if terminated or truncated:
                break

        self.assertTrue(terminated or truncated)
        env.close()

    def test_high_level_factory_defaults_to_local_formal_scenario_protocol(self):
        env = make_high_level_native_env(episode_hours=48)

        self.assertEqual(env.env.scenario_generator.config.episode_hours, 217)
        self.assertEqual(env.env.scenario_generator.config.weather_process, "block")
        self.assertTrue(env.env.scenario_generator.config.warm_start)

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


if __name__ == "__main__":
    unittest.main()
