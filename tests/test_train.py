import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import gymnasium  # noqa: F401

    HAVE_GYM = True
except ImportError:
    HAVE_GYM = False

from sim.environment import CCSEnv, CCSEnvConfig
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network


def _native_env() -> CCSEnv:
    return CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=2)),
        config=CCSEnvConfig(episode_hours=2),
    )


@unittest.skipUnless(HAVE_GYM, "gymnasium not installed")
class TrainPPOTests(unittest.TestCase):
    def test_train_ppo_builds_maskable_ppo_with_flat_multidiscrete_env(self):
        from sim import train

        captured = {}

        class FakeMaskablePPO:
            def __init__(self, policy, env, **kwargs):
                captured["policy"] = policy
                captured["env"] = env
                captured["kwargs"] = kwargs

            def learn(self, total_timesteps, **kwargs):
                captured["total_timesteps"] = total_timesteps
                captured["learn_kwargs"] = kwargs
                return self

        native = _native_env()
        fake_sb3_contrib = SimpleNamespace(MaskablePPO=FakeMaskablePPO)

        with (
            patch.dict(sys.modules, {"sb3_contrib": fake_sb3_contrib}),
            patch.object(train, "make_native_env", return_value=native) as make_native_env,
        ):
            model = train.train_ppo(
                total_timesteps=7,
                seed=11,
                gamma=0.95,
                episode_hours=2,
                warm_start=False,
                storage_shortfall_penalty=0.0,
                verbose=0,
                n_steps=4,
                batch_size=2,
            )

        self.assertIsInstance(model, FakeMaskablePPO)
        make_native_env.assert_called_once_with(
            episode_hours=2,
            warm_start=False,
            storage_shortfall_penalty=0.0,
            injection_reward_eur_per_t=0.0,
        )
        self.assertEqual(captured["policy"], "MlpPolicy")
        self.assertEqual(
            list(captured["env"].action_space.nvec),
            native.vessel_action_dims + native.well_rate_action_dims,
        )
        self.assertEqual(captured["kwargs"]["seed"], 11)
        self.assertEqual(captured["kwargs"]["gamma"], 0.95)
        self.assertEqual(captured["kwargs"]["verbose"], 0)
        self.assertEqual(captured["kwargs"]["n_steps"], 4)
        self.assertEqual(captured["kwargs"]["batch_size"], 2)
        self.assertEqual(captured["kwargs"]["device"], "auto")
        self.assertEqual(captured["total_timesteps"], 7)
        self.assertEqual(captured["learn_kwargs"]["progress_bar"], False)

    def test_make_native_env_passes_stress_disturbance_config(self):
        from sim import train

        captured = {}
        sentinel_env = object()

        def fake_build_phase1_env(**kwargs):
            captured.update(kwargs)
            return sentinel_env

        with patch.object(train, "build_phase1_env", side_effect=fake_build_phase1_env):
            env = train.make_native_env(
                episode_hours=720,
                capture_noise_std=0.50,
                initial_inventory_fill_max=0.80,
                leg_wave_slowdown_multiplier=2.0,
                leg_wave_speed_factor_floor=0.25,
            )

        self.assertIs(env, sentinel_env)
        scenario_config = captured["scenario_config"]
        self.assertEqual(captured["weather_mode"], "window")
        self.assertEqual(scenario_config.capture_noise_std, 0.50)
        self.assertEqual(scenario_config.emitter_initial_fill_range, (0.0, 0.80))
        self.assertEqual(scenario_config.terminal_initial_fill_range, (0.0, 0.80))
        self.assertEqual(scenario_config.reservoir_initial_pressure_fill_range, (0.0, 0.80))
        self.assertEqual(scenario_config.leg_wave_slowdown_multiplier, 2.0)
        self.assertEqual(scenario_config.leg_wave_speed_factor_floor, 0.25)

    def test_make_native_env_separates_episode_length_from_scenario_context(self):
        from sim import train

        captured = {}

        def fake_build_phase1_env(**kwargs):
            captured.update(kwargs)
            return object()

        with patch.object(train, "build_phase1_env", side_effect=fake_build_phase1_env):
            train.make_native_env(episode_hours=720, scenario_context_hours=169)

        self.assertEqual(captured["config"].episode_hours, 720)
        self.assertEqual(captured["scenario_config"].episode_hours, 889)

    def test_make_native_env_preserves_existing_positional_argument_order(self):
        from sim import train

        captured = {}

        def fake_build_phase1_env(**kwargs):
            captured.update(kwargs)
            return object()

        with patch.object(train, "build_phase1_env", side_effect=fake_build_phase1_env):
            train.make_native_env(720, 0.95)

        self.assertEqual(captured["config"].storage_target_rate, 0.95)
        self.assertEqual(captured["scenario_config"].episode_hours, 888)


if __name__ == "__main__":
    unittest.main()
