import unittest

from sim.control.baselines import greedy_shuttle_policy, idle_policy
from sim.environment import CCSEnv, CCSEnvConfig
from sim.metrics import (
    EpisodeMetrics,
    aggregate_metrics,
    evaluate,
    run_episode,
)
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network


def _env(
    episode_hours: int = 48,
    scenario_config: ScenarioConfig | None = None,
    **config,
) -> CCSEnv:
    scenario_config = scenario_config or ScenarioConfig(episode_hours=episode_hours)
    return CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(config=scenario_config),
        config=CCSEnvConfig(episode_hours=episode_hours, **config),
    )


class RunEpisodeTests(unittest.TestCase):
    def test_returns_metrics_consistent_with_env(self):
        env = _env()
        metrics = run_episode(env, greedy_shuttle_policy, seed=1)
        self.assertIsInstance(metrics, EpisodeMetrics)
        self.assertAlmostEqual(metrics.storage_rate, env.storage_rate())
        self.assertAlmostEqual(metrics.stored_t, env.cumulative_stored_t)
        self.assertAlmostEqual(metrics.net, env.ledger.net)
        self.assertAlmostEqual(metrics.total_cost, env.ledger.total_cost)
        self.assertAlmostEqual(metrics.storage_shortfall_penalty, env.ledger.storage_shortfall_penalty)
        self.assertAlmostEqual(metrics.vessel_fuel, env.ledger.vessel_fuel)
        self.assertAlmostEqual(metrics.conditioning, env.ledger.conditioning)
        self.assertAlmostEqual(metrics.reconditioning, env.ledger.reconditioning)
        self.assertAlmostEqual(metrics.loading, env.ledger.loading)
        self.assertAlmostEqual(metrics.unloading, env.ledger.unloading)
        self.assertAlmostEqual(metrics.total_cost_per_stored_t, metrics.total_cost / metrics.stored_t)
        self.assertEqual(metrics.horizon_hours, 48)

    def test_kpis_are_in_sensible_ranges(self):
        metrics = run_episode(_env(), greedy_shuttle_policy, seed=3)
        self.assertTrue(0.0 <= metrics.storage_rate <= 1.0)
        self.assertGreaterEqual(metrics.vented_t, 0.0)
        self.assertGreater(metrics.operating_cost, 0.0)
        self.assertGreaterEqual(metrics.throttle_hours, 0)
        self.assertTrue(0.0 <= metrics.min_pressure_margin_fraction <= 1.0)
        self.assertGreaterEqual(metrics.longest_venting_streak_hours, 0)

    def test_idle_policy_runs_minimum_injection_and_vents(self):
        # A long horizon guarantees the emitter buffers overflow under idling,
        # while wells still drain any terminal inventory at their minimum rate.
        metrics = run_episode(_env(episode_hours=168), idle_policy, seed=5)
        self.assertGreater(metrics.stored_t, 0.0)
        self.assertGreater(metrics.storage_rate, 0.0)
        self.assertGreater(metrics.vented_t, 0.0)
        self.assertIsNotNone(metrics.cost_per_stored_t)

    def test_shuttle_beats_idle_on_storage(self):
        idle = run_episode(_env(episode_hours=168), idle_policy, seed=7)
        shuttle = run_episode(_env(episode_hours=168), greedy_shuttle_policy, seed=7)
        self.assertGreater(shuttle.stored_t, idle.stored_t)
        self.assertLess(shuttle.vented_t, idle.vented_t)

    def test_deterministic_for_seed_and_policy(self):
        a = run_episode(_env(), greedy_shuttle_policy, seed=11).as_dict()
        b = run_episode(_env(), greedy_shuttle_policy, seed=11).as_dict()
        self.assertEqual(a, b)

    def test_in_transit_growth_obeys_mass_balance(self):
        # In-transit inventory grows by exactly captured - stored - vented.
        m = run_episode(_env(), greedy_shuttle_policy, seed=4)
        self.assertAlmostEqual(m.in_transit_growth_t, m.captured_t - m.stored_t - m.vented_t, places=3)

    def test_idle_accumulates_in_transit_inventory_without_losing_co2(self):
        # Idling stores nothing, so captured CO2 piles up in buffers but is not lost
        # in a short episode (loss rate ~ 0).
        quiet = ScenarioConfig(
            episode_hours=48,
            capture_noise_std=0.0,
            capture_outage_rate_per_week=0.0,
            randomize_initial_inventory=False,
        )
        m = run_episode(_env(scenario_config=quiet), idle_policy, seed=6)
        self.assertGreater(m.in_transit_growth_t, 0.0)
        self.assertEqual(m.loss_rate, 0.0)
        self.assertFalse(any("backlog" in key for key in m.as_dict()))

    def test_shuttle_grows_in_transit_inventory_less_than_idle(self):
        idle = run_episode(_env(), idle_policy, seed=8)
        shuttle = run_episode(_env(), greedy_shuttle_policy, seed=8)
        self.assertLess(shuttle.in_transit_growth_t, idle.in_transit_growth_t)


class HorizonModeTests(unittest.TestCase):
    def test_storage_goal_config_is_not_supported(self):
        with self.assertRaises(TypeError):
            CCSEnvConfig(episode_hours=72, storage_goal_t=2_000.0)

    def test_episode_runs_to_horizon_without_goal_termination(self):
        env = _env(episode_hours=72)
        m = run_episode(env, greedy_shuttle_policy, seed=1)

        self.assertEqual(m.elapsed_hours, 72)
        self.assertFalse(hasattr(m, "reached_target"))

    def test_step_end_is_time_limit_truncation(self):
        env = _env(episode_hours=4)
        env.reset(seed=1)
        terminated = truncated = False
        while not (terminated or truncated):
            _o, _r, terminated, truncated, _i = env.step(greedy_shuttle_policy(env))

        self.assertFalse(terminated)
        self.assertTrue(truncated)


class AggregateTests(unittest.TestCase):
    def test_evaluate_returns_per_episode_and_summary(self):
        episodes, summary = evaluate(_env(), greedy_shuttle_policy, seeds=[1, 2, 3])
        self.assertEqual(len(episodes), 3)
        self.assertIn("storage_rate", summary)
        self.assertIn("mean", summary["storage_rate"])
        self.assertIn("std", summary["storage_rate"])

    def test_aggregate_handles_single_episode(self):
        episodes = [run_episode(_env(), greedy_shuttle_policy, seed=1)]
        summary = aggregate_metrics(episodes)
        self.assertEqual(summary["storage_rate"]["std"], 0.0)

    def test_aggregate_skips_none_valued_fields(self):
        # Empty/no-storage records yield cost_per_stored_t = None; aggregation must not crash.
        summary = aggregate_metrics([EpisodeMetrics(cost_per_stored_t=None)])
        self.assertNotIn("cost_per_stored_t", summary)

    def test_report_renders_all_sections(self):
        text = run_episode(_env(), greedy_shuttle_policy, seed=2).report()
        for token in ("storage rate", "operating cost", "throttle hours", "pressure-risk"):
            self.assertIn(token, text)
        self.assertNotIn("revenue", text)
        self.assertNotIn("backlog", text)
        self.assertNotIn("goal", text.lower())

    def test_metrics_do_not_expose_storage_revenue(self):
        metrics = run_episode(_env(), greedy_shuttle_policy, seed=2)
        self.assertNotIn("revenue_storage", metrics.as_dict())


if __name__ == "__main__":
    unittest.main()
