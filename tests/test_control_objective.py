import unittest
from types import SimpleNamespace

from sim.control.objective import control_objective_value, control_objective_weights
from sim.economics import EconomicParameters


class ControlObjectiveWeightsTests(unittest.TestCase):
    def test_economic_mode_matches_rl_reward_weights(self):
        env = SimpleNamespace(
            config=SimpleNamespace(
                reward_mode="economic",
                vent_penalty_weight=2.0,
                operating_cost_weight=3.0,
                store_reward_eur_per_t=125.0,
                injection_reward_eur_per_t=0.0,
                vent_first_vent_eur_per_t=10_000.0,
                overflow_risk_eur_per_t=100.0,
                overflow_risk_lookahead_h=24.0,
            )
        )

        weights = control_objective_weights(env, EconomicParameters(carbon_price_eur_per_t=80.0))

        self.assertEqual(weights.mode, "economic")
        self.assertEqual(weights.vent_eur_per_t, 160.0)
        self.assertEqual(weights.operating_cost_weight, 3.0)
        self.assertEqual(weights.storage_reward_eur_per_t, 125.0)
        self.assertEqual(weights.overflow_risk_eur_per_t, 0.0)
        self.assertEqual(
            control_objective_value(
                weights,
                operating_cost=10.0,
                vented_t=2.0,
                stored_t=3.0,
            ),
            3.0 * 10.0 + 160.0 * 2.0 - 125.0 * 3.0,
        )

    def test_vent_first_mode_matches_rl_and_ignores_storage_credit(self):
        env = SimpleNamespace(
            config=SimpleNamespace(
                reward_mode="vent_first",
                vent_penalty_weight=2.0,
                operating_cost_weight=1.5,
                store_reward_eur_per_t=125.0,
                injection_reward_eur_per_t=80.0,
                vent_first_vent_eur_per_t=10_000.0,
                overflow_risk_eur_per_t=100.0,
                overflow_risk_lookahead_h=36.0,
            )
        )

        weights = control_objective_weights(
            env,
            EconomicParameters(carbon_price_eur_per_t=80.0),
            storage_reward_eur_per_t=999.0,
        )

        self.assertEqual(weights.mode, "vent_first")
        self.assertEqual(weights.vent_eur_per_t, 10_000.0)
        self.assertEqual(weights.operating_cost_weight, 1.5)
        self.assertEqual(weights.storage_reward_eur_per_t, 0.0)
        self.assertEqual(weights.overflow_risk_eur_per_t, 100.0)
        self.assertEqual(weights.overflow_risk_lookahead_h, 36.0)
        self.assertEqual(
            control_objective_value(
                weights,
                operating_cost=10.0,
                vented_t=2.0,
                stored_t=3.0,
                overflow_risk_t=4.0,
            ),
            1.5 * 10.0 + 10_000.0 * 2.0 + 100.0 * 4.0,
        )


if __name__ == "__main__":
    unittest.main()
