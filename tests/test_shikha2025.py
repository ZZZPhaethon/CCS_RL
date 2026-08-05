from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sim.control.shikha2025 import (
    Shikha2025Config,
    merge_subproblem_actions,
    projected_subgradient_update,
    shrinking_horizon_stages,
)


class Shikha2025Tests(unittest.TestCase):
    def test_paper_shrinking_windows_match_reported_subiteration_counts(self):
        self.assertEqual(
            shrinking_horizon_stages(240),
            ((0, 120), (60, 180), (120, 240)),
        )
        self.assertEqual(
            shrinking_horizon_stages(360),
            (
                (0, 120),
                (60, 180),
                (120, 240),
                (180, 300),
                (240, 360),
            ),
        )

    def test_configuration_rejects_invalid_window_and_tolerance(self):
        with self.assertRaises(ValueError):
            Shikha2025Config(active_window_h=60, fix_window_h=120)
        with self.assertRaises(ValueError):
            Shikha2025Config(tolerance_rel=0.0)

    def test_projected_subgradient_update_uses_equation_35_scaling(self):
        updated = projected_subgradient_update(
            {("source", 0): 2.0, ("terminal", 0): 1.0},
            {("source", 0): 1.0, ("terminal", 0): -2.0},
            objective_gap=10.0,
            step_size=1.0,
        )
        self.assertAlmostEqual(updated[("source", 0)], 4.0)
        self.assertEqual(updated[("terminal", 0)], 0.0)

    def test_merge_preserves_vessel_order_and_non_vessel_actions(self):
        env = SimpleNamespace(vessel_ids=["a", "b"])
        base = [
            {"vessels": [0, 0], "wells": [2]},
            {"vessels": [0, 0], "wells": [1]},
        ]
        results = {
            "a": SimpleNamespace(
                vessel_actions_by_hour={"a": [3, 4]}
            ),
            "b": SimpleNamespace(
                vessel_actions_by_hour={"b": [5, 6]}
            ),
        }
        with patch(
            "sim.control.shikha2025.greedy_warm_start_actions",
            return_value=base,
        ):
            merged = merge_subproblem_actions(env, 2, results)
        self.assertEqual(
            merged,
            [
                {"vessels": [3, 5], "wells": [2]},
                {"vessels": [4, 6], "wells": [1]},
            ],
        )


if __name__ == "__main__":
    unittest.main()
