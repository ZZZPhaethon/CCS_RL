from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import compare_reward_modes_bc as compare
from sim.entities import Emitter


class _Metric:
    storage_rate = 0.5
    loss_rate = 0.1
    stored_t = 100.0
    vented_t = 10.0
    operating_cost = 20.0
    vent_penalty = 30.0


class CompareRewardModesTests(unittest.TestCase):
    def test_eval_baselines_includes_idle_and_greedy_teacher_rows(self):
        args = SimpleNamespace(eval_seeds=[1, 2])

        with (
            patch.object(compare, "make_env", return_value=object()),
            patch.object(compare, "run_episode", return_value=_Metric()),
        ):
            rows = compare.eval_baselines(args)

        self.assertEqual([row["policy"] for row in rows], ["idle", "greedy_teacher"])
        self.assertEqual(rows[0]["actual_total_cost"], 50.0)
        self.assertEqual(rows[0]["actual_cost_per_stored_t"], 0.5)

    def test_yara_buffer_override_replaces_frozen_emitter(self):
        emitter = Emitter("yara_sluiskil", nominal_capture_tph=100.0, buffer_capacity_t=15_000.0)
        env = SimpleNamespace(network=SimpleNamespace(entities={"yara_sluiskil": emitter}))
        args = SimpleNamespace(yara_buffer_capacity=7_500.0)

        compare.apply_yara_buffer_capacity_override(env, args)

        self.assertEqual(env.network.entities["yara_sluiskil"].buffer_capacity_t, 7_500.0)
        self.assertEqual(emitter.buffer_capacity_t, 15_000.0)

    def test_yara_buffer_tag_marks_output_artifacts(self):
        self.assertEqual(
            compare.yara_buffer_tag(SimpleNamespace(yara_buffer_capacity=7_500.0)),
            "_yara7500",
        )


if __name__ == "__main__":
    unittest.main()
