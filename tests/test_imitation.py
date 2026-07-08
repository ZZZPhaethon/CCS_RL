import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sim.control import imitation


class ImitationTests(unittest.TestCase):
    def test_action_dimension_weights_only_upweight_nonwait_vessel_actions(self):
        actions = np.array(
            [
                [0, 1, 0, 2],
                [2, 0, 0, 1],
            ],
            dtype=np.int64,
        )

        weights = imitation.action_dimension_weights(actions, vessel_count=3, nonwait_weight=7.0)

        self.assertEqual(
            weights.tolist(),
            [
                [1.0, 7.0, 1.0, 1.0],
                [7.0, 1.0, 1.0, 1.0],
            ],
        )

    def test_bc_pretrain_passes_dimension_weights_to_behavior_clone(self):
        obs = np.zeros((2, 3), dtype=np.float32)
        acts = np.array([[0, 1, 0], [0, 0, 2]], dtype=np.int64)
        masks = np.ones((2, 5), dtype=bool)
        gym_env = SimpleNamespace(env=SimpleNamespace(vessel_ids=["ship_a", "ship_b"]))

        with (
            patch.object(imitation, "collect_demonstrations", return_value=(obs, acts, masks)),
            patch.object(imitation, "behavior_clone") as clone,
        ):
            _obs, _acts, _masks, weights = imitation.bc_pretrain(
                object(),
                gym_env,
                lambda _env: {},
                n_episodes=1,
                epochs=1,
                nonwait_weight=7.0,
            )

        self.assertEqual(weights.tolist(), [[1.0, 7.0, 1.0], [1.0, 1.0, 1.0]])
        self.assertEqual(clone.call_args.kwargs["weights"].tolist(), weights.tolist())


if __name__ == "__main__":
    unittest.main()
