import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from gymnasium import Env, spaces
from torch import nn

from sim.control import imitation
from sim.environment.forecast_encoder import TCNForecastExtractor


class _StructuredEnv(Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(-10.0, 10.0, (51,), np.float32),
                "forecast": spaces.Box(-10.0, 10.0, (168, 9), np.float32),
            }
        )
        self.action_space = spaces.MultiDiscrete([2, 3])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._observation(), {}

    def step(self, action):
        return self._observation(), 0.0, False, True, {}

    def _observation(self):
        return {
            key: np.zeros(space.shape, dtype=np.float32)
            for key, space in self.observation_space.spaces.items()
        }


def _tcn_model():
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        "MultiInputPolicy",
        _StructuredEnv(),
        policy_kwargs={"features_extractor_class": TCNForecastExtractor},
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        seed=7,
        device="cpu",
        verbose=0,
    )


def _structured_demonstrations(n=4):
    rng = np.random.default_rng(11)
    observations = {
        "state": rng.normal(size=(n, 51)).astype(np.float32),
        "forecast": rng.normal(size=(n, 168, 9)).astype(np.float32),
    }
    observations["state"][:, 0] = np.arange(n)
    observations["forecast"][:, 0, 0] = np.arange(n)
    actions = np.column_stack((np.arange(n) % 2, np.arange(n) % 3)).astype(np.int64)
    masks = np.ones((n, 5), dtype=bool)
    weights = np.column_stack(
        (np.arange(n, dtype=np.float32) + 1.0, np.arange(n, dtype=np.float32) + 2.0)
    )
    return observations, actions, masks, weights


def _convolution_parameters(model):
    return [
        parameter
        for module in model.policy.features_extractor.forecast_convolutions
        if isinstance(module, nn.Conv1d)
        for parameter in module.parameters()
    ]


def test_index_observations_preserves_dictionary_keys_and_row_order():
    observations = {
        "state": np.arange(12).reshape(4, 3),
        "forecast": np.arange(24).reshape(4, 2, 3),
    }

    indexed = imitation._index_observations(observations, np.array([3, 1]))

    assert list(indexed) == ["state", "forecast"]
    assert indexed["state"].tolist() == [[9, 10, 11], [3, 4, 5]]
    assert indexed["forecast"].tolist() == [
        observations["forecast"][3].tolist(),
        observations["forecast"][1].tolist(),
    ]


def test_observation_count_accepts_aligned_dictionary():
    observations = {
        "state": np.zeros((3, 51)),
        "forecast": np.zeros((3, 168, 9)),
    }

    assert imitation._observation_count(observations) == 3


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        ({}, "empty"),
        (
            {"state": np.zeros((2, 51)), "forecast": np.zeros((3, 168, 9))},
            "leading dimension",
        ),
    ],
)
def test_observation_count_rejects_invalid_dictionary(observations, message):
    with pytest.raises(ValueError, match=message):
        imitation._observation_count(observations)


def test_tensor_observations_preserves_keys_dtype_and_device():
    observations = {
        "state": np.ones((2, 51), dtype=np.float64),
        "forecast": np.ones((2, 168, 9), dtype=np.int16),
    }

    tensors = imitation._tensor_observations(observations, torch.device("cpu"))

    assert list(tensors) == ["state", "forecast"]
    assert all(value.dtype == torch.float32 for value in tensors.values())
    assert all(value.device.type == "cpu" for value in tensors.values())


def test_observation_helpers_preserve_legacy_array_behavior():
    observations = np.arange(12, dtype=np.float64).reshape(4, 3)

    tensors = imitation._tensor_observations(observations, torch.device("cpu"))

    assert imitation._observation_count(tensors) == 4
    assert tensors.dtype == torch.float32
    assert imitation._index_observations(tensors, torch.tensor([2, 0])).tolist() == [
        [6.0, 7.0, 8.0],
        [0.0, 1.0, 2.0],
    ]


def test_structured_behavior_clone_updates_real_tcn_with_dimension_weights():
    torch.manual_seed(7)
    model = _tcn_model()
    observations, actions, masks, weights = _structured_demonstrations()
    parameters = _convolution_parameters(model)
    before = [parameter.detach().clone() for parameter in parameters]

    imitation.behavior_clone(
        model,
        observations,
        actions,
        masks=masks,
        weights=weights,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        log=False,
    )

    assert any(not torch.equal(old, new) for old, new in zip(before, parameters))
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())


def test_structured_kickstart_lifecycle_keeps_alignment_and_updates_real_tcn():
    torch.manual_seed(7)
    model = _tcn_model()
    observations, actions, masks, weights = _structured_demonstrations()
    callback = imitation.make_kickstart_callback(
        observations,
        actions,
        masks,
        weights,
        total_timesteps=10,
        n_batches=1,
        batch_size=2,
        lr=1e-3,
    )
    callback.init_callback(model)
    callback.on_training_start({}, {})
    parameters = _convolution_parameters(model)
    before = [parameter.detach().clone() for parameter in parameters]
    sampled_indices = torch.tensor([3, 1])
    real_log_probs = imitation._masked_action_log_probs
    captured = {}

    def capture_log_probs(policy, obs, batch_actions, action_masks=None):
        captured["obs"] = obs
        captured["actions"] = batch_actions
        captured["masks"] = action_masks
        return real_log_probs(policy, obs, batch_actions, action_masks)

    with (
        patch("torch.randint", return_value=sampled_indices),
        patch.object(imitation, "_masked_action_log_probs", side_effect=capture_log_probs),
    ):
        callback.on_rollout_end()

    expected = sampled_indices.numpy()
    assert captured["obs"]["state"][:, 0].tolist() == expected.tolist()
    assert captured["obs"]["forecast"][:, 0, 0].tolist() == expected.tolist()
    assert captured["actions"].tolist() == actions[expected].tolist()
    assert captured["masks"].tolist() == masks[expected].tolist()
    assert callback._w[sampled_indices].tolist() == weights[expected].tolist()
    assert any(not torch.equal(old, new) for old, new in zip(before, parameters))
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())


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
