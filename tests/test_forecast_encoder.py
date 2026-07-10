import importlib
import importlib.util

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from sim.environment.forecast_gym import ForecastGymEnv
from sim.train import make_native_env


def _extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None, (
        "TCN forecast encoder module has not been implemented"
    )
    return importlib.import_module(module_name).TCNForecastExtractor


def _tcn_env():
    native = make_native_env(
        episode_hours=2,
        scenario_context_hours=169,
        scenario="northern_lights_phase1_3vessels",
        weather_mode="block",
    )
    return ForecastGymEnv(native, "tcn")


def test_real_tcn_observation_batch_encodes_to_128_features():
    env = _tcn_env()
    observation, _ = env.reset(seed=4)
    batch = {
        key: torch.as_tensor(np.stack([value, value]))
        for key, value in observation.items()
    }

    extractor = _extractor_class()(env.observation_space)
    output = extractor(batch)

    assert batch["forecast"].shape == (2, 168, 9)
    assert extractor.features_dim == 128
    assert output.shape == (2, 128)


def test_convolution_parameters_receive_finite_gradients():
    observation_space = _tcn_env().observation_space
    extractor = _extractor_class()(observation_space)
    batch = {
        "state": torch.randn(2, *observation_space["state"].shape),
        "forecast": torch.randn(2, *observation_space["forecast"].shape),
    }

    extractor(batch).sum().backward()

    convolutions = [
        module for module in extractor.modules() if isinstance(module, nn.Conv1d)
    ]
    assert len(convolutions) == 3
    for convolution in convolutions:
        for parameter in convolution.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_changing_forecast_changes_encoded_output_with_fixed_state():
    torch.manual_seed(7)
    observation_space = _tcn_env().observation_space
    extractor = _extractor_class()(observation_space)
    state = torch.randn(2, *observation_space["state"].shape)
    forecast = torch.randn(2, *observation_space["forecast"].shape)

    baseline = extractor({"state": state, "forecast": forecast})
    changed = extractor({"state": state, "forecast": forecast + 1.0})

    assert not torch.allclose(baseline, changed)


def test_architecture_and_input_dimensions_come_from_observation_space():
    observation_space = spaces.Dict(
        {
            "state": spaces.Box(-1.0, 1.0, (7,), np.float32),
            "forecast": spaces.Box(-1.0, 1.0, (33, 4), np.float32),
        }
    )
    extractor = _extractor_class()(observation_space)

    convolutions = [
        module for module in extractor.modules() if isinstance(module, nn.Conv1d)
    ]
    assert [
        (
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
        )
        for layer in convolutions
    ] == [
        (4, 32, (5,), (2,), (2,)),
        (32, 32, (5,), (2,), (2,)),
        (32, 32, (5,), (2,), (2,)),
    ]
    linear_shapes = {
        (module.in_features, module.out_features)
        for module in extractor.modules()
        if isinstance(module, nn.Linear)
    }
    assert (7, 64) in linear_shapes
    assert (32 * 5, 64) in linear_shapes
