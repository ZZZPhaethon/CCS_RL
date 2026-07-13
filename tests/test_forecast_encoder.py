import importlib
import importlib.util

import numpy as np
import pytest
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


def _gnn_extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name).GNNForecastExtractor


def _edge_gnn_extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name).EdgeGNNForecastExtractor


def _larger_mlp_extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name).LargerMLPForecastExtractor


def _tcn_env():
    native = make_native_env(
        episode_hours=2,
        scenario_context_hours=169,
        scenario="northern_lights_phase1_3vessels",
        weather_mode="block",
    )
    return ForecastGymEnv(native, "tcn")


def _gnn_env():
    native = make_native_env(
        episode_hours=2,
        scenario_context_hours=169,
        scenario="northern_lights_phase1_3vessels",
        weather_mode="block",
    )
    return ForecastGymEnv(native, "tcn_mode_destination")


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


def test_real_gnn_observation_batch_encodes_to_128_features():
    env = _gnn_env()
    observation, _ = env.reset(seed=4)
    batch = {
        key: torch.as_tensor(np.stack([value, value]))
        for key, value in observation.items()
    }

    extractor = _gnn_extractor_class()(env.observation_space)
    output = extractor(batch)

    assert batch["state"].shape == (2, 78)
    assert batch["forecast"].shape == (2, 168, 9)
    assert extractor.features_dim == 128
    assert output.shape == (2, 128)


def test_gnn_attention_parameters_receive_finite_gradients():
    observation_space = _gnn_env().observation_space
    extractor = _gnn_extractor_class()(observation_space)
    batch = {
        "state": torch.randn(2, *observation_space["state"].shape),
        "forecast": torch.randn(2, *observation_space["forecast"].shape),
    }

    extractor(batch).sum().backward()

    attention_layers = [
        module for module in extractor.modules() if isinstance(module, nn.MultiheadAttention)
    ]
    assert len(attention_layers) == 2
    for attention in attention_layers:
        for parameter in attention.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_gnn_mask_matches_ccs_logistics_graph():
    encoder = _gnn_extractor_class()(_gnn_env().observation_space).state_encoder
    allowed = ~encoder.attention_mask

    # Node order: 3 emitters, 3 vessels, terminal, well, reservoir.
    assert allowed[0, 3]
    assert allowed[3, 6]
    assert allowed[6, 7]
    assert allowed[7, 8]
    assert not allowed[0, 6]
    assert not allowed[3, 7]


def test_gnn_rejects_nonformal_current_state_layout():
    observation_space = spaces.Dict(
        {
            "state": spaces.Box(-1.0, 1.0, (77,), np.float32),
            "forecast": spaces.Box(-1.0, 1.0, (168, 9), np.float32),
        }
    )

    with np.testing.assert_raises_regex(ValueError, "78 current-state features"):
        _gnn_extractor_class()(observation_space)


@pytest.mark.parametrize(
    "extractor_class",
    [
        _larger_mlp_extractor_class,
        _edge_gnn_extractor_class,
    ],
)
def test_new_encoder_variants_produce_128_features_and_finite_gradients(
    extractor_class,
):
    env = _gnn_env()
    observation, _ = env.reset(seed=4)
    batch = {
        key: torch.as_tensor(np.stack([value, value]))
        for key, value in observation.items()
    }
    extractor = extractor_class()(env.observation_space)

    output = extractor(batch)
    output.sum().backward()

    assert batch["state"].shape == (2, 78)
    assert output.shape == (2, 128)
    for parameter in extractor.state_encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_edge_gnn_places_route_state_on_vessel_location_edges():
    encoder = _edge_gnn_extractor_class()(_gnn_env().observation_space).state_encoder
    state = torch.zeros(1, 78)
    travel_times = torch.arange(1.0, 13.0).reshape(3, 4)
    state[:, 39:51] = travel_times.reshape(1, -1)
    state[:, 11:32] = torch.tensor(
        [
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]
    )
    state[:, 66:78] = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0] * 3]
    )

    _nodes, edge_features, _globals = encoder._graph_inputs(state)

    destination_nodes = (6, 0, 1, 2)  # terminal, Brevik, Celsio, Yara
    expected_locations = (0, 1, 2)
    for vessel_index, vessel_node in enumerate(range(3, 6)):
        for slot, destination_node in enumerate(destination_nodes):
            forward = edge_features[0, destination_node, vessel_node, :3]
            reverse = edge_features[0, vessel_node, destination_node, :3]
            expected = torch.tensor(
                [
                    travel_times[vessel_index, slot],
                    float(slot == expected_locations[vessel_index]),
                    float(slot == 3),
                ]
            )
            torch.testing.assert_close(forward, expected)
            torch.testing.assert_close(reverse, expected)


def test_more_parameter_mlp_is_parameter_matched_to_edge_gnn_within_one_percent():
    observation_space = _gnn_env().observation_space
    edge_encoder = _edge_gnn_extractor_class()(observation_space).state_encoder
    mlp_encoder = _larger_mlp_extractor_class()(observation_space).state_encoder

    edge_parameters = sum(parameter.numel() for parameter in edge_encoder.parameters())
    mlp_parameters = sum(parameter.numel() for parameter in mlp_encoder.parameters())

    assert abs(edge_parameters - mlp_parameters) / edge_parameters < 0.01
    assert mlp_parameters > 5_056
