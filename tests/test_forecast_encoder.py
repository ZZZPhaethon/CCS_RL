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


def _stable_tcn_extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name).StableTCNForecastExtractor


def _fixed_scale_tcn_extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name).FixedScaleTCNForecastExtractor


def _future_mlp_extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name).FutureMLPForecastExtractor


def _past_mlp_extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name).PastMLPForecastExtractor


def _gated_past_mlp_extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name).GatedPastMLPForecastExtractor


def _balanced_edge_gnn_extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name).BalancedEdgeGNNForecastExtractor


def _balanced_edge_gnn_future_mlp_extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(
        module_name
    ).BalancedEdgeGNNFutureMLPForecastExtractor


def _future_conditioned_edge_gnn_extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(
        module_name
    ).FutureConditionedEdgeGNNForecastExtractor


def _fixed_scale_larger_mlp_extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name).FixedScaleLargerMLPForecastExtractor


def _fixed_scale_edge_gnn_extractor_class():
    module_name = "sim.environment.forecast_encoder"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name).FixedScaleEdgeGNNForecastExtractor


def _tcn_env():
    native = make_native_env(
        episode_hours=2,
        scenario_context_hours=169,
        scenario="northern_lights_phase1_3vessels",
        weather_mode="block",
    )
    return ForecastGymEnv(native, "tcn")


def _destination_env(variant):
    native = make_native_env(
        episode_hours=2,
        scenario_context_hours=169,
        scenario="northern_lights_phase1_3vessels",
        weather_mode="block",
    )
    return ForecastGymEnv(native, variant)


def _gnn_env():
    return _destination_env("tcn_mode_destination")


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
    env = _destination_env("tcn_mode_destination")
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
    encoder = _edge_gnn_extractor_class()(
        _destination_env("tcn_mode_destination").observation_space
    ).state_encoder
    state = torch.zeros(1, 78)
    travel_times = torch.arange(1.0, 13.0).reshape(3, 4)
    state[:, 39:51] = travel_times.reshape(1, -1)
    state[:, 11:32] = torch.tensor(
        [
            0.0, 0.0, 1.0, 0.25, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.50, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.75, 0.0, 1.0, 0.0,
        ]
    )
    state[:, 66:78] = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0] * 3]
    )

    _nodes, edge_features, _globals = encoder._graph_inputs(state)

    destination_nodes = (6, 0, 1, 2)  # terminal, Brevik, Celsio, Yara
    expected_locations = (0, 1, 2)
    progress = (0.25, 0.50, 0.75)
    for vessel_index, vessel_node in enumerate(range(3, 6)):
        for slot, destination_node in enumerate(destination_nodes):
            forward = edge_features[0, destination_node, vessel_node, :3]
            reverse = edge_features[0, vessel_node, destination_node, :3]
            expected = torch.tensor(
                [
                    travel_times[vessel_index, slot]
                    * (1.0 - progress[vessel_index] * float(slot == 3)),
                    float(slot == expected_locations[vessel_index]),
                    float(slot == 3),
                ]
            )
            torch.testing.assert_close(forward, expected)
            torch.testing.assert_close(reverse, expected)


def test_more_parameter_mlp_is_parameter_matched_to_edge_gnn_within_one_percent():
    observation_space = _destination_env("tcn_mode_destination").observation_space
    edge_encoder = _edge_gnn_extractor_class()(observation_space).state_encoder
    mlp_encoder = _larger_mlp_extractor_class()(observation_space).state_encoder

    edge_parameters = sum(parameter.numel() for parameter in edge_encoder.parameters())
    mlp_parameters = sum(parameter.numel() for parameter in mlp_encoder.parameters())

    assert abs(edge_parameters - mlp_parameters) / edge_parameters < 0.01
    assert mlp_parameters > 5_056


def test_stable_tcn_forecast_path_is_active_and_receives_gradients():
    torch.manual_seed(11)
    observation_space = _destination_env(
        "stable_tcn_mode_destination"
    ).observation_space
    extractor = _stable_tcn_extractor_class()(observation_space)
    batch = {
        "state": torch.randn(4, *observation_space["state"].shape),
        "forecast": torch.randn(4, *observation_space["forecast"].shape),
    }

    forecast_features = extractor(batch)[:, 64:]
    forecast_features.square().mean().backward()

    assert any(isinstance(module, nn.LayerNorm) for module in extractor.forecast_projection)
    assert not torch.allclose(forecast_features, torch.zeros_like(forecast_features))
    for parameter in extractor.forecast_convolutions.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_fixed_scale_tcn_has_non_affine_forecast_normalization():
    observation_space = _destination_env(
        "fixed_scale_tcn_mode_destination"
    ).observation_space
    extractor = _fixed_scale_tcn_extractor_class()(observation_space)
    normalizations = [
        module
        for module in extractor.forecast_projection
        if isinstance(module, nn.LayerNorm)
    ]

    assert len(normalizations) == 1
    assert not normalizations[0].elementwise_affine
    assert sum(parameter.numel() for parameter in extractor.parameters()) == 59_904


def test_future_mlp_is_parameter_matched_and_keeps_forecast_gradients_active():
    torch.manual_seed(11)
    observation_space = _destination_env(
        "future_mlp_mode_destination"
    ).observation_space
    mlp = _future_mlp_extractor_class()(observation_space)
    tcn = _fixed_scale_tcn_extractor_class()(observation_space)
    batch = {
        "state": torch.randn(4, *observation_space["state"].shape),
        "forecast": torch.randn(4, *observation_space["forecast"].shape),
    }

    forecast_features = mlp(batch)[:, 64:]
    forecast_features.square().mean().backward()

    mlp_forecast_parameters = sum(
        parameter.numel() for parameter in mlp.forecast_projection.parameters()
    )
    tcn_forecast_parameters = sum(
        parameter.numel()
        for module in (tcn.forecast_convolutions, tcn.forecast_projection)
        for parameter in module.parameters()
    )
    assert abs(mlp_forecast_parameters - tcn_forecast_parameters) / tcn_forecast_parameters < 0.01
    assert forecast_features.shape == (4, 64)
    assert torch.count_nonzero(forecast_features) > 0
    normalizations = [
        module
        for module in mlp.forecast_projection
        if isinstance(module, nn.LayerNorm)
    ]
    assert len(normalizations) == 1
    assert not normalizations[0].elementwise_affine
    for parameter in mlp.forecast_projection.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_past_mlp_encodes_three_modalities_and_keeps_history_gradients_active():
    torch.manual_seed(11)
    observation_space = _destination_env(
        "past24_mlp_mode_destination"
    ).observation_space
    extractor = _past_mlp_extractor_class()(observation_space)
    batch = {
        key: torch.randn(4, *space.shape)
        for key, space in observation_space.spaces.items()
    }

    features = extractor(batch)
    features[:, 64:128].square().mean().backward()

    assert batch["state"].shape == (4, 78)
    assert batch["past"].shape == (4, 24, 83)
    assert batch["forecast"].shape == (4, 168, 9)
    assert extractor.features_dim == 192
    assert features.shape == (4, 192)
    for parameter in extractor.past_projection.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_gated_past_mlp_starts_as_exact_baseline_then_opens_history_path():
    torch.manual_seed(11)
    observation_space = _destination_env(
        "gated_past24_mlp_mode_destination"
    ).observation_space
    extractor = _gated_past_mlp_extractor_class()(observation_space)
    batch = {
        key: torch.randn(4, *space.shape)
        for key, space in observation_space.spaces.items()
    }

    features = extractor(batch)
    baseline = torch.cat(
        (
            extractor.state_encoder(batch["state"]),
            extractor.forecast_projection(batch["forecast"]),
        ),
        dim=1,
    )

    torch.testing.assert_close(features, baseline)
    assert extractor.features_dim == 128
    torch.testing.assert_close(extractor.past_gate, torch.zeros(128))
    features.square().mean().backward()
    assert extractor.past_gate.grad is not None
    assert torch.count_nonzero(extractor.past_gate.grad) > 0

    extractor.zero_grad()
    with torch.no_grad():
        extractor.past_gate.fill_(0.1)
    extractor(batch).square().mean().backward()
    for parameter in extractor.past_projection.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_balanced_edge_gnn_normalizes_state_without_adding_parameters():
    torch.manual_seed(11)
    observation_space = _destination_env(
        "balanced_edge_gnn_mode_destination"
    ).observation_space
    balanced = _balanced_edge_gnn_extractor_class()(observation_space)
    unbalanced = _fixed_scale_edge_gnn_extractor_class()(observation_space)
    batch = {
        "state": torch.randn(4, *observation_space["state"].shape),
        "forecast": torch.randn(4, *observation_space["forecast"].shape),
    }

    features = balanced(batch)
    state_l2 = torch.linalg.vector_norm(features[:, :64], dim=1).mean()
    forecast_l2 = torch.linalg.vector_norm(features[:, 64:], dim=1).mean()
    normalizations = [
        module
        for module in balanced.state_encoder
        if isinstance(module, nn.LayerNorm)
    ]

    assert len(normalizations) == 1
    assert not normalizations[0].elementwise_affine
    assert 0.25 < float(state_l2 / forecast_l2) < 4.0
    assert sum(parameter.numel() for parameter in balanced.parameters()) == sum(
        parameter.numel() for parameter in unbalanced.parameters()
    )


def test_balanced_edge_gnn_future_mlp_keeps_both_modalities_active():
    torch.manual_seed(11)
    observation_space = _destination_env(
        "balanced_edge_gnn_future_mlp_mode_destination"
    ).observation_space
    extractor = _balanced_edge_gnn_future_mlp_extractor_class()(
        observation_space
    )
    batch = {
        "state": torch.randn(4, *observation_space["state"].shape),
        "forecast": torch.randn(4, *observation_space["forecast"].shape),
    }

    features = extractor(batch)
    features[:, 64:].square().mean().backward()
    state_l2 = torch.linalg.vector_norm(features[:, :64], dim=1).mean()
    forecast_l2 = torch.linalg.vector_norm(features[:, 64:], dim=1).mean()

    assert features.shape == (4, 128)
    assert 0.25 < float(state_l2 / forecast_l2) < 4.0
    assert sum(parameter.numel() for parameter in extractor.parameters()) == 92_763
    assert not hasattr(extractor, "forecast_convolutions")
    for parameter in extractor.forecast_projection.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_future_conditioned_edge_gnn_routes_every_forecast_channel_into_graph():
    torch.manual_seed(11)
    observation_space = _destination_env(
        "future_conditioned_edge_gnn_mode_destination"
    ).observation_space
    extractor = _future_conditioned_edge_gnn_extractor_class()(observation_space)
    state = torch.randn(4, *observation_space["state"].shape)
    forecast = torch.randn(
        4,
        *observation_space["forecast"].shape,
        requires_grad=True,
    )

    features = extractor({"state": state, "forecast": forecast})
    features.square().mean().backward()

    assert state.shape == (4, 78)
    assert forecast.shape == (4, 168, 9)
    assert features.shape == (4, 128)
    assert forecast.grad is not None
    channel_gradient = forecast.grad.abs().sum(dim=(0, 1))
    assert torch.all(channel_gradient > 0.0)
    for encoder in (
        extractor.graph_encoder.emitter_future_encoder,
        extractor.graph_encoder.well_future_encoder,
        extractor.graph_encoder.weather_future_encoder,
    ):
        for parameter in encoder.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    ("class_name", "variant"),
    [
        (
            "GatedResidualEdgeGNNForecastExtractor",
            "gated_residual_edge_gnn_mode_destination",
        ),
        (
            "EntityResidualEdgeGNNForecastExtractor",
            "entity_residual_edge_gnn_mode_destination",
        ),
    ],
)
def test_residual_fusions_preserve_direct_future_path(class_name, variant):
    torch.manual_seed(11)
    module = importlib.import_module("sim.environment.forecast_encoder")
    observation_space = _destination_env(variant).observation_space
    extractor = getattr(module, class_name)(observation_space)
    state = torch.randn(4, *observation_space["state"].shape)
    forecast = torch.randn(
        4,
        *observation_space["forecast"].shape,
        requires_grad=True,
    )

    features = extractor({"state": state, "forecast": forecast})
    expected_future = extractor.forecast_projection(
        extractor.forecast_convolutions(forecast.transpose(1, 2))
    )
    features.square().mean().backward()

    assert features.shape == (4, 128)
    torch.testing.assert_close(features[:, 64:].detach(), expected_future.detach())
    assert forecast.grad is not None
    assert torch.all(forecast.grad.abs().sum(dim=(0, 1)) > 0.0)


@pytest.mark.parametrize(
    "extractor_class",
    [
        _fixed_scale_larger_mlp_extractor_class,
        _fixed_scale_edge_gnn_extractor_class,
    ],
)
def test_fixed_scale_state_encoder_combinations_keep_forecast_gradients_active(
    extractor_class,
):
    torch.manual_seed(11)
    observation_space = _destination_env(
        "fixed_scale_tcn_mode_destination"
    ).observation_space
    extractor = extractor_class()(observation_space)
    batch = {
        "state": torch.randn(4, *observation_space["state"].shape),
        "forecast": torch.randn(4, *observation_space["forecast"].shape),
    }

    forecast_features = extractor(batch)[:, 64:]
    forecast_features.square().mean().backward()

    normalizations = [
        module
        for module in extractor.forecast_projection
        if isinstance(module, nn.LayerNorm)
    ]
    assert len(normalizations) == 1
    assert not normalizations[0].elementwise_affine
    assert torch.count_nonzero(forecast_features) > 0
    for parameter in extractor.forecast_convolutions.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_fixed_scale_larger_mlp_remains_parameter_matched_to_fixed_scale_edge_gnn():
    observation_space = _destination_env(
        "fixed_scale_tcn_mode_destination"
    ).observation_space
    edge_encoder = _fixed_scale_edge_gnn_extractor_class()(
        observation_space
    ).state_encoder
    mlp_encoder = _fixed_scale_larger_mlp_extractor_class()(
        observation_space
    ).state_encoder

    edge_parameters = sum(parameter.numel() for parameter in edge_encoder.parameters())
    mlp_parameters = sum(parameter.numel() for parameter in mlp_encoder.parameters())

    assert abs(edge_parameters - mlp_parameters) / edge_parameters < 0.01
