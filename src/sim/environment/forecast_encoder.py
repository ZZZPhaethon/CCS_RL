"""Stable-Baselines3 feature extractor for structured forecast observations."""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class TCNForecastExtractor(BaseFeaturesExtractor):
    """Encode current state and time-major forecast tensors separately."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__(
            observation_space,
            features_dim=state_features + forecast_features,
        )
        state_size = observation_space["state"].shape[0]
        forecast_steps, forecast_channels = observation_space["forecast"].shape

        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, state_features),
            nn.ReLU(),
        )
        self.forecast_convolutions = nn.Sequential(
            nn.Conv1d(forecast_channels, 32, kernel_size=5, stride=2, padding=2),
            self._forecast_activation(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            self._forecast_activation(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            self._forecast_activation(),
        )
        with torch.no_grad():
            convolution_output = self.forecast_convolutions(
                torch.zeros(1, forecast_channels, forecast_steps)
            )
        flatten_size = convolution_output.shape[1] * convolution_output.shape[2]
        self.forecast_projection = self._make_forecast_projection(
            flatten_size,
            forecast_features,
        )

    def _forecast_activation(self) -> nn.Module:
        return nn.ReLU()

    def _make_forecast_projection(
        self,
        flatten_size: int,
        forecast_features: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, forecast_features),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        state_features = self.state_encoder(observations["state"])
        forecast = observations["forecast"].transpose(1, 2)
        forecast_features = self.forecast_projection(
            self.forecast_convolutions(forecast)
        )
        return torch.cat((state_features, forecast_features), dim=1)


class StableTCNForecastExtractor(TCNForecastExtractor):
    """Keep forecast gradients alive with smooth activations and normalization."""

    def _forecast_activation(self) -> nn.Module:
        return nn.SiLU()

    def _make_forecast_projection(
        self,
        flatten_size: int,
        forecast_features: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, forecast_features),
            nn.LayerNorm(forecast_features),
            nn.SiLU(),
        )


class FixedScaleTCNForecastExtractor(StableTCNForecastExtractor):
    """Prevent BC from learning to suppress the normalized forecast modality."""

    def _make_forecast_projection(
        self,
        flatten_size: int,
        forecast_features: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, forecast_features),
            nn.LayerNorm(
                forecast_features,
                eps=1e-8,
                elementwise_affine=False,
            ),
            nn.SiLU(),
        )


class FutureMLPForecastExtractor(BaseFeaturesExtractor):
    """Encode state and a parameter-matched flattened forecast MLP separately."""

    FORECAST_HIDDEN_FEATURES = 35

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__(
            observation_space,
            features_dim=state_features + forecast_features,
        )
        state_size = observation_space["state"].shape[0]
        forecast_steps, forecast_channels = observation_space["forecast"].shape
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, state_features),
            nn.ReLU(),
        )
        self.forecast_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                forecast_steps * forecast_channels,
                self.FORECAST_HIDDEN_FEATURES,
            ),
            nn.SiLU(),
            nn.Linear(self.FORECAST_HIDDEN_FEATURES, forecast_features),
            nn.LayerNorm(
                forecast_features,
                eps=1e-8,
                elementwise_affine=False,
            ),
            nn.SiLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        state_features = self.state_encoder(observations["state"])
        forecast_features = self.forecast_projection(observations["forecast"])
        return torch.cat((state_features, forecast_features), dim=1)


class PastMLPForecastExtractor(FutureMLPForecastExtractor):
    """Add a causal flattened-history MLP beside state and future MLPs."""

    PAST_HIDDEN_FEATURES = 35

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        past_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__(
            observation_space,
            state_features=state_features,
            forecast_features=forecast_features,
        )
        past_steps, past_channels = observation_space["past"].shape
        self.past_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                past_steps * past_channels,
                self.PAST_HIDDEN_FEATURES,
            ),
            nn.SiLU(),
            nn.Linear(self.PAST_HIDDEN_FEATURES, past_features),
            nn.LayerNorm(
                past_features,
                eps=1e-8,
                elementwise_affine=False,
            ),
            nn.SiLU(),
        )
        self._features_dim = state_features + past_features + forecast_features

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        state_features = self.state_encoder(observations["state"])
        past_features = self.past_projection(observations["past"])
        forecast_features = self.forecast_projection(observations["forecast"])
        return torch.cat((state_features, past_features, forecast_features), dim=1)


class GatedPastMLPForecastExtractor(FutureMLPForecastExtractor):
    """Add a zero-initialized, feature-wise gated past correction."""

    PAST_HIDDEN_FEATURES = 35

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        past_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__(
            observation_space,
            state_features=state_features,
            forecast_features=forecast_features,
        )
        past_steps, past_channels = observation_space["past"].shape
        self.past_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                past_steps * past_channels,
                self.PAST_HIDDEN_FEATURES,
            ),
            nn.SiLU(),
            nn.Linear(self.PAST_HIDDEN_FEATURES, past_features),
            nn.LayerNorm(
                past_features,
                eps=1e-8,
                elementwise_affine=False,
            ),
            nn.SiLU(),
        )
        output_features = state_features + forecast_features
        self.past_correction = nn.Linear(past_features, output_features)
        self.past_gate = nn.Parameter(torch.zeros(output_features))

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        baseline = super().forward(observations)
        past_features = self.past_projection(observations["past"])
        correction = self.past_correction(past_features)
        return baseline + torch.tanh(self.past_gate) * correction


class _GraphAttentionBlock(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            features,
            num_heads=4,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(features)
        self.feed_forward = nn.Sequential(
            nn.Linear(features, 2 * features),
            nn.ReLU(),
            nn.Linear(2 * features, features),
        )
        self.output_norm = nn.LayerNorm(features)

    def forward(
        self,
        nodes: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _weights = self.attention(
            nodes,
            nodes,
            nodes,
            attn_mask=attention_mask,
            need_weights=False,
        )
        nodes = self.attention_norm(nodes + attended)
        return self.output_norm(nodes + self.feed_forward(nodes))


class _FormalCCSGraphStateEncoder(nn.Module):
    """Encode the fixed formal CCS state as a heterogeneous logistics graph."""

    STATE_SIZE = 78
    NODE_COUNT = 9
    RAW_NODE_FEATURES = 25

    def __init__(self, state_size: int, output_features: int) -> None:
        super().__init__()
        if state_size != self.STATE_SIZE:
            raise ValueError(
                "GNN forecast encoder requires 78 current-state features "
                f"for the formal 3-vessel layout, got {state_size}"
            )
        graph_features = 32
        self.node_embedding = nn.Sequential(
            nn.Linear(self.RAW_NODE_FEATURES, graph_features),
            nn.ReLU(),
        )
        self.graph_blocks = nn.ModuleList(
            [_GraphAttentionBlock(graph_features) for _ in range(2)]
        )
        self.projection = nn.Sequential(
            nn.Linear(self.NODE_COUNT * graph_features + 3, output_features),
            nn.ReLU(),
        )
        self.register_buffer("attention_mask", self._attention_mask())

    @classmethod
    def _attention_mask(cls) -> torch.Tensor:
        allowed = torch.eye(cls.NODE_COUNT, dtype=torch.bool)
        emitter_nodes = range(3)
        vessel_nodes = range(3, 6)
        terminal_node = 6
        well_node = 7
        reservoir_node = 8
        for vessel_node in vessel_nodes:
            for destination_node in [*emitter_nodes, terminal_node]:
                allowed[vessel_node, destination_node] = True
                allowed[destination_node, vessel_node] = True
        allowed[terminal_node, well_node] = True
        allowed[well_node, terminal_node] = True
        allowed[well_node, reservoir_node] = True
        allowed[reservoir_node, well_node] = True
        return ~allowed

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        nodes = state.new_zeros(
            batch_size,
            self.NODE_COUNT,
            self.RAW_NODE_FEATURES,
        )

        # Node order: emitters, vessels, terminal, well, reservoir.
        nodes[:, 0:3, 0:3] = state[:, 2:11].reshape(batch_size, 3, 3)
        nodes[:, 3:6, 0:7] = state[:, 11:32].reshape(batch_size, 3, 7)
        nodes[:, 3:6, 7:12] = state[:, 51:66].reshape(batch_size, 3, 5)
        nodes[:, 3:6, 12:16] = state[:, 66:78].reshape(batch_size, 3, 4)
        nodes[:, 3:6, 16:20] = state[:, 39:51].reshape(batch_size, 3, 4)
        nodes[:, 6, 0:2] = state[:, 32:34]
        nodes[:, 7, 0:3] = state[:, 34:37]
        nodes[:, 8, 0] = state[:, 37]

        nodes[:, 0:3, 20] = 1.0
        nodes[:, 3:6, 21] = 1.0
        nodes[:, 6, 22] = 1.0
        nodes[:, 7, 23] = 1.0
        nodes[:, 8, 24] = 1.0

        encoded = self.node_embedding(nodes)
        for block in self.graph_blocks:
            encoded = block(encoded, self.attention_mask)
        global_features = state[:, [0, 1, 38]]
        return self.projection(
            torch.cat((encoded.reshape(batch_size, -1), global_features), dim=1)
        )


class GNNForecastExtractor(TCNForecastExtractor):
    """Combine a graph-encoded formal CCS state with the existing TCN forecast."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__(
            observation_space,
            state_features=state_features,
            forecast_features=forecast_features,
        )
        self.state_encoder = _FormalCCSGraphStateEncoder(
            observation_space["state"].shape[0],
            state_features,
        )


class LargerMLPForecastExtractor(TCNForecastExtractor):
    """Use a larger current-state MLP with the unchanged TCN forecast path."""

    STATE_SIZE = 78
    HIDDEN_FEATURES = 262

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__(
            observation_space,
            state_features=state_features,
            forecast_features=forecast_features,
        )
        state_size = observation_space["state"].shape[0]
        if state_size != self.STATE_SIZE:
            raise ValueError(
                "larger MLP forecast encoder requires 78 current-state features "
                f"for the formal 3-vessel layout, got {state_size}"
            )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, self.HIDDEN_FEATURES),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_FEATURES, state_features),
            nn.ReLU(),
        )


class FixedScaleLargerMLPForecastExtractor(FixedScaleTCNForecastExtractor):
    """Pair the parameter-matched state MLP with the fixed-scale TCN."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__(
            observation_space,
            state_features=state_features,
            forecast_features=forecast_features,
        )
        state_size = observation_space["state"].shape[0]
        if state_size != LargerMLPForecastExtractor.STATE_SIZE:
            raise ValueError(
                "fixed-scale larger MLP forecast encoder requires 78 current-state "
                f"features for the formal 3-vessel layout, got {state_size}"
            )
        self.state_encoder = nn.Sequential(
            nn.Linear(
                state_size,
                LargerMLPForecastExtractor.HIDDEN_FEATURES,
            ),
            nn.ReLU(),
            nn.Linear(
                LargerMLPForecastExtractor.HIDDEN_FEATURES,
                state_features,
            ),
            nn.ReLU(),
        )


class _EdgeAwareAttentionBlock(nn.Module):
    def __init__(self, node_features: int, edge_features: int) -> None:
        super().__init__()
        self.num_heads = 4
        self.head_features = node_features // self.num_heads
        self.query = nn.Linear(node_features, node_features)
        self.key = nn.Linear(node_features + edge_features, node_features)
        self.value = nn.Linear(node_features + edge_features, node_features)
        self.attention_output = nn.Linear(node_features, node_features)
        self.attention_norm = nn.LayerNorm(node_features)
        self.feed_forward = nn.Sequential(
            nn.Linear(node_features, 2 * node_features),
            nn.ReLU(),
            nn.Linear(2 * node_features, node_features),
        )
        self.output_norm = nn.LayerNorm(node_features)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, node_features = nodes.shape
        queries = self.query(nodes).reshape(
            batch_size,
            node_count,
            self.num_heads,
            self.head_features,
        )
        senders = nodes[:, None, :, :].expand(-1, node_count, -1, -1)
        edge_inputs = torch.cat((senders, edge_features), dim=-1)
        keys = self.key(edge_inputs).reshape(
            batch_size,
            node_count,
            node_count,
            self.num_heads,
            self.head_features,
        )
        values = self.value(edge_inputs).reshape_as(keys)
        scores = torch.einsum("bihd,bijhd->bijh", queries, keys)
        scores = scores / self.head_features**0.5
        scores = scores.masked_fill(attention_mask[None, :, :, None], float("-inf"))
        weights = torch.softmax(scores, dim=2)
        attended = torch.einsum("bijh,bijhd->bihd", weights, values).reshape(
            batch_size,
            node_count,
            node_features,
        )
        nodes = self.attention_norm(nodes + self.attention_output(attended))
        return self.output_norm(nodes + self.feed_forward(nodes))


class _EdgeAwareCCSGraphStateEncoder(nn.Module):
    """Encode route state on directed vessel-location edges."""

    STATE_SIZE = 78
    NODE_COUNT = 9
    RAW_NODE_FEATURES = 13
    EDGE_FEATURES = 10
    DESTINATION_NODES = (6, 0, 1, 2)  # terminal, Brevik, Celsio, Yara

    def __init__(self, state_size: int, output_features: int) -> None:
        super().__init__()
        if state_size != self.STATE_SIZE:
            raise ValueError(
                "Edge-GNN forecast encoder requires 78 current-state features "
                f"for the formal 3-vessel layout, got {state_size}"
            )
        graph_features = 32
        self.node_embedding = nn.Sequential(
            nn.Linear(self.RAW_NODE_FEATURES, graph_features),
            nn.ReLU(),
        )
        self.graph_blocks = nn.ModuleList(
            [
                _EdgeAwareAttentionBlock(graph_features, self.EDGE_FEATURES)
                for _ in range(2)
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(self.NODE_COUNT * graph_features + 3, output_features),
            nn.ReLU(),
        )
        attention_mask, relation_features = self._graph_structure()
        self.register_buffer("attention_mask", attention_mask)
        self.register_buffer("relation_features", relation_features)

    @classmethod
    def _graph_structure(cls) -> tuple[torch.Tensor, torch.Tensor]:
        allowed = torch.eye(cls.NODE_COUNT, dtype=torch.bool)
        relations = torch.zeros(cls.NODE_COUNT, cls.NODE_COUNT, cls.EDGE_FEATURES)
        for node in range(cls.NODE_COUNT):
            relations[node, node, 3] = 1.0  # self
        for vessel_node in range(3, 6):
            for destination_node in cls.DESTINATION_NODES:
                allowed[destination_node, vessel_node] = True
                allowed[vessel_node, destination_node] = True
                relations[destination_node, vessel_node, 4] = 1.0  # vessel -> location
                relations[vessel_node, destination_node, 5] = 1.0  # location -> vessel
        for receiver, sender, relation_index in (
            (7, 6, 6),  # terminal -> well
            (6, 7, 7),  # well -> terminal
            (8, 7, 8),  # well -> reservoir
            (7, 8, 9),  # reservoir -> well
        ):
            allowed[receiver, sender] = True
            relations[receiver, sender, relation_index] = 1.0
        return ~allowed, relations

    def _graph_inputs(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = state.shape[0]
        nodes = state.new_zeros(batch_size, self.NODE_COUNT, self.RAW_NODE_FEATURES)

        # Node order: emitters, vessels, terminal, well, reservoir.
        nodes[:, 0:3, 0:3] = state[:, 2:11].reshape(batch_size, 3, 3)
        vessel_state = state[:, 11:32].reshape(batch_size, 3, 7)
        nodes[:, 3:6, 0:3] = vessel_state[:, :, [0, 1, 3]]
        nodes[:, 3:6, 3:8] = state[:, 51:66].reshape(batch_size, 3, 5)
        nodes[:, 6, 0:2] = state[:, 32:34]
        nodes[:, 7, 0:3] = state[:, 34:37]
        nodes[:, 8, 0] = state[:, 37]
        nodes[:, 0:3, 8] = 1.0
        nodes[:, 3:6, 9] = 1.0
        nodes[:, 6, 10] = 1.0
        nodes[:, 7, 11] = 1.0
        nodes[:, 8, 12] = 1.0

        edge_features = self.relation_features[None, :, :, :].expand(
            batch_size,
            -1,
            -1,
            -1,
        ).clone()
        travel_times = state[:, 39:51].reshape(batch_size, 3, 4)
        at_locations = vessel_state[:, :, [2, 4, 5, 6]]
        destinations = state[:, 66:78].reshape(batch_size, 3, 4)
        remaining_travel_times = travel_times * (
            1.0 - vessel_state[:, :, 3:4] * destinations
        )
        dynamic_features = torch.stack(
            (remaining_travel_times, at_locations, destinations),
            dim=-1,
        )
        for vessel_index, vessel_node in enumerate(range(3, 6)):
            for destination_slot, destination_node in enumerate(
                self.DESTINATION_NODES
            ):
                route = dynamic_features[:, vessel_index, destination_slot]
                edge_features[:, destination_node, vessel_node, :3] = route
                edge_features[:, vessel_node, destination_node, :3] = route

        return nodes, edge_features, state[:, [0, 1, 38]]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        nodes, edge_features, global_features = self._graph_inputs(state)
        encoded = self.node_embedding(nodes)
        for block in self.graph_blocks:
            encoded = block(encoded, edge_features, self.attention_mask)
        return self.projection(
            torch.cat((encoded.reshape(batch_size, -1), global_features), dim=1)
        )


class _TemporalContextEncoder(nn.Module):
    """Encode one entity-aligned future trajectory without mixing entities."""

    def __init__(
        self,
        input_channels: int,
        forecast_steps: int,
        output_features: int,
    ) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
        )
        with torch.no_grad():
            convolution_output = self.convolutions(
                torch.zeros(1, input_channels, forecast_steps)
            )
        flatten_size = convolution_output.shape[1] * convolution_output.shape[2]
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, output_features),
            nn.LayerNorm(
                output_features,
                eps=1e-8,
                elementwise_affine=False,
            ),
            nn.SiLU(),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolutions(trajectory.transpose(1, 2)))


class _FutureConditionedCCSGraphEncoder(_EdgeAwareCCSGraphStateEncoder):
    """Inject entity-aligned future trajectories before graph message passing."""

    FUTURE_FEATURES = 16

    def __init__(
        self,
        state_size: int,
        forecast_steps: int,
        output_features: int,
    ) -> None:
        super().__init__(state_size, output_features)
        graph_features = 32
        self.emitter_future_encoder = _TemporalContextEncoder(
            2,
            forecast_steps,
            self.FUTURE_FEATURES,
        )
        self.well_future_encoder = _TemporalContextEncoder(
            2,
            forecast_steps,
            self.FUTURE_FEATURES,
        )
        self.weather_future_encoder = _TemporalContextEncoder(
            1,
            forecast_steps,
            self.FUTURE_FEATURES,
        )
        self.node_future_fusion = nn.Sequential(
            nn.Linear(graph_features + self.FUTURE_FEATURES, graph_features),
            nn.LayerNorm(
                graph_features,
                eps=1e-8,
                elementwise_affine=False,
            ),
            nn.SiLU(),
        )
        conditioned_edge_features = self.EDGE_FEATURES + self.FUTURE_FEATURES
        self.graph_blocks = nn.ModuleList(
            [
                _EdgeAwareAttentionBlock(graph_features, conditioned_edge_features)
                for _ in range(2)
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(self.NODE_COUNT * graph_features + 3, output_features),
            nn.LayerNorm(
                output_features,
                eps=1e-8,
                elementwise_affine=False,
            ),
            nn.SiLU(),
        )

    def _future_contexts(
        self,
        forecast: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, forecast_steps, _channels = forecast.shape
        emitter_trajectories = torch.stack(
            (forecast[:, :, 0:3], forecast[:, :, 3:6]),
            dim=-1,
        ).permute(0, 2, 1, 3)
        emitter_context = self.emitter_future_encoder(
            emitter_trajectories.reshape(batch_size * 3, forecast_steps, 2)
        ).reshape(batch_size, 3, self.FUTURE_FEATURES)
        well_context = self.well_future_encoder(forecast[:, :, 6:8])
        weather_context = self.weather_future_encoder(forecast[:, :, 8:9])
        zeros = forecast.new_zeros
        node_context = torch.cat(
            (
                emitter_context,
                zeros(batch_size, 3, self.FUTURE_FEATURES),
                zeros(batch_size, 1, self.FUTURE_FEATURES),
                well_context[:, None, :],
                zeros(batch_size, 1, self.FUTURE_FEATURES),
            ),
            dim=1,
        )
        return node_context, weather_context

    def forward(
        self,
        state: torch.Tensor,
        forecast: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        nodes, edge_features, global_features = self._graph_inputs(state)
        node_context, weather_context = self._future_contexts(forecast)
        encoded = self.node_future_fusion(
            torch.cat((self.node_embedding(nodes), node_context), dim=-1)
        )
        vessel_location_edges = self.relation_features[:, :, 4:6].sum(dim=-1)
        edge_context = (
            weather_context[:, None, None, :]
            * vessel_location_edges[None, :, :, None]
        )
        conditioned_edges = torch.cat((edge_features, edge_context), dim=-1)
        for block in self.graph_blocks:
            encoded = block(encoded, conditioned_edges, self.attention_mask)
        return self.projection(
            torch.cat((encoded.reshape(batch_size, -1), global_features), dim=1)
        )


class FutureConditionedEdgeGNNForecastExtractor(BaseFeaturesExtractor):
    """Fuse current state and entity-aligned futures inside the Edge-GNN."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        output_features = state_features + forecast_features
        super().__init__(observation_space, features_dim=output_features)
        forecast_steps, forecast_channels = observation_space["forecast"].shape
        if forecast_channels != 9:
            raise ValueError(
                "future-conditioned Edge-GNN requires 9 forecast channels, "
                f"got {forecast_channels}"
            )
        self.graph_encoder = _FutureConditionedCCSGraphEncoder(
            observation_space["state"].shape[0],
            forecast_steps,
            output_features,
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.graph_encoder(observations["state"], observations["forecast"])


class EdgeGNNForecastExtractor(TCNForecastExtractor):
    """Combine an edge-aware CCS graph encoder with the unchanged TCN."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__(
            observation_space,
            state_features=state_features,
            forecast_features=forecast_features,
        )
        self.state_encoder = _EdgeAwareCCSGraphStateEncoder(
            observation_space["state"].shape[0],
            state_features,
        )


class FixedScaleEdgeGNNForecastExtractor(FixedScaleTCNForecastExtractor):
    """Pair the edge-aware state graph with the fixed-scale TCN."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__(
            observation_space,
            state_features=state_features,
            forecast_features=forecast_features,
        )
        self.state_encoder = _EdgeAwareCCSGraphStateEncoder(
            observation_space["state"].shape[0],
            state_features,
        )


class BalancedEdgeGNNForecastExtractor(FixedScaleTCNForecastExtractor):
    """Normalize both graph-state and forecast modalities before fusion."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__(
            observation_space,
            state_features=state_features,
            forecast_features=forecast_features,
        )
        self.state_encoder = nn.Sequential(
            _EdgeAwareCCSGraphStateEncoder(
                observation_space["state"].shape[0],
                state_features,
            ),
            nn.LayerNorm(
                state_features,
                eps=1e-8,
                elementwise_affine=False,
            ),
            nn.SiLU(),
        )


class BalancedEdgeGNNFutureMLPForecastExtractor(FutureMLPForecastExtractor):
    """Pair the normalized edge-aware state graph with the future MLP."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__(
            observation_space,
            state_features=state_features,
            forecast_features=forecast_features,
        )
        self.state_encoder = nn.Sequential(
            _EdgeAwareCCSGraphStateEncoder(
                observation_space["state"].shape[0],
                state_features,
            ),
            nn.LayerNorm(
                state_features,
                eps=1e-8,
                elementwise_affine=False,
            ),
            nn.SiLU(),
        )


class GatedResidualEdgeGNNForecastExtractor(BalancedEdgeGNNForecastExtractor):
    """Preserve both balanced branches while future-gating the graph state."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__(
            observation_space,
            state_features=state_features,
            forecast_features=forecast_features,
        )
        self.future_to_state = nn.Linear(forecast_features, state_features)
        self.interaction_gate = nn.Sequential(
            nn.Linear(state_features + forecast_features, state_features),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(
            state_features,
            eps=1e-8,
            elementwise_affine=False,
        )
        self.fusion_activation = nn.SiLU()

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        state_features = self.state_encoder(observations["state"])
        forecast = observations["forecast"].transpose(1, 2)
        forecast_features = self.forecast_projection(
            self.forecast_convolutions(forecast)
        )
        gate = self.interaction_gate(
            torch.cat((state_features, forecast_features), dim=1)
        )
        fused_state = self.fusion_activation(
            self.fusion_norm(
                state_features + gate * self.future_to_state(forecast_features)
            )
        )
        return torch.cat((fused_state, forecast_features), dim=1)


class EntityResidualEdgeGNNForecastExtractor(FixedScaleTCNForecastExtractor):
    """Keep a direct future path beside an entity-conditioned graph path."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__(
            observation_space,
            state_features=state_features,
            forecast_features=forecast_features,
        )
        del self.state_encoder
        forecast_steps, forecast_channels = observation_space["forecast"].shape
        if forecast_channels != 9:
            raise ValueError(
                "entity-residual Edge-GNN requires 9 forecast channels, "
                f"got {forecast_channels}"
            )
        self.graph_encoder = _FutureConditionedCCSGraphEncoder(
            observation_space["state"].shape[0],
            forecast_steps,
            state_features,
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        graph_features = self.graph_encoder(
            observations["state"],
            observations["forecast"],
        )
        forecast = observations["forecast"].transpose(1, 2)
        forecast_features = self.forecast_projection(
            self.forecast_convolutions(forecast)
        )
        return torch.cat((graph_features, forecast_features), dim=1)
