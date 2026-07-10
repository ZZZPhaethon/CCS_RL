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
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            convolution_output = self.forecast_convolutions(
                torch.zeros(1, forecast_channels, forecast_steps)
            )
        flatten_size = convolution_output.shape[1] * convolution_output.shape[2]
        self.forecast_projection = nn.Sequential(
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
