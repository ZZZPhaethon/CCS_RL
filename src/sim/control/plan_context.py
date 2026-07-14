"""Learned plan context for the 24-hour native-MPC replan cycle."""

from __future__ import annotations

import torch
from gymnasium import spaces
from torch import nn

from ..environment.forecast_encoder import FixedScaleTCNForecastExtractor


class CandidatePlanEncoder(nn.Module):
    """Predict continuous candidate logits from a replan observation."""

    def __init__(
        self,
        state_size: int,
        candidate_count: int,
        state_features: int = 64,
        forecast_features: int = 64,
    ) -> None:
        super().__init__()
        observation_space = spaces.Dict(
            {
                "state": spaces.Box(-10.0, 10.0, (int(state_size),), dtype=float),
                "forecast": spaces.Box(-10.0, 10.0, (168, 9), dtype=float),
            }
        )
        self.encoder = FixedScaleTCNForecastExtractor(
            observation_space,
            state_features=state_features,
            forecast_features=forecast_features,
        )
        self.classifier = nn.Sequential(
            nn.Linear(state_features + forecast_features, 64),
            nn.SiLU(),
            nn.Linear(64, int(candidate_count)),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encoder(observations))
