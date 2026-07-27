"""Recurrent bootstrapped quantile Q-network for joint residual Event control."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ForecastTCN(nn.Module):
    def __init__(self, steps: int, channels: int, output_features: int = 64):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
        )
        with torch.no_grad():
            flattened = self.convolutions(torch.zeros(1, channels, steps)).numel()
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, output_features),
            nn.LayerNorm(output_features, elementwise_affine=False),
            nn.SiLU(),
        )

    def forward(self, forecast: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolutions(forecast.transpose(1, 2)))


class ForecastMLP(nn.Module):
    def __init__(self, steps: int, channels: int, output_features: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(steps * channels, 35),
            nn.SiLU(),
            nn.Linear(35, output_features),
            nn.LayerNorm(output_features, elementwise_affine=False),
            nn.SiLU(),
        )

    def forward(self, forecast: torch.Tensor) -> torch.Tensor:
        return self.projection(forecast)


class EntityStateEncoder(nn.Module):
    def __init__(self, state_feature_names: list[str], output_features: int = 64):
        super().__init__()
        vessel_ids = sorted(
            {
                name.split(".")[1]
                for name in state_feature_names
                if name.startswith("greedy_proposal.")
            }
        )
        vessel_indices = []
        used = set()
        for vessel_id in vessel_ids:
            indices = [
                index
                for index, name in enumerate(state_feature_names)
                if name.startswith(f"{vessel_id}.")
                or name.startswith(f"greedy_proposal.{vessel_id}.")
            ]
            vessel_indices.append(indices)
            used.update(indices)
        if not vessel_indices or len({len(indices) for indices in vessel_indices}) != 1:
            raise ValueError("entity encoder requires equal per-vessel feature blocks")
        global_indices = [
            index for index in range(len(state_feature_names)) if index not in used
        ]
        self.register_buffer(
            "vessel_indices", torch.as_tensor(vessel_indices, dtype=torch.long)
        )
        self.register_buffer(
            "global_indices", torch.as_tensor(global_indices, dtype=torch.long)
        )
        self.vessel_encoder = nn.Sequential(
            nn.Linear(len(vessel_indices[0]), 32), nn.SiLU()
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(len(global_indices), 32), nn.SiLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(32 * (len(vessel_ids) + 1), output_features), nn.SiLU()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        vessel_state = state[:, self.vessel_indices].reshape(
            batch * self.vessel_indices.shape[0], self.vessel_indices.shape[1]
        )
        vessels = self.vessel_encoder(vessel_state).reshape(batch, -1)
        global_features = self.global_encoder(
            state.index_select(1, self.global_indices)
        )
        return self.fusion(torch.cat((global_features, vessels), dim=1))


class RecurrentBootstrappedQuantileQ(nn.Module):
    """Direct joint-action Q distribution with persistent randomized priors."""

    def __init__(
        self,
        state_feature_names: list[str],
        forecast_shape: tuple[int, int],
        action_count: int,
        *,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        forecast_mean: np.ndarray,
        forecast_std: np.ndarray,
        return_scale: float,
        heads: int = 5,
        quantiles: int = 51,
        hidden_size: int = 128,
        prior_scale: float = 0.25,
        forecast_encoder: str = "tcn",
    ) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.heads = int(heads)
        self.quantiles = int(quantiles)
        self.hidden_size = int(hidden_size)
        self.return_scale = float(return_scale)
        self.prior_scale = float(prior_scale)
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))
        self.register_buffer(
            "forecast_mean", torch.as_tensor(forecast_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "forecast_std", torch.as_tensor(forecast_std, dtype=torch.float32)
        )
        self.state_encoder = EntityStateEncoder(state_feature_names, 64)
        if forecast_encoder == "tcn":
            self.forecast_encoder = ForecastTCN(*forecast_shape, output_features=64)
        elif forecast_encoder == "small_mlp":
            self.forecast_encoder = ForecastMLP(*forecast_shape, output_features=64)
        else:
            raise ValueError(f"unknown forecast encoder: {forecast_encoder}")
        self.action_embedding = nn.Embedding(
            self.action_count + 1, 32, padding_idx=self.action_count
        )
        self.input_projection = nn.Sequential(
            nn.Linear(64 + 64 + 32 + 2, hidden_size), nn.SiLU()
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.value = nn.Linear(hidden_size, self.heads * self.quantiles)
        self.advantage = nn.Linear(
            hidden_size, self.heads * self.action_count * self.quantiles
        )
        self.prior = nn.Linear(
            hidden_size, self.heads * self.action_count * self.quantiles
        )
        for parameter in self.prior.parameters():
            parameter.requires_grad_(False)

    def initial_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        device = device or self.state_mean.device
        return torch.zeros(1, int(batch_size), self.hidden_size, device=device)

    def forward(
        self,
        states: torch.Tensor,
        forecasts: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        previous_durations: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent, next_hidden = self.recurrent_features(
            states,
            forecasts,
            previous_actions,
            previous_rewards,
            previous_durations,
            hidden,
        )
        return self.quantiles_from_features(recurrent), next_hidden

    def recurrent_features(
        self,
        states: torch.Tensor,
        forecasts: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        previous_durations: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence = states.shape[:2]
        normalized_state = (states - self.state_mean) / self.state_std
        normalized_forecast = (forecasts - self.forecast_mean) / self.forecast_std
        state_features = self.state_encoder(
            normalized_state.reshape(batch * sequence, -1)
        )
        forecast_features = self.forecast_encoder(
            normalized_forecast.reshape(batch * sequence, *forecasts.shape[2:])
        )
        safe_actions = torch.where(
            previous_actions >= 0,
            previous_actions,
            torch.full_like(previous_actions, self.action_count),
        )
        action_features = self.action_embedding(safe_actions).reshape(
            batch * sequence, -1
        )
        reward_features = torch.tanh(
            previous_rewards.reshape(batch * sequence, 1) / self.return_scale
        )
        duration_features = (
            previous_durations.reshape(batch * sequence, 1) / 168.0
        ).clamp(0.0, 1.0)
        inputs = self.input_projection(
            torch.cat(
                (
                    state_features,
                    forecast_features,
                    action_features,
                    reward_features,
                    duration_features,
                ),
                dim=1,
            )
        ).reshape(batch, sequence, -1)
        recurrent, next_hidden = self.gru(inputs, hidden)
        return recurrent, next_hidden

    def quantiles_from_features(self, recurrent: torch.Tensor) -> torch.Tensor:
        batch, sequence = recurrent.shape[:2]
        value = self.value(recurrent).reshape(
            batch, sequence, self.heads, 1, self.quantiles
        )
        advantage = self.advantage(recurrent).reshape(
            batch, sequence, self.heads, self.action_count, self.quantiles
        )
        prior = self.prior(recurrent.detach()).reshape_as(advantage)
        q = value + advantage - advantage.mean(dim=3, keepdim=True)
        q = q + self.prior_scale * prior
        return q


class ActionAlignedForecastResidual(nn.Module):
    """Condition forecast corrections on each vessel's candidate destination."""

    WINDOW_HOURS = (24, 72, 168)

    def __init__(
        self,
        joint_actions: np.ndarray,
        forecast_channel_names: list[str],
        hidden_size: int,
        heads: int,
        quantiles: int,
    ) -> None:
        super().__init__()
        names = list(forecast_channel_names)
        capture_names = [name for name in names if name.startswith("capture.")]
        emitter_ids = [name.split(".", 1)[1] for name in capture_names]
        if not emitter_ids:
            raise ValueError("action-aligned forecast requires emitter capture channels")
        capture_indices = [names.index(f"capture.{emitter}") for emitter in emitter_ids]
        available_indices = [
            names.index(f"emitter_available.{emitter}") for emitter in emitter_ids
        ]
        well_indices = [
            index
            for index, name in enumerate(names)
            if name.startswith("well_available.")
        ]
        injectivity_indices = [
            index for index, name in enumerate(names) if name.startswith("injectivity.")
        ]
        weather_indices = [
            index for index, name in enumerate(names) if name.startswith("weather.")
        ]
        if not well_indices or not injectivity_indices or not weather_indices:
            raise ValueError(
                "action-aligned forecast requires well, injectivity, and weather channels"
            )
        joint_array = np.asarray(joint_actions, dtype=np.int64)
        local_action_count = int(joint_array.max()) + 1
        follow_action = 2 + len(emitter_ids)
        if local_action_count != follow_action + 1:
            raise ValueError("unexpected vessel action schema for aligned forecast")
        self.register_buffer(
            "joint_action_indices", torch.as_tensor(joint_array, dtype=torch.long)
        )
        self.register_buffer(
            "capture_indices", torch.as_tensor(capture_indices, dtype=torch.long)
        )
        self.register_buffer(
            "available_indices", torch.as_tensor(available_indices, dtype=torch.long)
        )
        self.register_buffer(
            "well_indices", torch.as_tensor(well_indices, dtype=torch.long)
        )
        self.register_buffer(
            "injectivity_indices",
            torch.as_tensor(injectivity_indices, dtype=torch.long),
        )
        self.register_buffer(
            "weather_indices", torch.as_tensor(weather_indices, dtype=torch.long)
        )
        self.follow_action = int(follow_action)
        self.local_action_embedding = nn.Embedding(local_action_count, 8)
        summary_size = len(self.WINDOW_HOURS) * 6
        self.vessel_encoder = nn.Sequential(
            nn.Linear(summary_size + 8, 32),
            nn.SiLU(),
        )
        vessel_count = int(joint_array.shape[1])
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_size + vessel_count * 32, 64),
            nn.SiLU(),
            nn.Linear(64, heads * quantiles),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.heads = int(heads)
        self.quantiles = int(quantiles)

    def _window_statistics(self, forecast: torch.Tensor):
        means = []
        minimums = []
        steps = int(forecast.shape[1])
        for hours in self.WINDOW_HOURS:
            values = forecast[:, : min(int(hours), steps)]
            means.append(values.mean(dim=1))
            minimums.append(values.amin(dim=1))
        return torch.stack(means, dim=1), torch.stack(minimums, dim=1)

    def _local_action_summaries(self, forecast: torch.Tensor) -> torch.Tensor:
        means, minimums = self._window_statistics(forecast)
        capture = means.index_select(2, self.capture_indices)
        available = means.index_select(2, self.available_indices)
        available_min = minimums.index_select(2, self.available_indices)
        well = means.index_select(2, self.well_indices).mean(dim=2)
        injectivity = means.index_select(2, self.injectivity_indices).mean(dim=2)
        weather = means.index_select(2, self.weather_indices).mean(dim=2)
        weather_min = minimums.index_select(2, self.weather_indices).mean(dim=2)
        zeros = torch.zeros_like(weather)
        summaries = []
        for action in range(self.follow_action + 1):
            if action == self.follow_action:
                summaries.append(
                    torch.zeros(
                        forecast.shape[0],
                        len(self.WINDOW_HOURS) * 6,
                        dtype=forecast.dtype,
                        device=forecast.device,
                    )
                )
                continue
            if action == 0:
                selected_capture = capture.mean(dim=2)
                selected_available = available.mean(dim=2)
                selected_available_min = available_min.mean(dim=2)
            elif action == 1:
                selected_capture = zeros
                selected_available = zeros
                selected_available_min = zeros
            else:
                emitter_index = action - 2
                selected_capture = capture[:, :, emitter_index]
                selected_available = available[:, :, emitter_index]
                selected_available_min = available_min[:, :, emitter_index]
            summaries.append(
                torch.cat(
                    (
                        selected_capture,
                        selected_available,
                        selected_available_min,
                        well,
                        injectivity,
                        0.5 * (weather + weather_min),
                    ),
                    dim=1,
                )
            )
        return torch.stack(summaries, dim=1)

    def forward(
        self,
        recurrent: torch.Tensor,
        forecast: torch.Tensor,
    ) -> torch.Tensor:
        local_summaries = self._local_action_summaries(forecast)
        aligned = local_summaries[:, self.joint_action_indices]
        action_features = self.local_action_embedding(self.joint_action_indices)
        action_features = action_features.unsqueeze(0).expand(
            forecast.shape[0], -1, -1, -1
        )
        full_vessels = self.vessel_encoder(
            torch.cat((aligned, action_features), dim=-1)
        ).flatten(start_dim=2)
        no_future_vessels = self.vessel_encoder(
            torch.cat((torch.zeros_like(aligned), action_features), dim=-1)
        ).flatten(start_dim=2)
        repeated_state = recurrent[:, None].expand(-1, aligned.shape[1], -1)
        full = self.residual_head(
            torch.cat((repeated_state, full_vessels), dim=-1)
        )
        no_future = self.residual_head(
            torch.cat((repeated_state, no_future_vessels), dim=-1)
        )
        residual = (full - no_future).reshape(
            forecast.shape[0],
            aligned.shape[1],
            self.heads,
            self.quantiles,
        )
        return residual.permute(0, 2, 1, 3)


class ETAAlignedForecastResidual(nn.Module):
    """Align each candidate destination with its weather-adjusted arrival time."""

    LOAD_HOURS = 10
    ARRIVAL_WINDOW_HOURS = 6
    RECEIVING_WINDOW_HOURS = 12
    SUMMARY_FEATURES = (
        "arrival_fraction",
        "arrival_within_forecast",
        "travel_weather_mean",
        "travel_weather_min",
        "current_emitter_fill",
        "emitter_fill_at_arrival",
        "emitter_overflow_at_arrival",
        "capture_fill_delta",
        "capture_near_arrival",
        "destination_available_before_arrival",
        "destination_available_near_arrival",
        "destination_available_min",
        "current_terminal_fill",
        "projected_terminal_fill",
        "terminal_headroom",
        "well_available_mean",
        "well_available_min",
        "injectivity_mean",
        "terminal_receiving_score",
        "remaining_forecast_fraction",
    )
    TERMINAL_DRAIN_HOURS = 9150.0 / 285.3881278538813
    EMITTER_FILL_HOURS = {
        "brevik": 7500.0 / 56.0,
        "celsio": 7500.0 / 48.0,
        "yara_sluiskil": 15000.0 / 110.0,
    }
    EMITTER_TO_TERMINAL_HOURS = {
        "brevik": 26.111597584824608,
        "celsio": 30.893879429659957,
        "yara_sluiskil": 43.811310826100545,
    }

    def __init__(
        self,
        joint_actions: np.ndarray,
        state_feature_names: list[str],
        forecast_channel_names: list[str],
        episode_hours: int,
        hidden_size: int,
        heads: int,
        quantiles: int,
    ) -> None:
        super().__init__()
        state_names = list(state_feature_names)
        forecast_names = list(forecast_channel_names)
        capture_names = [
            name for name in forecast_names if name.startswith("capture.")
        ]
        emitter_ids = [name.split(".", 1)[1] for name in capture_names]
        vessel_ids = sorted(
            {
                name.split(".")[1]
                for name in state_names
                if name.startswith("greedy_proposal.")
            }
        )
        if not emitter_ids or not vessel_ids:
            raise ValueError("ETA-aligned forecast requires emitter and vessel features")
        destination_ids = ["oygarden_terminal", *emitter_ids]
        travel_indices = []
        for vessel_id in vessel_ids:
            travel_indices.append(
                [
                    state_names.index(
                        f"{vessel_id}.to_{destination_id}.travel_hours_now"
                    )
                    for destination_id in destination_ids
                ]
            )
        capture_indices = [
            forecast_names.index(f"capture.{emitter_id}")
            for emitter_id in emitter_ids
        ]
        available_indices = [
            forecast_names.index(f"emitter_available.{emitter_id}")
            for emitter_id in emitter_ids
        ]
        well_indices = [
            index
            for index, name in enumerate(forecast_names)
            if name.startswith("well_available.")
        ]
        injectivity_indices = [
            index
            for index, name in enumerate(forecast_names)
            if name.startswith("injectivity.")
        ]
        weather_indices = [
            index
            for index, name in enumerate(forecast_names)
            if name.startswith("weather.")
        ]
        if not well_indices or not injectivity_indices or not weather_indices:
            raise ValueError(
                "ETA-aligned forecast requires well, injectivity, and weather channels"
            )
        joint_array = np.asarray(joint_actions, dtype=np.int64)
        local_action_count = int(joint_array.max()) + 1
        follow_action = 2 + len(emitter_ids)
        if (
            joint_array.shape[1] != len(vessel_ids)
            or local_action_count != follow_action + 1
        ):
            raise ValueError("unexpected vessel action schema for ETA-aligned forecast")
        self.register_buffer(
            "joint_action_indices", torch.as_tensor(joint_array, dtype=torch.long)
        )
        self.register_buffer(
            "travel_indices", torch.as_tensor(travel_indices, dtype=torch.long)
        )
        self.register_buffer(
            "emitter_fill_indices",
            torch.as_tensor(
                [state_names.index(f"{emitter_id}.fill") for emitter_id in emitter_ids],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "capture_indices", torch.as_tensor(capture_indices, dtype=torch.long)
        )
        self.register_buffer(
            "available_indices", torch.as_tensor(available_indices, dtype=torch.long)
        )
        self.register_buffer(
            "well_indices", torch.as_tensor(well_indices, dtype=torch.long)
        )
        self.register_buffer(
            "injectivity_indices",
            torch.as_tensor(injectivity_indices, dtype=torch.long),
        )
        self.register_buffer(
            "weather_indices", torch.as_tensor(weather_indices, dtype=torch.long)
        )
        self.register_buffer(
            "fill_hours",
            torch.as_tensor(
                [self.EMITTER_FILL_HOURS.get(name, 168.0) for name in emitter_ids],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "return_hours",
            torch.as_tensor(
                [
                    self.EMITTER_TO_TERMINAL_HOURS.get(name, 24.0)
                    for name in emitter_ids
                ],
                dtype=torch.float32,
            ),
        )
        self.weather_now_index = state_names.index("weather.speed_now")
        self.terminal_fill_index = state_names.index("oygarden_terminal.fill")
        self.episode_hours = float(episode_hours)
        self.follow_action = int(follow_action)
        self.local_action_embedding = nn.Embedding(local_action_count, 8)
        self.summary_size = len(self.SUMMARY_FEATURES)
        self.register_buffer(
            "summary_feature_mask",
            torch.ones(self.summary_size, dtype=torch.float32),
            persistent=False,
        )
        self.vessel_encoder = nn.Sequential(
            nn.Linear(self.summary_size + 8, 32),
            nn.SiLU(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_size + len(vessel_ids) * 32, 64),
            nn.SiLU(),
            nn.Linear(64, heads * quantiles),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.heads = int(heads)
        self.quantiles = int(quantiles)

    @staticmethod
    def _arrival_hours(
        work_hours: torch.Tensor,
        weather: torch.Tensor,
        start_hours: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = int(weather.shape[1])
        time = torch.arange(steps, device=weather.device)[None, :]
        cumulative = weather.clamp_min(0.0).cumsum(dim=1)
        if start_hours is None:
            start_hours = torch.zeros_like(work_hours)
            completed = torch.zeros_like(work_hours)
        else:
            start_indices = (start_hours.long() - 1).clamp(0, steps - 1)
            completed = cumulative.gather(1, start_indices[:, None]).squeeze(1)
            completed = torch.where(
                start_hours > 0.0, completed, torch.zeros_like(completed)
            )
        target = completed + work_hours.clamp_min(0.0)
        reached = (cumulative >= target[:, None]) & (
            time >= start_hours.long()[:, None]
        )
        within = reached.any(dim=1)
        first = reached.to(torch.int64).argmax(dim=1).to(weather.dtype) + 1.0
        arrival = torch.where(within, first, torch.full_like(first, float(steps)))
        arrival = torch.where(work_hours <= 1e-6, start_hours, arrival)
        return arrival.clamp(0.0, float(steps)), within.to(weather.dtype)

    @staticmethod
    def _masked_stats(
        values: torch.Tensor,
        start_hours: torch.Tensor,
        end_hours: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = int(values.shape[1])
        time = torch.arange(steps, device=values.device)[None, :]
        mask = (time >= start_hours.long()[:, None]) & (
            time < end_hours.long()[:, None].clamp(max=steps)
        )
        count = mask.sum(dim=1).clamp_min(1)
        mean = (values * mask.to(values.dtype)).sum(dim=1) / count
        minimum = values.masked_fill(~mask, torch.inf).amin(dim=1)
        minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
        return mean, minimum

    @staticmethod
    def _gather_completed(values: torch.Tensor, hours: torch.Tensor) -> torch.Tensor:
        indices = (hours.long() - 1).clamp(0, values.shape[1] - 1)
        gathered = values.gather(1, indices[:, None]).squeeze(1)
        return torch.where(hours > 0.0, gathered, torch.zeros_like(gathered))

    def _receiving_features(
        self,
        terminal_fill: torch.Tensor,
        well: torch.Tensor,
        injectivity: torch.Tensor,
        receiving_hours: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        start = (receiving_hours - self.RECEIVING_WINDOW_HOURS).clamp_min(0.0)
        end = receiving_hours + self.RECEIVING_WINDOW_HOURS + 1
        well_mean, well_min = self._masked_stats(well, start, end)
        injectivity_mean, _injectivity_min = self._masked_stats(
            injectivity, start, end
        )
        effective_injection = (well * injectivity).cumsum(dim=1)
        injected = self._gather_completed(effective_injection, receiving_hours)
        projected_terminal_fill = terminal_fill - injected / self.TERMINAL_DRAIN_HOURS
        terminal_headroom = 1.0 - projected_terminal_fill.clamp(0.0, 1.0)
        receiving_score = terminal_headroom * well_mean * injectivity_mean
        return (
            projected_terminal_fill,
            terminal_headroom,
            well_mean,
            well_min,
            injectivity_mean,
            receiving_score,
        )

    def _local_action_summaries(
        self, state: torch.Tensor, forecast: torch.Tensor
    ) -> torch.Tensor:
        batch, steps = forecast.shape[:2]
        weather = forecast.index_select(2, self.weather_indices).mean(dim=2)
        well = forecast.index_select(2, self.well_indices).mean(dim=2)
        injectivity = forecast.index_select(2, self.injectivity_indices).mean(dim=2)
        captures = forecast.index_select(2, self.capture_indices)
        availability = forecast.index_select(2, self.available_indices)
        emitter_fill = state.index_select(1, self.emitter_fill_indices)
        terminal_fill = state[:, self.terminal_fill_index]
        current_weather = state[:, self.weather_now_index].clamp_min(1e-3)
        cumulative_capture = captures.cumsum(dim=1)
        projected_fill = (
            emitter_fill[:, None, :]
            + cumulative_capture / self.fill_hours[None, None, :]
        )
        overflow_reached = projected_fill >= 1.0
        overflow_within = overflow_reached.any(dim=1)
        overflow_first = (
            overflow_reached.to(torch.int64).argmax(dim=1).to(forecast.dtype) + 1.0
        )
        overflow_hours = torch.where(
            overflow_within,
            overflow_first,
            torch.full_like(overflow_first, float(steps)),
        )
        urgent_emitter = projected_fill.amax(dim=1).argmax(dim=1)
        urgent_hours = overflow_hours.gather(
            1, urgent_emitter[:, None]
        ).squeeze(1)
        urgent_within = overflow_within.gather(
            1, urgent_emitter[:, None]
        ).squeeze(1).to(forecast.dtype)

        vessel_summaries = []
        for vessel_index in range(self.travel_indices.shape[0]):
            action_summaries = []
            for action in range(self.follow_action + 1):
                if action == self.follow_action:
                    action_summaries.append(
                        torch.zeros(
                            batch,
                            self.summary_size,
                            dtype=forecast.dtype,
                            device=forecast.device,
                        )
                    )
                    continue
                if action == 0:
                    arrival_hours = urgent_hours
                    within = urgent_within
                    emitter_index = urgent_emitter
                    receiving_hours = arrival_hours
                else:
                    destination_index = action - 1
                    travel_now = (
                        state[:, self.travel_indices[vessel_index, destination_index]]
                        * self.episode_hours
                    )
                    work_hours = travel_now * current_weather
                    arrival_hours, within = self._arrival_hours(work_hours, weather)
                    emitter_index = (
                        torch.zeros(
                            batch, dtype=torch.long, device=forecast.device
                        )
                        if action == 1
                        else torch.full(
                            (batch,),
                            action - 2,
                            dtype=torch.long,
                            device=forecast.device,
                        )
                    )
                    if action == 1:
                        receiving_hours = arrival_hours
                    else:
                        departure_hours = (
                            arrival_hours + float(self.LOAD_HOURS)
                        ).clamp(max=float(steps))
                        return_work = self.return_hours[action - 2].expand(batch)
                        receiving_hours, _return_within = self._arrival_hours(
                            return_work, weather, departure_hours
                        )

                travel_mean, travel_min = self._masked_stats(
                    weather, torch.zeros_like(arrival_hours), arrival_hours
                )
                arrival_start = (
                    arrival_hours - self.ARRIVAL_WINDOW_HOURS
                ).clamp_min(0.0)
                arrival_end = arrival_hours + self.ARRIVAL_WINDOW_HOURS + 1
                if action == 1:
                    current_fill = torch.zeros_like(terminal_fill)
                    arrival_fill = torch.zeros_like(terminal_fill)
                    overflow = torch.zeros_like(terminal_fill)
                    capture_delta = torch.zeros_like(terminal_fill)
                    capture_near = torch.zeros_like(terminal_fill)
                    available_before = torch.zeros_like(terminal_fill)
                    available_near = torch.zeros_like(terminal_fill)
                    available_min = torch.zeros_like(terminal_fill)
                else:
                    current_fill = emitter_fill.gather(
                        1, emitter_index[:, None]
                    ).squeeze(1)
                    selected_projected = projected_fill.gather(
                        2,
                        emitter_index[:, None, None].expand(-1, steps, 1),
                    ).squeeze(2)
                    completed_fill = self._gather_completed(
                        selected_projected, arrival_hours
                    )
                    arrival_fill = torch.where(
                        arrival_hours > 0.0, completed_fill, current_fill
                    )
                    overflow = (arrival_fill - 1.0).clamp_min(0.0)
                    selected_capture = captures.gather(
                        2,
                        emitter_index[:, None, None].expand(-1, steps, 1),
                    ).squeeze(2)
                    selected_available = availability.gather(
                        2,
                        emitter_index[:, None, None].expand(-1, steps, 1),
                    ).squeeze(2)
                    capture_delta = arrival_fill - current_fill
                    capture_near, _capture_min = self._masked_stats(
                        selected_capture, arrival_start, arrival_end
                    )
                    available_before, _available_before_min = self._masked_stats(
                        selected_available,
                        torch.zeros_like(arrival_hours),
                        arrival_hours,
                    )
                    available_near, available_min = self._masked_stats(
                        selected_available, arrival_start, arrival_end
                    )
                receiving = self._receiving_features(
                    terminal_fill,
                    well,
                    injectivity,
                    receiving_hours,
                )
                remaining = (
                    float(steps) - receiving_hours
                ).clamp_min(0.0) / max(1.0, float(steps))
                action_summaries.append(
                    torch.stack(
                        (
                            arrival_hours / max(1.0, float(steps)),
                            within,
                            travel_mean,
                            travel_min,
                            current_fill,
                            arrival_fill,
                            overflow,
                            capture_delta,
                            capture_near,
                            available_before,
                            available_near,
                            available_min,
                            terminal_fill,
                            *receiving,
                            remaining,
                        ),
                        dim=1,
                    )
                )
            vessel_summaries.append(torch.stack(action_summaries, dim=1))
        return torch.stack(vessel_summaries, dim=1)

    def forward(
        self,
        recurrent: torch.Tensor,
        state: torch.Tensor,
        forecast: torch.Tensor,
        baseline_forecast: torch.Tensor,
    ) -> torch.Tensor:
        full_local = self._local_action_summaries(state, forecast)
        baseline_local = self._local_action_summaries(state, baseline_forecast)
        full_aligned = self._aligned_summaries(full_local)
        baseline_aligned = self._aligned_summaries(baseline_local)
        mask = self.summary_feature_mask.to(
            dtype=full_aligned.dtype, device=full_aligned.device
        )
        full_aligned = baseline_aligned + (
            full_aligned - baseline_aligned
        ) * mask
        return self._predict_quantiles(
            recurrent, full_aligned
        ) - self._predict_quantiles(recurrent, baseline_aligned)

    def joint_quantiles(
        self,
        recurrent: torch.Tensor,
        state: torch.Tensor,
        forecast: torch.Tensor,
    ) -> torch.Tensor:
        """Predict full Q distributions from state and action-aligned future."""

        local = self._local_action_summaries(state, forecast)
        return self._predict_quantiles(recurrent, self._aligned_summaries(local))

    def _aligned_summaries(self, local: torch.Tensor) -> torch.Tensor:
        aligned = []
        for vessel_index in range(self.joint_action_indices.shape[1]):
            actions = self.joint_action_indices[:, vessel_index]
            aligned.append(local[:, vessel_index, actions])
        return torch.stack(aligned, dim=2)

    def _predict_quantiles(
        self, recurrent: torch.Tensor, aligned: torch.Tensor
    ) -> torch.Tensor:
        action_features = self.local_action_embedding(self.joint_action_indices)
        action_features = action_features.unsqueeze(0).expand(
            recurrent.shape[0], -1, -1, -1
        )
        vessels = self.vessel_encoder(
            torch.cat((aligned, action_features), dim=-1)
        ).flatten(start_dim=2)
        repeated_state = recurrent[:, None].expand(
            -1, self.joint_action_indices.shape[0], -1
        )
        predicted = self.residual_head(
            torch.cat((repeated_state, vessels), dim=-1)
        ).reshape(
            recurrent.shape[0],
            self.joint_action_indices.shape[0],
            self.heads,
            self.quantiles,
        )
        return predicted.permute(0, 2, 1, 3)


class ArrivalTimeForecastResidual(nn.Module):
    """Read destination forecasts only around a weather-adjusted arrival time."""

    ARRIVAL_WINDOW_HOURS = 6
    SUMMARY_FEATURES = (
        "arrival_fraction",
        "arrival_within_forecast",
        "capture_near_arrival",
        "destination_available_mean",
        "destination_available_min",
        "well_available_mean",
        "well_available_min",
        "injectivity_mean",
    )

    def __init__(
        self,
        joint_actions: np.ndarray,
        state_feature_names: list[str],
        forecast_channel_names: list[str],
        episode_hours: int,
        hidden_size: int,
        heads: int,
        quantiles: int,
    ) -> None:
        super().__init__()
        state_names = list(state_feature_names)
        forecast_names = list(forecast_channel_names)
        capture_names = [
            name for name in forecast_names if name.startswith("capture.")
        ]
        emitter_ids = [name.split(".", 1)[1] for name in capture_names]
        vessel_ids = sorted(
            {
                name.split(".")[1]
                for name in state_names
                if name.startswith("greedy_proposal.")
            }
        )
        if not emitter_ids or not vessel_ids:
            raise ValueError(
                "arrival-time forecast requires emitter and vessel features"
            )
        destination_ids = ["oygarden_terminal", *emitter_ids]
        travel_indices = [
            [
                state_names.index(
                    f"{vessel_id}.to_{destination_id}.travel_hours_now"
                )
                for destination_id in destination_ids
            ]
            for vessel_id in vessel_ids
        ]
        follow_action = 2 + len(emitter_ids)
        proposal_indices = [
            [
                state_names.index(
                    f"greedy_proposal.{vessel_id}.native_action_{action}"
                )
                for action in range(follow_action)
            ]
            for vessel_id in vessel_ids
        ]
        capture_indices = [
            forecast_names.index(f"capture.{emitter_id}")
            for emitter_id in emitter_ids
        ]
        available_indices = [
            forecast_names.index(f"emitter_available.{emitter_id}")
            for emitter_id in emitter_ids
        ]
        well_indices = [
            index
            for index, name in enumerate(forecast_names)
            if name.startswith("well_available.")
        ]
        injectivity_indices = [
            index
            for index, name in enumerate(forecast_names)
            if name.startswith("injectivity.")
        ]
        weather_indices = [
            index
            for index, name in enumerate(forecast_names)
            if name.startswith("weather.")
        ]
        if not well_indices or not injectivity_indices or not weather_indices:
            raise ValueError(
                "arrival-time forecast requires well, injectivity, and weather channels"
            )
        joint_array = np.asarray(joint_actions, dtype=np.int64)
        local_action_count = int(joint_array.max()) + 1
        if (
            joint_array.shape[1] != len(vessel_ids)
            or local_action_count != follow_action + 1
        ):
            raise ValueError(
                "unexpected vessel action schema for arrival-time forecast"
            )
        self.register_buffer(
            "joint_action_indices", torch.as_tensor(joint_array, dtype=torch.long)
        )
        self.register_buffer(
            "travel_indices", torch.as_tensor(travel_indices, dtype=torch.long)
        )
        self.register_buffer(
            "proposal_indices", torch.as_tensor(proposal_indices, dtype=torch.long)
        )
        self.register_buffer(
            "capture_indices", torch.as_tensor(capture_indices, dtype=torch.long)
        )
        self.register_buffer(
            "available_indices", torch.as_tensor(available_indices, dtype=torch.long)
        )
        self.register_buffer(
            "well_indices", torch.as_tensor(well_indices, dtype=torch.long)
        )
        self.register_buffer(
            "injectivity_indices",
            torch.as_tensor(injectivity_indices, dtype=torch.long),
        )
        self.register_buffer(
            "weather_indices", torch.as_tensor(weather_indices, dtype=torch.long)
        )
        self.weather_now_index = state_names.index("weather.speed_now")
        self.episode_hours = float(episode_hours)
        self.follow_action = int(follow_action)
        self.local_action_embedding = nn.Embedding(local_action_count, 8)
        self.vessel_encoder = nn.Sequential(
            nn.Linear(len(self.SUMMARY_FEATURES) + 8, 32),
            nn.SiLU(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_size + len(vessel_ids) * 32, 64),
            nn.SiLU(),
            nn.Linear(64, heads * quantiles),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.heads = int(heads)
        self.quantiles = int(quantiles)

    def _local_action_summaries(
        self, state: torch.Tensor, forecast: torch.Tensor
    ) -> torch.Tensor:
        batch, steps = forecast.shape[:2]
        weather = forecast.index_select(2, self.weather_indices).mean(dim=2)
        well = forecast.index_select(2, self.well_indices).mean(dim=2)
        injectivity = forecast.index_select(2, self.injectivity_indices).mean(dim=2)
        captures = forecast.index_select(2, self.capture_indices)
        availability = forecast.index_select(2, self.available_indices)
        current_weather = state[:, self.weather_now_index].clamp_min(1e-3)
        zero = torch.zeros(
            batch, dtype=forecast.dtype, device=forecast.device
        )
        native_by_vessel = []
        for vessel_index in range(self.travel_indices.shape[0]):
            native = [
                torch.zeros(
                    batch,
                    len(self.SUMMARY_FEATURES),
                    dtype=forecast.dtype,
                    device=forecast.device,
                )
            ]
            for action in range(1, self.follow_action):
                destination_index = action - 1
                travel_now = (
                    state[:, self.travel_indices[vessel_index, destination_index]]
                    * self.episode_hours
                )
                work_hours = travel_now * current_weather
                arrival_hours, within = ETAAlignedForecastResidual._arrival_hours(
                    work_hours, weather
                )
                arrival_start = (
                    arrival_hours - self.ARRIVAL_WINDOW_HOURS
                ).clamp_min(0.0)
                arrival_end = arrival_hours + self.ARRIVAL_WINDOW_HOURS + 1
                if action == 1:
                    capture_near = zero
                    available_mean = zero
                    available_min = zero
                    well_mean, well_min = ETAAlignedForecastResidual._masked_stats(
                        well, arrival_start, arrival_end
                    )
                    injectivity_mean, _injectivity_min = (
                        ETAAlignedForecastResidual._masked_stats(
                            injectivity, arrival_start, arrival_end
                        )
                    )
                else:
                    emitter_index = action - 2
                    capture_near, _capture_min = (
                        ETAAlignedForecastResidual._masked_stats(
                            captures[:, :, emitter_index],
                            arrival_start,
                            arrival_end,
                        )
                    )
                    available_mean, available_min = (
                        ETAAlignedForecastResidual._masked_stats(
                            availability[:, :, emitter_index],
                            arrival_start,
                            arrival_end,
                        )
                    )
                    well_mean = zero
                    well_min = zero
                    injectivity_mean = zero
                native.append(
                    torch.stack(
                        (
                            arrival_hours / max(1.0, float(steps)),
                            within,
                            capture_near,
                            available_mean,
                            available_min,
                            well_mean,
                            well_min,
                            injectivity_mean,
                        ),
                        dim=1,
                    )
                )
            native = torch.stack(native, dim=1)
            proposal = state[:, self.proposal_indices[vessel_index]]
            follow = torch.einsum("baf,ba->bf", native, proposal)
            native_by_vessel.append(torch.cat((native, follow[:, None]), dim=1))
        return torch.stack(native_by_vessel, dim=1)

    def forward(
        self,
        state_features: torch.Tensor,
        state: torch.Tensor,
        forecast: torch.Tensor,
        baseline_forecast: torch.Tensor,
    ) -> torch.Tensor:
        full_local = self._local_action_summaries(state, forecast)
        baseline_local = self._local_action_summaries(state, baseline_forecast)
        full_aligned = []
        baseline_aligned = []
        for vessel_index in range(self.joint_action_indices.shape[1]):
            actions = self.joint_action_indices[:, vessel_index]
            full_aligned.append(full_local[:, vessel_index, actions])
            baseline_aligned.append(baseline_local[:, vessel_index, actions])
        full_aligned = torch.stack(full_aligned, dim=2)
        baseline_aligned = torch.stack(baseline_aligned, dim=2)
        action_features = self.local_action_embedding(self.joint_action_indices)
        action_features = action_features.unsqueeze(0).expand(
            forecast.shape[0], -1, -1, -1
        )
        full_vessels = self.vessel_encoder(
            torch.cat((full_aligned, action_features), dim=-1)
        ).flatten(start_dim=2)
        baseline_vessels = self.vessel_encoder(
            torch.cat((baseline_aligned, action_features), dim=-1)
        ).flatten(start_dim=2)
        repeated_state = state_features[:, None].expand(
            -1, self.joint_action_indices.shape[0], -1
        )
        full = self.residual_head(
            torch.cat((repeated_state, full_vessels), dim=-1)
        )
        baseline = self.residual_head(
            torch.cat((repeated_state, baseline_vessels), dim=-1)
        )
        residual = (full - baseline).reshape(
            forecast.shape[0],
            self.joint_action_indices.shape[0],
            self.heads,
            self.quantiles,
        )
        return residual.permute(0, 2, 1, 3)


class StructuredActionRecurrentQuantileQ(RecurrentBootstrappedQuantileQ):
    """Recurrent Q-network that shares value structure across joint actions."""

    def __init__(
        self,
        state_feature_names: list[str],
        forecast_shape: tuple[int, int],
        joint_actions: np.ndarray | list[list[int]],
        *,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        forecast_mean: np.ndarray,
        forecast_std: np.ndarray,
        return_scale: float,
        heads: int = 5,
        quantiles: int = 51,
        hidden_size: int = 128,
        prior_scale: float = 0.25,
        action_embedding_size: int = 16,
        action_feature_size: int = 64,
        forecast_encoder: str = "tcn",
        forecast_channel_names: list[str] | None = None,
        episode_hours: int = 720,
    ) -> None:
        joint_array = np.asarray(joint_actions, dtype=np.int64)
        if joint_array.ndim != 2 or len(joint_array) == 0:
            raise ValueError("joint_actions must be a non-empty action-by-vessel matrix")
        if (joint_array < 0).any():
            raise ValueError("joint action indices must be non-negative")
        base_forecast_encoder = (
            "tcn"
            if forecast_encoder in ("action_aligned", "eta_aligned")
            else forecast_encoder
        )
        super().__init__(
            state_feature_names,
            forecast_shape,
            len(joint_array),
            state_mean=state_mean,
            state_std=state_std,
            forecast_mean=forecast_mean,
            forecast_std=forecast_std,
            return_scale=return_scale,
            heads=heads,
            quantiles=quantiles,
            hidden_size=hidden_size,
            prior_scale=prior_scale,
            forecast_encoder=base_forecast_encoder,
        )
        del self.advantage
        del self.prior
        self.action_feature_size = int(action_feature_size)
        self.register_buffer(
            "joint_action_indices", torch.as_tensor(joint_array, dtype=torch.long)
        )
        cardinalities = [int(joint_array[:, index].max()) + 1 for index in range(joint_array.shape[1])]
        self.structured_action_embeddings = nn.ModuleList(
            [nn.Embedding(size, action_embedding_size) for size in cardinalities]
        )
        self.structured_action_fusion = nn.Sequential(
            nn.Linear(len(cardinalities) * action_embedding_size, action_feature_size),
            nn.SiLU(),
            nn.LayerNorm(action_feature_size, elementwise_affine=False),
        )
        self.structured_query = nn.Linear(
            hidden_size, heads * quantiles * action_feature_size
        )
        self.structured_prior_embeddings = nn.ModuleList(
            [nn.Embedding(size, action_embedding_size) for size in cardinalities]
        )
        self.structured_prior_fusion = nn.Sequential(
            nn.Linear(len(cardinalities) * action_embedding_size, action_feature_size),
            nn.SiLU(),
            nn.LayerNorm(action_feature_size, elementwise_affine=False),
        )
        self.structured_prior_query = nn.Linear(
            hidden_size, heads * quantiles * action_feature_size
        )
        for module in (
            self.structured_prior_embeddings,
            self.structured_prior_fusion,
            self.structured_prior_query,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.action_aligned_residual = None
        self.eta_aligned_residual = None
        if forecast_encoder == "action_aligned":
            if forecast_channel_names is None:
                raise ValueError(
                    "action-aligned forecast encoder requires forecast channel names"
                )
            self.action_aligned_residual = ActionAlignedForecastResidual(
                joint_array,
                forecast_channel_names,
                hidden_size,
                heads,
                quantiles,
            )
        elif forecast_encoder == "eta_aligned":
            if forecast_channel_names is None:
                raise ValueError(
                    "ETA-aligned forecast encoder requires forecast channel names"
                )
            self.eta_aligned_residual = ETAAlignedForecastResidual(
                joint_array,
                state_feature_names,
                forecast_channel_names,
                episode_hours,
                hidden_size,
                heads,
                quantiles,
            )

    def forward(
        self,
        states: torch.Tensor,
        forecasts: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        previous_durations: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.action_aligned_residual is None
            and self.eta_aligned_residual is None
        ):
            return super().forward(
                states,
                forecasts,
                previous_actions,
                previous_rewards,
                previous_durations,
                hidden,
            )
        base_forecasts = self.forecast_mean.reshape(1, 1, 1, -1).expand_as(
            forecasts
        )
        recurrent, next_hidden = self.recurrent_features(
            states,
            base_forecasts,
            previous_actions,
            previous_rewards,
            previous_durations,
            hidden,
        )
        base_q = self.quantiles_from_features(recurrent)
        batch, sequence = recurrent.shape[:2]
        if self.eta_aligned_residual is not None:
            residual = self.eta_aligned_residual(
                recurrent.reshape(batch * sequence, -1),
                states.reshape(batch * sequence, -1),
                forecasts.reshape(batch * sequence, *forecasts.shape[2:]),
                base_forecasts.reshape(
                    batch * sequence, *base_forecasts.shape[2:]
                ),
            ).reshape_as(base_q)
        else:
            normalized_forecast = (
                forecasts - self.forecast_mean
            ) / self.forecast_std
            residual = self.action_aligned_residual(
                recurrent.reshape(batch * sequence, -1),
                normalized_forecast.reshape(
                    batch * sequence, *normalized_forecast.shape[2:]
                ),
            ).reshape_as(base_q)
        return base_q + residual, next_hidden

    def _action_features(self, embeddings, fusion) -> torch.Tensor:
        pieces = [
            embedding(self.joint_action_indices[:, vessel_index])
            for vessel_index, embedding in enumerate(embeddings)
        ]
        return fusion(torch.cat(pieces, dim=-1))

    def quantiles_from_features(self, recurrent: torch.Tensor) -> torch.Tensor:
        batch, sequence = recurrent.shape[:2]
        action_features = self._action_features(
            self.structured_action_embeddings, self.structured_action_fusion
        )
        query = self.structured_query(recurrent).reshape(
            batch,
            sequence,
            self.heads,
            self.quantiles,
            self.action_feature_size,
        )
        advantage = torch.einsum("bshqk,ak->bshaq", query, action_features)
        advantage = advantage / np.sqrt(self.action_feature_size)
        prior_action_features = self._action_features(
            self.structured_prior_embeddings, self.structured_prior_fusion
        )
        prior_query = self.structured_prior_query(recurrent.detach()).reshape_as(query)
        prior = torch.einsum(
            "bshqk,ak->bshaq", prior_query, prior_action_features
        ) / np.sqrt(self.action_feature_size)
        value = self.value(recurrent).reshape(
            batch, sequence, self.heads, 1, self.quantiles
        )
        q = value + advantage - advantage.mean(dim=3, keepdim=True)
        return q + self.prior_scale * prior


class StatelessGRUGate(nn.Module):
    """The exact one-step GRU transform for a zero incoming hidden state."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_r, input_z, input_n = F.linear(
            inputs, self.weight_ih, self.bias_ih
        ).chunk(3, dim=-1)
        hidden_r, hidden_z, hidden_n = self.bias_hh.chunk(3)
        reset = torch.sigmoid(input_r + hidden_r)
        update = torch.sigmoid(input_z + hidden_z)
        candidate = torch.tanh(input_n + reset * hidden_n)
        return (1.0 - update) * candidate


class StatelessFutureMLPResidual(nn.Module):
    """Use current vessel state and destination-specific raw forecasts per action."""

    def __init__(
        self,
        joint_actions: np.ndarray,
        state_feature_names: list[str],
        forecast_shape: tuple[int, int],
        forecast_channel_names: list[str],
        hidden_size: int,
        heads: int,
        quantiles: int,
        temporal_attention: bool = False,
    ) -> None:
        super().__init__()
        state_names = list(state_feature_names)
        forecast_names = list(forecast_channel_names)
        capture_names = [
            name for name in forecast_names if name.startswith("capture.")
        ]
        emitter_ids = [name.split(".", 1)[1] for name in capture_names]
        vessel_ids = sorted(
            {
                name.split(".")[1]
                for name in state_names
                if name.startswith("greedy_proposal.")
            }
        )
        if not emitter_ids or not vessel_ids:
            raise ValueError("stateless small MLP requires emitter and vessel features")
        joint_array = np.asarray(joint_actions, dtype=np.int64)
        follow_action = 2 + len(emitter_ids)
        local_action_count = int(joint_array.max()) + 1
        if (
            joint_array.shape[1] != len(vessel_ids)
            or local_action_count != follow_action + 1
        ):
            raise ValueError("unexpected vessel action schema for stateless small MLP")

        vessel_indices = []
        proposal_indices = []
        for vessel_id in vessel_ids:
            vessel_indices.append(
                [
                    index
                    for index, name in enumerate(state_names)
                    if name.startswith(f"{vessel_id}.")
                    or name.startswith(f"greedy_proposal.{vessel_id}.")
                ]
            )
            proposal_indices.append(
                [
                    state_names.index(
                        f"greedy_proposal.{vessel_id}.native_action_{action}"
                    )
                    for action in range(follow_action)
                ]
            )
        if len({len(indices) for indices in vessel_indices}) != 1:
            raise ValueError("stateless small MLP requires equal vessel state blocks")

        capture_indices = [
            forecast_names.index(f"capture.{emitter_id}")
            for emitter_id in emitter_ids
        ]
        available_indices = [
            forecast_names.index(f"emitter_available.{emitter_id}")
            for emitter_id in emitter_ids
        ]
        well_indices = [
            index
            for index, name in enumerate(forecast_names)
            if name.startswith("well_available.")
        ]
        injectivity_indices = [
            index
            for index, name in enumerate(forecast_names)
            if name.startswith("injectivity.")
        ]
        weather_indices = [
            index
            for index, name in enumerate(forecast_names)
            if name.startswith("weather.")
        ]
        if not well_indices or not injectivity_indices or not weather_indices:
            raise ValueError(
                "stateless small MLP requires well, injectivity, and weather channels"
            )

        self.register_buffer(
            "joint_action_indices", torch.as_tensor(joint_array, dtype=torch.long)
        )
        self.register_buffer(
            "vessel_indices", torch.as_tensor(vessel_indices, dtype=torch.long)
        )
        self.register_buffer(
            "proposal_indices", torch.as_tensor(proposal_indices, dtype=torch.long)
        )
        self.register_buffer(
            "capture_indices", torch.as_tensor(capture_indices, dtype=torch.long)
        )
        self.register_buffer(
            "available_indices", torch.as_tensor(available_indices, dtype=torch.long)
        )
        self.register_buffer(
            "well_indices", torch.as_tensor(well_indices, dtype=torch.long)
        )
        self.register_buffer(
            "injectivity_indices",
            torch.as_tensor(injectivity_indices, dtype=torch.long),
        )
        self.register_buffer(
            "weather_indices", torch.as_tensor(weather_indices, dtype=torch.long)
        )
        self.follow_action = int(follow_action)
        self.local_action_count = int(local_action_count)
        self.future_steps = int(forecast_shape[0])
        self.temporal_attention = bool(temporal_attention)
        if self.temporal_attention:
            self.attention_heads = 4
            self.attention_head_size = 8
            attention_size = self.attention_heads * self.attention_head_size
            query_input_size = len(vessel_indices[0]) + self.follow_action
            self.query_projection = nn.Sequential(
                nn.Linear(query_input_size, attention_size),
                nn.SiLU(),
            )
            self.key_projection = nn.Linear(7, attention_size)
            self.value_projection = nn.Sequential(
                nn.Linear(7, attention_size),
                nn.SiLU(),
            )
            self.attention_output = nn.Sequential(
                nn.Linear(attention_size, 32),
                nn.SiLU(),
            )
        else:
            local_input_size = (
                len(vessel_indices[0])
                + self.follow_action
                + self.future_steps * 5
            )
            self.local_encoder = nn.Sequential(
                nn.Linear(local_input_size, 64),
                nn.SiLU(),
                nn.Linear(64, 32),
                nn.SiLU(),
            )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_size + len(vessel_ids) * 32, 64),
            nn.SiLU(),
            nn.Linear(64, heads * quantiles),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.heads = int(heads)
        self.quantiles = int(quantiles)

    def _selected_future(
        self,
        state: torch.Tensor,
        forecast: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps = forecast.shape[:2]
        if steps != self.future_steps:
            raise ValueError(
                f"stateless small MLP expected {self.future_steps} forecast steps"
            )
        captures = forecast.index_select(2, self.capture_indices)
        available = forecast.index_select(2, self.available_indices)
        zeros = torch.zeros(
            batch, steps, 2, dtype=forecast.dtype, device=forecast.device
        )
        native_capture = torch.cat((zeros, captures), dim=2)
        native_available = torch.cat((zeros, available), dim=2)
        proposals = state[:, self.proposal_indices]
        follow_capture = torch.einsum("btp,bvp->bvt", native_capture, proposals)
        follow_available = torch.einsum("btp,bvp->bvt", native_available, proposals)
        vessel_count = int(self.vessel_indices.shape[0])
        native_capture = native_capture[:, None].expand(-1, vessel_count, -1, -1)
        native_available = native_available[:, None].expand(
            -1, vessel_count, -1, -1
        )
        local_capture = torch.cat(
            (native_capture.permute(0, 1, 3, 2), follow_capture[:, :, None]),
            dim=2,
        )
        local_available = torch.cat(
            (native_available.permute(0, 1, 3, 2), follow_available[:, :, None]),
            dim=2,
        )
        well = forecast.index_select(2, self.well_indices).mean(dim=2)
        injectivity = forecast.index_select(2, self.injectivity_indices).mean(dim=2)
        weather = forecast.index_select(2, self.weather_indices).mean(dim=2)
        global_shape = (
            batch,
            vessel_count,
            self.local_action_count,
            steps,
        )
        selected = torch.stack(
            (
                local_capture,
                local_available,
                well[:, None, None].expand(global_shape),
                injectivity[:, None, None].expand(global_shape),
                weather[:, None, None].expand(global_shape),
            ),
            dim=-1,
        )
        native_actions = torch.eye(
            self.follow_action, dtype=state.dtype, device=state.device
        )
        native_actions = native_actions[None, None].expand(
            batch, vessel_count, -1, -1
        )
        local_actions = torch.cat((native_actions, proposals[:, :, None]), dim=2)
        return selected, local_actions

    def _local_features(
        self,
        state: torch.Tensor,
        normalized_state: torch.Tensor,
        normalized_forecast: torch.Tensor,
        *,
        zero_future: bool,
    ) -> torch.Tensor:
        selected, local_actions = self._selected_future(
            state, normalized_forecast
        )
        if zero_future:
            selected = torch.zeros_like(selected)
        vessel_state = normalized_state[:, self.vessel_indices]
        vessel_state = vessel_state[:, :, None].expand(
            -1, -1, self.local_action_count, -1
        )
        if self.temporal_attention:
            query = self.query_projection(
                torch.cat((vessel_state, local_actions), dim=3)
            )
            time = torch.linspace(
                0.0,
                1.0,
                self.future_steps,
                dtype=selected.dtype,
                device=selected.device,
            )
            time_features = torch.stack((time, time.square()), dim=1)
            time_features = time_features.reshape(
                1, 1, 1, self.future_steps, 2
            ).expand(*selected.shape[:-1], 2)
            temporal_inputs = torch.cat((selected, time_features), dim=4)
            keys = self.key_projection(temporal_inputs)
            values = self.value_projection(temporal_inputs)
            query = query.reshape(
                *query.shape[:-1],
                self.attention_heads,
                self.attention_head_size,
            )
            keys = keys.reshape(
                *keys.shape[:-1],
                self.attention_heads,
                self.attention_head_size,
            )
            values = values.reshape_as(keys)
            scores = torch.einsum(
                "bvahd,bvathd->bvaht", query, keys
            ) / np.sqrt(self.attention_head_size)
            weights = torch.softmax(scores, dim=-1)
            attended = torch.einsum(
                "bvaht,bvathd->bvahd", weights, values
            ).flatten(start_dim=3)
            return self.attention_output(attended)
        inputs = torch.cat(
            (vessel_state, local_actions, selected.flatten(start_dim=3)),
            dim=3,
        )
        batch, vessels, actions = inputs.shape[:3]
        return self.local_encoder(inputs.reshape(batch * vessels * actions, -1)).reshape(
            batch, vessels, actions, -1
        )

    def _joint_features(self, local_features: torch.Tensor) -> torch.Tensor:
        pieces = [
            local_features[:, vessel_index, self.joint_action_indices[:, vessel_index]]
            for vessel_index in range(self.joint_action_indices.shape[1])
        ]
        return torch.cat(pieces, dim=2)

    def forward(
        self,
        state_features: torch.Tensor,
        state: torch.Tensor,
        normalized_state: torch.Tensor,
        normalized_forecast: torch.Tensor,
    ) -> torch.Tensor:
        full_local = self._local_features(
            state,
            normalized_state,
            normalized_forecast,
            zero_future=False,
        )
        baseline_local = self._local_features(
            state,
            normalized_state,
            normalized_forecast,
            zero_future=True,
        )
        full_joint = self._joint_features(full_local)
        baseline_joint = self._joint_features(baseline_local)
        repeated_state = state_features[:, None].expand(
            -1, self.joint_action_indices.shape[0], -1
        )
        full = self.residual_head(torch.cat((repeated_state, full_joint), dim=2))
        baseline = self.residual_head(
            torch.cat((repeated_state, baseline_joint), dim=2)
        )
        residual = (full - baseline).reshape(
            state.shape[0],
            self.joint_action_indices.shape[0],
            self.heads,
            self.quantiles,
        )
        return residual.permute(0, 2, 1, 3)


class WindowSummaryForecastResidual(nn.Module):
    """Use fixed-horizon availability, injectivity, and weather summaries."""

    def __init__(
        self,
        joint_actions: np.ndarray,
        forecast_channel_names: list[str],
        horizons: tuple[int, ...],
        hidden_size: int,
        heads: int,
        quantiles: int,
    ) -> None:
        super().__init__()
        forecast_names = list(forecast_channel_names)
        available_indices = [
            index
            for index, name in enumerate(forecast_names)
            if name.startswith("emitter_available.")
        ]
        well_indices = [
            index
            for index, name in enumerate(forecast_names)
            if name.startswith("well_available.")
        ]
        injectivity_indices = [
            index
            for index, name in enumerate(forecast_names)
            if name.startswith("injectivity.")
        ]
        weather_indices = [
            index
            for index, name in enumerate(forecast_names)
            if name.startswith("weather.")
        ]
        if not available_indices or not well_indices:
            raise ValueError(
                "window summary requires emitter and well availability channels"
            )
        if not injectivity_indices or not weather_indices:
            raise ValueError(
                "window summary requires injectivity and weather channels"
            )
        if not horizons or min(horizons) <= 0:
            raise ValueError("window summary horizons must be positive")
        joint_array = np.asarray(joint_actions, dtype=np.int64)
        if max(horizons) > 168:
            raise ValueError("window summary horizon exceeds 168-hour forecast")

        self.register_buffer(
            "joint_action_indices", torch.as_tensor(joint_array, dtype=torch.long)
        )
        self.register_buffer(
            "available_indices",
            torch.as_tensor(available_indices, dtype=torch.long),
        )
        self.register_buffer(
            "well_indices", torch.as_tensor(well_indices, dtype=torch.long)
        )
        self.register_buffer(
            "injectivity_indices",
            torch.as_tensor(injectivity_indices, dtype=torch.long),
        )
        self.register_buffer(
            "weather_indices", torch.as_tensor(weather_indices, dtype=torch.long)
        )
        self.horizons = tuple(int(horizon) for horizon in horizons)
        summary_size = len(self.horizons) * (
            len(available_indices)
            + len(well_indices)
            + len(injectivity_indices)
            + 2
        )
        self.summary_encoder = nn.Sequential(
            nn.Linear(summary_size, 32),
            nn.SiLU(),
        )
        cardinalities = [
            int(joint_array[:, index].max()) + 1
            for index in range(joint_array.shape[1])
        ]
        self.action_embeddings = nn.ModuleList(
            [nn.Embedding(size, 8) for size in cardinalities]
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(len(cardinalities) * 8, 32),
            nn.SiLU(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_size + 64, 64),
            nn.SiLU(),
            nn.Linear(64, heads * quantiles),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.heads = int(heads)
        self.quantiles = int(quantiles)

    def _summaries(self, forecast: torch.Tensor) -> torch.Tensor:
        pieces = []
        for horizon in self.horizons:
            window = forecast[:, : min(horizon, forecast.shape[1])]
            emitter_available = window.index_select(
                2, self.available_indices
            ).mean(dim=1)
            well_available = window.index_select(
                2, self.well_indices
            ).mean(dim=1)
            injectivity_min = window.index_select(
                2, self.injectivity_indices
            ).amin(dim=1)
            weather = window.index_select(2, self.weather_indices)
            weather_mean = weather.mean(dim=(1, 2), keepdim=False)[:, None]
            weather_min = weather.amin(dim=(1, 2), keepdim=False)[:, None]
            pieces.extend(
                (
                    emitter_available,
                    well_available,
                    injectivity_min,
                    weather_mean,
                    weather_min,
                )
            )
        return torch.cat(pieces, dim=1)

    def _predict(
        self, state_features: torch.Tensor, forecast: torch.Tensor
    ) -> torch.Tensor:
        summary = self.summary_encoder(self._summaries(forecast))
        action_pieces = [
            embedding(self.joint_action_indices[:, vessel_index])
            for vessel_index, embedding in enumerate(self.action_embeddings)
        ]
        action_features = self.action_encoder(torch.cat(action_pieces, dim=1))
        batch = state_features.shape[0]
        joint_actions = self.joint_action_indices.shape[0]
        inputs = torch.cat(
            (
                state_features[:, None].expand(-1, joint_actions, -1),
                summary[:, None].expand(-1, joint_actions, -1),
                action_features[None].expand(batch, -1, -1),
            ),
            dim=2,
        )
        predicted = self.residual_head(inputs).reshape(
            batch, joint_actions, self.heads, self.quantiles
        )
        return predicted.permute(0, 2, 1, 3)

    def forward(
        self,
        state_features: torch.Tensor,
        forecast: torch.Tensor,
        baseline_forecast: torch.Tensor,
    ) -> torch.Tensor:
        return self._predict(state_features, forecast) - self._predict(
            state_features, baseline_forecast
        )


class StatelessStructuredActionQuantileQ(nn.Module):
    """Structured Q-network with current-state and optional future inputs."""

    is_stateless = True

    def __init__(
        self,
        state_feature_names: list[str],
        forecast_shape: tuple[int, int],
        joint_actions: np.ndarray | list[list[int]],
        *,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        forecast_mean: np.ndarray,
        forecast_std: np.ndarray,
        return_scale: float,
        heads: int = 5,
        quantiles: int = 51,
        hidden_size: int = 128,
        prior_scale: float = 0.25,
        action_embedding_size: int = 16,
        action_feature_size: int = 64,
        forecast_encoder: str = "state_only",
        forecast_channel_names: list[str] | None = None,
        episode_hours: int = 720,
    ) -> None:
        super().__init__()
        joint_array = np.asarray(joint_actions, dtype=np.int64)
        if joint_array.ndim != 2 or len(joint_array) == 0:
            raise ValueError("joint_actions must be a non-empty action-by-vessel matrix")
        if (joint_array < 0).any():
            raise ValueError("joint action indices must be non-negative")
        if forecast_encoder not in (
            "state_only",
            "small_mlp",
            "temporal_attention",
            "action_aligned",
            "arrival_time",
            "eta_aligned",
            "eta_joint",
            "window_summary_24_72",
            "window_summary_168",
            "window_summary_24_72_168",
            "window_summary_joint_168",
        ):
            raise ValueError(
                "stateless forecast encoder must be state_only, small_mlp, "
                "temporal_attention, action_aligned, arrival_time, eta_aligned, "
                "eta_joint, or a window_summary encoder"
            )
        self.action_count = int(len(joint_array))
        self.heads = int(heads)
        self.quantiles = int(quantiles)
        self.hidden_size = int(hidden_size)
        self.return_scale = float(return_scale)
        self.prior_scale = float(prior_scale)
        self.action_feature_size = int(action_feature_size)
        self.forecast_encoder_name = str(forecast_encoder)
        self.register_buffer(
            "state_mean", torch.as_tensor(state_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "state_std", torch.as_tensor(state_std, dtype=torch.float32)
        )
        if forecast_encoder in (
            "small_mlp",
            "temporal_attention",
            "action_aligned",
            "arrival_time",
            "eta_aligned",
            "eta_joint",
            "window_summary_24_72",
            "window_summary_168",
            "window_summary_24_72_168",
            "window_summary_joint_168",
        ):
            self.register_buffer(
                "forecast_mean",
                torch.as_tensor(forecast_mean, dtype=torch.float32),
            )
            self.register_buffer(
                "forecast_std",
                torch.as_tensor(forecast_std, dtype=torch.float32),
            )
        self.state_encoder = EntityStateEncoder(state_feature_names, 64)
        self.state_projection = nn.Sequential(
            nn.Linear(64, hidden_size),
            nn.SiLU(),
        )
        self.stateless_gate = StatelessGRUGate(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, self.heads * self.quantiles)
        self.register_buffer(
            "joint_action_indices", torch.as_tensor(joint_array, dtype=torch.long)
        )
        cardinalities = [
            int(joint_array[:, index].max()) + 1
            for index in range(joint_array.shape[1])
        ]
        self.structured_action_embeddings = nn.ModuleList(
            [nn.Embedding(size, action_embedding_size) for size in cardinalities]
        )
        self.structured_action_fusion = nn.Sequential(
            nn.Linear(
                len(cardinalities) * action_embedding_size,
                action_feature_size,
            ),
            nn.SiLU(),
            nn.LayerNorm(action_feature_size, elementwise_affine=False),
        )
        self.structured_query = nn.Linear(
            hidden_size, heads * quantiles * action_feature_size
        )
        self.structured_prior_embeddings = nn.ModuleList(
            [nn.Embedding(size, action_embedding_size) for size in cardinalities]
        )
        self.structured_prior_fusion = nn.Sequential(
            nn.Linear(
                len(cardinalities) * action_embedding_size,
                action_feature_size,
            ),
            nn.SiLU(),
            nn.LayerNorm(action_feature_size, elementwise_affine=False),
        )
        self.structured_prior_query = nn.Linear(
            hidden_size, heads * quantiles * action_feature_size
        )
        for module in (
            self.structured_prior_embeddings,
            self.structured_prior_fusion,
            self.structured_prior_query,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.small_mlp_residual = None
        self.temporal_attention_residual = None
        self.action_aligned_residual = None
        self.arrival_time_residual = None
        self.eta_aligned_residual = None
        self.eta_joint_q = None
        self.window_summary_residual = None
        self.window_summary_joint_q = None
        if forecast_encoder == "small_mlp":
            if forecast_channel_names is None:
                raise ValueError(
                    "stateless small MLP requires forecast channel names"
                )
            self.small_mlp_residual = StatelessFutureMLPResidual(
                joint_array,
                state_feature_names,
                forecast_shape,
                forecast_channel_names,
                hidden_size,
                heads,
                quantiles,
            )
        elif forecast_encoder == "temporal_attention":
            if forecast_channel_names is None:
                raise ValueError(
                    "temporal attention requires forecast channel names"
                )
            self.temporal_attention_residual = StatelessFutureMLPResidual(
                joint_array,
                state_feature_names,
                forecast_shape,
                forecast_channel_names,
                hidden_size,
                heads,
                quantiles,
                temporal_attention=True,
            )
        elif forecast_encoder == "action_aligned":
            if forecast_channel_names is None:
                raise ValueError(
                    "action-aligned forecast encoder requires forecast channel names"
                )
            self.action_aligned_residual = ActionAlignedForecastResidual(
                joint_array,
                forecast_channel_names,
                hidden_size,
                heads,
                quantiles,
            )
        elif forecast_encoder == "arrival_time":
            if forecast_channel_names is None:
                raise ValueError(
                    "arrival-time forecast encoder requires forecast channel names"
                )
            self.arrival_time_residual = ArrivalTimeForecastResidual(
                joint_array,
                state_feature_names,
                forecast_channel_names,
                episode_hours,
                hidden_size,
                heads,
                quantiles,
            )
        elif forecast_encoder == "eta_aligned":
            if forecast_channel_names is None:
                raise ValueError(
                    "ETA-aligned forecast encoder requires forecast channel names"
                )
            self.eta_aligned_residual = ETAAlignedForecastResidual(
                joint_array,
                state_feature_names,
                forecast_channel_names,
                episode_hours,
                hidden_size,
                heads,
                quantiles,
            )
        elif forecast_encoder == "eta_joint":
            if forecast_channel_names is None:
                raise ValueError(
                    "joint ETA forecast encoder requires forecast channel names"
                )
            self.eta_joint_q = ETAAlignedForecastResidual(
                joint_array,
                state_feature_names,
                forecast_channel_names,
                episode_hours,
                hidden_size,
                heads,
                quantiles,
            )
        elif forecast_encoder.startswith("window_summary_"):
            if forecast_channel_names is None:
                raise ValueError(
                    "window summary forecast encoder requires forecast channel names"
                )
            horizons = {
                "window_summary_24_72": (24, 72),
                "window_summary_168": (168,),
                "window_summary_24_72_168": (24, 72, 168),
                "window_summary_joint_168": (168,),
            }[forecast_encoder]
            module = WindowSummaryForecastResidual(
                joint_array,
                forecast_channel_names,
                horizons,
                hidden_size,
                heads,
                quantiles,
            )
            if forecast_encoder == "window_summary_joint_168":
                self.window_summary_joint_q = module
            else:
                self.window_summary_residual = module

    @classmethod
    def from_reset_recurrent(
        cls,
        source: StructuredActionRecurrentQuantileQ,
        state_feature_names: list[str],
        forecast_shape: tuple[int, int],
        joint_actions: np.ndarray | list[list[int]],
    ) -> "StatelessStructuredActionQuantileQ":
        if source.action_aligned_residual is not None or source.eta_aligned_residual is not None:
            raise ValueError("stateless conversion requires a state-only base network")
        padding = source.action_embedding.weight[source.action_count]
        if not torch.equal(padding, torch.zeros_like(padding)):
            raise ValueError("previous-action padding embedding must be exactly zero")

        target = cls(
            state_feature_names,
            forecast_shape,
            joint_actions,
            state_mean=source.state_mean.detach().cpu().numpy(),
            state_std=source.state_std.detach().cpu().numpy(),
            forecast_mean=source.forecast_mean.detach().cpu().numpy(),
            forecast_std=source.forecast_std.detach().cpu().numpy(),
            return_scale=source.return_scale,
            heads=source.heads,
            quantiles=source.quantiles,
            hidden_size=source.hidden_size,
            prior_scale=source.prior_scale,
            action_embedding_size=source.structured_action_embeddings[
                0
            ].embedding_dim,
            action_feature_size=source.action_feature_size,
            forecast_encoder="state_only",
        ).to(source.state_mean.device)

        shared_modules = (
            "state_encoder",
            "value",
            "structured_action_embeddings",
            "structured_action_fusion",
            "structured_query",
            "structured_prior_embeddings",
            "structured_prior_fusion",
            "structured_prior_query",
        )
        for name in shared_modules:
            getattr(target, name).load_state_dict(getattr(source, name).state_dict())

        with torch.no_grad():
            forecast_constant = source.forecast_encoder(
                torch.zeros(
                    (1, *forecast_shape),
                    dtype=source.state_mean.dtype,
                    device=source.state_mean.device,
                )
            )[0]
            projection = source.input_projection[0]
            target_projection = target.state_projection[0]
            target_projection.weight.copy_(projection.weight[:, :64])
            target_projection.bias.copy_(
                projection.bias
                + projection.weight[:, 64:128].matmul(forecast_constant)
            )
            target.stateless_gate.weight_ih.copy_(source.gru.weight_ih_l0)
            target.stateless_gate.bias_ih.copy_(source.gru.bias_ih_l0)
            target.stateless_gate.bias_hh.copy_(source.gru.bias_hh_l0)
        return target

    def _action_features(self, embeddings, fusion) -> torch.Tensor:
        pieces = [
            embedding(self.joint_action_indices[:, vessel_index])
            for vessel_index, embedding in enumerate(embeddings)
        ]
        return fusion(torch.cat(pieces, dim=-1))

    def quantiles_from_features(self, features: torch.Tensor) -> torch.Tensor:
        batch, sequence = features.shape[:2]
        action_features = self._action_features(
            self.structured_action_embeddings, self.structured_action_fusion
        )
        query = self.structured_query(features).reshape(
            batch,
            sequence,
            self.heads,
            self.quantiles,
            self.action_feature_size,
        )
        advantage = torch.einsum("bshqk,ak->bshaq", query, action_features)
        advantage = advantage / np.sqrt(self.action_feature_size)
        prior_action_features = self._action_features(
            self.structured_prior_embeddings, self.structured_prior_fusion
        )
        prior_query = self.structured_prior_query(features.detach()).reshape_as(
            query
        )
        prior = torch.einsum(
            "bshqk,ak->bshaq", prior_query, prior_action_features
        ) / np.sqrt(self.action_feature_size)
        value = self.value(features).reshape(
            batch, sequence, self.heads, 1, self.quantiles
        )
        q = value + advantage - advantage.mean(dim=3, keepdim=True)
        return q + self.prior_scale * prior

    def forward(
        self,
        states: torch.Tensor,
        forecasts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence = states.shape[:2]
        normalized_state = (states - self.state_mean) / self.state_std
        state_features = self.state_encoder(
            normalized_state.reshape(batch * sequence, -1)
        )
        projected = self.state_projection(state_features)
        features = self.stateless_gate(projected).reshape(batch, sequence, -1)
        q = self.quantiles_from_features(features)
        if self.window_summary_joint_q is not None:
            if forecasts is None:
                raise ValueError("future-aware stateless model requires forecasts")
            return self.window_summary_joint_q._predict(
                features.reshape(batch * sequence, -1),
                forecasts.reshape(batch * sequence, *forecasts.shape[2:]),
            ).reshape_as(q)
        if self.eta_joint_q is not None:
            if forecasts is None:
                raise ValueError("future-aware stateless model requires forecasts")
            return self.eta_joint_q.joint_quantiles(
                features.reshape(batch * sequence, -1),
                states.reshape(batch * sequence, -1),
                forecasts.reshape(batch * sequence, *forecasts.shape[2:]),
            ).reshape_as(q)
        if (
            self.small_mlp_residual is None
            and self.temporal_attention_residual is None
            and self.action_aligned_residual is None
            and self.arrival_time_residual is None
            and self.eta_aligned_residual is None
            and self.window_summary_residual is None
        ):
            return q
        if forecasts is None:
            raise ValueError("future-aware stateless model requires forecasts")
        raw_future_residual = (
            self.small_mlp_residual
            if self.small_mlp_residual is not None
            else self.temporal_attention_residual
        )
        if raw_future_residual is not None:
            normalized_forecast = (
                forecasts - self.forecast_mean
            ) / self.forecast_std
            residual = raw_future_residual(
                features.reshape(batch * sequence, -1),
                states.reshape(batch * sequence, -1),
                normalized_state.reshape(batch * sequence, -1),
                normalized_forecast.reshape(
                    batch * sequence, *normalized_forecast.shape[2:]
                ),
            ).reshape_as(q)
            return q + residual
        if self.action_aligned_residual is not None:
            normalized_forecast = (
                forecasts - self.forecast_mean
            ) / self.forecast_std
            residual = self.action_aligned_residual(
                features.reshape(batch * sequence, -1),
                normalized_forecast.reshape(
                    batch * sequence, *normalized_forecast.shape[2:]
                ),
            ).reshape_as(q)
            return q + residual
        base_forecasts = self.forecast_mean.reshape(1, 1, 1, -1).expand_as(
            forecasts
        )
        if self.window_summary_residual is not None:
            residual = self.window_summary_residual(
                features.reshape(batch * sequence, -1),
                forecasts.reshape(batch * sequence, *forecasts.shape[2:]),
                base_forecasts.reshape(
                    batch * sequence, *base_forecasts.shape[2:]
                ),
            ).reshape_as(q)
            return q + residual
        if self.arrival_time_residual is not None:
            residual = self.arrival_time_residual(
                features.reshape(batch * sequence, -1),
                states.reshape(batch * sequence, -1),
                forecasts.reshape(batch * sequence, *forecasts.shape[2:]),
                base_forecasts.reshape(
                    batch * sequence, *base_forecasts.shape[2:]
                ),
            ).reshape_as(q)
            return q + residual
        residual = self.eta_aligned_residual(
            features.reshape(batch * sequence, -1),
            states.reshape(batch * sequence, -1),
            forecasts.reshape(batch * sequence, *forecasts.shape[2:]),
            base_forecasts.reshape(
                batch * sequence, *base_forecasts.shape[2:]
            ),
        ).reshape_as(q)
        return q + residual


def quantile_huber_loss(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    *,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Pairwise QR-DQN loss; final axes are predicted and target quantiles."""

    if predicted.shape[-1] <= 0 or targets.shape[-1] <= 0:
        raise ValueError("quantile tensors must have non-empty final axes")
    error = targets.unsqueeze(-2) - predicted.unsqueeze(-1)
    absolute_error = error.abs()
    huber = torch.where(
        absolute_error <= kappa,
        0.5 * error.square(),
        kappa * (absolute_error - 0.5 * kappa),
    )
    count = predicted.shape[-1]
    tau = (
        torch.arange(count, device=predicted.device, dtype=predicted.dtype) + 0.5
    ) / count
    view_shape = [1] * (error.ndim - 2) + [count, 1]
    weight = (tau.reshape(view_shape) - (error.detach() < 0).to(predicted.dtype)).abs()
    return (weight * huber / kappa).mean(dim=(-2, -1))
