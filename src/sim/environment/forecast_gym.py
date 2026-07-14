"""Gymnasium adapter exposing current state and exact future forecasts."""

from __future__ import annotations

from typing import Literal

import numpy as np
from gymnasium import Env, spaces

from .env import CCSEnv
from .forecast import (
    current_state_feature_names,
    current_state_observation,
    future_forecast_observation,
)
from .gym_adapter import flat_action_mask, native_action_from_flat
from .vessel_mode import (
    vessel_operation_mode_observation,
    vessel_sailing_destination_observation,
)

ObservationVariant = Literal[
    "state",
    "flat",
    "tcn",
    "state_mode",
    "tcn_mode",
    "tcn_mode_destination",
    "gnn_mode_destination",
    "larger_mlp_mode_destination",
    "edge_gnn_mode_destination",
    "fixed_scale_larger_mlp_mode_destination",
    "fixed_scale_edge_gnn_mode_destination",
    "stable_tcn_mode_destination",
    "fixed_scale_tcn_mode_destination",
]


def variant_uses_operation_modes(variant: str) -> bool:
    return variant in {
        "state_mode",
        "tcn_mode",
        "tcn_mode_destination",
        "gnn_mode_destination",
        "larger_mlp_mode_destination",
        "edge_gnn_mode_destination",
        "fixed_scale_larger_mlp_mode_destination",
        "fixed_scale_edge_gnn_mode_destination",
        "stable_tcn_mode_destination",
        "fixed_scale_tcn_mode_destination",
    }


def variant_uses_sailing_destinations(variant: str) -> bool:
    return variant in {
        "tcn_mode_destination",
        "gnn_mode_destination",
        "larger_mlp_mode_destination",
        "edge_gnn_mode_destination",
        "fixed_scale_larger_mlp_mode_destination",
        "fixed_scale_edge_gnn_mode_destination",
        "stable_tcn_mode_destination",
        "fixed_scale_tcn_mode_destination",
    }


def variant_base_encoder(variant: str) -> str:
    if variant == "state_mode":
        return "state"
    if variant in {"tcn_mode", "tcn_mode_destination"}:
        return "tcn"
    if variant == "gnn_mode_destination":
        return "gnn"
    if variant == "larger_mlp_mode_destination":
        return "larger_mlp"
    if variant == "edge_gnn_mode_destination":
        return "edge_gnn"
    if variant == "fixed_scale_larger_mlp_mode_destination":
        return "fixed_scale_larger_mlp"
    if variant == "fixed_scale_edge_gnn_mode_destination":
        return "fixed_scale_edge_gnn"
    if variant == "stable_tcn_mode_destination":
        return "stable_tcn"
    if variant == "fixed_scale_tcn_mode_destination":
        return "fixed_scale_tcn"
    if variant in {"state", "flat", "tcn"}:
        return variant
    raise ValueError(f"unknown forecast observation variant: {variant}")


def forecast_policy_observation(
    env: CCSEnv,
    variant: ObservationVariant,
    *,
    timeout: bool = False,
):
    """Return the observation representation selected for a forecast policy."""
    state = np.asarray(current_state_observation(env), dtype=np.float32)
    if timeout:
        _apply_timeout_disturbances_to_observation(env, state)
    if variant_uses_operation_modes(variant):
        modes = np.asarray(vessel_operation_mode_observation(env), dtype=np.float32)
        state = np.concatenate((state, modes)).astype(np.float32, copy=False)
    if variant_uses_sailing_destinations(variant):
        destinations = np.asarray(
            vessel_sailing_destination_observation(env),
            dtype=np.float32,
        )
        state = np.concatenate((state, destinations)).astype(np.float32, copy=False)
    base_variant = variant_base_encoder(variant)
    if base_variant == "state":
        return state

    forecast = np.asarray(future_forecast_observation(env), dtype=np.float32)
    if base_variant == "flat":
        return np.concatenate((state, forecast.reshape(-1))).astype(np.float32)
    if base_variant in {
        "tcn",
        "gnn",
        "larger_mlp",
        "edge_gnn",
        "fixed_scale_larger_mlp",
        "fixed_scale_edge_gnn",
        "stable_tcn",
        "fixed_scale_tcn",
    }:
        return {"state": state, "forecast": forecast}
    raise AssertionError(f"unhandled forecast observation variant: {variant}")


def _apply_timeout_disturbances_to_observation(
    env: CCSEnv,
    state: np.ndarray,
) -> None:
    """Align exogenous state features with the timeout hour without mutation."""
    assert env.scenario is not None
    assert env.simulator is not None
    index = env.scenario.step_index(env.simulator.state.time_h)
    feature_index = {
        name: position
        for position, name in enumerate(current_state_feature_names(env))
    }
    for emitter_id in env.emitter_ids:
        values = env.scenario.emitter_availability.get(emitter_id)
        if values is not None:
            state[feature_index[f"{emitter_id}.availability"]] = values[index]
    for well_id in env.well_ids:
        availability = env.scenario.well_available.get(well_id)
        if availability is not None:
            state[feature_index[f"{well_id}.available"]] = float(availability[index])
        injectivity = env.scenario.injectivity_factor.get(well_id)
        if injectivity is not None:
            state[feature_index[f"{well_id}.injectivity"]] = injectivity[index]


class ForecastGymEnv(Env):
    """A Gymnasium view of ``CCSEnv`` with selectable forecast observations."""

    metadata = {"render_modes": []}

    def __init__(self, env: CCSEnv, variant: ObservationVariant):
        super().__init__()
        self.env = env
        self.variant = variant
        self.action_space = spaces.MultiDiscrete(
            env.vessel_action_dims + env.well_rate_action_dims
        )
        state_size = len(current_state_feature_names(env))
        if variant_uses_operation_modes(variant):
            state_size += 5 * len(env.vessel_ids)
        if variant_uses_sailing_destinations(variant):
            state_size += len(env.vessel_ids) * (
                len(env.terminal_ids) + len(env.emitter_ids)
            )
        base_variant = variant_base_encoder(variant)
        if base_variant == "state":
            self.observation_space = spaces.Box(
                -10.0, 10.0, (state_size,), np.float32
            )
        elif base_variant == "flat":
            self.observation_space = spaces.Box(
                -10.0, 10.0, (state_size + 168 * 9,), np.float32
            )
        elif base_variant in {
            "tcn",
            "gnn",
            "larger_mlp",
            "edge_gnn",
            "fixed_scale_larger_mlp",
            "fixed_scale_edge_gnn",
            "stable_tcn",
            "fixed_scale_tcn",
        }:
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(-10.0, 10.0, (state_size,), np.float32),
                    "forecast": spaces.Box(-10.0, 10.0, (168, 9), np.float32),
                }
            )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.env.reset(seed=episode_seed)
        return forecast_policy_observation(self.env, self.variant), {}

    def step(self, action):
        _observation, reward, terminated, truncated, info = self.env.step(
            native_action_from_flat(self.env, action)
        )
        return (
            forecast_policy_observation(
                self.env,
                self.variant,
                timeout=truncated,
            ),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def action_masks(self):
        return flat_action_mask(
            self.env.vessel_action_mask(), self.env.well_rate_action_mask()
        )


def make_forecast_ppo_policy(
    model,
    variant: ObservationVariant,
    deterministic: bool = False,
):
    """Wrap a forecast PPO model as a native ``policy(env) -> action``."""

    def policy(env: CCSEnv):
        observation = forecast_policy_observation(env, variant)
        masks = flat_action_mask(
            env.vessel_action_mask(), env.well_rate_action_mask()
        )
        action, _state = model.predict(
            observation,
            deterministic=deterministic,
            action_masks=masks,
        )
        return native_action_from_flat(env, action)

    return policy
