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
    replan_phase_observation,
)
from .gym_adapter import flat_action_mask, native_action_from_flat
from .past import PAST_HOURS, PastObservationBuffer
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
    "future_mlp",
    "future_mlp_mode",
    "future_mlp_mode_destination",
    "gated_past24_mlp_mode_destination",
    "past24_mlp_mode_destination",
    "past24_zero_mlp_mode_destination",
    "balanced_edge_gnn_mode_destination",
    "balanced_edge_gnn_future_mlp_mode_destination",
    "future_conditioned_edge_gnn_mode_destination",
    "gated_residual_edge_gnn_mode_destination",
    "entity_residual_edge_gnn_mode_destination",
    "fixed_scale_larger_mlp_mode_destination",
    "fixed_scale_edge_gnn_mode_destination",
    "stable_tcn_mode_destination",
    "fixed_scale_tcn_mode_destination",
    "fixed_scale_tcn_mode_destination_replan_phase",
    "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate",
    "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context",
]

ORACLE_CANDIDATE_COUNT = 8


def variant_uses_operation_modes(variant: str) -> bool:
    return variant in {
        "state_mode",
        "tcn_mode",
        "tcn_mode_destination",
        "gnn_mode_destination",
        "larger_mlp_mode_destination",
        "edge_gnn_mode_destination",
        "future_mlp_mode",
        "future_mlp_mode_destination",
        "gated_past24_mlp_mode_destination",
        "past24_mlp_mode_destination",
        "past24_zero_mlp_mode_destination",
        "balanced_edge_gnn_mode_destination",
        "balanced_edge_gnn_future_mlp_mode_destination",
        "future_conditioned_edge_gnn_mode_destination",
        "gated_residual_edge_gnn_mode_destination",
        "entity_residual_edge_gnn_mode_destination",
        "fixed_scale_larger_mlp_mode_destination",
        "fixed_scale_edge_gnn_mode_destination",
        "stable_tcn_mode_destination",
        "fixed_scale_tcn_mode_destination",
        "fixed_scale_tcn_mode_destination_replan_phase",
        "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate",
        "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context",
    }


def variant_uses_sailing_destinations(variant: str) -> bool:
    return variant in {
        "tcn_mode_destination",
        "gnn_mode_destination",
        "larger_mlp_mode_destination",
        "edge_gnn_mode_destination",
        "future_mlp_mode_destination",
        "gated_past24_mlp_mode_destination",
        "past24_mlp_mode_destination",
        "past24_zero_mlp_mode_destination",
        "balanced_edge_gnn_mode_destination",
        "balanced_edge_gnn_future_mlp_mode_destination",
        "future_conditioned_edge_gnn_mode_destination",
        "gated_residual_edge_gnn_mode_destination",
        "entity_residual_edge_gnn_mode_destination",
        "fixed_scale_larger_mlp_mode_destination",
        "fixed_scale_edge_gnn_mode_destination",
        "stable_tcn_mode_destination",
        "fixed_scale_tcn_mode_destination",
        "fixed_scale_tcn_mode_destination_replan_phase",
        "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate",
        "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context",
    }


def variant_uses_replan_phase(variant: str) -> bool:
    return variant in {
        "fixed_scale_tcn_mode_destination_replan_phase",
        "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate",
        "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context",
    }


def variant_uses_past(variant: str) -> bool:
    return variant in {
        "gated_past24_mlp_mode_destination",
        "past24_mlp_mode_destination",
        "past24_zero_mlp_mode_destination",
    }


def variant_uses_zero_past(variant: str) -> bool:
    return variant == "past24_zero_mlp_mode_destination"


def variant_uses_oracle_candidate(variant: str) -> bool:
    return variant == "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate"


def variant_uses_learned_plan_context(variant: str) -> bool:
    return variant == "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context"


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
    if variant in {
        "future_mlp",
        "future_mlp_mode",
        "future_mlp_mode_destination",
    }:
        return "future_mlp"
    if variant == "gated_past24_mlp_mode_destination":
        return "gated_past_mlp"
    if variant in {
        "past24_mlp_mode_destination",
        "past24_zero_mlp_mode_destination",
    }:
        return "past_mlp"
    if variant == "balanced_edge_gnn_mode_destination":
        return "balanced_edge_gnn"
    if variant == "balanced_edge_gnn_future_mlp_mode_destination":
        return "balanced_edge_gnn_future_mlp"
    if variant == "future_conditioned_edge_gnn_mode_destination":
        return "future_conditioned_edge_gnn"
    if variant == "gated_residual_edge_gnn_mode_destination":
        return "gated_residual_edge_gnn"
    if variant == "entity_residual_edge_gnn_mode_destination":
        return "entity_residual_edge_gnn"
    if variant == "fixed_scale_larger_mlp_mode_destination":
        return "fixed_scale_larger_mlp"
    if variant == "fixed_scale_edge_gnn_mode_destination":
        return "fixed_scale_edge_gnn"
    if variant == "stable_tcn_mode_destination":
        return "stable_tcn"
    if variant in {
        "fixed_scale_tcn_mode_destination",
        "fixed_scale_tcn_mode_destination_replan_phase",
        "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate",
        "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context",
    }:
        return "fixed_scale_tcn"
    if variant in {"state", "flat", "tcn"}:
        return variant
    raise ValueError(f"unknown forecast observation variant: {variant}")


def forecast_policy_observation(
    env: CCSEnv,
    variant: ObservationVariant,
    *,
    timeout: bool = False,
    oracle_candidate_index: int | None = None,
    learned_plan_context: np.ndarray | None = None,
    past_observation: np.ndarray | None = None,
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
    if variant_uses_replan_phase(variant):
        assert env.simulator is not None
        phase = np.asarray(
            replan_phase_observation(env.simulator.state.time_h),
            dtype=np.float32,
        )
        state = np.concatenate((state, phase)).astype(np.float32, copy=False)
    if variant_uses_oracle_candidate(variant):
        if oracle_candidate_index is None or not (
            0 <= int(oracle_candidate_index) < ORACLE_CANDIDATE_COUNT
        ):
            raise ValueError("oracle candidate variant requires a valid candidate index")
        candidate = np.zeros(ORACLE_CANDIDATE_COUNT, dtype=np.float32)
        candidate[int(oracle_candidate_index)] = 1.0
        state = np.concatenate((state, candidate)).astype(np.float32, copy=False)
    if variant_uses_learned_plan_context(variant):
        context = np.asarray(learned_plan_context, dtype=np.float32)
        if context.shape != (ORACLE_CANDIDATE_COUNT,) or not np.all(np.isfinite(context)):
            raise ValueError("learned plan-context variant requires a finite [8] context")
        state = np.concatenate((state, context)).astype(np.float32, copy=False)
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
        "future_mlp",
        "gated_past_mlp",
        "past_mlp",
        "balanced_edge_gnn",
        "balanced_edge_gnn_future_mlp",
        "future_conditioned_edge_gnn",
        "gated_residual_edge_gnn",
        "entity_residual_edge_gnn",
        "fixed_scale_larger_mlp",
        "fixed_scale_edge_gnn",
        "stable_tcn",
        "fixed_scale_tcn",
    }:
        observation = {"state": state, "forecast": forecast}
        if base_variant in {"gated_past_mlp", "past_mlp"}:
            expected_shape = (
                PAST_HOURS,
                len(state) + len(env.vessel_action_dims) + len(env.well_rate_action_dims) + 1,
            )
            past = np.asarray(past_observation, dtype=np.float32)
            if past.shape != expected_shape or not np.all(np.isfinite(past)):
                raise ValueError(
                    f"past observation must be finite with shape {expected_shape}"
                )
            observation["past"] = past
        return observation
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

    def __init__(
        self,
        env: CCSEnv,
        variant: ObservationVariant,
        oracle_candidate_index: int | None = None,
        learned_plan_context: np.ndarray | None = None,
    ):
        super().__init__()
        self.env = env
        self.variant = variant
        self.oracle_candidate_index = oracle_candidate_index
        self.learned_plan_context = learned_plan_context
        self._past_buffer: PastObservationBuffer | None = None
        self._current_policy_state: np.ndarray | None = None
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
        if variant_uses_replan_phase(variant):
            state_size += 2
        if variant_uses_oracle_candidate(variant):
            state_size += ORACLE_CANDIDATE_COUNT
        if variant_uses_learned_plan_context(variant):
            state_size += ORACLE_CANDIDATE_COUNT
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
            "future_mlp",
            "gated_past_mlp",
            "past_mlp",
            "balanced_edge_gnn",
            "balanced_edge_gnn_future_mlp",
            "future_conditioned_edge_gnn",
            "gated_residual_edge_gnn",
            "entity_residual_edge_gnn",
            "fixed_scale_larger_mlp",
            "fixed_scale_edge_gnn",
            "stable_tcn",
            "fixed_scale_tcn",
        }:
            observation_spaces = {
                "state": spaces.Box(-10.0, 10.0, (state_size,), np.float32),
                "forecast": spaces.Box(-10.0, 10.0, (168, 9), np.float32),
            }
            if base_variant in {"gated_past_mlp", "past_mlp"}:
                action_dimensions = [
                    *env.vessel_action_dims,
                    *env.well_rate_action_dims,
                ]
                self._past_buffer = PastObservationBuffer(
                    state_size,
                    action_dimensions,
                    hours=PAST_HOURS,
                )
                observation_spaces["past"] = spaces.Box(
                    -10.0,
                    10.0,
                    (PAST_HOURS, self._past_buffer.row_size),
                    np.float32,
                )
            self.observation_space = spaces.Dict(observation_spaces)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.env.reset(seed=episode_seed)
        if self._past_buffer is not None:
            self._past_buffer.reset()
        observation = forecast_policy_observation(
            self.env,
            self.variant,
            oracle_candidate_index=self.oracle_candidate_index,
            learned_plan_context=self.learned_plan_context,
            past_observation=(
                self._past_buffer.observation(zero=variant_uses_zero_past(self.variant))
                if self._past_buffer is not None
                else None
            ),
        )
        self._current_policy_state = (
            np.asarray(observation["state"], dtype=np.float32)
            if self._past_buffer is not None
            else None
        )
        return observation, {}

    def step(self, action):
        if self._past_buffer is not None:
            if self._current_policy_state is None:
                raise RuntimeError("Call reset() before step().")
            self._past_buffer.append(self._current_policy_state, np.asarray(action))
        _observation, reward, terminated, truncated, info = self.env.step(
            native_action_from_flat(self.env, action)
        )
        observation = forecast_policy_observation(
            self.env,
            self.variant,
            timeout=truncated,
            oracle_candidate_index=self.oracle_candidate_index,
            learned_plan_context=self.learned_plan_context,
            past_observation=(
                self._past_buffer.observation(zero=variant_uses_zero_past(self.variant))
                if self._past_buffer is not None
                else None
            ),
        )
        self._current_policy_state = (
            np.asarray(observation["state"], dtype=np.float32)
            if self._past_buffer is not None
            else None
        )
        return (
            observation,
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
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
):
    """Wrap a forecast PPO model as a native ``policy(env) -> action``."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    sampling_rng = rng or np.random.default_rng()
    past_buffer: PastObservationBuffer | None = None
    last_time_h: float | None = None

    def policy(env: CCSEnv):
        nonlocal past_buffer, last_time_h
        assert env.simulator is not None
        time_h = float(env.simulator.state.time_h)
        if variant_uses_past(variant):
            state_size = (
                len(current_state_feature_names(env))
                + 5 * len(env.vessel_ids)
                + len(env.vessel_ids)
                * (len(env.terminal_ids) + len(env.emitter_ids))
            )
            if past_buffer is None:
                past_buffer = PastObservationBuffer(
                    state_size,
                    [*env.vessel_action_dims, *env.well_rate_action_dims],
                    hours=PAST_HOURS,
                )
            if last_time_h is None or time_h <= last_time_h:
                past_buffer.reset()
        observation = forecast_policy_observation(
            env,
            variant,
            past_observation=(
                past_buffer.observation(zero=variant_uses_zero_past(variant))
                if past_buffer is not None
                else None
            ),
        )
        masks = flat_action_mask(
            env.vessel_action_mask(), env.well_rate_action_mask()
        )
        if deterministic or temperature == 1.0:
            action, _state = model.predict(
                observation,
                deterministic=deterministic,
                action_masks=masks,
            )
        else:
            import torch

            policy_module = model.policy
            observation_tensor, _vectorized = policy_module.obs_to_tensor(observation)
            mask_tensor = torch.as_tensor(
                masks[None, ...],
                device=policy_module.device,
            )
            with torch.no_grad():
                features = policy_module.extract_features(observation_tensor)
                if policy_module.share_features_extractor:
                    latent_pi, _latent_vf = policy_module.mlp_extractor(features)
                else:
                    policy_features, _value_features = features
                    latent_pi = policy_module.mlp_extractor.forward_actor(
                        policy_features
                    )
                distribution = policy_module._get_action_dist_from_latent(latent_pi)
                distribution.apply_masking(mask_tensor)
                probabilities = [
                    categorical.probs[0].detach().cpu().numpy()
                    for categorical in distribution.distributions
                ]
            sharpened = []
            for values in probabilities:
                values = np.power(values, 1.0 / temperature)
                sharpened.append(values / values.sum())
            action = np.asarray(
                [
                    sampling_rng.choice(len(values), p=values)
                    for values in sharpened
                ],
                dtype=np.int64,
            )
        if past_buffer is not None:
            past_buffer.append(observation["state"], np.asarray(action))
            last_time_h = time_h
        return native_action_from_flat(env, action)

    return policy
