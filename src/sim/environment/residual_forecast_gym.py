"""Forecast observation wrapper for residual control around Greedy."""

from __future__ import annotations

import copy

import numpy as np
from gymnasium import Env, spaces

from ..control.baselines import greedy_shuttle_policy
from .env import VESSEL_GO_EMITTER_BASE, VESSEL_GO_TERMINAL, VESSEL_WAIT
from .forecast_gym import ForecastGymEnv, forecast_policy_observation
from .gym_adapter import flat_action_mask


def _base_vessel_onehot(env, vessel_actions) -> np.ndarray:
    rows = []
    for action, dimension in zip(vessel_actions, env.vessel_action_dims):
        onehot = np.zeros(int(dimension), dtype=np.float32)
        onehot[int(action)] = 1.0
        rows.append(onehot)
    return np.concatenate(rows) if rows else np.empty(0, dtype=np.float32)


def augment_residual_observation(
    observation,
    env,
    vessel_actions,
    include_episode_progress: bool = False,
):
    """Append the Greedy vessel proposal to the policy's current-state branch."""

    proposal = _base_vessel_onehot(env, vessel_actions)
    if include_episode_progress:
        proposal = np.concatenate(
            (
                proposal,
                np.asarray([env.t / max(1, env.n_steps)], dtype=np.float32),
            )
        )
    if isinstance(observation, dict):
        augmented = dict(observation)
        augmented["state"] = np.concatenate(
            (np.asarray(observation["state"], dtype=np.float32), proposal)
        ).astype(np.float32, copy=False)
        return augmented
    return np.concatenate(
        (np.asarray(observation, dtype=np.float32), proposal)
    ).astype(np.float32, copy=False)


class GreedyResidualForecastGymEnv(Env):
    """RL chooses FOLLOW Greedy or a legal vessel-action override.

    Wells always follow Greedy.  This keeps the residual action focused on the
    routing decisions that differ between Greedy and MPC and makes all-FOLLOW
    reproduce the Greedy controller exactly.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env,
        variant: str,
        include_episode_progress: bool = False,
        greedy_control_variate: bool = False,
    ):
        super().__init__()
        self.env = env
        self.variant = variant
        self.include_episode_progress = bool(include_episode_progress)
        self.greedy_control_variate = bool(greedy_control_variate)
        self.forecast_env = ForecastGymEnv(env, variant)
        self.follow_indices = np.asarray(env.vessel_action_dims, dtype=np.int64)
        self.action_space = spaces.MultiDiscrete(
            [int(dimension) + 1 for dimension in env.vessel_action_dims]
        )
        self.observation_space = self._augmented_observation_space(
            self.forecast_env.observation_space
        )
        self._base_action = None
        self._greedy_baseline_rewards = None

    def _augmented_observation_space(self, original):
        proposal_size = int(sum(self.env.vessel_action_dims)) + int(
            self.include_episode_progress
        )
        if isinstance(original, spaces.Dict):
            mapping = dict(original.spaces)
            state_size = int(mapping["state"].shape[0]) + proposal_size
            mapping["state"] = spaces.Box(
                -10.0,
                10.0,
                (state_size,),
                np.float32,
            )
            return spaces.Dict(mapping)
        return spaces.Box(
            -10.0,
            10.0,
            (int(original.shape[0]) + proposal_size,),
            np.float32,
        )

    def _refresh_base_action(self) -> None:
        self._base_action = greedy_shuttle_policy(self.env)

    def _observation(self, observation):
        return augment_residual_observation(
            observation,
            self.env,
            self._base_action["vessels"],
            self.include_episode_progress,
        )

    def reset(self, *, seed=None, options=None):
        observation, info = self.forecast_env.reset(seed=seed, options=options)
        self._refresh_base_action()
        self._greedy_baseline_rewards = (
            self._build_greedy_reward_baseline()
            if self.greedy_control_variate
            else None
        )
        return self._observation(observation), info

    def reset_native_seed(self, seed: int):
        """Reset with an exact scenario seed, without Gym's seed remapping.

        Formal native-policy evaluations use scenario seeds directly.  This
        helper keeps residual evaluations on exactly the same scenarios.
        """

        self.env.reset(seed=int(seed))
        observation = forecast_policy_observation(self.env, self.variant)
        self._refresh_base_action()
        self._greedy_baseline_rewards = (
            self._build_greedy_reward_baseline()
            if self.greedy_control_variate
            else None
        )
        return self._observation(observation), {}

    def _build_greedy_reward_baseline(self) -> np.ndarray:
        baseline_env = copy.deepcopy(self.env)
        rewards = []
        done = False
        while not done:
            action = greedy_shuttle_policy(baseline_env)
            _observation, reward, terminated, truncated, _info = baseline_env.step(
                action
            )
            rewards.append(float(reward))
            done = terminated or truncated
        if len(rewards) != self.env.n_steps:
            raise RuntimeError(
                "Greedy reward baseline length does not match the episode horizon"
            )
        return np.asarray(rewards, dtype=np.float64)

    def step(self, action):
        step_index = int(self.env.t)
        residual = np.asarray(action, dtype=np.int64).reshape(-1)
        native_masks = self.env.vessel_action_mask()
        native_action = self.native_action(action)
        base_vessels = [int(value) for value in self._base_action["vessels"]]
        native_vessels = [int(value) for value in native_action["vessels"]]
        logical_overrides = sum(
            int(choice) != int(follow) and any(mask[1:])
            for choice, follow, mask in zip(
                residual,
                self.follow_indices,
                native_masks,
            )
        )
        physical_overrides = sum(
            actual != base
            for actual, base in zip(native_vessels, base_vessels)
        )
        decision_dimensions = sum(any(mask[1:]) for mask in native_masks)
        current_berths = dict(self.env.simulator.state.vessel_berths)
        early_terminal_departures = 0
        emitter_to_emitter_legs = 0
        for vessel_id, actual, base in zip(
            self.env.vessel_ids,
            native_vessels,
            base_vessels,
        ):
            if actual == base:
                continue
            berth = current_berths.get(vessel_id)
            if (
                base == VESSEL_WAIT
                and actual == VESSEL_GO_TERMINAL
                and berth in self.env.emitter_ids
            ):
                early_terminal_departures += 1
            if actual >= VESSEL_GO_EMITTER_BASE and berth in self.env.emitter_ids:
                emitter_index = actual - VESSEL_GO_EMITTER_BASE
                if (
                    0 <= emitter_index < len(self.env.emitter_ids)
                    and self.env.emitter_ids[emitter_index] != berth
                ):
                    emitter_to_emitter_legs += 1
        flat_action = np.asarray(
            [*native_action["vessels"], *native_action["wells"]],
            dtype=np.int64,
        )
        observation, reward, terminated, truncated, info = self.forecast_env.step(
            flat_action
        )
        raw_economic_reward = float(reward)
        greedy_baseline_reward = (
            float(self._greedy_baseline_rewards[step_index])
            if self._greedy_baseline_rewards is not None
            else 0.0
        )
        control_variate_reward = raw_economic_reward - greedy_baseline_reward
        if self.greedy_control_variate:
            reward = control_variate_reward
        info = dict(info)
        info.update(
            {
                "residual_decision_dimensions": int(decision_dimensions),
                "residual_logical_overrides": int(logical_overrides),
                "residual_physical_overrides": int(physical_overrides),
                "residual_early_terminal_departures": int(early_terminal_departures),
                "residual_emitter_to_emitter_legs": int(emitter_to_emitter_legs),
                "residual_raw_economic_reward": raw_economic_reward,
                "residual_greedy_baseline_reward": greedy_baseline_reward,
                "residual_control_variate_reward": control_variate_reward,
            }
        )
        self._refresh_base_action()
        return (
            self._observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def native_action(self, residual_action):
        if self._base_action is None:
            raise RuntimeError("Call reset() before requesting a residual action.")
        residual = np.asarray(residual_action, dtype=np.int64).reshape(-1)
        if residual.shape != self.follow_indices.shape:
            raise ValueError(
                f"expected {len(self.follow_indices)} vessel actions, got {len(residual)}"
            )
        vessels = []
        for choice, follow, base in zip(
            residual,
            self.follow_indices,
            self._base_action["vessels"],
        ):
            vessels.append(int(base) if int(choice) == int(follow) else int(choice))
        return {
            "vessels": vessels,
            "wells": [int(value) for value in self._base_action["wells"]],
        }

    def follow_action(self) -> np.ndarray:
        return self.follow_indices.copy()

    def action_masks(self) -> np.ndarray:
        vessel_masks = [list(values) + [True] for values in self.env.vessel_action_mask()]
        return flat_action_mask(vessel_masks, [])


def residual_policy_observation(
    env,
    variant: str,
    include_episode_progress: bool = False,
):
    base_action = greedy_shuttle_policy(env)
    observation = forecast_policy_observation(env, variant)
    return (
        augment_residual_observation(
            observation,
            env,
            base_action["vessels"],
            include_episode_progress,
        ),
        base_action,
    )


def residual_native_action(env, residual_action, base_action):
    residual = np.asarray(residual_action, dtype=np.int64).reshape(-1)
    follows = np.asarray(env.vessel_action_dims, dtype=np.int64)
    if residual.shape != follows.shape:
        raise ValueError(f"expected {len(follows)} vessel actions, got {len(residual)}")
    vessels = [
        int(base) if int(choice) == int(follow) else int(choice)
        for choice, follow, base in zip(residual, follows, base_action["vessels"])
    ]
    return {
        "vessels": vessels,
        "wells": [int(value) for value in base_action["wells"]],
    }
