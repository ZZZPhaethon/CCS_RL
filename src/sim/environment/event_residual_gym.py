"""Event-based joint-action residual control around the Greedy policy."""

from __future__ import annotations

import itertools

import numpy as np
from gymnasium import Env, spaces

from .residual_forecast_gym import GreedyResidualForecastGymEnv


class EventJointResidualGymEnv(Env):
    """Expose one categorical joint residual action at meaningful states.

    Native actions identical to Greedy are masked, so FOLLOW is the unique label
    for unmodified Greedy behaviour.  After a joint action is applied for one
    physical hour, hours with no genuine override choice are automatically run
    under Greedy until the next event.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env,
        variant: str = "future_mlp_mode",
        *,
        include_episode_progress: bool = True,
        greedy_control_variate: bool = True,
        hourly_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < float(hourly_gamma) <= 1.0:
            raise ValueError("hourly_gamma must be in (0, 1]")
        self.residual_env = GreedyResidualForecastGymEnv(
            env,
            variant,
            include_episode_progress=include_episode_progress,
            greedy_control_variate=greedy_control_variate,
        )
        self.env = env
        self.variant = variant
        self.hourly_gamma = float(hourly_gamma)
        self.observation_space = self.residual_env.observation_space
        self.residual_dimensions = tuple(
            int(value) for value in self.residual_env.action_space.nvec
        )
        self.joint_actions = tuple(
            itertools.product(*(range(size) for size in self.residual_dimensions))
        )
        self._joint_action_array = np.asarray(self.joint_actions, dtype=np.int64)
        self.action_space = spaces.Discrete(len(self.joint_actions))
        follow_tuple = tuple(int(value) for value in self.residual_env.follow_indices)
        self.follow_action_index = self.joint_actions.index(follow_tuple)
        self._observation = None

    def reset(self, *, seed=None, options=None):
        observation, info = self.residual_env.reset(seed=seed, options=options)
        return self._after_reset(observation, info)

    def reset_native_seed(self, seed: int):
        observation, info = self.residual_env.reset_native_seed(int(seed))
        return self._after_reset(observation, info)

    def _after_reset(self, observation, info):
        self._observation = observation
        observation, skipped_hours, skipped_reward, done, last_info = self._skip_forced_hours()
        if done:
            raise RuntimeError("episode ended before the first residual decision")
        info = dict(info)
        info.update(last_info)
        info.update(
            {
                "event_duration_h": int(skipped_hours),
                "event_skipped_reward": float(skipped_reward),
            }
        )
        return observation, info

    def decode_action(self, action) -> np.ndarray:
        index = int(np.asarray(action).reshape(-1)[0])
        if index < 0 or index >= len(self.joint_actions):
            raise ValueError(f"joint action index out of range: {index}")
        return self._joint_action_array[index].copy()

    def encode_action(self, residual_action) -> int:
        residual = tuple(int(value) for value in np.asarray(residual_action).reshape(-1))
        if len(residual) != len(self.residual_dimensions):
            raise ValueError(
                f"expected {len(self.residual_dimensions)} residual choices, got {len(residual)}"
            )
        index = 0
        for choice, dimension in zip(residual, self.residual_dimensions):
            if choice < 0 or choice >= dimension:
                raise ValueError(f"residual choice {choice} outside dimension {dimension}")
            index = index * dimension + choice
        return int(index)

    def _per_vessel_masks(self) -> list[np.ndarray]:
        native_masks = self.env.vessel_action_mask()
        base_actions = self.residual_env._base_action["vessels"]
        masks = []
        for native_mask, base_action, follow_index in zip(
            native_masks,
            base_actions,
            self.residual_env.follow_indices,
        ):
            mask = np.asarray([*native_mask, True], dtype=bool)
            # FOLLOW is the sole label for the Greedy physical action.
            mask[int(base_action)] = False
            mask[int(follow_index)] = True
            masks.append(mask)
        return masks

    def action_masks(self) -> np.ndarray:
        per_vessel = self._per_vessel_masks()
        allowed = np.ones(len(self.joint_actions), dtype=bool)
        for vessel_index, mask in enumerate(per_vessel):
            allowed &= mask[self._joint_action_array[:, vessel_index]]
        if not allowed[self.follow_action_index]:
            raise RuntimeError("FOLLOW joint action must always be legal")
        return allowed

    def has_override_choice(self) -> bool:
        return int(self.action_masks().sum()) > 1

    def step(self, action):
        action_index = int(np.asarray(action).reshape(-1)[0])
        masks = self.action_masks()
        if action_index < 0 or action_index >= len(masks) or not masks[action_index]:
            raise ValueError(f"illegal event residual action: {action_index}")

        residual_action = self.decode_action(action_index)
        observation, reward, terminated, truncated, info = self.residual_env.step(
            residual_action
        )
        self._observation = observation
        total_reward = float(reward)
        duration_h = 1
        done = bool(terminated or truncated)
        last_info = dict(info)

        if not done:
            (
                observation,
                skipped_hours,
                skipped_reward,
                done,
                skipped_info,
            ) = self._skip_forced_hours()
            total_reward += (self.hourly_gamma ** duration_h) * skipped_reward
            duration_h += skipped_hours
            if skipped_info:
                last_info = skipped_info

        last_info = dict(last_info)
        last_info.update(
            {
                "event_action_index": action_index,
                "event_residual_action": residual_action.tolist(),
                "event_duration_h": int(duration_h),
                "event_reward": float(total_reward),
            }
        )
        return observation, total_reward, False, done, last_info

    def _skip_forced_hours(self):
        observation = self._observation
        duration_h = 0
        total_reward = 0.0
        done = False
        last_info = {}
        while self.env.t < self.env.n_steps and not self.has_override_choice():
            observation, reward, terminated, truncated, info = self.residual_env.step(
                self.residual_env.follow_action()
            )
            total_reward += (self.hourly_gamma ** duration_h) * float(reward)
            duration_h += 1
            done = bool(terminated or truncated)
            last_info = dict(info)
            self._observation = observation
            if done:
                break
        return observation, duration_h, total_reward, done, last_info

    def follow_action(self) -> int:
        return int(self.follow_action_index)
