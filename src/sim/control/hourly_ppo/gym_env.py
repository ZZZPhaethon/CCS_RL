"""One-policy-decision-per-hour Gymnasium interface for centralized PPO."""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional training dependency.
    raise ImportError(
        "HourlyCentralizedPPOEnv requires gymnasium. "
        "Install with `pip install gymnasium`."
    ) from exc

from sim.control.cplex_milp import _terminal_cleanup_cost_for_state
from sim.control.event_based.rl.observation_encoder import (
    FORECAST_WINDOWS_H,
    high_level_observation,
    high_level_observation_size,
    validated_future_summary_windows,
)
from sim.environment import CCSEnv
from sim.environment.gym_adapter import (
    flat_action_mask,
    native_action_from_flat,
)


class HourlyCentralizedPPOEnv(gym.Env):
    """Expose direct vessel dispatch with exactly one decision per physical hour.

    The policy observes the current physical state plus the shared structured
    future summary. Its ``MultiDiscrete`` action is passed directly to
    :class:`CCSEnv`; no goal executor, Greedy default, event trigger, residual
    action, or behaviour-cloning policy is used.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: CCSEnv,
        *,
        future_summary_windows_h: tuple[int, ...] = FORECAST_WINDOWS_H,
        episode_seed_min: int = 0,
        episode_seed_max: int = 2**31 - 2,
        max_simulator_hour_steps: int | None = None,
        include_terminal_cleanup_reward: bool = True,
    ) -> None:
        super().__init__()
        if episode_seed_min > episode_seed_max:
            raise ValueError(
                "episode_seed_min must not exceed episode_seed_max."
            )
        if env.well_rate_action_dims:
            raise ValueError(
                "Hourly centralized PPO requires automatic well control."
            )
        if abs(float(env.network.time_step_hours) - 1.0) > 1e-9:
            raise ValueError(
                "Hourly centralized PPO requires a 1 h physical time step."
            )
        if max_simulator_hour_steps is not None and (
            int(max_simulator_hour_steps) != max_simulator_hour_steps
            or max_simulator_hour_steps <= 0
        ):
            raise ValueError(
                "max_simulator_hour_steps must be a positive integer."
            )

        self.env = env
        self.future_summary_windows_h = validated_future_summary_windows(
            future_summary_windows_h
        )
        self.episode_seed_min = int(episode_seed_min)
        self.episode_seed_max = int(episode_seed_max)
        self.max_simulator_hour_steps = (
            int(max_simulator_hour_steps)
            if max_simulator_hour_steps is not None
            else None
        )
        self.include_terminal_cleanup_reward = bool(
            include_terminal_cleanup_reward
        )
        self.action_space = spaces.MultiDiscrete(env.vessel_action_dims)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(
                high_level_observation_size(
                    env,
                    self.future_summary_windows_h,
                ),
            ),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        episode_seed = int(
            self.np_random.integers(
                self.episode_seed_min,
                self.episode_seed_max + 1,
            )
        )
        self.env.reset(seed=episode_seed)
        return self._observation(), {"episode_seed": episode_seed}

    def step(self, action):
        """Apply one direct joint vessel action and advance exactly one hour."""

        if self._budget_exhausted():
            return (
                self._observation(),
                0.0,
                False,
                True,
                {
                    "simulator_budget_exhausted": True,
                    "native_steps": 0,
                    "elapsed_hours": 0.0,
                },
            )

        assert self.env.simulator is not None
        started_at_h = float(self.env.simulator.state.time_h)
        _native_observation, reward, terminated, truncated, info = self.env.step(
            native_action_from_flat(self.env, action)
        )
        elapsed_hours = float(self.env.simulator.state.time_h) - started_at_h
        if abs(elapsed_hours - 1.0) > 1e-9:
            raise RuntimeError(
                "Hourly PPO transition did not advance exactly one physical hour."
            )

        info = dict(info)
        info.update(
            {
                "decision_interval_h": 1.0,
                "elapsed_hours": elapsed_hours,
                "native_steps": 1,
            }
        )
        episode_complete = bool(truncated and self.env.t >= self.env.n_steps)
        if episode_complete and self.include_terminal_cleanup_reward:
            cleanup_cost = float(
                _terminal_cleanup_cost_for_state(
                    self.env,
                    self.env.cost_model.parameters,
                )
            )
            reward -= cleanup_cost * float(self.env.config.reward_scale)
            info["terminal_cleanup_operating_cost_eur"] = cleanup_cost
            info["terminal_cleanup_included_in_reward"] = True
            # The cleanup value closes the fixed-horizon objective. Mark this as
            # a true MDP terminal so PPO does not bootstrap beyond it.
            terminated = True
            truncated = False
        else:
            info["terminal_cleanup_operating_cost_eur"] = 0.0
            info["terminal_cleanup_included_in_reward"] = False

        if self._budget_exhausted() and not (terminated or truncated):
            truncated = True
            info["simulator_budget_exhausted"] = True
        else:
            info["simulator_budget_exhausted"] = False

        return (
            self._observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def action_masks(self) -> np.ndarray:
        """Return the per-vessel legal-action masks used by MaskablePPO."""

        return flat_action_mask(
            self.env.vessel_action_mask(),
            self.env.well_rate_action_mask(),
        )

    def training_simulator_usage(self) -> dict[str, int | float]:
        """Expose this worker's physical-step counter to vector environments."""

        return self.env.simulator_step_usage().as_dict()

    def _observation(self) -> np.ndarray:
        return high_level_observation(
            self.env,
            self.future_summary_windows_h,
        ).astype(np.float32, copy=False)

    def _budget_exhausted(self) -> bool:
        if self.max_simulator_hour_steps is None:
            return False
        return (
            self.env.simulator_step_usage().hour_steps
            >= float(self.max_simulator_hour_steps) - 1e-9
        )
