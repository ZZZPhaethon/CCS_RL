"""Gymnasium adapter so RL libraries can train against :class:`CCSEnv`.

``CCSGymEnv`` exposes the native env as a standard ``gymnasium.Env`` with a flat
``MultiDiscrete`` action space: vessel destinations followed by discrete well
rate-level indices. ``action_masks()`` exposes the flattened vessel and well
masks in the same dimension order for policies that support discrete masking.

The episode boundary is reported as ``truncated`` (never ``terminated``), which
tells the trainer to bootstrap ``V(s_T)`` instead of zeroing the future - the
operation continues past the 168 h training window, it is not a true terminal.

This module is the only place that imports numpy/gymnasium; the simulation core
stays dependency-free.
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("CCSGymEnv requires gymnasium. Install with `pip install gymnasium`.") from exc

from .env import CCSEnv


def flat_action_mask(
    vessel_mask: list[list[bool]],
    well_mask: list[list[bool]],
) -> np.ndarray:
    """Flatten vessel and well legality masks into MultiDiscrete order."""
    return np.array(
        [legal for dimension in [*vessel_mask, *well_mask] for legal in dimension],
        dtype=bool,
    )


def flat_action_from_native(env: CCSEnv, native: dict[str, list]) -> np.ndarray:
    """Concatenate a native action dict into flat MultiDiscrete order.

    Inverse of :func:`native_action_from_flat`: ``[vessels..., wells...]``.
    """
    vessels = [int(a) for a in native["vessels"]]
    wells = [int(a) for a in native["wells"]]
    expected = len(env.vessel_ids) + len(env.well_ids)
    flat = vessels + wells
    if len(flat) != expected:
        raise ValueError(f"Expected {expected} flat action entries, got {len(flat)}.")
    return np.asarray(flat, dtype=np.int64)


def native_action_from_flat(env: CCSEnv, action) -> dict[str, list[int]]:
    """Split a flat MultiDiscrete action into the native env action dict."""
    flat = np.asarray(action, dtype=np.int64).reshape(-1)
    vessel_count = len(env.vessel_ids)
    well_count = len(env.well_ids)
    expected = vessel_count + well_count
    if len(flat) != expected:
        raise ValueError(f"Expected {expected} flat action entries, got {len(flat)}.")
    return {
        "vessels": [int(a) for a in flat[:vessel_count]],
        "wells": [int(a) for a in flat[vessel_count:expected]],
    }


class CCSGymEnv(gym.Env):
    """A ``gymnasium.Env`` view over a :class:`CCSEnv`."""

    metadata = {"render_modes": []}

    def __init__(self, env: CCSEnv) -> None:
        super().__init__()
        self.env = env
        self.action_space = spaces.MultiDiscrete(
            env.vessel_action_dims + env.well_rate_action_dims
        )
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(env.observation_size,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Draw a fresh per-episode scenario seed from the (optionally seeded)
        # np_random, so episodes are varied yet reproducible: domain randomization.
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        obs = self.env.reset(seed=episode_seed)
        return self._to_array(obs), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self._native_action(action))
        return self._to_array(obs), float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return flat_action_mask(
            self.env.vessel_action_mask(),
            self.env.well_rate_action_mask(),
        )

    def _to_array(self, obs: list[float]) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def _native_action(self, action) -> dict[str, list]:
        return native_action_from_flat(self.env, action)


def make_ppo_policy(model, deterministic: bool = False):
    """Wrap a trained flat-action PPO model as a metrics ``policy(env) -> action``.

    Lets the trained policy be scored by the same ``sim.metrics`` harness as the
    heuristic baselines, on the native :class:`CCSEnv`.

    ``deterministic`` defaults to ``False``: a still-stochastic policy relies on
    sampling to dispatch vessels, so an argmax (``deterministic=True``) action can
    collapse to all-WAIT and score like the idle baseline even when the sampled
    policy stores well. Use ``deterministic=True`` only once the policy has become
    confident (peaked) on its dispatch actions.
    """

    def policy(env: CCSEnv) -> dict[str, list]:
        obs = np.asarray(env._observation(), dtype=np.float32)
        masks = flat_action_mask(env.vessel_action_mask(), env.well_rate_action_mask())
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
        return native_action_from_flat(env, action)

    return policy
