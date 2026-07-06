"""Behavior cloning data collection for hybrid-action policies.

The old behavior-cloning loss targeted a flat discrete MaskablePPO action. The
collector now records hybrid demonstrator actions; the supervised loss must be
reintroduced with a policy distribution that has categorical vessel heads and
categorical well-rate heads.
"""

from __future__ import annotations

import numpy as np

from ..environment.gym_adapter import CCSGymEnv
from ..metrics import Policy


def collect_demonstrations(
    gym_env: CCSGymEnv,
    demo_policy: Policy,
    n_episodes: int,
    seed0: int = 0,
) -> tuple[np.ndarray, list[dict[str, list]]]:
    """Roll the demonstrator and record ``(obs, action)`` pairs."""
    obs_rows: list[np.ndarray] = []
    act_rows: list[dict[str, list]] = []
    for i in range(n_episodes):
        obs, _ = gym_env.reset(seed=seed0 + i)
        done = False
        while not done:
            action = demo_policy(gym_env.env)  # demonstrator acts on the native env
            obs_rows.append(np.asarray(obs, dtype=np.float32))
            act_rows.append(action)
            native_obs, _reward, terminated, truncated, _info = gym_env.env.step(action)
            obs = gym_env._to_array(native_obs)
            done = terminated or truncated
    return np.asarray(obs_rows, dtype=np.float32), act_rows


def behavior_clone(
    model,
    observations: np.ndarray,
    actions: list[dict[str, list]],
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> None:
    """Supervise ``model.policy`` to imitate the demonstrator's actions in place."""
    raise NotImplementedError(
        "Behavior cloning must be updated for the hybrid action distribution "
        "(discrete vessels + discrete well-rate indices)."
    )
