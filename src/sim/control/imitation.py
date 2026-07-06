"""Behavior cloning for the hybrid-action MaskablePPO policy.

Warm-starting PPO from a demonstrator (e.g. ``greedy_shuttle_policy``) gets the
agent past the early exploration phase where it collapses to idling. The policy
head is a masked MultiCategorical over discrete vessel destinations and discrete
well-rate indices; cloning maximises the log-probability of the demonstrator's
actions under that same masked distribution.
"""

from __future__ import annotations

import numpy as np

from ..environment.gym_adapter import CCSGymEnv, flat_action_from_native
from ..metrics import Policy


def collect_demonstrations(
    gym_env: CCSGymEnv,
    demo_policy: Policy,
    n_episodes: int,
    seed0: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the demonstrator and record ``(obs, flat_action, action_mask)``.

    ``obs`` is the array observation the policy consumes, ``flat_action`` is the
    demonstrator action in flat MultiDiscrete order, and ``action_mask`` is the
    legality mask at that state (so cloning matches training-time masking).
    """
    obs_rows: list[np.ndarray] = []
    act_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    for i in range(n_episodes):
        obs, _ = gym_env.reset(seed=seed0 + i)
        done = False
        while not done:
            native_action = demo_policy(gym_env.env)  # demonstrator acts on the native env
            obs_rows.append(np.asarray(obs, dtype=np.float32))
            act_rows.append(flat_action_from_native(gym_env.env, native_action))
            mask_rows.append(np.asarray(gym_env.action_masks(), dtype=bool))
            native_obs, _reward, terminated, truncated, _info = gym_env.env.step(native_action)
            obs = gym_env._to_array(native_obs)
            done = terminated or truncated
    return (
        np.asarray(obs_rows, dtype=np.float32),
        np.asarray(act_rows, dtype=np.int64),
        np.asarray(mask_rows, dtype=bool),
    )


def behavior_clone(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    masks: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    log: bool = True,
) -> None:
    """Supervise ``model.policy`` to imitate the demonstrator's actions in place.

    Minimises ``-log pi(a_demo | s)`` under the masked MultiCategorical head via
    ``policy.evaluate_actions``. Mutates the policy weights (a warm start for PPO).

    ``weights`` gives a per-sample loss weight, used to up-weight the rare
    "real decision" steps (a vessel actually dispatching) so the clone is not
    drowned by the WAIT-dominated majority.
    """
    import torch

    policy = model.policy
    device = policy.device
    obs_t = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=device)
    act_t = torch.as_tensor(np.asarray(actions, dtype=np.int64), device=device)
    mask_t = (
        torch.as_tensor(np.asarray(masks, dtype=bool), device=device)
        if masks is not None
        else None
    )
    w_t = (
        torch.as_tensor(np.asarray(weights, dtype=np.float32), device=device)
        if weights is not None
        else None
    )
    n = obs_t.shape[0]
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    policy.set_training_mode(True)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        running = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_masks = mask_t[idx] if mask_t is not None else None
            _values, log_prob, _entropy = policy.evaluate_actions(
                obs_t[idx], act_t[idx], action_masks=batch_masks
            )
            nll = -log_prob
            if w_t is not None:
                wb = w_t[idx]
                loss = (nll * wb).sum() / wb.sum().clamp_min(1e-8)
            else:
                loss = nll.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * idx.shape[0]
        if log:
            print(f"[bc] epoch {epoch + 1}/{epochs}  weighted_nll={running / n:.4f}", flush=True)
    policy.set_training_mode(False)


def decision_step_weights(
    actions: np.ndarray,
    vessel_count: int,
    nonwait_weight: float = 10.0,
) -> np.ndarray:
    """Weight steps where any vessel does something other than WAIT (index 0).

    The demonstrator WAITs on almost every step (loading), so plain BC is
    dominated by WAIT. Up-weighting the rare dispatch steps focuses the clone on
    the decisions that actually matter and helps the argmax (deterministic)
    policy learn to dispatch instead of collapsing to idle.
    """
    vessel_actions = np.asarray(actions, dtype=np.int64)[:, :vessel_count]
    is_decision = (vessel_actions != 0).any(axis=1)
    return np.where(is_decision, float(nonwait_weight), 1.0).astype(np.float32)


def bc_pretrain(
    model,
    gym_env: CCSGymEnv,
    demo_policy: Policy,
    n_episodes: int = 20,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed0: int = 0,
    nonwait_weight: float = 10.0,
) -> None:
    """Collect demonstrations from ``demo_policy`` and behavior-clone into ``model``.

    ``nonwait_weight`` up-weights the loss on dispatch (non-WAIT) steps.
    """
    obs, acts, masks = collect_demonstrations(gym_env, demo_policy, n_episodes, seed0=seed0)
    if len(obs) == 0:
        raise ValueError("No demonstrations collected; check demo_policy / episode length.")
    vessel_count = len(gym_env.env.vessel_ids)
    weights = decision_step_weights(acts, vessel_count, nonwait_weight=nonwait_weight)
    n_decision = int((weights > 1.0).sum())
    print(
        f"[bc] collected {len(obs)} pairs from {n_episodes} episodes; "
        f"{n_decision} dispatch steps up-weighted x{nonwait_weight:g}",
        flush=True,
    )
    behavior_clone(
        model, obs, acts, masks=masks, weights=weights,
        epochs=epochs, batch_size=batch_size, lr=lr,
    )
