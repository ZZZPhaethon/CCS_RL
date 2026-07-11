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


def _observation_count(observations) -> int:
    """Return the common batch size for array or dictionary observations."""
    if isinstance(observations, dict):
        if not observations:
            raise ValueError("Observation dictionary cannot be empty.")
        counts = {key: value.shape[0] for key, value in observations.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(
                "Observation dictionary values must share a leading dimension; "
                f"got {counts}."
            )
        return next(iter(counts.values()))
    return observations.shape[0]


def _tensor_observations(observations, device):
    """Convert array or dictionary observations to float tensors on ``device``."""
    import torch

    if isinstance(observations, dict):
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in observations.items()
        }
    return torch.as_tensor(observations, dtype=torch.float32, device=device)


def _index_observations(observations, idx):
    """Apply one row index to every component of an observation batch."""
    if isinstance(observations, dict):
        return {key: value[idx] for key, value in observations.items()}
    return observations[idx]


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
    observations: np.ndarray | dict[str, np.ndarray],
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
    obs_t = _tensor_observations(observations, device)
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
    n = _observation_count(obs_t)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    policy.set_training_mode(True)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        running = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_obs = _index_observations(obs_t, idx)
            batch_masks = mask_t[idx] if mask_t is not None else None
            if w_t is not None and w_t.ndim == 2:
                log_prob = _masked_action_log_probs(policy, batch_obs, act_t[idx], batch_masks)
                nll = -log_prob
            else:
                _values, log_prob, _entropy = policy.evaluate_actions(
                    batch_obs, act_t[idx], action_masks=batch_masks
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


def _masked_action_log_probs(policy, obs, actions, action_masks=None):
    """Per-action-dimension log probabilities under the masked policy."""
    import torch

    features = policy.extract_features(obs)
    if policy.share_features_extractor:
        latent_pi, _latent_vf = policy.mlp_extractor(features)
    else:
        pi_features, _vf_features = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
    distribution = policy._get_action_dist_from_latent(latent_pi)
    if action_masks is not None:
        distribution.apply_masking(action_masks)
    if not hasattr(distribution, "distributions"):
        return distribution.log_prob(actions).reshape(-1, 1)
    actions = actions.view(-1, len(distribution.action_dims))
    return torch.stack(
        [
            dist.log_prob(action)
            for dist, action in zip(distribution.distributions, torch.unbind(actions, dim=1))
        ],
        dim=1,
    )


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


def action_dimension_weights(
    actions: np.ndarray,
    vessel_count: int,
    nonwait_weight: float = 10.0,
) -> np.ndarray:
    """Per-action weights that up-weight only non-WAIT vessel decisions."""
    actions = np.asarray(actions, dtype=np.int64)
    weights = np.ones(actions.shape, dtype=np.float32)
    vessel_actions = actions[:, :vessel_count]
    weights[:, :vessel_count] = np.where(
        vessel_actions != 0,
        float(nonwait_weight),
        1.0,
    )
    return weights


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect demonstrations from ``demo_policy`` and behavior-clone into ``model``.

    ``nonwait_weight`` up-weights the loss on dispatch (non-WAIT) steps. Returns
    the ``(obs, actions, masks, weights)`` demo arrays so they can be reused for
    kickstarting during PPO fine-tune.
    """
    obs, acts, masks = collect_demonstrations(gym_env, demo_policy, n_episodes, seed0=seed0)
    if len(obs) == 0:
        raise ValueError("No demonstrations collected; check demo_policy / episode length.")
    vessel_count = len(gym_env.env.vessel_ids)
    weights = action_dimension_weights(acts, vessel_count, nonwait_weight=nonwait_weight)
    n_decision = int((weights[:, :vessel_count] > 1.0).sum())
    print(
        f"[bc] collected {len(obs)} pairs from {n_episodes} episodes; "
        f"{n_decision} dispatch actions up-weighted x{nonwait_weight:g}",
        flush=True,
    )
    behavior_clone(
        model, obs, acts, masks=masks, weights=weights,
        epochs=epochs, batch_size=batch_size, lr=lr,
    )
    return obs, acts, masks, weights


def make_kickstart_callback(
    observations: np.ndarray | dict[str, np.ndarray],
    actions: np.ndarray,
    masks: np.ndarray | None,
    weights: np.ndarray | None,
    total_timesteps: int,
    coef0: float = 1.0,
    n_batches: int = 4,
    batch_size: int = 256,
    lr: float = 3e-4,
    verbose: int = 0,
):
    """A callback that interleaves decaying BC updates into PPO fine-tuning.

    Kickstarting: before each PPO update, take a few weighted behavior-cloning
    gradient steps toward the demonstrator, with a coefficient that decays
    linearly to 0 over training. This anchors the policy to the teacher so RL
    refines it instead of drifting away (which degraded plain BC+PPO).
    """
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    class _KickstartBC(BaseCallback):
        def __init__(self):
            super().__init__(verbose)
            self._opt = None

        def _on_training_start(self) -> None:
            policy = self.model.policy
            device = policy.device
            self._obs = _tensor_observations(observations, device)
            self._act = torch.as_tensor(np.asarray(actions, dtype=np.int64), device=device)
            self._mask = (
                torch.as_tensor(np.asarray(masks, dtype=bool), device=device)
                if masks is not None else None
            )
            self._w = (
                torch.as_tensor(np.asarray(weights, dtype=np.float32), device=device)
                if weights is not None else None
            )
            _observation_count(self._obs)
            self._opt = torch.optim.Adam(policy.parameters(), lr=lr)

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            progress = min(1.0, self.num_timesteps / max(1, total_timesteps))
            coef = coef0 * (1.0 - progress)
            if coef <= 1e-6:
                return
            policy = self.model.policy
            policy.set_training_mode(True)
            n = _observation_count(self._obs)
            bc_val = 0.0
            for _ in range(n_batches):
                idx = torch.randint(0, n, (batch_size,), device=policy.device)
                batch_obs = _index_observations(self._obs, idx)
                bm = self._mask[idx] if self._mask is not None else None
                if self._w is not None and self._w.ndim == 2:
                    log_prob = _masked_action_log_probs(policy, batch_obs, self._act[idx], bm)
                    nll = -log_prob
                else:
                    _v, log_prob, _e = policy.evaluate_actions(batch_obs, self._act[idx], action_masks=bm)
                    nll = -log_prob
                if self._w is not None:
                    wb = self._w[idx]
                    bc = (nll * wb).sum() / wb.sum().clamp_min(1e-8)
                else:
                    bc = nll.mean()
                loss = coef * bc
                self._opt.zero_grad()
                loss.backward()
                self._opt.step()
                bc_val = float(bc.item())
            if self.verbose:
                print(f"[kickstart] t={self.num_timesteps} coef={coef:.3f} bc_nll={bc_val:.3f}", flush=True)

    return _KickstartBC()
