"""Train the formal direct hourly Masked Double-DQN baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sim.control.event_based.rl.observation_encoder import (
    FORECAST_WINDOWS_H,
    FUTURE_SUMMARY_REPRESENTATION_ID,
    future_summary_feature_names,
)
from sim.control.hourly_ppo.gym_env import HourlyCentralizedPPOEnv
from sim.control.hourly_ppo.train_hourly_ppo import (
    evaluate_seed,
    make_hourly_native_env,
)
from sim.simulator import SimulatorStepUsage

from .gym_env import HourlyJointActionDQNEnv
from .model import MaskedDoubleDQNPolicy, QNetwork


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


@dataclass
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminals: torch.Tensor
    next_action_masks: torch.Tensor


class ReplayBuffer:
    """Preallocated replay storage with masked next actions."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_count: int,
    ) -> None:
        if min(capacity, observation_dim, action_count) <= 0:
            raise ValueError("replay dimensions must be positive")
        self.capacity = int(capacity)
        self.states = np.empty(
            (capacity, observation_dim), dtype=np.float32
        )
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminals = np.empty(capacity, dtype=bool)
        self.next_action_masks = np.empty(
            (capacity, action_count), dtype=bool
        )
        self.position = 0
        self.size = 0

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        terminals: np.ndarray,
        next_action_masks: np.ndarray,
    ) -> None:
        count = len(states)
        if count <= 0 or count > self.capacity:
            raise ValueError("invalid replay insertion size")
        indices = (self.position + np.arange(count)) % self.capacity
        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.terminals[indices] = terminals
        self.next_action_masks[indices] = next_action_masks
        self.position = int((self.position + count) % self.capacity)
        self.size = min(self.capacity, self.size + count)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> ReplayBatch:
        if self.size < batch_size:
            raise ValueError("replay buffer does not contain a full batch")
        indices = rng.integers(0, self.size, size=batch_size)
        return ReplayBatch(
            states=torch.as_tensor(self.states[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_states=torch.as_tensor(
                self.next_states[indices], device=device
            ),
            terminals=torch.as_tensor(
                self.terminals[indices], device=device, dtype=torch.float32
            ),
            next_action_masks=torch.as_tensor(
                self.next_action_masks[indices], device=device
            ),
        )


def double_dqn_loss(
    online_network: QNetwork,
    target_network: QNetwork,
    batch: ReplayBatch,
    *,
    gamma: float,
) -> torch.Tensor:
    """Huber Double-DQN loss with next-state legality masking."""

    predicted = online_network(batch.states).gather(
        1, batch.actions[:, None]
    ).squeeze(1)
    with torch.no_grad():
        online_next = online_network(batch.next_states).masked_fill(
            ~batch.next_action_masks,
            -torch.inf,
        )
        next_actions = online_next.argmax(dim=1)
        target_next = target_network(batch.next_states).gather(
            1, next_actions[:, None]
        ).squeeze(1)
        targets = batch.rewards + (
            float(gamma) * (1.0 - batch.terminals) * target_next
        )
    return F.smooth_l1_loss(predicted, targets)


def _epsilon(
    physical_steps: int,
    max_physical_steps: int,
    start: float,
    final: float,
    fraction: float,
) -> float:
    anneal_steps = max(1.0, float(max_physical_steps) * float(fraction))
    progress = min(1.0, physical_steps / anneal_steps)
    return float(start + progress * (final - start))


def _select_actions(
    network: QNetwork,
    states: np.ndarray,
    masks: np.ndarray,
    *,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> np.ndarray:
    network.eval()
    with torch.no_grad():
        q_values = network(torch.as_tensor(states, device=device))
        q_values = q_values.masked_fill(
            ~torch.as_tensor(masks, device=device), -torch.inf
        )
        greedy = q_values.argmax(dim=1).cpu().numpy()
    actions = greedy.astype(np.int64, copy=True)
    explore = rng.random(len(states)) < float(epsilon)
    for index in np.flatnonzero(explore):
        actions[index] = int(rng.choice(np.flatnonzero(masks[index])))
    return actions


def _save_checkpoint(
    path: Path,
    network: QNetwork,
    *,
    observation_dim: int,
    action_dims: tuple[int, ...],
    hidden_sizes: tuple[int, ...],
    physical_steps: int,
    gradient_updates: int,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "kind": "hourly_masked_double_dqn",
            "network_state_dict": {
                key: value.detach().cpu()
                for key, value in network.state_dict().items()
            },
            "observation_dim": int(observation_dim),
            "action_dims": list(action_dims),
            "action_count": int(np.prod(action_dims)),
            "hidden_sizes": list(hidden_sizes),
            "physical_steps": int(physical_steps),
            "gradient_updates": int(gradient_updates),
            "configuration": config,
        },
        path,
    )


def default_run_dir(*, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / "hourly_dqn" / f"seed{seed}__{timestamp}"


def train_hourly_dqn(
    *,
    seed: int = 0,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    future_summary_windows_h: tuple[int, ...] = FORECAST_WINDOWS_H,
    weather_mode: str = "window",
    warm_start: bool = True,
    scenario_protocol: str = "unified_window_v1",
    gamma: float = 1.0,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    reward_scale: float = 1e-6,
    num_envs: int = 8,
    device: str = "cpu",
    log_dir: Path | None = None,
    max_simulator_hour_steps: int = 9_505_319,
    training_seed_min: int = 100_000,
    training_seed_max: int = 999_999,
    validation_seeds: tuple[int, ...] = (),
    validation_every_simulator_hour_steps: int | None = None,
    hidden_sizes: tuple[int, ...] = (256, 256),
    replay_capacity: int = 1_000_000,
    learning_starts: int = 100_000,
    gradient_steps_per_vector_step: int = 2,
    target_update_interval: int = 10_000,
    epsilon_start: float = 1.0,
    epsilon_final: float = 0.05,
    epsilon_fraction: float = 0.20,
    log_every_simulator_hour_steps: int = 100_000,
) -> Path:
    """Train from scratch under a hard bottom-level simulator-hour cap."""

    if gamma != 1.0:
        raise ValueError("Formal hourly DQN fixes gamma=1.0.")
    if min(
        batch_size,
        num_envs,
        max_simulator_hour_steps,
        replay_capacity,
        learning_starts,
        gradient_steps_per_vector_step,
        target_update_interval,
        log_every_simulator_hour_steps,
    ) <= 0:
        raise ValueError("training sizes and intervals must be positive")
    if learning_starts < batch_size:
        raise ValueError("learning_starts must cover at least one batch")
    if training_seed_min > training_seed_max:
        raise ValueError("training_seed_min must not exceed training_seed_max")
    if any(
        training_seed_min <= value <= training_seed_max
        for value in validation_seeds
    ):
        raise ValueError("training and validation seeds overlap")
    if not 0.0 <= epsilon_final <= epsilon_start <= 1.0:
        raise ValueError("epsilon must satisfy 0 <= final <= start <= 1")
    if not 0.0 < epsilon_fraction <= 1.0:
        raise ValueError("epsilon_fraction must be in (0, 1]")

    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError(
            "train_hourly_dqn requires stable-baselines3 for vector workers"
        ) from exc

    run_dir = log_dir or default_run_dir(seed=seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    effective_budget_stop = (
        max_simulator_hour_steps
        - max_simulator_hour_steps % num_envs
    )
    if effective_budget_stop <= 0:
        raise ValueError("budget must cover at least one vector step")
    if validation_seeds and validation_every_simulator_hour_steps is None:
        validation_every_simulator_hour_steps = max(
            num_envs,
            effective_budget_stop // 10,
        )

    def make_worker(rank: int):
        def build_worker():
            native = make_hourly_native_env(
                scenario=scenario,
                episode_hours=episode_hours,
                forecast_context_hours=forecast_context_hours,
                weather_mode=weather_mode,
                warm_start=warm_start,
                scenario_protocol=scenario_protocol,
                reward_scale=reward_scale,
            )
            hourly = HourlyCentralizedPPOEnv(
                native,
                future_summary_windows_h=future_summary_windows_h,
                episode_seed_min=training_seed_min,
                episode_seed_max=training_seed_max,
            )
            return HourlyJointActionDQNEnv(hourly)

        return build_worker

    vec_env = (
        DummyVecEnv([make_worker(0)])
        if num_envs == 1
        else SubprocVecEnv(
            [make_worker(rank) for rank in range(num_envs)],
            start_method="spawn",
        )
    )
    vec_env.seed(seed)
    observations = vec_env.reset()
    states = np.asarray(observations["state"], dtype=np.float32)
    masks = np.asarray(observations["action_mask"], dtype=bool)
    observation_dim = int(states.shape[1])
    action_count = int(masks.shape[1])
    action_dims = tuple(
        int(value) for value in vec_env.get_attr("action_dims")[0]
    )
    if action_count != int(np.prod(action_dims)):
        raise RuntimeError("joint action enumeration does not match action dimensions")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    torch_device = torch.device(device)
    online = QNetwork(observation_dim, action_count, hidden_sizes).to(torch_device)
    target = QNetwork(observation_dim, action_count, hidden_sizes).to(torch_device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
    replay = ReplayBuffer(replay_capacity, observation_dim, action_count)

    schema_native = make_hourly_native_env(
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        weather_mode=weather_mode,
        warm_start=warm_start,
        scenario_protocol=scenario_protocol,
        reward_scale=reward_scale,
    )
    validation_env = (
        make_hourly_native_env(
            scenario=scenario,
            episode_hours=episode_hours,
            forecast_context_hours=forecast_context_hours,
            weather_mode=weather_mode,
            warm_start=warm_start,
            scenario_protocol=scenario_protocol,
            reward_scale=reward_scale,
        )
        if validation_seeds
        else None
    )
    config: dict[str, Any] = {
        "interface_version": 1,
        "paper_name": "Hourly Masked Double DQN",
        "algorithm": "Masked Double DQN",
        "training_from_scratch": True,
        "decision_interval_h": 1.0,
        "physical_time_step_h": 1.0,
        "direct_native_action": True,
        "action_space": "Discrete enumeration of MultiDiscrete vessel dispatch",
        "action_dims": list(action_dims),
        "joint_action_count": action_count,
        "legal_action_masks": True,
        "uses_goal_executor": False,
        "uses_event_trigger": False,
        "uses_greedy_default": False,
        "uses_residual_actions": False,
        "uses_bc_warm_start": False,
        "well_control": "automatic_max",
        "scenario": scenario,
        "scenario_protocol": scenario_protocol,
        "weather_mode": weather_mode,
        "warm_start": warm_start,
        "episode_hours": episode_hours,
        "forecast_context_hours": forecast_context_hours,
        "future_summary_representation_id": FUTURE_SUMMARY_REPRESENTATION_ID,
        "future_summary_windows_h": list(future_summary_windows_h),
        "future_summary_feature_names": list(
            future_summary_feature_names(
                schema_native,
                future_summary_windows_h,
            )
        ),
        "valid_fraction_feature": False,
        "objective": "negative realised economic cost plus terminal cleanup cost",
        "reward_scale": reward_scale,
        "gamma": gamma,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_envs": num_envs,
        "hidden_sizes": list(hidden_sizes),
        "replay_capacity": replay_capacity,
        "learning_starts": learning_starts,
        "gradient_steps_per_vector_step": gradient_steps_per_vector_step,
        "target_update_interval": target_update_interval,
        "epsilon_start": epsilon_start,
        "epsilon_final": epsilon_final,
        "epsilon_fraction": epsilon_fraction,
        "max_simulator_hour_steps": max_simulator_hour_steps,
        "effective_vectorized_budget_stop": effective_budget_stop,
        "training_seed_min": training_seed_min,
        "training_seed_max": training_seed_max,
        "validation_seeds": list(validation_seeds),
        "validation_every_simulator_hour_steps": (
            validation_every_simulator_hour_steps
        ),
        "checkpoint_selection": (
            "minimum_validation_mean_total_cost" if validation_seeds else None
        ),
        "model_seed": seed,
        "device": device,
        "parameter_count": sum(parameter.numel() for parameter in online.parameters()),
    }
    _write_json(run_dir / "config.json", config)

    physical_steps = 0
    gradient_updates = 0
    next_target_update = target_update_interval
    next_log = log_every_simulator_hour_steps
    next_validation = float(
        validation_every_simulator_hour_steps
        if validation_every_simulator_hour_steps is not None
        else "inf"
    )
    last_validation_steps = -1
    best_validation_cost = float("inf")
    validation_history: list[dict[str, Any]] = []
    recent_losses: list[float] = []
    started_at = perf_counter()

    def run_validation() -> None:
        nonlocal best_validation_cost, last_validation_steps
        if validation_env is None:
            return
        policy = MaskedDoubleDQNPolicy(
            online,
            action_dims,
            device=torch_device,
        )
        records = [
            evaluate_seed(
                policy,
                validation_env,
                seed=validation_seed,
                future_summary_windows_h=future_summary_windows_h,
            )
            for validation_seed in validation_seeds
        ]
        mean_cost = float(
            np.mean([float(row["total_cost_eur"]) for row in records])
        )
        is_best = mean_cost < best_validation_cost
        payload = {
            "training_simulator_hour_steps": physical_steps,
            "gradient_updates": gradient_updates,
            "is_best": is_best,
            "mean_total_cost_eur": mean_cost,
            "validation_seeds": list(validation_seeds),
            "per_seed": records,
        }
        validation_history.append(payload)
        _write_json(
            run_dir / "validation" / f"simulator_hour_{physical_steps}.json",
            payload,
        )
        if is_best:
            best_validation_cost = mean_cost
            _save_checkpoint(
                run_dir / "masked_double_dqn_best_validation.pt",
                online,
                observation_dim=observation_dim,
                action_dims=action_dims,
                hidden_sizes=hidden_sizes,
                physical_steps=physical_steps,
                gradient_updates=gradient_updates,
                config=config,
            )
            _write_json(
                run_dir / "validation" / "best.json",
                {
                    **payload,
                    "model_path": str(
                        run_dir / "masked_double_dqn_best_validation.pt"
                    ),
                },
            )
        last_validation_steps = physical_steps
        online.train()

    try:
        while physical_steps < effective_budget_stop:
            epsilon = _epsilon(
                physical_steps,
                effective_budget_stop,
                epsilon_start,
                epsilon_final,
                epsilon_fraction,
            )
            actions = _select_actions(
                online,
                states,
                masks,
                epsilon=epsilon,
                rng=rng,
                device=torch_device,
            )
            next_observations, rewards, dones, infos = vec_env.step(actions)
            next_states = np.asarray(
                next_observations["state"], dtype=np.float32
            )
            next_masks = np.asarray(
                next_observations["action_mask"], dtype=bool
            )
            terminals = np.asarray(
                [
                    bool(done)
                    and not bool(info.get("TimeLimit.truncated", False))
                    for done, info in zip(dones, infos)
                ],
                dtype=bool,
            )
            replay.add(
                states,
                actions,
                np.asarray(rewards, dtype=np.float32),
                next_states,
                terminals,
                next_masks,
            )
            states = next_states
            masks = next_masks
            physical_steps += num_envs

            if physical_steps >= learning_starts and replay.size >= batch_size:
                online.train()
                for _ in range(gradient_steps_per_vector_step):
                    batch = replay.sample(batch_size, rng, torch_device)
                    loss = double_dqn_loss(
                        online,
                        target,
                        batch,
                        gamma=gamma,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                    optimizer.step()
                    gradient_updates += 1
                    recent_losses.append(float(loss.item()))
                    if len(recent_losses) > 1000:
                        del recent_losses[: len(recent_losses) - 1000]

            if physical_steps >= next_target_update:
                target.load_state_dict(online.state_dict())
                while next_target_update <= physical_steps:
                    next_target_update += target_update_interval

            if physical_steps >= next_log:
                payload = {
                    "training_simulator_hour_steps": physical_steps,
                    "gradient_updates": gradient_updates,
                    "epsilon": epsilon,
                    "replay_size": replay.size,
                    "mean_recent_loss": (
                        float(np.mean(recent_losses)) if recent_losses else None
                    ),
                    "elapsed_seconds": perf_counter() - started_at,
                }
                print(json.dumps(payload, sort_keys=True), flush=True)
                _write_json(run_dir / "latest_training_metrics.json", payload)
                while next_log <= physical_steps:
                    next_log += log_every_simulator_hour_steps

            if physical_steps >= next_validation:
                run_validation()
                interval = float(validation_every_simulator_hour_steps)
                while next_validation <= physical_steps:
                    next_validation += interval
    finally:
        worker_usage = vec_env.env_method("training_simulator_usage")
        vec_env.close()

    if validation_seeds and last_validation_steps != physical_steps:
        run_validation()
    _save_checkpoint(
        run_dir / "masked_double_dqn_final.pt",
        online,
        observation_dim=observation_dim,
        action_dims=action_dims,
        hidden_sizes=hidden_sizes,
        physical_steps=physical_steps,
        gradient_updates=gradient_updates,
        config=config,
    )
    usage = SimulatorStepUsage(
        calls=sum(int(row["simulator_step_calls"]) for row in worker_usage),
        simulated_hours=sum(
            float(row["simulator_simulated_hours"]) for row in worker_usage
        ),
    )
    if usage.calls != physical_steps:
        raise RuntimeError(
            f"vector steps recorded {physical_steps} calls, workers recorded {usage.calls}"
        )
    training_complete = {
        "state": "simulator_budget_stop_reached",
        **usage.as_dict(),
        "max_simulator_hour_steps": max_simulator_hour_steps,
        "effective_vectorized_budget_stop": effective_budget_stop,
        "simulator_budget_fraction": usage.hour_steps / max_simulator_hour_steps,
        "simulator_budget_stop_reached": (
            usage.hour_steps >= effective_budget_stop - 1e-9
        ),
        "gradient_updates": gradient_updates,
        "replay_size": replay.size,
        "elapsed_seconds": perf_counter() - started_at,
        "model_path": str(run_dir / "masked_double_dqn_final.pt"),
        "best_validation_model_path": (
            str(run_dir / "masked_double_dqn_best_validation.pt")
            if validation_seeds
            else None
        ),
        "validation_history": validation_history,
    }
    _write_json(run_dir / "training_complete.json", training_complete)
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the formal direct hourly Masked Double-DQN baseline."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", default="northern_lights_phase1_3vessels")
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument(
        "--future-summary-windows-h",
        type=int,
        nargs="+",
        default=list(FORECAST_WINDOWS_H),
    )
    parser.add_argument("--weather-mode", choices=("window", "block"), default="window")
    parser.add_argument(
        "--scenario-protocol",
        choices=("unified_window_v1", "local_formal"),
        default="unified_window_v1",
    )
    parser.add_argument(
        "--warm-start", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--reward-scale", type=float, default=1e-6)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument(
        "--max-simulator-hour-steps", type=int, default=9_505_319
    )
    parser.add_argument("--training-seed-min", type=int, default=100_000)
    parser.add_argument("--training-seed-max", type=int, default=999_999)
    parser.add_argument("--validation-seeds", type=int, nargs="*", default=[])
    parser.add_argument(
        "--validation-every-simulator-hour-steps", type=int, default=None
    )
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=100_000)
    parser.add_argument("--gradient-steps-per-vector-step", type=int, default=2)
    parser.add_argument("--target-update-interval", type=int, default=10_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-fraction", type=float, default=0.20)
    parser.add_argument(
        "--log-every-simulator-hour-steps", type=int, default=100_000
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = train_hourly_dqn(
        seed=args.seed,
        scenario=args.scenario,
        episode_hours=args.episode_hours,
        forecast_context_hours=args.forecast_context_hours,
        future_summary_windows_h=tuple(args.future_summary_windows_h),
        weather_mode=args.weather_mode,
        warm_start=args.warm_start,
        scenario_protocol=args.scenario_protocol,
        gamma=args.gamma,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reward_scale=args.reward_scale,
        num_envs=args.num_envs,
        device=args.device,
        log_dir=args.log_dir,
        max_simulator_hour_steps=args.max_simulator_hour_steps,
        training_seed_min=args.training_seed_min,
        training_seed_max=args.training_seed_max,
        validation_seeds=tuple(args.validation_seeds),
        validation_every_simulator_hour_steps=(
            args.validation_every_simulator_hour_steps
        ),
        hidden_sizes=tuple(args.hidden_sizes),
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        gradient_steps_per_vector_step=args.gradient_steps_per_vector_step,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_fraction=args.epsilon_fraction,
        log_every_simulator_hour_steps=args.log_every_simulator_hour_steps,
    )
    print(f"Saved hourly masked Double-DQN under: {run_dir}")


if __name__ == "__main__":
    main()
