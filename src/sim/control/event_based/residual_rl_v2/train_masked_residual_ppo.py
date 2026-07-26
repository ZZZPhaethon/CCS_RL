"""Train MaskablePPO with persistent rule-counterfactual rewards.

使用持续规则反事实奖励训练 MaskablePPO。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
from functools import partial
import json
import math
from pathlib import Path
from typing import Any

from sim.control.event_based.residual_rl.observation import residual_feature_names
from sim.control.event_based.residual_rl.train_residual_ppo import (
    _make_progress_callback,
    _planned_timesteps,
    _write_json,
)
from sim.control.event_based.rl.reward import HighLevelRewardConfig

from .evaluation import evaluate_seeds, validation_metrics
from .factory import (
    make_masked_residual_gym_env,
    make_masked_residual_native_env,
)


DEFAULT_VALIDATION_SEEDS = tuple(range(2_000_001, 2_000_009))


def default_run_dir(
    *,
    scenario: str,
    episode_hours: int,
    decision_interval_h: float,
    seed: int,
) -> Path:
    """Return a unique v2 training directory under ``logs``.

    返回 ``logs`` 下唯一的 v2 训练目录。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = (
        f"{scenario}__{episode_hours}h__decision{decision_interval_h:g}h"
        f"__masked_residual_v2__seed{seed}__{timestamp}"
    )
    return Path("logs") / "residual_rl_v2" / label


def _make_validation_callback(
    *,
    run_dir: Path,
    scenario: str,
    episode_hours: int,
    forecast_context_hours: int,
    decision_interval_h: float,
    event_triggered: bool,
    weather_mode: str,
    hard_scenario_probability: float,
    reward: HighLevelRewardConfig,
    validation_seeds: tuple[int, ...],
    validation_every_steps: int,
    cvar_tail_fraction: float,
    tail_vent_penalty_eur_per_t: float,
):
    """Create masked fixed-seed validation and best-model saving.

    创建带掩码的固定 seed 验证和最优模型保存。
    """
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError("Training requires stable-baselines3.") from exc

    class MaskedValidationCallback(BaseCallback):
        """Evaluate physical metrics without using training reward alone.

        使用物理指标评估，而不是仅依赖训练奖励。
        """

        def __init__(self) -> None:
            super().__init__()
            self.next_evaluation = max(1, int(validation_every_steps))
            self.last_evaluated = -1
            self.best_loss = math.inf
            self.validation_dir = run_dir / "validation"
            self.validation_env = None
            self.stream = None
            self.writer = None

        def _on_training_start(self) -> None:
            self.validation_dir.mkdir(parents=True, exist_ok=False)
            self.validation_env = make_masked_residual_native_env(
                scenario=scenario,
                episode_hours=episode_hours,
                forecast_context_hours=forecast_context_hours,
                decision_interval_h=decision_interval_h,
                event_triggered=event_triggered,
                weather_mode=weather_mode,
                hard_scenario_probability=hard_scenario_probability,
                reward=reward,
            )
            self.stream = (self.validation_dir / "metrics.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            )
            fields = (
                "timesteps",
                "mean_total_cost_eur",
                "mean_stored_t",
                "mean_vented_t",
                "worst_vented_t",
                "cvar_vented_t",
                "mean_effective_intervention_rate",
                "hard_violations",
                "selection_loss",
                "is_best",
            )
            self.writer = csv.DictWriter(self.stream, fieldnames=fields)
            self.writer.writeheader()

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_evaluation:
                self._evaluate()
                while self.next_evaluation <= self.num_timesteps:
                    self.next_evaluation += max(
                        1,
                        int(validation_every_steps),
                    )
            return True

        def _on_training_end(self) -> None:
            if self.last_evaluated != self.num_timesteps:
                self._evaluate()
            if self.stream is not None:
                self.stream.close()

        def _evaluate(self) -> None:
            assert self.validation_env is not None
            records = evaluate_seeds(
                self.model,
                self.validation_env,
                validation_seeds,
            )
            metrics = validation_metrics(
                records,
                cvar_tail_fraction=cvar_tail_fraction,
                tail_vent_penalty_eur_per_t=(
                    tail_vent_penalty_eur_per_t
                ),
            )
            is_best = metrics["selection_loss"] < self.best_loss
            if is_best:
                self.best_loss = metrics["selection_loss"]
                self.model.save(
                    run_dir / "maskable_residual_v2_best_validation"
                )
                _write_json(
                    self.validation_dir / "best.json",
                    {
                        "timesteps": self.num_timesteps,
                        **metrics,
                        "model_path": str(
                            run_dir
                            / "maskable_residual_v2_best_validation"
                        ),
                    },
                )
            row = {
                "timesteps": self.num_timesteps,
                **metrics,
                "is_best": int(is_best),
            }
            if self.writer is not None and self.stream is not None:
                self.writer.writerow(row)
                self.stream.flush()
            _write_json(
                self.validation_dir / f"step_{self.num_timesteps}.json",
                {
                    "timesteps": self.num_timesteps,
                    "validation_seeds": list(validation_seeds),
                    "metrics": metrics,
                    "per_seed": records,
                },
            )
            effective_rate = metrics[
                "mean_effective_intervention_rate"
            ]
            print(
                "Masked residual validation | "
                f"step={self.num_timesteps} | "
                f"mean_cost=EUR {metrics['mean_total_cost_eur']:,.0f} | "
                f"worst_vent={metrics['worst_vented_t']:,.1f} t | "
                f"CVaR_vent={metrics['cvar_vented_t']:,.1f} t | "
                f"effective={100.0 * effective_rate:.1f}% | "
                f"best={'yes' if is_best else 'no'}",
                flush=True,
            )
            self.last_evaluated = self.num_timesteps

    return MaskedValidationCallback()


def train_masked_residual_ppo(
    *,
    total_timesteps: int = 100_000,
    seed: int = 0,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    decision_interval_h: float = 24.0,
    event_triggered: bool = True,
    weather_mode: str = "window",
    hard_scenario_probability: float = 0.30,
    num_envs: int = 4,
    vec_env_backend: str = "subproc",
    training_seed_min: int = 100_000,
    training_seed_max: int = 999_999,
    validation_seeds: tuple[int, ...] = DEFAULT_VALIDATION_SEEDS,
    validation_every_steps: int = 5_000,
    cvar_tail_fraction: float = 0.25,
    tail_vent_penalty_eur_per_t: float = 500.0,
    gamma: float = 1.0,
    n_steps: int = 256,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    entropy_coefficient: float = 0.01,
    device: str = "cpu",
    log_dir: Path | None = None,
    status_every_steps: int = 100,
    reward: HighLevelRewardConfig | None = None,
):
    """Train masked residual PPO and save final plus validation-best models.

    训练掩码残差 PPO，并保存最终模型和验证集最优模型。
    """
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
        )
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecMonitor,
        )
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError(
            "Training requires stable-baselines3 and sb3-contrib."
        ) from exc

    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if vec_env_backend not in {"dummy", "subproc"}:
        raise ValueError("vec_env_backend must be 'dummy' or 'subproc'.")
    validation_seeds = tuple(int(value) for value in validation_seeds)
    if not validation_seeds:
        raise ValueError("At least one validation seed is required.")
    overlap = [
        value
        for value in validation_seeds
        if training_seed_min <= value <= training_seed_max
    ]
    if overlap:
        raise ValueError(
            f"Training and validation seeds overlap: {overlap}."
        )

    run_dir = log_dir or default_run_dir(
        scenario=scenario,
        episode_hours=episode_hours,
        decision_interval_h=decision_interval_h,
        seed=seed,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    reward_config = reward or HighLevelRewardConfig()
    env_kwargs = {
        "scenario": scenario,
        "episode_hours": episode_hours,
        "forecast_context_hours": forecast_context_hours,
        "decision_interval_h": decision_interval_h,
        "event_triggered": event_triggered,
        "weather_mode": weather_mode,
        "hard_scenario_probability": hard_scenario_probability,
        "reward": reward_config,
        "episode_seed_min": training_seed_min,
        "episode_seed_max": training_seed_max,
    }
    env_fns = [
        partial(make_masked_residual_gym_env, **env_kwargs)
        for _ in range(num_envs)
    ]
    if vec_env_backend == "subproc" and num_envs > 1:
        vector_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vector_env = DummyVecEnv(env_fns)
    vector_env.seed(seed)
    vector_env = VecMonitor(
        vector_env,
        filename=str(run_dir / "monitor.csv"),
    )

    probe = make_masked_residual_native_env(
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        decision_interval_h=decision_interval_h,
        event_triggered=event_triggered,
        weather_mode=weather_mode,
        hard_scenario_probability=hard_scenario_probability,
        reward=reward_config,
    )
    planned_timesteps = _planned_timesteps(
        total_timesteps,
        n_steps,
        num_envs,
    )
    _write_json(
        run_dir / "config.json",
        {
            "interface_version": 2,
            "algorithm": "maskable_residual_ppo_v2",
            "scenario": scenario,
            "episode_hours": episode_hours,
            "forecast_context_hours": forecast_context_hours,
            "decision_interval_h": decision_interval_h,
            "event_triggered": event_triggered,
            "weather_mode": weather_mode,
            "hard_scenario_probability": hard_scenario_probability,
            "requested_timesteps": total_timesteps,
            "planned_timesteps": planned_timesteps,
            "seed": seed,
            "training_seed_range": [
                training_seed_min,
                training_seed_max,
            ],
            "validation_seeds": list(validation_seeds),
            "validation_every_steps": validation_every_steps,
            "cvar_tail_fraction": cvar_tail_fraction,
            "tail_vent_penalty_eur_per_t": (
                tail_vent_penalty_eur_per_t
            ),
            "gamma": gamma,
            "n_steps_per_env": n_steps,
            "num_envs": num_envs,
            "vec_env_backend": vec_env_backend,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "entropy_coefficient": entropy_coefficient,
            "device": device,
            "action_count": probe.action_count,
            "action_labels": list(probe.codec.labels()),
            "dynamic_action_masks": True,
            "priority_requires_two_eligible_vessels": True,
            "reward_mode": "persistent_rule_counterfactual_delta",
            "injection_control": (
                "highest_feasible_rate_by_rule_executor"
            ),
            "observation_size": probe.observation_size,
            "observation_features": list(
                residual_feature_names(probe.env)
            ),
            "high_level_reward": asdict(reward_config),
        },
    )

    callbacks = CallbackList(
        [
            _make_progress_callback(
                total_timesteps=planned_timesteps,
                run_dir=run_dir,
                report_every_steps=status_every_steps,
            ),
            _make_validation_callback(
                run_dir=run_dir,
                scenario=scenario,
                episode_hours=episode_hours,
                forecast_context_hours=forecast_context_hours,
                decision_interval_h=decision_interval_h,
                event_triggered=event_triggered,
                weather_mode=weather_mode,
                hard_scenario_probability=hard_scenario_probability,
                reward=reward_config,
                validation_seeds=validation_seeds,
                validation_every_steps=validation_every_steps,
                cvar_tail_fraction=cvar_tail_fraction,
                tail_vent_penalty_eur_per_t=(
                    tail_vent_penalty_eur_per_t
                ),
            ),
            CheckpointCallback(
                save_freq=max(
                    1,
                    planned_timesteps // (10 * num_envs),
                ),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="maskable_residual_v2",
            ),
        ]
    )
    model = MaskablePPO(
        "MlpPolicy",
        vector_env,
        seed=seed,
        gamma=float(gamma),
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        ent_coef=entropy_coefficient,
        device=device,
        verbose=0,
    )
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )
        model.save(run_dir / "maskable_residual_v2_final")
        _write_json(
            run_dir / "training_complete.json",
            {
                "state": "completed",
                "requested_timesteps": total_timesteps,
                "planned_timesteps": planned_timesteps,
                "final_model_path": str(
                    run_dir / "maskable_residual_v2_final"
                ),
                "best_validation_model_path": str(
                    run_dir
                    / "maskable_residual_v2_best_validation"
                ),
            },
        )
    finally:
        vector_env.close()
    return model, run_dir


def main() -> None:
    """Train masked residual PPO from PowerShell or another terminal.

    从 PowerShell 或其他终端训练掩码残差 PPO。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenario",
        default="northern_lights_phase1_3vessels",
    )
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument("--decision-interval-h", type=float, default=24.0)
    parser.add_argument(
        "--event-triggered",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--weather-mode",
        choices=("window", "block"),
        default="window",
    )
    parser.add_argument("--hard-scenario-probability", type=float, default=0.30)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--vec-env",
        choices=("dummy", "subproc"),
        default="subproc",
    )
    parser.add_argument("--training-seed-min", type=int, default=100_000)
    parser.add_argument("--training-seed-max", type=int, default=999_999)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_VALIDATION_SEEDS),
    )
    parser.add_argument("--validation-every-steps", type=int, default=5_000)
    parser.add_argument("--cvar-tail-fraction", type=float, default=0.25)
    parser.add_argument(
        "--tail-vent-penalty-eur-per-t",
        type=float,
        default=500.0,
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--status-every-steps", type=int, default=100)
    parser.add_argument("--reward-scale", type=float, default=1e-6)
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    _model, run_dir = train_masked_residual_ppo(
        total_timesteps=args.timesteps,
        seed=args.seed,
        scenario=args.scenario,
        episode_hours=args.episode_hours,
        forecast_context_hours=args.forecast_context_hours,
        decision_interval_h=args.decision_interval_h,
        event_triggered=args.event_triggered,
        weather_mode=args.weather_mode,
        hard_scenario_probability=args.hard_scenario_probability,
        num_envs=args.num_envs,
        vec_env_backend=args.vec_env,
        training_seed_min=args.training_seed_min,
        training_seed_max=args.training_seed_max,
        validation_seeds=tuple(args.validation_seeds),
        validation_every_steps=args.validation_every_steps,
        cvar_tail_fraction=args.cvar_tail_fraction,
        tail_vent_penalty_eur_per_t=(
            args.tail_vent_penalty_eur_per_t
        ),
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        entropy_coefficient=args.ent_coef,
        device=args.device,
        log_dir=args.log_dir,
        status_every_steps=args.status_every_steps,
        reward=HighLevelRewardConfig(reward_scale=args.reward_scale),
    )
    print(f"Saved masked residual PPO v2 under: {run_dir}")


if __name__ == "__main__":
    main()

