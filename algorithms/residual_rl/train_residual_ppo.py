"""Train event-triggered residual PPO over a safe rule dispatcher.

在安全规则调度器之上训练事件触发残差 PPO。
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
from time import perf_counter
from typing import Any

from algorithms.rl.reward import HighLevelRewardConfig

from .evaluation import evaluate_seeds, validation_metrics
from .factory import make_residual_gym_env, make_residual_native_env
from .observation import residual_feature_names


HOURLY_DISCOUNT = 0.999
DEFAULT_VALIDATION_SEEDS = tuple(range(2_000_001, 2_000_009))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one UTF-8 JSON status or metadata file.

    写入一个 UTF-8 JSON 状态或元数据文件。
    """
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def default_run_dir(
    *,
    scenario: str,
    episode_hours: int,
    decision_interval_h: float,
    seed: int,
) -> Path:
    """Return a unique residual-RL directory under ``logs``.

    返回 ``logs`` 下唯一的残差 RL 训练目录。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = (
        f"{scenario}__{episode_hours}h__decision{decision_interval_h:g}h"
        f"__residual__seed{seed}__{timestamp}"
    )
    return Path("logs") / "residual_rl" / label


def _planned_timesteps(
    requested_timesteps: int,
    n_steps: int,
    num_envs: int,
) -> int:
    """Return PPO's effective total after complete vector rollouts.

    返回完成整批向量 rollout 后 PPO 的实际总步数。
    """
    rollout_size = int(n_steps) * int(num_envs)
    if requested_timesteps <= 0 or rollout_size <= 0:
        raise ValueError("timesteps, n_steps, and num_envs must be positive.")
    return int(math.ceil(requested_timesteps / rollout_size) * rollout_size)


def _make_progress_callback(
    *,
    total_timesteps: int,
    run_dir: Path,
    report_every_steps: int,
):
    """Create readable one-line progress and persistent status logging.

    创建易读的单行进度输出与持久化状态日志。
    """
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError("Residual PPO requires stable-baselines3.") from exc

    class ResidualProgressCallback(BaseCallback):
        """Write progress without dynamic terminal redraws.

        写入进度，同时避免终端动态重绘造成的文本拼接。
        """

        def __init__(self) -> None:
            super().__init__()
            self.started_at = 0.0
            self.last_report = 0
            self.last_console = 0
            self.report_every = max(1, int(report_every_steps))
            self.console_every = max(
                self.report_every,
                math.ceil(total_timesteps / 20),
            )
            self.stream = None
            self.writer = None

        def _on_training_start(self) -> None:
            self.started_at = perf_counter()
            self.stream = (run_dir / "training_metrics.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            )
            self.writer = csv.DictWriter(
                self.stream,
                fieldnames=(
                    "timesteps",
                    "target_timesteps",
                    "elapsed_seconds",
                    "decisions_per_second",
                    "episode_reward_mean",
                    "episode_length_mean",
                    "policy_loss",
                    "value_loss",
                    "entropy_loss",
                    "approx_kl",
                ),
            )
            self.writer.writeheader()
            self._record("running", console=False)

        def _on_step(self) -> bool:
            if self.num_timesteps - self.last_report >= self.report_every:
                self._record("running")
            return True

        def _on_training_end(self) -> None:
            self._record("completed", console=True)
            if self.stream is not None:
                self.stream.close()

        def _record(self, state: str, console: bool | None = None) -> None:
            elapsed = max(1e-9, perf_counter() - self.started_at)
            metrics = self.logger.name_to_value
            row = {
                "timesteps": self.num_timesteps,
                "target_timesteps": total_timesteps,
                "elapsed_seconds": elapsed,
                "decisions_per_second": self.num_timesteps / elapsed,
                "episode_reward_mean": _episode_mean(
                    self.model.ep_info_buffer,
                    "r",
                ),
                "episode_length_mean": _episode_mean(
                    self.model.ep_info_buffer,
                    "l",
                ),
                "policy_loss": _scalar(
                    metrics.get("train/policy_gradient_loss")
                ),
                "value_loss": _scalar(metrics.get("train/value_loss")),
                "entropy_loss": _scalar(metrics.get("train/entropy_loss")),
                "approx_kl": _scalar(metrics.get("train/approx_kl")),
            }
            if self.writer is not None and self.stream is not None:
                self.writer.writerow(row)
                self.stream.flush()
            _write_json(
                run_dir / "status.json",
                {
                    "state": state,
                    "timesteps": self.num_timesteps,
                    "target_timesteps": total_timesteps,
                    "progress_fraction": min(
                        1.0,
                        self.num_timesteps / total_timesteps,
                    ),
                    "latest_metrics": row,
                },
            )
            due = (
                self.num_timesteps - self.last_console >= self.console_every
            )
            should_print = due if console is None else console
            if should_print:
                print(
                    "Residual PPO | "
                    f"{100.0 * self.num_timesteps / total_timesteps:5.1f}% | "
                    f"{self.num_timesteps}/{total_timesteps} decisions | "
                    f"{row['decisions_per_second']:.1f} decision/s | "
                    f"mean_reward={_format(row['episode_reward_mean'])} | "
                    f"value_loss={_format(row['value_loss'])} | "
                    f"kl={_format(row['approx_kl'])}",
                    flush=True,
                )
                self.last_console = self.num_timesteps
            self.last_report = self.num_timesteps

    return ResidualProgressCallback()


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
    """Create fixed-seed validation and best-model selection.

    创建固定 seed 验证与最优模型选择。
    """
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError("Residual PPO requires stable-baselines3.") from exc

    class ResidualValidationCallback(BaseCallback):
        """Evaluate tail risk and save the best validation checkpoint.

        评估尾部风险并保存验证集最优检查点。
        """

        def __init__(self) -> None:
            super().__init__()
            self.next_evaluation = max(1, int(validation_every_steps))
            self.last_evaluated = -1
            self.best_loss = math.inf
            self.stream = None
            self.writer = None
            self.validation_dir = run_dir / "validation"
            self.validation_env = None

        def _on_training_start(self) -> None:
            self.validation_dir.mkdir(parents=True, exist_ok=False)
            self.validation_env = make_residual_native_env(
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
            self.writer = csv.DictWriter(
                self.stream,
                fieldnames=(
                    "timesteps",
                    "mean_total_cost_eur",
                    "mean_stored_t",
                    "mean_vented_t",
                    "worst_vented_t",
                    "cvar_vented_t",
                    "hard_violations",
                    "selection_loss",
                    "is_best",
                ),
            )
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
                tail_vent_penalty_eur_per_t=tail_vent_penalty_eur_per_t,
            )
            is_best = metrics["selection_loss"] < self.best_loss
            if is_best:
                self.best_loss = metrics["selection_loss"]
                self.model.save(run_dir / "ppo_residual_best_validation")
                _write_json(
                    self.validation_dir / "best.json",
                    {
                        "timesteps": self.num_timesteps,
                        **metrics,
                        "model_path": str(
                            run_dir / "ppo_residual_best_validation"
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
            print(
                "Residual validation | "
                f"step={self.num_timesteps} | "
                f"mean_cost=EUR {metrics['mean_total_cost_eur']:,.0f} | "
                f"worst_vent={metrics['worst_vented_t']:,.1f} t | "
                f"CVaR_vent={metrics['cvar_vented_t']:,.1f} t | "
                f"best={'yes' if is_best else 'no'}",
                flush=True,
            )
            self.last_evaluated = self.num_timesteps

    return ResidualValidationCallback()


def _scalar(value: Any) -> float | None:
    """Convert an optional logger value to float.

    将可选日志值转换为浮点数。
    """
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _episode_mean(episodes: Any, key: str) -> float | None:
    """Return one mean Monitor episode field.

    返回一个 Monitor 回合字段的均值。
    """
    values = [
        float(episode[key])
        for episode in episodes
        if episode.get(key) is not None
    ]
    return sum(values) / len(values) if values else None


def _format(value: float | None) -> str:
    """Format an optional training scalar.

    格式化可选训练标量。
    """
    return "n/a" if value is None else f"{value:.3f}"


def train_residual_ppo(
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
    gamma: float | None = None,
    n_steps: int = 256,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    entropy_coefficient: float = 0.01,
    device: str = "cpu",
    log_dir: Path | None = None,
    status_every_steps: int = 100,
    reward: HighLevelRewardConfig | None = None,
):
    """Train residual PPO and save both final and validation-best models.

    训练残差 PPO，并保存最终模型和验证集最优模型。
    """
    try:
        from stable_baselines3 import PPO
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
        raise ImportError("Residual PPO requires stable-baselines3.") from exc

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
        partial(make_residual_gym_env, **env_kwargs)
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

    probe = make_residual_native_env(
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        decision_interval_h=decision_interval_h,
        event_triggered=event_triggered,
        weather_mode=weather_mode,
        hard_scenario_probability=hard_scenario_probability,
        reward=reward_config,
    )
    effective_gamma = float(gamma) if gamma is not None else (
        1.0 if event_triggered else HOURLY_DISCOUNT ** decision_interval_h
    )
    planned_timesteps = _planned_timesteps(
        total_timesteps,
        n_steps,
        num_envs,
    )
    _write_json(
        run_dir / "config.json",
        {
            "interface_version": 1,
            "algorithm": "residual_ppo",
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
            "tail_vent_penalty_eur_per_t": tail_vent_penalty_eur_per_t,
            "gamma": effective_gamma,
            "n_steps_per_env": n_steps,
            "num_envs": num_envs,
            "vec_env_backend": vec_env_backend,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "entropy_coefficient": entropy_coefficient,
            "device": device,
            "action_count": probe.action_count,
            "action_labels": list(probe.codec.labels()),
            "injection_control": "highest_feasible_rate_by_rule_executor",
            "observation_size": probe.observation_size,
            "observation_features": list(
                residual_feature_names(probe.env)
            ),
            "high_level_reward": asdict(reward_config),
        },
    )

    progress_callback = _make_progress_callback(
        total_timesteps=planned_timesteps,
        run_dir=run_dir,
        report_every_steps=status_every_steps,
    )
    validation_callback = _make_validation_callback(
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
        tail_vent_penalty_eur_per_t=tail_vent_penalty_eur_per_t,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, planned_timesteps // (10 * num_envs)),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_residual",
    )
    model = PPO(
        "MlpPolicy",
        vector_env,
        seed=seed,
        gamma=effective_gamma,
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
            callback=CallbackList(
                [
                    progress_callback,
                    validation_callback,
                    checkpoint_callback,
                ]
            ),
            progress_bar=False,
        )
        model.save(run_dir / "ppo_residual_final")
        _write_json(
            run_dir / "training_complete.json",
            {
                "state": "completed",
                "requested_timesteps": total_timesteps,
                "planned_timesteps": planned_timesteps,
                "final_model_path": str(run_dir / "ppo_residual_final"),
                "best_validation_model_path": str(
                    run_dir / "ppo_residual_best_validation"
                ),
            },
        )
    finally:
        vector_env.close()
    return model, run_dir


def main() -> None:
    """Train residual PPO from the command line.

    从命令行训练残差 PPO。
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
    parser.add_argument("--weather-mode", choices=("window", "block"), default="window")
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
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--status-every-steps", type=int, default=100)
    parser.add_argument("--reward-scale", type=float, default=1e-6)
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    _model, run_dir = train_residual_ppo(
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
        tail_vent_penalty_eur_per_t=args.tail_vent_penalty_eur_per_t,
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
    print(f"Saved residual PPO under: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
