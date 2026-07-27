"""Train masked residual PPO with a staged difficulty curriculum.

使用分阶段难度课程训练掩码残差 PPO。
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

from .curriculum import (
    DEFAULT_CURRICULUM,
    CurriculumStage,
    curriculum_stage_index,
    make_curriculum_masked_residual_gym_env,
    parse_curriculum_specs,
    validate_curriculum,
)
from .evaluation import evaluate_seeds, validation_metrics
from .factory import make_masked_residual_native_env
from .train_masked_residual_ppo import DEFAULT_VALIDATION_SEEDS


DEFAULT_HARD_VALIDATION_SEEDS = tuple(range(3_000_001, 3_000_009))


def default_curriculum_run_dir(
    *,
    scenario: str,
    episode_hours: int,
    decision_interval_h: float,
    seed: int,
) -> Path:
    """Return a unique curriculum run directory under ``logs``.

    返回 ``logs`` 下唯一的课程训练目录。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = (
        f"{scenario}__{episode_hours}h__decision{decision_interval_h:g}h"
        f"__masked_residual_v2_curriculum__seed{seed}__{timestamp}"
    )
    return Path("logs") / "residual_rl_v2" / label


def _make_curriculum_callback(
    *,
    run_dir: Path,
    planned_timesteps: int,
    stages: tuple[CurriculumStage, ...],
):
    """Create a callback that updates every vector environment.

    创建用于更新全部向量环境的课程回调。
    """
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError("Training requires stable-baselines3.") from exc

    class CurriculumCallback(BaseCallback):
        """Change difficulty only at declared curriculum boundaries.

        仅在预先声明的课程边界上改变难度。
        """

        def __init__(self) -> None:
            super().__init__()
            self.active_stage = -1
            self.normal_episodes = 0
            self.hard_episodes = 0
            self.stream = None
            self.writer = None

        def _on_training_start(self) -> None:
            curriculum_dir = run_dir / "curriculum"
            curriculum_dir.mkdir(parents=True, exist_ok=False)
            self.stream = (curriculum_dir / "transitions.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            )
            self.writer = csv.DictWriter(
                self.stream,
                fieldnames=(
                    "timesteps",
                    "progress_fraction",
                    "stage_index",
                    "hard_probability",
                    "normal_episodes_completed",
                    "hard_episodes_completed",
                ),
            )
            self.writer.writeheader()
            self._activate_stage(0)

        def _on_step(self) -> bool:
            self._count_completed_episodes()
            fraction = min(
                1.0,
                self.num_timesteps / max(1, planned_timesteps),
            )
            index = curriculum_stage_index(stages, fraction)
            if index != self.active_stage:
                self._activate_stage(index)
            return True

        def _on_training_end(self) -> None:
            _write_json(
                run_dir / "curriculum" / "summary.json",
                {
                    "final_stage_index": self.active_stage,
                    "normal_episodes_completed": self.normal_episodes,
                    "hard_episodes_completed": self.hard_episodes,
                    "stages": [asdict(stage) for stage in stages],
                },
            )
            if self.stream is not None:
                self.stream.close()

        def _count_completed_episodes(self) -> None:
            dones = self.locals.get("dones", ())
            infos = self.locals.get("infos", ())
            for done, info in zip(dones, infos):
                if not bool(done):
                    continue
                difficulty = info.get("scenario_difficulty")
                if difficulty == "hard":
                    self.hard_episodes += 1
                elif difficulty == "normal":
                    self.normal_episodes += 1

        def _activate_stage(self, index: int) -> None:
            stage = stages[index]
            returned = self.training_env.env_method(
                "set_hard_probability",
                stage.hard_probability,
            )
            if any(
                abs(float(value) - stage.hard_probability) > 1e-12
                for value in returned
            ):
                raise RuntimeError("Not all environments changed difficulty.")
            self.active_stage = index
            row = {
                "timesteps": self.num_timesteps,
                "progress_fraction": (
                    self.num_timesteps / max(1, planned_timesteps)
                ),
                "stage_index": index,
                "hard_probability": stage.hard_probability,
                "normal_episodes_completed": self.normal_episodes,
                "hard_episodes_completed": self.hard_episodes,
            }
            if self.writer is not None and self.stream is not None:
                self.writer.writerow(row)
                self.stream.flush()
            _write_json(
                run_dir / "curriculum" / "status.json",
                row,
            )
            print(
                "Curriculum stage | "
                f"step={self.num_timesteps} | "
                f"stage={index + 1}/{len(stages)} | "
                f"hard_probability={stage.hard_probability:.0%}",
                flush=True,
            )

    return CurriculumCallback()


def _make_dual_validation_callback(
    *,
    run_dir: Path,
    scenario: str,
    episode_hours: int,
    forecast_context_hours: int,
    decision_interval_h: float,
    event_triggered: bool,
    weather_mode: str,
    reward: HighLevelRewardConfig,
    normal_validation_seeds: tuple[int, ...],
    hard_validation_seeds: tuple[int, ...],
    validation_every_steps: int,
    cvar_tail_fraction: float,
    tail_vent_penalty_eur_per_t: float,
):
    """Validate on separate normal and fully hard scenario sets.

    分别在普通场景集和全困难场景集上进行验证。
    """
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError("Training requires stable-baselines3.") from exc

    class DualValidationCallback(BaseCallback):
        """Select the best model using balanced normal/hard validation.

        使用均衡的普通/困难验证选择最优模型。
        """

        def __init__(self) -> None:
            super().__init__()
            self.next_evaluation = max(1, int(validation_every_steps))
            self.last_evaluated = -1
            self.best_loss = math.inf
            self.validation_dir = run_dir / "validation"
            self.normal_env = None
            self.hard_env = None
            self.stream = None
            self.writer = None

        def _on_training_start(self) -> None:
            self.validation_dir.mkdir(parents=True, exist_ok=False)
            common = {
                "scenario": scenario,
                "episode_hours": episode_hours,
                "forecast_context_hours": forecast_context_hours,
                "decision_interval_h": decision_interval_h,
                "event_triggered": event_triggered,
                "weather_mode": weather_mode,
                "reward": reward,
            }
            self.normal_env = make_masked_residual_native_env(
                hard_scenario_probability=0.0,
                **common,
            )
            self.hard_env = make_masked_residual_native_env(
                hard_scenario_probability=1.0,
                **common,
            )
            self.stream = (self.validation_dir / "metrics.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            )
            fields = (
                "timesteps",
                "normal_mean_cost_eur",
                "normal_mean_stored_t",
                "normal_mean_vented_t",
                "normal_worst_vented_t",
                "normal_cvar_vented_t",
                "hard_mean_cost_eur",
                "hard_mean_stored_t",
                "hard_mean_vented_t",
                "hard_worst_vented_t",
                "hard_cvar_vented_t",
                "hard_violations",
                "combined_selection_loss",
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
            assert self.normal_env is not None
            assert self.hard_env is not None
            normal_records = evaluate_seeds(
                self.model,
                self.normal_env,
                normal_validation_seeds,
            )
            hard_records = evaluate_seeds(
                self.model,
                self.hard_env,
                hard_validation_seeds,
            )
            metric_kwargs = {
                "cvar_tail_fraction": cvar_tail_fraction,
                "tail_vent_penalty_eur_per_t": (
                    tail_vent_penalty_eur_per_t
                ),
            }
            normal = validation_metrics(normal_records, **metric_kwargs)
            hard = validation_metrics(hard_records, **metric_kwargs)
            combined_loss = 0.5 * (
                normal["selection_loss"] + hard["selection_loss"]
            )
            is_best = combined_loss < self.best_loss
            if is_best:
                self.best_loss = combined_loss
                self.model.save(
                    run_dir / "maskable_residual_v2_best_validation"
                )
                _write_json(
                    self.validation_dir / "best.json",
                    {
                        "timesteps": self.num_timesteps,
                        "combined_selection_loss": combined_loss,
                        "normal": normal,
                        "hard": hard,
                        "model_path": str(
                            run_dir
                            / "maskable_residual_v2_best_validation"
                        ),
                    },
                )
            row = {
                "timesteps": self.num_timesteps,
                "normal_mean_cost_eur": normal["mean_total_cost_eur"],
                "normal_mean_stored_t": normal["mean_stored_t"],
                "normal_mean_vented_t": normal["mean_vented_t"],
                "normal_worst_vented_t": normal["worst_vented_t"],
                "normal_cvar_vented_t": normal["cvar_vented_t"],
                "hard_mean_cost_eur": hard["mean_total_cost_eur"],
                "hard_mean_stored_t": hard["mean_stored_t"],
                "hard_mean_vented_t": hard["mean_vented_t"],
                "hard_worst_vented_t": hard["worst_vented_t"],
                "hard_cvar_vented_t": hard["cvar_vented_t"],
                "hard_violations": (
                    normal["hard_violations"] + hard["hard_violations"]
                ),
                "combined_selection_loss": combined_loss,
                "is_best": int(is_best),
            }
            if self.writer is not None and self.stream is not None:
                self.writer.writerow(row)
                self.stream.flush()
            _write_json(
                self.validation_dir / f"step_{self.num_timesteps}.json",
                {
                    "timesteps": self.num_timesteps,
                    "combined_selection_loss": combined_loss,
                    "normal_validation_seeds": list(
                        normal_validation_seeds
                    ),
                    "hard_validation_seeds": list(hard_validation_seeds),
                    "normal": {
                        "metrics": normal,
                        "per_seed": normal_records,
                    },
                    "hard": {
                        "metrics": hard,
                        "per_seed": hard_records,
                    },
                },
            )
            print(
                "Curriculum validation | "
                f"step={self.num_timesteps} | "
                f"normal_CVaR={normal['cvar_vented_t']:,.1f} t | "
                f"hard_CVaR={hard['cvar_vented_t']:,.1f} t | "
                f"best={'yes' if is_best else 'no'}",
                flush=True,
            )
            self.last_evaluated = self.num_timesteps

    return DualValidationCallback()


def train_curriculum_masked_residual_ppo(
    *,
    total_timesteps: int = 40_000,
    seed: int = 0,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    decision_interval_h: float = 24.0,
    event_triggered: bool = True,
    weather_mode: str = "window",
    curriculum: tuple[CurriculumStage, ...] = DEFAULT_CURRICULUM,
    num_envs: int = 4,
    vec_env_backend: str = "subproc",
    training_seed_min: int = 100_000,
    training_seed_max: int = 999_999,
    normal_validation_seeds: tuple[int, ...] = DEFAULT_VALIDATION_SEEDS,
    hard_validation_seeds: tuple[int, ...] = (
        DEFAULT_HARD_VALIDATION_SEEDS
    ),
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
    """Train MaskablePPO using a real staged curriculum.

    使用真正的分阶段课程训练 MaskablePPO。
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

    stages = validate_curriculum(tuple(curriculum))
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if vec_env_backend not in {"dummy", "subproc"}:
        raise ValueError("vec_env_backend must be 'dummy' or 'subproc'.")
    normal_validation_seeds = tuple(
        int(value) for value in normal_validation_seeds
    )
    hard_validation_seeds = tuple(
        int(value) for value in hard_validation_seeds
    )
    if not normal_validation_seeds or not hard_validation_seeds:
        raise ValueError("Both validation seed sets must be non-empty.")
    validation_values = (
        normal_validation_seeds + hard_validation_seeds
    )
    overlap = [
        value
        for value in validation_values
        if training_seed_min <= value <= training_seed_max
    ]
    if overlap:
        raise ValueError(
            f"Training and validation seeds overlap: {overlap}."
        )
    if set(normal_validation_seeds) & set(hard_validation_seeds):
        raise ValueError("Normal and hard validation seeds must differ.")

    run_dir = log_dir or default_curriculum_run_dir(
        scenario=scenario,
        episode_hours=episode_hours,
        decision_interval_h=decision_interval_h,
        seed=seed,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    reward_config = reward or HighLevelRewardConfig()
    initial_probability = stages[0].hard_probability
    env_kwargs = {
        "scenario": scenario,
        "episode_hours": episode_hours,
        "forecast_context_hours": forecast_context_hours,
        "decision_interval_h": decision_interval_h,
        "event_triggered": event_triggered,
        "weather_mode": weather_mode,
        "initial_hard_probability": initial_probability,
        "reward": reward_config,
        "episode_seed_min": training_seed_min,
        "episode_seed_max": training_seed_max,
    }
    env_fns = [
        partial(make_curriculum_masked_residual_gym_env, **env_kwargs)
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
        hard_scenario_probability=initial_probability,
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
            "training_regime": "curriculum",
            "scenario": scenario,
            "episode_hours": episode_hours,
            "forecast_context_hours": forecast_context_hours,
            "decision_interval_h": decision_interval_h,
            "event_triggered": event_triggered,
            "weather_mode": weather_mode,
            "hard_scenario_probability": (
                stages[-1].hard_probability
            ),
            "curriculum_stages": [asdict(stage) for stage in stages],
            "requested_timesteps": total_timesteps,
            "planned_timesteps": planned_timesteps,
            "seed": seed,
            "training_seed_range": [
                training_seed_min,
                training_seed_max,
            ],
            "normal_validation_seeds": list(normal_validation_seeds),
            "hard_validation_seeds": list(hard_validation_seeds),
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
            _make_curriculum_callback(
                run_dir=run_dir,
                planned_timesteps=planned_timesteps,
                stages=stages,
            ),
            _make_progress_callback(
                total_timesteps=planned_timesteps,
                run_dir=run_dir,
                report_every_steps=status_every_steps,
            ),
            _make_dual_validation_callback(
                run_dir=run_dir,
                scenario=scenario,
                episode_hours=episode_hours,
                forecast_context_hours=forecast_context_hours,
                decision_interval_h=decision_interval_h,
                event_triggered=event_triggered,
                weather_mode=weather_mode,
                reward=reward_config,
                normal_validation_seeds=normal_validation_seeds,
                hard_validation_seeds=hard_validation_seeds,
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
                name_prefix="maskable_residual_v2_curriculum",
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
                "training_regime": "curriculum",
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
    """Train curriculum MaskablePPO from a terminal.

    从终端训练课程版 MaskablePPO。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=40_000)
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
    parser.add_argument(
        "--curriculum-stages",
        nargs="+",
        default=[
            "0.00:0.00",
            "0.20:0.15",
            "0.40:0.30",
            "0.70:0.50",
        ],
        metavar="FRACTION:HARD_PROBABILITY",
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--vec-env",
        choices=("dummy", "subproc"),
        default="subproc",
    )
    parser.add_argument("--training-seed-min", type=int, default=100_000)
    parser.add_argument("--training-seed-max", type=int, default=999_999)
    parser.add_argument(
        "--normal-validation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_VALIDATION_SEEDS),
    )
    parser.add_argument(
        "--hard-validation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_HARD_VALIDATION_SEEDS),
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

    stages = parse_curriculum_specs(args.curriculum_stages)
    _model, run_dir = train_curriculum_masked_residual_ppo(
        total_timesteps=args.timesteps,
        seed=args.seed,
        scenario=args.scenario,
        episode_hours=args.episode_hours,
        forecast_context_hours=args.forecast_context_hours,
        decision_interval_h=args.decision_interval_h,
        event_triggered=args.event_triggered,
        weather_mode=args.weather_mode,
        curriculum=stages,
        num_envs=args.num_envs,
        vec_env_backend=args.vec_env,
        training_seed_min=args.training_seed_min,
        training_seed_max=args.training_seed_max,
        normal_validation_seeds=tuple(
            args.normal_validation_seeds
        ),
        hard_validation_seeds=tuple(args.hard_validation_seeds),
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
    print(f"Saved curriculum masked residual PPO v2 under: {run_dir}")


if __name__ == "__main__":
    main()

