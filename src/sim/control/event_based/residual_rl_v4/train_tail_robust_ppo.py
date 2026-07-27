"""Train one tail-robust residual PPO policy with failure replay.

使用失败场景重放训练单个面向尾部风险的 residual PPO 策略。
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
from sim.control.event_based.residual_rl_v2.curriculum import (
    CurriculumStage,
    parse_curriculum_specs,
    validate_curriculum,
)
from sim.control.event_based.residual_rl_v2.evaluation import (
    evaluate_seeds,
    validation_metrics,
)
from sim.control.event_based.residual_rl_v2.train_curriculum_masked_residual_ppo import (
    _make_curriculum_callback,
)
from sim.control.event_based.residual_rl_v3.risk_gate import (
    AdaptiveRiskGateConfig,
)
from sim.control.event_based.rl.reward import HighLevelRewardConfig
from sim.control.event_based.rl.observation_encoder import (
    FORECAST_WINDOWS_H,
    FUTURE_SUMMARY_REPRESENTATION_ID,
)

from .factory import (
    make_tail_replay_gym_env,
    make_tail_robust_native_env,
)
from .model_selection import (
    ReferenceValidationMetrics,
    TailRiskSelectionConfig,
    discover_reference_v3_run,
    load_reference_validation,
    score_validation_checkpoint,
)


DEFAULT_TAIL_CURRICULUM = (
    CurriculumStage(0.00, 0.10),
    CurriculumStage(0.20, 0.25),
    CurriculumStage(0.40, 0.40),
    CurriculumStage(0.60, 0.55),
    CurriculumStage(0.80, 0.40),
)
DEFAULT_NORMAL_VALIDATION_SEEDS = tuple(
    range(2_000_001, 2_000_021)
)
DEFAULT_HARD_VALIDATION_SEEDS = tuple(
    range(3_000_001, 3_000_021)
)


def default_run_dir(
    *,
    scenario: str,
    episode_hours: int,
    decision_interval_h: float,
    seed: int,
) -> Path:
    """Return a unique v4 training directory under ``logs``.

    返回 ``logs`` 下唯一的 v4 训练目录。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = (
        f"{scenario}__{episode_hours}h__decision{decision_interval_h:g}h"
        f"__residual_v4_tail_replay__seed{seed}__{timestamp}"
    )
    return Path("logs") / "residual_rl_v4" / label


def _make_tail_validation_callback(
    *,
    run_dir: Path,
    scenario: str,
    episode_hours: int,
    forecast_context_hours: int,
    future_summary_windows_h: tuple[int, ...],
    decision_interval_h: float,
    event_triggered: bool,
    weather_mode: str,
    scenario_protocol: str,
    override_windows_h: tuple[tuple[float, float], ...],
    reward: HighLevelRewardConfig,
    gate: AdaptiveRiskGateConfig,
    gate_mode: str,
    outside_risk_intervention_penalty: float,
    normal_validation_seeds: tuple[int, ...],
    hard_validation_seeds: tuple[int, ...],
    validation_every_steps: int,
    cvar_tail_fraction: float,
    selection: TailRiskSelectionConfig,
    reference: ReferenceValidationMetrics | None,
):
    """Create constrained normal/hard tail-risk validation.

    创建带约束的普通/困难尾部风险验证。
    """
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError("Training requires stable-baselines3.") from exc

    class TailValidationCallback(BaseCallback):
        """Save checkpoints using constraints before the scalar loss.

        先按约束、再按标量损失保存 checkpoint。
        """

        def __init__(self) -> None:
            super().__init__()
            self.next_evaluation = max(1, int(validation_every_steps))
            self.last_evaluated = -1
            self.best_rank = (math.inf, math.inf, math.inf)
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
                "future_summary_windows_h": future_summary_windows_h,
                "decision_interval_h": decision_interval_h,
                "event_triggered": event_triggered,
                "weather_mode": weather_mode,
                "scenario_protocol": scenario_protocol,
                "override_windows_h": override_windows_h,
                "reward": reward,
                "gate": gate,
                "gate_mode": gate_mode,
                "outside_risk_intervention_penalty": (
                    outside_risk_intervention_penalty
                ),
            }
            self.normal_env = make_tail_robust_native_env(
                hard_scenario_probability=0.0,
                **common,
            )
            self.hard_env = make_tail_robust_native_env(
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
                "normal_cvar_vented_t",
                "hard_mean_cost_eur",
                "hard_mean_stored_t",
                "hard_mean_vented_t",
                "hard_cvar_vented_t",
                "hard_worst_vented_t",
                "hard_violations",
                "failed_constraints",
                "qualified",
                "robust_selection_loss",
                "replay_buffer_size",
                "replay_episode_rate",
                "is_best",
            )
            self.writer = csv.DictWriter(
                self.stream,
                fieldnames=fields,
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
            replay = self._replay_diagnostics()
            _write_json(
                run_dir / "replay_summary.json",
                {
                    "timesteps": self.num_timesteps,
                    "workers": replay["workers"],
                    "mean_buffer_size": replay["mean_buffer_size"],
                    "mean_replay_episode_rate": (
                        replay["mean_replay_episode_rate"]
                    ),
                },
            )
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
                "tail_vent_penalty_eur_per_t": 0.0,
            }
            normal = validation_metrics(
                normal_records,
                **metric_kwargs,
            )
            hard = validation_metrics(
                hard_records,
                **metric_kwargs,
            )
            score = score_validation_checkpoint(
                normal,
                hard,
                config=selection,
                reference=reference,
            )
            rank = tuple(score.pop("rank"))
            is_best = rank < self.best_rank
            if is_best:
                self.best_rank = rank
                self.model.save(
                    run_dir / "maskable_residual_v4_best_validation"
                )
                _write_json(
                    self.validation_dir / "best.json",
                    {
                        "timesteps": self.num_timesteps,
                        **score,
                        "normal": normal,
                        "hard": hard,
                        "reference": (
                            asdict(reference)
                            if reference is not None
                            else None
                        ),
                        "model_path": str(
                            run_dir
                            / "maskable_residual_v4_best_validation"
                        ),
                    },
                )

            replay = self._replay_diagnostics()
            row = {
                "timesteps": self.num_timesteps,
                "normal_mean_cost_eur": normal[
                    "mean_total_cost_eur"
                ],
                "normal_mean_stored_t": normal["mean_stored_t"],
                "normal_mean_vented_t": normal["mean_vented_t"],
                "normal_cvar_vented_t": normal["cvar_vented_t"],
                "hard_mean_cost_eur": hard["mean_total_cost_eur"],
                "hard_mean_stored_t": hard["mean_stored_t"],
                "hard_mean_vented_t": hard["mean_vented_t"],
                "hard_cvar_vented_t": hard["cvar_vented_t"],
                "hard_worst_vented_t": hard["worst_vented_t"],
                "hard_violations": (
                    normal["hard_violations"]
                    + hard["hard_violations"]
                ),
                "failed_constraints": score["failed_constraints"],
                "qualified": int(score["qualified"]),
                "robust_selection_loss": score[
                    "robust_selection_loss"
                ],
                "replay_buffer_size": replay["mean_buffer_size"],
                "replay_episode_rate": replay[
                    "mean_replay_episode_rate"
                ],
                "is_best": int(is_best),
            }
            if self.writer is not None and self.stream is not None:
                self.writer.writerow(row)
                self.stream.flush()
            _write_json(
                self.validation_dir / f"step_{self.num_timesteps}.json",
                {
                    "timesteps": self.num_timesteps,
                    **score,
                    "normal_validation_seeds": list(
                        normal_validation_seeds
                    ),
                    "hard_validation_seeds": list(
                        hard_validation_seeds
                    ),
                    "normal": {
                        "metrics": normal,
                        "per_seed": normal_records,
                    },
                    "hard": {
                        "metrics": hard,
                        "per_seed": hard_records,
                    },
                    "replay": replay,
                },
            )
            print(
                "Residual v4 validation | "
                f"step={self.num_timesteps} | "
                f"normal_vent={normal['mean_vented_t']:,.1f} t | "
                f"hard_CVaR={hard['cvar_vented_t']:,.1f} t | "
                f"hard_worst={hard['worst_vented_t']:,.1f} t | "
                f"qualified={'yes' if score['qualified'] else 'no'} | "
                f"best={'yes' if is_best else 'no'}",
                flush=True,
            )
            self.last_evaluated = self.num_timesteps

        def _replay_diagnostics(self) -> dict[str, Any]:
            """Collect replay snapshots from all vector workers.

            收集全部向量 worker 的回放快照。
            """
            workers = self.training_env.env_method(
                "get_replay_snapshot"
            )
            return {
                "workers": workers,
                "mean_buffer_size": sum(
                    float(item["buffer_size"]) for item in workers
                )
                / len(workers),
                "mean_replay_episode_rate": sum(
                    float(item["replay_episode_rate"])
                    for item in workers
                )
                / len(workers),
            }

    return TailValidationCallback()


def train_residual_v4(
    *,
    total_timesteps: int = 100_000,
    seed: int = 0,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    future_summary_windows_h: tuple[int, ...] = FORECAST_WINDOWS_H,
    decision_interval_h: float = 24.0,
    event_triggered: bool = True,
    weather_mode: str = "window",
    scenario_protocol: str = "v4_mixed_window",
    override_windows_h: tuple[tuple[float, float], ...] = (),
    curriculum: tuple[
        CurriculumStage, ...
    ] = DEFAULT_TAIL_CURRICULUM,
    replay_probability: float = 0.30,
    replay_capacity: int = 20,
    minimum_replay_pool: int = 4,
    gate: AdaptiveRiskGateConfig | None = None,
    gate_mode: str = "soft",
    outside_risk_intervention_penalty: float = 0.02,
    num_envs: int = 4,
    vec_env_backend: str = "subproc",
    training_seed_min: int = 100_000,
    training_seed_max: int = 999_999,
    normal_validation_seeds: tuple[
        int, ...
    ] = DEFAULT_NORMAL_VALIDATION_SEEDS,
    hard_validation_seeds: tuple[
        int, ...
    ] = DEFAULT_HARD_VALIDATION_SEEDS,
    validation_every_steps: int = 10_000,
    cvar_tail_fraction: float = 0.20,
    selection: TailRiskSelectionConfig | None = None,
    reference_run_dir: Path | None = None,
    use_reference_constraints: bool = True,
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
    """Train a single v4 policy with curriculum and failure replay.

    使用课程学习和失败重放训练单个 v4 策略。
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
    gate_config = gate or AdaptiveRiskGateConfig(
        hours_to_overflow_threshold_h=144.0,
        fill_ratio_threshold=0.70,
    )
    selection_config = selection or TailRiskSelectionConfig()
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
    _validate_seed_partitions(
        training_seed_min=training_seed_min,
        training_seed_max=training_seed_max,
        normal_validation_seeds=normal_validation_seeds,
        hard_validation_seeds=hard_validation_seeds,
    )

    reference = _resolve_reference(
        enabled=use_reference_constraints,
        requested_run=reference_run_dir,
        scenario=scenario,
        normal_validation_seeds=normal_validation_seeds,
        hard_validation_seeds=hard_validation_seeds,
    )
    run_dir = log_dir or default_run_dir(
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
        "future_summary_windows_h": future_summary_windows_h,
        "decision_interval_h": decision_interval_h,
        "event_triggered": event_triggered,
        "weather_mode": weather_mode,
        "scenario_protocol": scenario_protocol,
        "override_windows_h": override_windows_h,
        "initial_hard_probability": initial_probability,
        "reward": reward_config,
        "gate": gate_config,
        "gate_mode": gate_mode,
        "outside_risk_intervention_penalty": (
            outside_risk_intervention_penalty
        ),
        "episode_seed_min": training_seed_min,
        "episode_seed_max": training_seed_max,
        "replay_probability": replay_probability,
        "replay_capacity": replay_capacity,
        "minimum_replay_pool": minimum_replay_pool,
    }
    env_fns = [
        partial(make_tail_replay_gym_env, **env_kwargs)
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

    probe = make_tail_robust_native_env(
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        future_summary_windows_h=future_summary_windows_h,
        decision_interval_h=decision_interval_h,
        event_triggered=event_triggered,
        weather_mode=weather_mode,
        scenario_protocol=scenario_protocol,
        hard_scenario_probability=initial_probability,
        reward=reward_config,
        gate=gate_config,
        gate_mode=gate_mode,
        outside_risk_intervention_penalty=(
            outside_risk_intervention_penalty
        ),
        override_windows_h=override_windows_h,
    )
    planned_timesteps = _planned_timesteps(
        total_timesteps,
        n_steps,
        num_envs,
    )
    _write_training_config(
        run_dir=run_dir,
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        future_summary_windows_h=future_summary_windows_h,
        decision_interval_h=decision_interval_h,
        event_triggered=event_triggered,
        weather_mode=weather_mode,
        scenario_protocol=scenario_protocol,
        override_windows_h=override_windows_h,
        stages=stages,
        replay_probability=replay_probability,
        replay_capacity=replay_capacity,
        minimum_replay_pool=minimum_replay_pool,
        gate=gate_config,
        gate_mode=gate_mode,
        outside_risk_intervention_penalty=(
            outside_risk_intervention_penalty
        ),
        total_timesteps=total_timesteps,
        planned_timesteps=planned_timesteps,
        seed=seed,
        training_seed_min=training_seed_min,
        training_seed_max=training_seed_max,
        normal_validation_seeds=normal_validation_seeds,
        hard_validation_seeds=hard_validation_seeds,
        validation_every_steps=validation_every_steps,
        cvar_tail_fraction=cvar_tail_fraction,
        selection=selection_config,
        reference=reference,
        gamma=gamma,
        n_steps=n_steps,
        num_envs=num_envs,
        vec_env_backend=vec_env_backend,
        batch_size=batch_size,
        learning_rate=learning_rate,
        entropy_coefficient=entropy_coefficient,
        device=device,
        probe=probe,
        reward=reward_config,
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
            _make_tail_validation_callback(
                run_dir=run_dir,
                scenario=scenario,
                episode_hours=episode_hours,
                forecast_context_hours=forecast_context_hours,
                future_summary_windows_h=future_summary_windows_h,
                decision_interval_h=decision_interval_h,
                event_triggered=event_triggered,
                weather_mode=weather_mode,
                scenario_protocol=scenario_protocol,
                override_windows_h=override_windows_h,
                reward=reward_config,
                gate=gate_config,
                gate_mode=gate_mode,
                outside_risk_intervention_penalty=(
                    outside_risk_intervention_penalty
                ),
                normal_validation_seeds=normal_validation_seeds,
                hard_validation_seeds=hard_validation_seeds,
                validation_every_steps=validation_every_steps,
                cvar_tail_fraction=cvar_tail_fraction,
                selection=selection_config,
                reference=reference,
            ),
            CheckpointCallback(
                save_freq=max(
                    1,
                    planned_timesteps // (10 * num_envs),
                ),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="maskable_residual_v4",
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
        model.save(run_dir / "maskable_residual_v4_final")
        best = json.loads(
            (run_dir / "validation" / "best.json").read_text(
                encoding="utf-8"
            )
        )
        _write_json(
            run_dir / "training_complete.json",
            {
                "state": "completed",
                "training_regime": "tail_curriculum_failure_replay",
                "requested_timesteps": total_timesteps,
                "planned_timesteps": planned_timesteps,
                "best_validation_qualified": best["qualified"],
                "final_model_path": str(
                    run_dir / "maskable_residual_v4_final"
                ),
                "best_validation_model_path": str(
                    run_dir
                    / "maskable_residual_v4_best_validation"
                ),
            },
        )
    finally:
        vector_env.close()
    return model, run_dir


def _validate_seed_partitions(
    *,
    training_seed_min: int,
    training_seed_max: int,
    normal_validation_seeds: tuple[int, ...],
    hard_validation_seeds: tuple[int, ...],
) -> None:
    """Prevent training, validation, and difficulty-set leakage.

    防止训练集、验证集及难度集合之间发生 seed 泄漏。
    """
    if training_seed_min > training_seed_max:
        raise ValueError(
            "training_seed_min must not exceed training_seed_max."
        )
    if not normal_validation_seeds or not hard_validation_seeds:
        raise ValueError("Both validation seed sets must be non-empty.")
    validation = normal_validation_seeds + hard_validation_seeds
    overlap = [
        seed
        for seed in validation
        if training_seed_min <= seed <= training_seed_max
    ]
    if overlap:
        raise ValueError(
            f"Training and validation seeds overlap: {overlap}."
        )
    if set(normal_validation_seeds) & set(hard_validation_seeds):
        raise ValueError("Normal and hard validation seeds must differ.")


def _resolve_reference(
    *,
    enabled: bool,
    requested_run: Path | None,
    scenario: str,
    normal_validation_seeds: tuple[int, ...],
    hard_validation_seeds: tuple[int, ...],
) -> ReferenceValidationMetrics | None:
    """Resolve and load the compatible v3 comparison checkpoint.

    解析并加载兼容的 v3 对照 checkpoint。
    """
    if not enabled:
        return None
    run_dir = requested_run or discover_reference_v3_run(
        scenario=scenario,
        normal_validation_seeds=normal_validation_seeds,
        hard_validation_seeds=hard_validation_seeds,
    )
    if run_dir is None:
        raise FileNotFoundError(
            "No compatible v3 seed0 reference run was found. "
            "Pass --reference-run-dir or use "
            "--no-reference-constraints for a smoke test."
        )
    return load_reference_validation(
        run_dir,
        normal_validation_seeds=normal_validation_seeds,
        hard_validation_seeds=hard_validation_seeds,
    )


def _write_training_config(
    *,
    run_dir: Path,
    scenario: str,
    episode_hours: int,
    forecast_context_hours: int,
    future_summary_windows_h: tuple[int, ...],
    decision_interval_h: float,
    event_triggered: bool,
    weather_mode: str,
    scenario_protocol: str,
    override_windows_h: tuple[tuple[float, float], ...],
    stages: tuple[CurriculumStage, ...],
    replay_probability: float,
    replay_capacity: int,
    minimum_replay_pool: int,
    gate: AdaptiveRiskGateConfig,
    gate_mode: str,
    outside_risk_intervention_penalty: float,
    total_timesteps: int,
    planned_timesteps: int,
    seed: int,
    training_seed_min: int,
    training_seed_max: int,
    normal_validation_seeds: tuple[int, ...],
    hard_validation_seeds: tuple[int, ...],
    validation_every_steps: int,
    cvar_tail_fraction: float,
    selection: TailRiskSelectionConfig,
    reference: ReferenceValidationMetrics | None,
    gamma: float,
    n_steps: int,
    num_envs: int,
    vec_env_backend: str,
    batch_size: int,
    learning_rate: float,
    entropy_coefficient: float,
    device: str,
    probe,
    reward: HighLevelRewardConfig,
) -> None:
    """Persist all settings required for reproducible training.

    保存可复现训练所需的全部设置。
    """
    _write_json(
        run_dir / "config.json",
        {
            "interface_version": 4,
            "algorithm": "maskable_residual_ppo_v4",
            "training_regime": "tail_curriculum_failure_replay",
            "scenario": scenario,
            "episode_hours": episode_hours,
            "forecast_context_hours": forecast_context_hours,
            "future_summary_representation_id": (
                FUTURE_SUMMARY_REPRESENTATION_ID
            ),
            "future_summary_windows_h": list(
                future_summary_windows_h
            ),
            "decision_interval_h": decision_interval_h,
            "event_triggered": event_triggered,
            "weather_mode": weather_mode,
            "scenario_protocol": scenario_protocol,
            "override_windows_h": [
                [float(start), float(end)]
                for start, end in override_windows_h
            ],
            "hard_scenario_probability": (
                stages[-1].hard_probability
            ),
            "curriculum_stages": [asdict(stage) for stage in stages],
            "failure_replay": {
                "probability": replay_probability,
                "capacity_per_worker": replay_capacity,
                "minimum_pool_per_worker": minimum_replay_pool,
                "source": "training_seeds_only",
            },
            "risk_gate": asdict(gate),
            "risk_gate_mode": gate_mode,
            "outside_risk_intervention_penalty": (
                outside_risk_intervention_penalty
            ),
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
            "tail_risk_selection": asdict(selection),
            "reference_validation": (
                asdict(reference) if reference is not None else None
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
            "reward_mode": (
                "persistent_rule_delta_plus_soft_risk_penalty"
            ),
            "observation_size": probe.observation_size,
            "observation_features": list(
                residual_feature_names(
                    probe.env,
                    future_summary_windows_h,
                )
            ),
            "high_level_reward": asdict(reward),
        },
    )


def _parse_override_window(value: str) -> tuple[float, float]:
    try:
        start_text, end_text = value.split("-", 1)
        start, end = float(start_text), float(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "override windows must use START-END"
        ) from exc
    if start < 0.0 or end < start:
        raise argparse.ArgumentTypeError(
            "override window must satisfy 0 <= START <= END"
        )
    return start, end


def main() -> None:
    """Run one v4 policy-seed training experiment.

    运行一次 v4 策略 seed 训练实验。
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
    parser.add_argument(
        "--future-summary-windows-h",
        type=int,
        nargs="*",
        default=list(FORECAST_WINDOWS_H),
        help=(
            "Increasing future-summary horizons in hours; pass the "
            "option with no values for a state-only ablation."
        ),
    )
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
        "--scenario-protocol",
        choices=("v4_mixed_window", "unified_window_v1"),
        default="v4_mixed_window",
    )
    parser.add_argument(
        "--override-windows-h",
        type=_parse_override_window,
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--curriculum-stages",
        nargs="+",
        default=[
            "0.00:0.10",
            "0.20:0.25",
            "0.40:0.40",
            "0.60:0.55",
            "0.80:0.40",
        ],
    )
    parser.add_argument("--replay-probability", type=float, default=0.30)
    parser.add_argument("--replay-capacity", type=int, default=20)
    parser.add_argument("--minimum-replay-pool", type=int, default=4)
    parser.add_argument(
        "--gate-mode",
        choices=("off", "soft", "hard"),
        default="soft",
    )
    parser.add_argument("--risk-hours-threshold-h", type=float, default=144)
    parser.add_argument("--risk-fill-threshold", type=float, default=0.70)
    parser.add_argument(
        "--outside-risk-penalty",
        type=float,
        default=0.02,
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
        default=list(DEFAULT_NORMAL_VALIDATION_SEEDS),
    )
    parser.add_argument(
        "--hard-validation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_HARD_VALIDATION_SEEDS),
    )
    parser.add_argument("--validation-every-steps", type=int, default=10_000)
    parser.add_argument("--cvar-tail-fraction", type=float, default=0.20)
    parser.add_argument(
        "--normal-cvar-weight",
        type=float,
        default=37.5,
    )
    parser.add_argument(
        "--hard-cvar-weight",
        type=float,
        default=112.5,
    )
    parser.add_argument(
        "--hard-worst-weight",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--normal-vent-degradation-limit",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--hard-worst-improvement-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--reference-run-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--reference-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
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

    gate = AdaptiveRiskGateConfig(
        hours_to_overflow_threshold_h=(
            args.risk_hours_threshold_h
        ),
        fill_ratio_threshold=args.risk_fill_threshold,
    )
    selection = TailRiskSelectionConfig(
        normal_cvar_weight_eur_per_t=args.normal_cvar_weight,
        hard_cvar_weight_eur_per_t=args.hard_cvar_weight,
        hard_worst_weight_eur_per_t=args.hard_worst_weight,
        normal_vent_degradation_limit=(
            args.normal_vent_degradation_limit
        ),
        hard_worst_improvement_fraction=(
            args.hard_worst_improvement_fraction
        ),
    )
    _model, run_dir = train_residual_v4(
        total_timesteps=args.timesteps,
        seed=args.seed,
        scenario=args.scenario,
        episode_hours=args.episode_hours,
        forecast_context_hours=args.forecast_context_hours,
        future_summary_windows_h=tuple(
            args.future_summary_windows_h
        ),
        decision_interval_h=args.decision_interval_h,
        event_triggered=args.event_triggered,
        weather_mode=args.weather_mode,
        scenario_protocol=args.scenario_protocol,
        override_windows_h=tuple(args.override_windows_h),
        curriculum=parse_curriculum_specs(args.curriculum_stages),
        replay_probability=args.replay_probability,
        replay_capacity=args.replay_capacity,
        minimum_replay_pool=args.minimum_replay_pool,
        gate=gate,
        gate_mode=args.gate_mode,
        outside_risk_intervention_penalty=(
            args.outside_risk_penalty
        ),
        num_envs=args.num_envs,
        vec_env_backend=args.vec_env,
        training_seed_min=args.training_seed_min,
        training_seed_max=args.training_seed_max,
        normal_validation_seeds=tuple(
            args.normal_validation_seeds
        ),
        hard_validation_seeds=tuple(
            args.hard_validation_seeds
        ),
        validation_every_steps=args.validation_every_steps,
        cvar_tail_fraction=args.cvar_tail_fraction,
        selection=selection,
        reference_run_dir=args.reference_run_dir,
        use_reference_constraints=args.reference_constraints,
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        entropy_coefficient=args.ent_coef,
        device=args.device,
        log_dir=args.log_dir,
        status_every_steps=args.status_every_steps,
        reward=HighLevelRewardConfig(
            reward_scale=args.reward_scale
        ),
    )
    print(f"Saved residual PPO v4 under: {run_dir}")


if __name__ == "__main__":
    main()
