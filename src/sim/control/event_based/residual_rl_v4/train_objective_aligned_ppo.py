"""Train the formal E1 Event-Residual PPO without changing legacy v4."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from sim.control.event_based.residual_rl.observation import (
    residual_feature_names,
)
from sim.control.event_based.residual_rl.train_residual_ppo import (
    DEFAULT_VALIDATION_SEEDS,
    _planned_timesteps,
    _write_json,
)
from sim.control.event_based.residual_rl_v2.evaluation import (
    evaluate_seeds,
    validation_metrics,
)
from sim.control.event_based.residual_rl_v2.gym_env import (
    MaskedResidualGymEnv,
)
from sim.control.event_based.residual_rl_v3.risk_gate import (
    AdaptiveRiskGateConfig,
)
from sim.control.event_based.rl.observation_encoder import (
    FORECAST_WINDOWS_H,
    FUTURE_SUMMARY_REPRESENTATION_ID,
)
from sim.control.event_based.rl.reward import HighLevelRewardConfig
from sim.simulator import SimulatorStepCounter

from .factory import make_tail_robust_native_env


def default_run_dir(
    *,
    scenario: str,
    episode_hours: int,
    decision_interval_h: float,
    seed: int,
) -> Path:
    """Return a unique directory for a formal E1 training run."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = (
        f"{scenario}__{episode_hours}h__decision{decision_interval_h:g}h"
        f"__event_residual_e1__seed{seed}__{timestamp}"
    )
    return Path("logs") / "event_residual_e1" / label


def _validate_seed_partition(
    *,
    training_seed_min: int,
    training_seed_max: int,
    validation_seeds: tuple[int, ...],
) -> None:
    if training_seed_min > training_seed_max:
        raise ValueError(
            "training_seed_min must not exceed training_seed_max."
        )
    if not validation_seeds:
        raise ValueError("At least one validation seed is required.")
    overlap = [
        seed
        for seed in validation_seeds
        if training_seed_min <= seed <= training_seed_max
    ]
    if overlap:
        raise ValueError(
            f"Training and validation seeds overlap: {overlap}."
        )


def _make_training_callback(
    *,
    run_dir: Path,
    simulator_step_counter: SimulatorStepCounter,
    max_simulator_hour_steps: int | None,
    paired_step_hours: float,
    validation_env,
    validation_seeds: tuple[int, ...],
    validation_every_simulator_hour_steps: int | None,
):
    """Stop on the paired budget and select checkpoints by validation cost."""

    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError(
            "Training requires stable-baselines3."
        ) from exc

    class ObjectiveAlignedCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.best_cost = float("inf")
            self.last_validation_usage = -1.0
            self.next_validation_usage = (
                float(validation_every_simulator_hour_steps)
                if validation_every_simulator_hour_steps is not None
                else float("inf")
            )

        def _budget_exhausted(self) -> bool:
            if max_simulator_hour_steps is None:
                return False
            remaining = (
                float(max_simulator_hour_steps)
                - simulator_step_counter.snapshot().hour_steps
            )
            return remaining < paired_step_hours - 1e-9

        def _on_training_start(self) -> None:
            (run_dir / "validation").mkdir(
                parents=True,
                exist_ok=False,
            )
            self._write_status("running")

        def _on_step(self) -> bool:
            usage = simulator_step_counter.snapshot().hour_steps
            if usage + 1e-9 >= self.next_validation_usage:
                self._evaluate(usage)
                interval = float(
                    validation_every_simulator_hour_steps
                )
                while self.next_validation_usage <= usage + 1e-9:
                    self.next_validation_usage += interval
            exhausted = self._budget_exhausted()
            self._write_status(
                "simulator_budget_exhausted"
                if exhausted
                else "running"
            )
            return not exhausted

        def _on_training_end(self) -> None:
            usage = simulator_step_counter.snapshot().hour_steps
            if abs(usage - self.last_validation_usage) > 1e-9:
                self._evaluate(usage)
            self._write_status(
                "simulator_budget_exhausted"
                if self._budget_exhausted()
                else "completed"
            )

        def _evaluate(self, usage: float) -> None:
            records = evaluate_seeds(
                self.model,
                validation_env,
                validation_seeds,
            )
            metrics = validation_metrics(
                records,
                tail_vent_penalty_eur_per_t=0.0,
                hard_violation_penalty_eur=0.0,
            )
            mean_cost = float(metrics["mean_total_cost_eur"])
            is_best = mean_cost < self.best_cost
            if is_best:
                self.best_cost = mean_cost
                self.model.save(
                    run_dir / "event_residual_e1_best_validation"
                )
                _write_json(
                    run_dir / "validation" / "best.json",
                    {
                        "training_simulator_hour_steps": usage,
                        "selection_metric": "mean_total_cost_eur",
                        "metrics": metrics,
                        "validation_seeds": list(validation_seeds),
                        "per_seed": records,
                        "model_path": str(
                            run_dir
                            / "event_residual_e1_best_validation"
                        ),
                    },
                )
            _write_json(
                run_dir
                / "validation"
                / f"simulator_hour_{usage:g}.json",
                {
                    "training_simulator_hour_steps": usage,
                    "is_best": is_best,
                    "metrics": metrics,
                    "validation_seeds": list(validation_seeds),
                    "per_seed": records,
                },
            )
            self.last_validation_usage = usage

        def _write_status(self, state: str) -> None:
            usage = simulator_step_counter.snapshot()
            _write_json(
                run_dir / "status.json",
                {
                    "state": state,
                    "ppo_timesteps": self.num_timesteps,
                    **usage.as_dict(),
                    "max_simulator_hour_steps": (
                        max_simulator_hour_steps
                    ),
                    "simulator_budget_fraction": (
                        usage.hour_steps / max_simulator_hour_steps
                        if max_simulator_hour_steps is not None
                        else None
                    ),
                    "simulator_budget_exhausted": (
                        self._budget_exhausted()
                    ),
                },
            )

    return ObjectiveAlignedCallback()


def train_objective_aligned_event_residual_ppo(
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
    scenario_protocol: str = "unified_window_v1",
    hard_scenario_probability: float = 0.0,
    gate: AdaptiveRiskGateConfig | None = None,
    gate_mode: str = "hard",
    override_windows_h: tuple[tuple[float, float], ...] = (),
    training_seed_min: int = 100_000,
    training_seed_max: int = 999_999,
    validation_seeds: tuple[int, ...] = DEFAULT_VALIDATION_SEEDS,
    validation_every_simulator_hour_steps: int | None = None,
    max_simulator_hour_steps: int | None = None,
    gamma: float = 1.0,
    n_steps: int = 256,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    entropy_coefficient: float = 0.01,
    reward_scale: float = 1e-6,
    device: str = "cpu",
    verbose: int = 0,
    log_dir: Path | None = None,
):
    """Train v4 architecture from scratch for the E1 economic objective."""

    if abs(float(gamma) - 1.0) > 1e-12:
        raise ValueError("Formal Event-Residual PPO uses gamma=1.0.")
    if max_simulator_hour_steps is not None and (
        int(max_simulator_hour_steps) != max_simulator_hour_steps
        or max_simulator_hour_steps <= 0
    ):
        raise ValueError(
            "max_simulator_hour_steps must be a positive integer."
        )
    if (
        validation_every_simulator_hour_steps is not None
        and validation_every_simulator_hour_steps <= 0
    ):
        raise ValueError(
            "validation_every_simulator_hour_steps must be positive."
        )
    validation_seeds = tuple(int(value) for value in validation_seeds)
    _validate_seed_partition(
        training_seed_min=training_seed_min,
        training_seed_max=training_seed_max,
        validation_seeds=validation_seeds,
    )
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError(
            "Training requires stable-baselines3 and sb3-contrib."
        ) from exc

    run_dir = log_dir or default_run_dir(
        scenario=scenario,
        episode_hours=episode_hours,
        decision_interval_h=decision_interval_h,
        seed=seed,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    reward = HighLevelRewardConfig.objective_aligned(
        reward_scale=reward_scale,
    )
    gate_config = gate or AdaptiveRiskGateConfig()
    simulator_step_counter = SimulatorStepCounter()
    common_env = {
        "scenario": scenario,
        "episode_hours": episode_hours,
        "forecast_context_hours": forecast_context_hours,
        "future_summary_windows_h": future_summary_windows_h,
        "decision_interval_h": decision_interval_h,
        "event_triggered": event_triggered,
        "weather_mode": weather_mode,
        "scenario_protocol": scenario_protocol,
        "hard_scenario_probability": hard_scenario_probability,
        "reward": reward,
        "gate": gate_config,
        "gate_mode": gate_mode,
        "outside_risk_intervention_penalty": 0.0,
        "override_windows_h": override_windows_h,
    }
    native_env = make_tail_robust_native_env(
        **common_env,
        simulator_step_counter=simulator_step_counter,
        max_simulator_hour_steps=max_simulator_hour_steps,
    )
    gym_env = Monitor(
        MaskedResidualGymEnv(
            native_env,
            episode_seed_min=training_seed_min,
            episode_seed_max=training_seed_max,
        ),
        filename=str(run_dir / "monitor"),
    )
    validation_env = make_tail_robust_native_env(**common_env)
    if (
        validation_every_simulator_hour_steps is None
        and max_simulator_hour_steps is not None
    ):
        validation_every_simulator_hour_steps = max(
            2,
            max_simulator_hour_steps // 10,
        )
    planned_timesteps = _planned_timesteps(
        total_timesteps,
        n_steps,
        1,
    )
    paired_step_hours = (
        2.0 * native_env.env.network.time_step_hours
    )
    _write_json(
        run_dir / "config.json",
        {
            "interface_version": 1,
            "algorithm": "event_residual_ppo_objective_aligned",
            "architecture": "residual_rl_v4",
            "legacy_v4_training_entry_unchanged": True,
            "training_regime": "e1_objective_aligned_from_scratch",
            "reward_semantics": (
                "scaled_greedy_counterfactual_cost_minus_actual_cost"
            ),
            "scenario": scenario,
            "episode_hours": episode_hours,
            "forecast_context_hours": forecast_context_hours,
            "future_summary_representation_id": (
                FUTURE_SUMMARY_REPRESENTATION_ID
            ),
            "future_summary_windows_h": list(
                future_summary_windows_h
            ),
            "observation_feature_names": list(
                residual_feature_names(
                    native_env.env,
                    future_summary_windows_h,
                )
            ),
            "decision_interval_h": decision_interval_h,
            "event_triggered": event_triggered,
            "weather_mode": weather_mode,
            "scenario_protocol": scenario_protocol,
            "hard_scenario_probability": hard_scenario_probability,
            "gate": asdict(gate_config),
            "gate_mode": gate_mode,
            "outside_risk_intervention_penalty": 0.0,
            "override_windows_h": [
                list(window) for window in override_windows_h
            ],
            "well_control_mode": "automatic_max",
            "reward": asdict(reward),
            "requested_timesteps": total_timesteps,
            "planned_timesteps": planned_timesteps,
            "max_simulator_hour_steps": max_simulator_hour_steps,
            "paired_actual_counterfactual_step_hours": (
                paired_step_hours
            ),
            "training_seed_min": training_seed_min,
            "training_seed_max": training_seed_max,
            "validation_seeds": list(validation_seeds),
            "validation_every_simulator_hour_steps": (
                validation_every_simulator_hour_steps
            ),
            "checkpoint_selection": "minimum_validation_mean_total_cost",
            "seed": seed,
            "gamma": 1.0,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "entropy_coefficient": entropy_coefficient,
            "device": device,
            "action_count": native_env.action_count,
            "observation_size": native_env.observation_size,
        },
    )
    callback = _make_training_callback(
        run_dir=run_dir,
        simulator_step_counter=simulator_step_counter,
        max_simulator_hour_steps=max_simulator_hour_steps,
        paired_step_hours=paired_step_hours,
        validation_env=validation_env,
        validation_seeds=validation_seeds,
        validation_every_simulator_hour_steps=(
            validation_every_simulator_hour_steps
        ),
    )
    model = MaskablePPO(
        "MlpPolicy",
        gym_env,
        seed=seed,
        gamma=1.0,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        ent_coef=entropy_coefficient,
        device=device,
        verbose=verbose,
    )
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )
        model.save(run_dir / "event_residual_e1_final")
        usage = simulator_step_counter.snapshot()
        budget_exhausted = (
            max_simulator_hour_steps is not None
            and float(max_simulator_hour_steps) - usage.hour_steps
            < paired_step_hours - 1e-9
        )
        _write_json(
            run_dir / "training_complete.json",
            {
                "state": (
                    "simulator_budget_exhausted"
                    if budget_exhausted
                    else "completed"
                ),
                "requested_timesteps": total_timesteps,
                "planned_timesteps": planned_timesteps,
                **usage.as_dict(),
                "max_simulator_hour_steps": (
                    max_simulator_hour_steps
                ),
                "simulator_budget_fraction": (
                    usage.hour_steps / max_simulator_hour_steps
                    if max_simulator_hour_steps is not None
                    else None
                ),
                "simulator_budget_exhausted": budget_exhausted,
                "final_model_path": str(
                    run_dir / "event_residual_e1_final"
                ),
                "best_validation_model_path": str(
                    run_dir
                    / "event_residual_e1_best_validation"
                ),
            },
        )
    finally:
        gym_env.close()
    return model, run_dir


def main() -> None:
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
    )
    parser.add_argument("--decision-interval-h", type=float, default=24.0)
    parser.add_argument(
        "--event-triggered",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gate-mode",
        choices=("off", "soft", "hard"),
        default="hard",
    )
    parser.add_argument("--risk-hours-threshold-h", type=float, default=48.0)
    parser.add_argument("--risk-fill-threshold", type=float, default=0.80)
    parser.add_argument(
        "--max-simulator-hour-steps",
        type=int,
        default=None,
        help="Set to the measured B_4800 for formal E1 training.",
    )
    parser.add_argument(
        "--validation-every-simulator-hour-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_VALIDATION_SEEDS),
    )
    parser.add_argument("--training-seed-min", type=int, default=100_000)
    parser.add_argument("--training-seed-max", type=int, default=999_999)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--reward-scale", type=float, default=1e-6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    _model, run_dir = train_objective_aligned_event_residual_ppo(
        total_timesteps=args.timesteps,
        seed=args.seed,
        scenario=args.scenario,
        episode_hours=args.episode_hours,
        forecast_context_hours=args.forecast_context_hours,
        future_summary_windows_h=tuple(
            args.future_summary_windows_h
        ),
        decision_interval_h=args.decision_interval_h,
        gate=AdaptiveRiskGateConfig(
            hours_to_overflow_threshold_h=(
                args.risk_hours_threshold_h
            ),
            fill_ratio_threshold=args.risk_fill_threshold,
        ),
        gate_mode=args.gate_mode,
        training_seed_min=args.training_seed_min,
        training_seed_max=args.training_seed_max,
        validation_seeds=tuple(args.validation_seeds),
        validation_every_simulator_hour_steps=(
            args.validation_every_simulator_hour_steps
        ),
        max_simulator_hour_steps=(
            args.max_simulator_hour_steps
        ),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        entropy_coefficient=args.ent_coef,
        reward_scale=args.reward_scale,
        device=args.device,
        verbose=args.verbose,
        log_dir=args.log_dir,
    )
    print(f"Saved formal Event-Residual PPO under: {run_dir}")


if __name__ == "__main__":
    main()
