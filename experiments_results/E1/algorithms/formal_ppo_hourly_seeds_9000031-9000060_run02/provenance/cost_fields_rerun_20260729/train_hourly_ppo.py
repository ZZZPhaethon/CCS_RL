"""Train the formal one-decision-per-hour centralized MaskablePPO baseline."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim.control.cplex_milp import _terminal_cleanup_cost_for_state
from sim.control.event_based.residual_rl_v4.scenario import (
    ReplayableDifficultyScenarioGenerator,
)
from sim.control.event_based.rl.observation_encoder import (
    FORECAST_WINDOWS_H,
    FUTURE_SUMMARY_REPRESENTATION_ID,
    future_summary_feature_names,
    high_level_observation,
)
from sim.environment import CCSEnv, CCSEnvConfig, build_phase1_env
from sim.environment.gym_adapter import (
    flat_action_mask,
    native_action_from_flat,
)
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from sim.simulator import (
    SimulatorStepCounter,
    SimulatorStepUsage,
)

from .gym_env import HourlyCentralizedPPOEnv


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def make_hourly_native_env(
    *,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    weather_mode: str = "window",
    warm_start: bool = True,
    scenario_protocol: str = "unified_window_v1",
    reward_scale: float = 1e-6,
    simulator_step_counter: SimulatorStepCounter | None = None,
) -> CCSEnv:
    """Build the objective-aligned physical environment used by hourly PPO."""

    if weather_mode not in {"window", "block"}:
        raise ValueError(
            "Hourly PPO supports weather_mode 'window' or 'block'."
        )
    if episode_hours <= 0 or forecast_context_hours < max(
        FORECAST_WINDOWS_H
    ):
        raise ValueError(
            "episode_hours must be positive and forecast_context_hours must "
            "cover the 168 h summary."
        )
    if reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive.")

    scenario_hours = episode_hours + forecast_context_hours
    if scenario_protocol == "local_formal":
        scenario_generator = ScenarioGenerator(
            config=ScenarioConfig(
                episode_hours=scenario_hours,
                time_step_hours=1.0,
                weather_process=weather_mode,
                warm_start=warm_start,
            )
        )
    elif scenario_protocol == "unified_window_v1":
        scenario_generator = ReplayableDifficultyScenarioGenerator(
            episode_hours=scenario_hours,
            weather_process=weather_mode,
            hard_probability=0.0,
            scenario_protocol=scenario_protocol,
        )
    else:
        raise ValueError(f"unknown scenario protocol: {scenario_protocol}")

    return build_phase1_env(
        scenario=scenario,
        scenario_generator=scenario_generator,
        weather_mode=weather_mode,
        config=CCSEnvConfig(
            episode_hours=episode_hours,
            include_goal_obs=False,
            include_weather_obs=False,
            reward_scale=reward_scale,
            injection_reward_eur_per_t=0.0,
            store_reward_eur_per_t=0.0,
            vent_penalty_weight=1.0,
            operating_cost_weight=1.0,
            reward_mode="economic",
            well_control_mode="automatic_max",
        ),
        simulator_step_counter=simulator_step_counter,
    )


def evaluate_seed(
    model,
    env: CCSEnv,
    *,
    seed: int,
    future_summary_windows_h: tuple[int, ...] = FORECAST_WINDOWS_H,
) -> dict[str, int | float]:
    """Evaluate a deterministic hourly policy on one fixed physical seed."""

    env.reset(seed=int(seed))
    decisions = 0
    while env.t < env.n_steps:
        observation = high_level_observation(
            env,
            future_summary_windows_h,
        )
        masks = flat_action_mask(
            env.vessel_action_mask(),
            env.well_rate_action_mask(),
        )
        action, _state = model.predict(
            observation,
            deterministic=True,
            action_masks=masks,
        )
        env.step(native_action_from_flat(env, action))
        decisions += 1

    cleanup_cost = float(
        _terminal_cleanup_cost_for_state(
            env,
            env.cost_model.parameters,
        )
    )
    episode_operating_cost = float(env.ledger.operating_cost)
    episode_total_cost = float(env.ledger.total_cost)
    total_cost = episode_total_cost + cleanup_cost
    stored_t = float(env.cumulative_stored_t)
    captured_t = float(env.cumulative_captured_t)
    unit_cost = (
        total_cost / stored_t if stored_t > 0.0 else float("inf")
    )
    return {
        "seed": int(seed),
        "decisions": decisions,
        "simulated_hours": float(env.simulator.state.time_h),
        "episode_vessel_fuel_eur": float(env.ledger.vessel_fuel),
        "episode_conditioning_eur": float(env.ledger.conditioning),
        "episode_reconditioning_eur": float(env.ledger.reconditioning),
        "episode_loading_eur": float(env.ledger.loading),
        "episode_unloading_eur": float(env.ledger.unloading),
        "episode_operating_cost_eur": episode_operating_cost,
        "episode_vent_penalty_eur": float(env.ledger.vent_penalty),
        "episode_storage_shortfall_penalty_eur": float(
            env.ledger.storage_shortfall_penalty
        ),
        "episode_total_cost_eur": episode_total_cost,
        "terminal_cleanup_operating_cost_eur": cleanup_cost,
        "total_cost_eur": total_cost,
        "operating_cost_eur": episode_operating_cost + cleanup_cost,
        "vented_t": float(env.ledger.vented_t),
        "stored_t": stored_t,
        "captured_t": captured_t,
        "storage_rate": stored_t / captured_t if captured_t > 0.0 else 1.0,
        "loss_rate": (
            float(env.ledger.vented_t) / captured_t
            if captured_t > 0.0
            else 0.0
        ),
        "cost_per_stored_t_eur": unit_cost,
        "unit_total_cost_eur_per_t": unit_cost,
    }


def default_run_dir(
    *,
    scenario: str,
    episode_hours: int,
    seed: int,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("logs")
        / "hourly_ppo"
        / f"{scenario}__{episode_hours}h__seed{seed}__{timestamp}"
    )


def train_hourly_ppo(
    *,
    total_timesteps: int = 10_000_000,
    seed: int = 0,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    future_summary_windows_h: tuple[int, ...] = FORECAST_WINDOWS_H,
    weather_mode: str = "window",
    warm_start: bool = True,
    scenario_protocol: str = "unified_window_v1",
    gamma: float = 1.0,
    n_steps: int = 2048,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    entropy_coefficient: float = 0.01,
    reward_scale: float = 1e-6,
    num_envs: int = 1,
    device: str = "cpu",
    verbose: int = 1,
    log_dir: Path | None = None,
    max_simulator_hour_steps: int | None = None,
    training_seed_min: int = 100_000,
    training_seed_max: int = 999_999,
    validation_seeds: tuple[int, ...] = (),
    validation_every_simulator_hour_steps: int | None = None,
) -> Path:
    """Train from scratch and select checkpoints only on validation seeds."""

    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if gamma != 1.0:
        raise ValueError("Formal hourly PPO fixes gamma=1.0.")
    if training_seed_min > training_seed_max:
        raise ValueError(
            "training_seed_min must not exceed training_seed_max."
        )
    overlap = [
        validation_seed
        for validation_seed in validation_seeds
        if training_seed_min <= validation_seed <= training_seed_max
    ]
    if overlap:
        raise ValueError(
            f"training and validation seeds overlap: {sorted(overlap)[:5]}"
        )
    if max_simulator_hour_steps is not None and (
        int(max_simulator_hour_steps) != max_simulator_hour_steps
        or max_simulator_hour_steps <= 0
    ):
        raise ValueError(
            "max_simulator_hour_steps must be a positive integer."
        )
    if validation_every_simulator_hour_steps is not None and (
        validation_every_simulator_hour_steps <= 0
    ):
        raise ValueError(
            "validation_every_simulator_hour_steps must be positive."
        )

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError(
            "train_hourly_ppo requires stable-baselines3 and sb3-contrib."
        ) from exc

    run_dir = log_dir or default_run_dir(
        scenario=scenario,
        episode_hours=episode_hours,
        seed=seed,
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    counter = SimulatorStepCounter()
    native_env = make_hourly_native_env(
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        weather_mode=weather_mode,
        warm_start=warm_start,
        scenario_protocol=scenario_protocol,
        reward_scale=reward_scale,
        simulator_step_counter=counter,
    )
    effective_budget_stop = max_simulator_hour_steps
    if max_simulator_hour_steps is not None and num_envs > 1:
        effective_budget_stop = (
            max_simulator_hour_steps
            - max_simulator_hour_steps % num_envs
        )
        if effective_budget_stop <= 0:
            raise ValueError(
                "max_simulator_hour_steps must cover at least one "
                "vectorized environment step."
            )
    if num_envs == 1:
        gym_env = Monitor(
            HourlyCentralizedPPOEnv(
                native_env,
                future_summary_windows_h=future_summary_windows_h,
                episode_seed_min=training_seed_min,
                episode_seed_max=training_seed_max,
                max_simulator_hour_steps=max_simulator_hour_steps,
            ),
            filename=str(run_dir / "monitor"),
        )
    else:
        def make_worker(rank: int):
            def build_worker():
                worker_env = make_hourly_native_env(
                    scenario=scenario,
                    episode_hours=episode_hours,
                    forecast_context_hours=forecast_context_hours,
                    weather_mode=weather_mode,
                    warm_start=warm_start,
                    scenario_protocol=scenario_protocol,
                    reward_scale=reward_scale,
                )
                return Monitor(
                    HourlyCentralizedPPOEnv(
                        worker_env,
                        future_summary_windows_h=(
                            future_summary_windows_h
                        ),
                        episode_seed_min=training_seed_min,
                        episode_seed_max=training_seed_max,
                    ),
                    filename=str(run_dir / f"monitor_{rank}"),
                )

            return build_worker

        gym_env = SubprocVecEnv(
            [make_worker(rank) for rank in range(num_envs)],
            start_method="spawn",
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
    if (
        validation_seeds
        and validation_every_simulator_hour_steps is None
        and max_simulator_hour_steps is not None
    ):
        validation_every_simulator_hour_steps = max(
            1,
            max_simulator_hour_steps // 10,
        )

    config = {
        "interface_version": 1,
        "paper_name": "Hourly Centralized Maskable PPO",
        "algorithm": "MaskablePPO",
        "training_from_scratch": True,
        "decision_interval_h": 1.0,
        "physical_time_step_h": 1.0,
        "direct_native_action": True,
        "action_space": "MultiDiscrete per-vessel dispatch",
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
        "future_summary_representation_id": (
            FUTURE_SUMMARY_REPRESENTATION_ID
        ),
        "future_summary_windows_h": list(future_summary_windows_h),
        "future_summary_feature_names": list(
            future_summary_feature_names(
                native_env,
                future_summary_windows_h,
            )
        ),
        "valid_fraction_feature": False,
        "objective": (
            "negative realised economic cost plus terminal cleanup cost"
        ),
        "reward_scale": reward_scale,
        "gamma": gamma,
        "n_steps": n_steps,
        "num_envs": num_envs,
        "vectorization": (
            "single_env" if num_envs == 1 else "SubprocVecEnv"
        ),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "entropy_coefficient": entropy_coefficient,
        "requested_timesteps": total_timesteps,
        "max_simulator_hour_steps": max_simulator_hour_steps,
        "effective_vectorized_budget_stop": effective_budget_stop,
        "training_seed_min": training_seed_min,
        "training_seed_max": training_seed_max,
        "validation_seeds": list(validation_seeds),
        "validation_every_simulator_hour_steps": (
            validation_every_simulator_hour_steps
        ),
        "checkpoint_selection": (
            "minimum_validation_mean_total_cost"
            if validation_seeds
            else None
        ),
        "model_seed": seed,
        "device": device,
    }
    _write_json(run_dir / "config.json", config)

    class BudgetValidationCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.best_validation_cost = float("inf")
            self.last_validation_usage = -1.0
            self.next_validation_usage = float(
                validation_every_simulator_hour_steps
                if validation_every_simulator_hour_steps is not None
                else "inf"
            )

        def _on_step(self) -> bool:
            usage = self._current_usage()
            if usage + 1e-9 >= self.next_validation_usage:
                self._run_validation(usage)
                interval = float(
                    validation_every_simulator_hour_steps
                )
                while self.next_validation_usage <= usage + 1e-9:
                    self.next_validation_usage += interval
            return not self._budget_exhausted()

        def _on_training_end(self) -> None:
            usage = self._current_usage()
            if (
                validation_seeds
                and abs(usage - self.last_validation_usage) > 1e-9
            ):
                self._run_validation(usage)

        def _run_validation(self, usage: float) -> None:
            if validation_env is None:
                return
            records = [
                evaluate_seed(
                    self.model,
                    validation_env,
                    seed=validation_seed,
                    future_summary_windows_h=future_summary_windows_h,
                )
                for validation_seed in validation_seeds
            ]
            mean_cost = float(
                np.mean(
                    [float(row["total_cost_eur"]) for row in records]
                )
            )
            is_best = mean_cost < self.best_validation_cost
            payload = {
                "training_simulator_hour_steps": usage,
                "is_best": is_best,
                "mean_total_cost_eur": mean_cost,
                "validation_seeds": list(validation_seeds),
                "per_seed": records,
            }
            _write_json(
                run_dir
                / "validation"
                / f"simulator_hour_{usage:g}.json",
                payload,
            )
            if is_best:
                self.best_validation_cost = mean_cost
                self.model.save(
                    run_dir / "ppo_hourly_best_validation"
                )
                _write_json(
                    run_dir / "validation" / "best.json",
                    {
                        **payload,
                        "model_path": str(
                            run_dir / "ppo_hourly_best_validation"
                        ),
                    },
                )
            self.last_validation_usage = usage

        def _budget_exhausted(self) -> bool:
            return (
                effective_budget_stop is not None
                and self._current_usage()
                >= float(effective_budget_stop) - 1e-9
            )

        def _current_usage(self) -> float:
            if num_envs == 1:
                return counter.snapshot().hour_steps
            # Every vector worker either advances one physical hour or the
            # callback stops the rollout. Therefore SB3's aggregate transition
            # counter is exactly the summed bottom-level simulator-hour use.
            return float(self.num_timesteps)

    callback = BudgetValidationCallback()
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
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
    )
    model.save(run_dir / "ppo_hourly_final")

    if num_envs == 1:
        usage = counter.snapshot()
    else:
        worker_usage = gym_env.env_method("training_simulator_usage")
        usage = SimulatorStepUsage(
            calls=sum(int(row["simulator_step_calls"]) for row in worker_usage),
            simulated_hours=sum(
                float(row["simulator_simulated_hours"])
                for row in worker_usage
            ),
        )
    exhausted = (
        max_simulator_hour_steps is not None
        and usage.hour_steps
        >= float(max_simulator_hour_steps) - 1e-9
    )
    budget_stop_reached = (
        effective_budget_stop is not None
        and usage.hour_steps
        >= float(effective_budget_stop) - 1e-9
    )
    _write_json(
        run_dir / "training_complete.json",
        {
            "state": (
                "simulator_budget_exhausted"
                if exhausted
                else (
                    "simulator_budget_stop_reached"
                    if budget_stop_reached
                    else "completed"
                )
            ),
            **usage.as_dict(),
            "max_simulator_hour_steps": max_simulator_hour_steps,
            "effective_vectorized_budget_stop": effective_budget_stop,
            "simulator_budget_fraction": (
                usage.hour_steps / max_simulator_hour_steps
                if max_simulator_hour_steps is not None
                else None
            ),
            "simulator_budget_exhausted": exhausted,
            "simulator_budget_stop_reached": budget_stop_reached,
            "model_path": str(run_dir / "ppo_hourly_final"),
            "best_validation_model_path": (
                str(run_dir / "ppo_hourly_best_validation")
                if validation_seeds
                else None
            ),
        },
    )
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the direct one-action-per-hour centralized MaskablePPO "
            "baseline."
        )
    )
    parser.add_argument("--timesteps", type=int, default=10_000_000)
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
        nargs="+",
        default=list(FORECAST_WINDOWS_H),
    )
    parser.add_argument(
        "--weather-mode",
        choices=("window", "block"),
        default="window",
    )
    parser.add_argument(
        "--scenario-protocol",
        choices=("unified_window_v1", "local_formal"),
        default="unified_window_v1",
    )
    parser.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--reward-scale", type=float, default=1e-6)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument(
        "--max-simulator-hour-steps",
        type=int,
        default=None,
    )
    parser.add_argument("--training-seed-min", type=int, default=100_000)
    parser.add_argument("--training-seed-max", type=int, default=999_999)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--validation-every-simulator-hour-steps",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = train_hourly_ppo(
        total_timesteps=args.timesteps,
        seed=args.seed,
        scenario=args.scenario,
        episode_hours=args.episode_hours,
        forecast_context_hours=args.forecast_context_hours,
        future_summary_windows_h=tuple(
            args.future_summary_windows_h
        ),
        weather_mode=args.weather_mode,
        warm_start=args.warm_start,
        scenario_protocol=args.scenario_protocol,
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        entropy_coefficient=args.ent_coef,
        reward_scale=args.reward_scale,
        num_envs=args.num_envs,
        device=args.device,
        verbose=args.verbose,
        log_dir=args.log_dir,
        max_simulator_hour_steps=args.max_simulator_hour_steps,
        training_seed_min=args.training_seed_min,
        training_seed_max=args.training_seed_max,
        validation_seeds=tuple(args.validation_seeds),
        validation_every_simulator_hour_steps=(
            args.validation_every_simulator_hour_steps
        ),
    )
    print(f"Saved hourly PPO model and metrics under: {run_dir}")


if __name__ == "__main__":
    main()
