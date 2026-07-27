"""Train PPO on sparse 24-hour CCS dispatch decisions.

在稀疏的 24 小时 CCS 调度决策上训练 PPO。

The learner chooses a high-level goal while the fast rule executor performs
hourly actions. Native MPC is intentionally excluded from the training loop and
remains an evaluation baseline because it is too expensive for large PPO runs.

学习器选择高层目标，而快速规则执行器负责每小时动作。原生 MPC 有意不放入训练内环，因为它对大规模 PPO
训练过于昂贵；它保持作为评估基线。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from sim.environment import CCSEnvConfig, build_phase1_env
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from sim.simulator import SimulatorStepCounter

from sim.control.event_based.residual_rl_v4.scenario import (
    ReplayableDifficultyScenarioGenerator,
)
from sim.control.event_based.hybrid import (
    GoalAwareNativeMpcExecutor,
    GoalAwareRuleExecutor,
)

from .gym_env import HighLevelDispatchGymEnv
from .high_level_env import HighLevelDispatchEnv, HighLevelEnvConfig
from .observation_encoder import (
    FORECAST_WINDOWS_H,
    FUTURE_SUMMARY_REPRESENTATION_ID,
    future_summary_feature_names,
)
from .reward import HighLevelRewardConfig


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 training-status or configuration JSON document.

    写入 UTF-8 格式的训练状态或配置 JSON 文档。
    """
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tensorboard_log_dir(run_dir: Path) -> str | None:
    """Return a TensorBoard path only when TensorBoard is installed.

    仅在已安装 TensorBoard 时返回对应日志路径。
    """
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        return None
    path = run_dir / "tensorboard"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _make_status_callback(
    *,
    total_timesteps: int,
    run_dir: Path,
    report_every_steps: int,
    simulator_step_counter: SimulatorStepCounter,
    max_simulator_hour_steps: int | None,
    progress_mode: str = "lines",
):
    """Create a PPO callback that persists readable live training status.

    ``tqdm`` redraws a progress bar on one terminal line.  Some Windows
    PowerShell hosts render those redraws as concatenated text, so ``lines``
    is the default: it emits one complete record roughly every 5%.  ``bar``
    remains available for terminals that render dynamic progress bars well.

    ``tqdm`` 会在同一终端行上重绘进度条。部分 Windows PowerShell 主机将这些
    重绘显示为拼接文本，因此默认 ``lines``：大约每完成 5% 输出一条完整记录。
    对能正确渲染动态进度条的终端，仍可选择 ``bar``。
    """
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError(
            "High-level PPO progress logging requires stable-baselines3."
        ) from exc

    if progress_mode not in {"lines", "bar"}:
        raise ValueError("progress_mode must be either 'lines' or 'bar'.")
    try:
        from tqdm.auto import tqdm as tqdm_factory
    except ImportError:  # pragma: no cover - depends on the runtime environment.
        tqdm_factory = None
    if progress_mode == "bar" and tqdm_factory is None:
        raise ImportError("progress_mode='bar' requires tqdm.")

    def progress_write(message: str) -> None:
        if tqdm_factory is None:
            print(message, flush=True)
        else:
            tqdm_factory.write(message)

    class TqdmStatusCallback(BaseCallback):
        """Show readable progress and write CSV/JSON training metrics.

        显示可读的高层训练进度，并写入 CSV/JSON 指标。
        """

        def __init__(self) -> None:
            super().__init__()
            self.report_every = max(1, int(report_every_steps))
            self.progress_mode = progress_mode
            self.console_every = max(
                self.report_every,
                math.ceil(total_timesteps / 20),
            )
            self.last_report_step = 0
            self.last_progress_step = 0
            self.last_console_step = 0
            self.started_at = 0.0
            self.progress_bar = None
            self.metrics_file = None
            self.metrics_writer = None

        def _on_training_start(self) -> None:
            self.started_at = perf_counter()
            if self.progress_mode == "bar":
                self.progress_bar = tqdm_factory(
                    total=total_timesteps,
                    desc="High-level PPO",
                    unit="decision",
                    dynamic_ncols=True,
                )
            self.metrics_file = (run_dir / "training_metrics.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            )
            self.metrics_writer = csv.DictWriter(
                self.metrics_file,
                fieldnames=(
                    "timesteps",
                    "target_timesteps",
                    "simulator_step_calls",
                    "simulator_hour_steps",
                    "max_simulator_hour_steps",
                    "simulator_budget_fraction",
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
            self.metrics_writer.writeheader()
            self._record("running")

        def _on_step(self) -> bool:
            if self.progress_bar is not None:
                delta = self.num_timesteps - self.last_progress_step
                if delta > 0:
                    self.progress_bar.update(delta)
                    self.last_progress_step = self.num_timesteps
            if self.num_timesteps - self.last_report_step >= self.report_every:
                self._record("running")
            exhausted = self._budget_exhausted()
            if exhausted:
                self._record("simulator_budget_exhausted")
            return not exhausted

        def _on_training_end(self) -> None:
            state = (
                "simulator_budget_exhausted"
                if self._budget_exhausted()
                else "completed"
            )
            self._record(state)
            if self.progress_bar is not None:
                self.progress_bar.close()
            if self.metrics_file is not None:
                self.metrics_file.close()

        def _record(self, state: str) -> None:
            elapsed_seconds = max(1e-9, perf_counter() - self.started_at)
            metrics = self.logger.name_to_value
            simulator_usage = simulator_step_counter.snapshot()
            budget_fraction = (
                simulator_usage.hour_steps / max_simulator_hour_steps
                if max_simulator_hour_steps is not None
                else None
            )
            row = {
                "timesteps": self.num_timesteps,
                "target_timesteps": total_timesteps,
                "simulator_step_calls": simulator_usage.calls,
                "simulator_hour_steps": simulator_usage.hour_steps,
                "max_simulator_hour_steps": max_simulator_hour_steps,
                "simulator_budget_fraction": budget_fraction,
                "elapsed_seconds": elapsed_seconds,
                "decisions_per_second": self.num_timesteps / elapsed_seconds,
                "episode_reward_mean": _logger_scalar(
                    metrics,
                    "rollout/ep_rew_mean",
                )
                or _episode_buffer_mean(self.model.ep_info_buffer, "r"),
                "episode_length_mean": _logger_scalar(
                    metrics,
                    "rollout/ep_len_mean",
                )
                or _episode_buffer_mean(self.model.ep_info_buffer, "l"),
                "policy_loss": _logger_scalar(metrics, "train/policy_gradient_loss"),
                "value_loss": _logger_scalar(metrics, "train/value_loss"),
                "entropy_loss": _logger_scalar(metrics, "train/entropy_loss"),
                "approx_kl": _logger_scalar(metrics, "train/approx_kl"),
            }
            if self.metrics_writer is not None and self.metrics_file is not None:
                self.metrics_writer.writerow(row)
                self.metrics_file.flush()
            _write_json(
                run_dir / "status.json",
                {
                    "state": state,
                    "timesteps": self.num_timesteps,
                    "target_timesteps": total_timesteps,
                    "progress_fraction": min(1.0, self.num_timesteps / total_timesteps),
                    **simulator_usage.as_dict(),
                    "max_simulator_hour_steps": max_simulator_hour_steps,
                    "simulator_budget_fraction": budget_fraction,
                    "simulator_budget_exhausted": self._budget_exhausted(),
                    "elapsed_seconds": elapsed_seconds,
                    "decisions_per_second": row["decisions_per_second"],
                    "latest_metrics": row,
                },
            )
            if self.progress_bar is not None:
                self.progress_bar.set_postfix(
                    fps=f"{row['decisions_per_second']:.2f}",
                    reward=_format_optional(row["episode_reward_mean"]),
                )
            self._write_console_status(state, row)
            self.last_report_step = self.num_timesteps

        def _write_console_status(self, state: str, row: dict[str, Any]) -> None:
            """Emit one complete progress line when line-mode reporting is due.

            在行模式报告时机到达时，输出一条完整的进度记录。
            """
            if self.progress_mode != "lines":
                return
            is_final = state == "completed"
            if (
                not is_final
                and self.num_timesteps - self.last_console_step < self.console_every
            ):
                return
            self.last_console_step = self.num_timesteps
            progress_write(
                "High-level PPO | "
                f"{100.0 * self.num_timesteps / total_timesteps:5.1f}% | "
                f"{self.num_timesteps}/{total_timesteps} decisions | "
                f"{row['simulator_hour_steps']:.0f} simulator h | "
                f"{row['decisions_per_second']:.1f} decision/s | "
                f"mean_reward={_format_optional(row['episode_reward_mean'])} | "
                f"value_loss={_format_optional(row['value_loss'])} | "
                f"kl={_format_optional(row['approx_kl'])}"
            )

        @staticmethod
        def _budget_exhausted() -> bool:
            usage = simulator_step_counter.snapshot()
            return (
                max_simulator_hour_steps is not None
                and usage.hour_steps
                >= float(max_simulator_hour_steps) - 1e-9
            )

    return TqdmStatusCallback()


def _logger_scalar(values: dict[str, Any], name: str) -> float | None:
    """Read one scalar metric from Stable-Baselines3's current logger state.

    从 Stable-Baselines3 当前日志状态中读取一个标量指标。
    """
    value = values.get(name)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _episode_buffer_mean(episodes: Any, key: str) -> float | None:
    """Return a mean Monitor episode field before SB3's next logger dump.

    在 SB3 下次写入 logger 前，返回 Monitor 回合缓冲中一个字段的均值。
    """
    values: list[float] = []
    for episode in episodes:
        value = episode.get(key)
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _format_optional(value: float | None) -> str:
    """Format a possibly unavailable scalar for a compact tqdm postfix.

    为精简 tqdm 后缀格式化可能缺失的标量值。
    """
    return "n/a" if value is None else f"{value:.3f}"


def _planned_total_timesteps(requested_timesteps: int, rollout_steps: int) -> int:
    """Return PPO's effective total after completing whole rollout batches.

    返回 PPO 完成整个 rollout 批次后的实际总步数。
    """
    if requested_timesteps <= 0 or rollout_steps <= 0:
        raise ValueError("timesteps and n_steps must both be positive.")
    return int(math.ceil(requested_timesteps / rollout_steps) * rollout_steps)


def default_run_dir(
    *,
    scenario: str,
    episode_hours: int,
    decision_interval_h: float,
    seed: int,
) -> Path:
    """Return a unique configuration-labelled training directory under ``logs``.

    返回 ``logs`` 下方按配置命名的唯一训练目录。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = (
        f"{scenario}__{episode_hours}h__decision{decision_interval_h:g}h"
        f"__seed{seed}__{timestamp}"
    )
    return Path("logs") / "high_level_rl" / label


def make_high_level_native_env(
    *,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 169,
    future_summary_windows_h: tuple[int, ...] = FORECAST_WINDOWS_H,
    decision_interval_h: float = 24.0,
    event_triggered: bool = True,
    weather_mode: str = "block",
    warm_start: bool = True,
    scenario_protocol: str = "local_formal",
    executor: str = "rule",
    reward: HighLevelRewardConfig | None = None,
    simulator_step_counter: SimulatorStepCounter | None = None,
    max_simulator_hour_steps: int | None = None,
) -> HighLevelDispatchEnv:
    """Build the sparse RL environment with forecast context beyond episode end.

    构建在回合终点之后仍保留预测上下文的稀疏 RL 环境。
    """
    if weather_mode not in {"window", "block"}:
        raise ValueError("High-level PPO currently supports weather_mode 'window' or 'block'.")
    if scenario_protocol == "local_formal":
        scenario_generator = ScenarioGenerator(
            config=ScenarioConfig(
                episode_hours=episode_hours + forecast_context_hours,
                time_step_hours=1.0,
                weather_process=weather_mode,
                warm_start=warm_start,
            )
        )
    elif scenario_protocol == "unified_window_v1":
        scenario_generator = ReplayableDifficultyScenarioGenerator(
            episode_hours=episode_hours + forecast_context_hours,
            weather_process=weather_mode,
            hard_probability=0.0,
            scenario_protocol=scenario_protocol,
        )
    else:
        raise ValueError(f"unknown scenario protocol: {scenario_protocol}")
    reward_config = reward or HighLevelRewardConfig.objective_aligned()
    physical_env = build_phase1_env(
        scenario=scenario,
        scenario_generator=scenario_generator,
        weather_mode=weather_mode,
        config=CCSEnvConfig(
            episode_hours=episode_hours,
            include_goal_obs=False,
            reward_scale=reward_config.reward_scale,
            injection_reward_eur_per_t=0.0,
            store_reward_eur_per_t=0.0,
            vent_penalty_weight=1.0,
            operating_cost_weight=1.0,
            reward_mode="economic",
            well_control_mode="automatic_max",
        ),
        simulator_step_counter=simulator_step_counter,
    )
    if executor == "rule":
        executor_factory = GoalAwareRuleExecutor
    elif executor == "mpc":
        executor_factory = GoalAwareNativeMpcExecutor
    else:
        raise ValueError("executor must be 'rule' or 'mpc'.")
    return HighLevelDispatchEnv(
        physical_env,
        config=HighLevelEnvConfig(
            decision_interval_h=decision_interval_h,
            event_triggered=event_triggered,
            reward=reward_config,
            future_summary_windows_h=future_summary_windows_h,
            max_simulator_hour_steps=max_simulator_hour_steps,
        ),
        executor_factory=executor_factory,
    )


def train_high_level_ppo(
    *,
    total_timesteps: int = 50_000,
    seed: int = 0,
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    future_summary_windows_h: tuple[int, ...] = FORECAST_WINDOWS_H,
    decision_interval_h: float = 24.0,
    event_triggered: bool = True,
    weather_mode: str = "window",
    warm_start: bool = True,
    scenario_protocol: str = "unified_window_v1",
    gamma: float = 1.0,
    n_steps: int = 256,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    entropy_coefficient: float = 0.01,
    device: str = "cpu",
    verbose: int = 0,
    log_dir: Path | None = None,
    status_every_steps: int = 10,
    progress_mode: str = "lines",
    reward: HighLevelRewardConfig | None = None,
    max_simulator_hour_steps: int | None = None,
):
    """Train MaskablePPO over sparse dispatch goals and return the model.

    在稀疏调度目标上训练 PPO，并返回训练后的模型。
    """
    if abs(float(gamma) - 1.0) > 1e-12:
        raise ValueError("Formal Centralized PPO uses gamma=1.0.")
    if max_simulator_hour_steps is not None and (
        int(max_simulator_hour_steps) != max_simulator_hour_steps
        or max_simulator_hour_steps <= 0
    ):
        raise ValueError(
            "max_simulator_hour_steps must be a positive integer."
        )
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError(
            "train_high_level_ppo requires stable-baselines3 and sb3-contrib."
        ) from exc

    run_dir = log_dir or default_run_dir(
        scenario=scenario,
        episode_hours=episode_hours,
        decision_interval_h=decision_interval_h,
        seed=seed,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    simulator_step_counter = SimulatorStepCounter()
    native_env = make_high_level_native_env(
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        future_summary_windows_h=future_summary_windows_h,
        decision_interval_h=decision_interval_h,
        event_triggered=event_triggered,
        weather_mode=weather_mode,
        warm_start=warm_start,
        scenario_protocol=scenario_protocol,
        reward=reward,
        simulator_step_counter=simulator_step_counter,
        max_simulator_hour_steps=max_simulator_hour_steps,
    )
    gym_env = Monitor(
        HighLevelDispatchGymEnv(native_env),
        filename=str(run_dir / "monitor"),
    )
    effective_gamma = 1.0
    planned_timesteps = _planned_total_timesteps(total_timesteps, n_steps)
    _write_json(
        run_dir / "config.json",
        {
            "interface_version": 3,
            "algorithm": "MaskablePPO",
            "dynamic_action_masks": True,
            "action_mask_semantics": (
                "all_vessel_preferences_are_safe_intentions"
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
            "future_summary_feature_names": list(
                future_summary_feature_names(
                    native_env.env,
                    future_summary_windows_h,
                )
            ),
            "decision_interval_h": decision_interval_h,
            "event_triggered": event_triggered,
            "weather_mode": weather_mode,
            "warm_start": warm_start,
            "scenario_protocol": scenario_protocol,
            "requested_timesteps": total_timesteps,
            "planned_timesteps": planned_timesteps,
            "max_simulator_hour_steps": max_simulator_hour_steps,
            "seed": seed,
            "gamma": effective_gamma,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "entropy_coefficient": entropy_coefficient,
            "device": device,
            "progress_mode": progress_mode,
            "action_count": native_env.action_count,
            "observation_size": native_env.observation_size,
            "high_level_reward": asdict(native_env.config.reward),
        },
    )
    status_callback = _make_status_callback(
        total_timesteps=planned_timesteps,
        run_dir=run_dir,
        report_every_steps=status_every_steps,
        simulator_step_counter=simulator_step_counter,
        max_simulator_hour_steps=max_simulator_hour_steps,
        progress_mode=progress_mode,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, planned_timesteps // 10),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_high_level",
    )
    model = MaskablePPO(
        "MlpPolicy",
        gym_env,
        seed=seed,
        gamma=effective_gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        ent_coef=entropy_coefficient,
        device=device,
        verbose=verbose,
        tensorboard_log=_tensorboard_log_dir(run_dir),
    )
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([status_callback, checkpoint_callback]),
            progress_bar=False,
        )
        model.save(run_dir / "ppo_high_level_final")
        simulator_usage = simulator_step_counter.snapshot()
        simulator_budget_exhausted = (
            max_simulator_hour_steps is not None
            and simulator_usage.hour_steps
            >= float(max_simulator_hour_steps) - 1e-9
        )
        _write_json(
            run_dir / "training_complete.json",
            {
                "state": (
                    "simulator_budget_exhausted"
                    if simulator_budget_exhausted
                    else "completed"
                ),
                "requested_timesteps": total_timesteps,
                "planned_timesteps": planned_timesteps,
                **simulator_usage.as_dict(),
                "max_simulator_hour_steps": max_simulator_hour_steps,
                "simulator_budget_fraction": (
                    simulator_usage.hour_steps
                    / max_simulator_hour_steps
                    if max_simulator_hour_steps is not None
                    else None
                ),
                "simulator_budget_exhausted": (
                    simulator_budget_exhausted
                ),
                "model_path": str(run_dir / "ppo_high_level_final"),
            },
        )
    finally:
        gym_env.close()
    return model


def main() -> None:
    """Train from command-line settings and save the resulting PPO model.

    根据命令行设置训练并保存 PPO 模型。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", default="northern_lights_phase1_3vessels")
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument(
        "--future-summary-windows-h",
        type=int,
        nargs="*",
        default=list(FORECAST_WINDOWS_H),
        help=(
            "Increasing shared future-summary horizons; pass no values "
            "for a state-only ablation."
        ),
    )
    parser.add_argument("--decision-interval-h", type=float, default=24.0)
    parser.add_argument("--weather-mode", choices=("window", "block"), default="window")
    parser.add_argument(
        "--scenario-protocol",
        choices=("local_formal", "unified_window_v1"),
        default="unified_window_v1",
    )
    parser.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--event-triggered",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replan on operational events before the maximum interval. "
            "/ 在最大间隔前遇到运行事件时重规划。"
        ),
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--max-simulator-hour-steps",
        type=int,
        default=None,
        help=(
            "Hard cap on bottom-level one-hour simulator advances. "
            "Set this to B_4800 for formal training."
        ),
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=HighLevelRewardConfig().reward_scale,
        help="High-level realised-reward scale. / 高层实际结果奖励的缩放系数。",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PPO device; CPU is recommended for MLP policies. / PPO 设备；MLP 策略建议使用 CPU。",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="PPO entropy coefficient. / PPO 熵系数。",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Training directory under logs. / logs 下的训练目录。",
    )
    parser.add_argument(
        "--status-every-steps",
        type=int,
        default=10,
        help="Progress/status write interval. / 进度与状态写入间隔。",
    )
    parser.add_argument(
        "--progress-mode",
        choices=("lines", "bar"),
        default="lines",
        help=(
            "Console progress style; lines avoids PowerShell redraw artifacts. "
            "/ 终端进度样式；lines 可避免 PowerShell 重绘乱码。"
        ),
    )
    args = parser.parse_args()
    run_dir = args.log_dir or default_run_dir(
        scenario=args.scenario,
        episode_hours=args.episode_hours,
        decision_interval_h=args.decision_interval_h,
        seed=args.seed,
    )
    model = train_high_level_ppo(
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
        warm_start=args.warm_start,
        scenario_protocol=args.scenario_protocol,
        gamma=args.gamma,
        entropy_coefficient=args.ent_coef,
        device=args.device,
        log_dir=run_dir,
        status_every_steps=args.status_every_steps,
        progress_mode=args.progress_mode,
        reward=HighLevelRewardConfig.objective_aligned(
            reward_scale=args.reward_scale
        ),
        max_simulator_hour_steps=args.max_simulator_hour_steps,
    )
    print(f"Saved PPO model and metrics under: {run_dir}")
    print("High-level gamma: 1.00000000")


if __name__ == "__main__":
    main()
