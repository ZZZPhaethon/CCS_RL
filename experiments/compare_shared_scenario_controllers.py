"""Compare rule, PPO, and rollout MPC on identical pre-generated scenarios.

在完全相同的预生成场景上比较规则、PPO 与 rollout MPC。

For every seed, one disturbance trajectory covering the physical episode plus
the MPC forecast context is sampled once. Independent controller environments
receive deep copies of that trajectory and execute only the physical episode.

对于每个 seed，只采样一次覆盖物理回合与 MPC 预测上下文的扰动轨迹。各控制器
环境接收该轨迹的独立深拷贝，并且只执行物理回合时域。
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import (
    DispatchGoal,
    GoalAwareNativeMpcExecutor,
    GoalAwareRuleExecutor,
    evaluate_executor,
)
from algorithms.rl.high_level_env import HighLevelDispatchEnv, HighLevelEnvConfig
from algorithms.rl.reward import HARD_VIOLATION_CODES, HighLevelRewardConfig
from Simulation.control.baselines import balanced_capture_assignment
from Simulation.environment import CCSEnvConfig, build_phase1_env
from Simulation.scenario_generation import Scenario, ScenarioConfig, ScenarioGenerator


RAW_COLUMNS = (
    "controller",
    "seed",
    "captured_t",
    "stored_t",
    "vented_t",
    "storage_rate",
    "vent_rate",
    "operating_cost_eur",
    "total_cost_eur",
    "unit_total_cost_eur_per_t",
    "episode_reward",
    "wall_clock_seconds",
    "hard_violations",
    "violation_counts",
    "decisions",
    "mean_decision_interval_h",
    "action_counts",
    "trigger_counts",
)
SUMMARY_METRICS = tuple(
    column
    for column in RAW_COLUMNS
    if column
    not in {
        "controller",
        "seed",
        "violation_counts",
        "action_counts",
        "trigger_counts",
        # PPO uses its shaped high-level reward while the other controllers
        # expose the simulator reward, so this field is diagnostic only.
        # PPO 使用高层塑形奖励，其他控制器使用仿真器奖励，因此该字段仅供诊断。
        "episode_reward",
    }
)


class FixedScenarioGenerator:
    """Return an independent copy of one already sampled scenario.

    返回一份已采样场景的独立副本。
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = deepcopy(scenario)

    def sample(self, _network, seed: int | None = None) -> Scenario:
        """Ignore the reset seed and return the fixed physical trajectory.

        忽略 reset seed，并返回固定物理轨迹。
        """
        return deepcopy(self.scenario)


def parse_args() -> argparse.Namespace:
    """Parse reproducible shared-scenario comparison settings.

    解析可复现的共享场景比较参数。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-run-dir", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        default="northern_lights_phase1_3vessels",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument("--replan-hours", type=float, default=24.0)
    parser.add_argument("--planning-horizon-hours", type=int, default=168)
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=("rule", "ppo", "rollout_mpc"),
        default=("rule", "ppo", "rollout_mpc"),
    )
    parser.add_argument(
        "--weather-mode",
        choices=("window", "block"),
        default="window",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the paired comparison and persist raw, summary, and metadata files.

    运行配对比较并保存原始、汇总和元数据文件。
    """
    args = parse_args()
    _validate_args(args)
    output_paths = _output_paths(args.output_dir)
    _guard_output_paths(output_paths, args.overwrite)

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise ImportError("Shared comparison requires stable-baselines3.") from exc

    ppo_config = json.loads(
        (args.ppo_run_dir / "config.json").read_text(encoding="utf-8")
    )
    _validate_ppo_config(args, ppo_config)
    reward_config = HighLevelRewardConfig(**ppo_config["high_level_reward"])
    model = PPO.load(args.ppo_run_dir / "ppo_high_level_final", device="cpu")

    scenario_generator = ScenarioGenerator(
        config=ScenarioConfig(
            episode_hours=args.episode_hours + args.forecast_context_hours,
            time_step_hours=1.0,
            weather_process=args.weather_mode,
        )
    )
    probe = build_phase1_env(
        scenario=args.scenario,
        scenario_generator=scenario_generator,
        weather_mode=args.weather_mode,
        config=_physical_env_config(args),
    )
    goal = DispatchGoal(
        emitter_to_vessel=balanced_capture_assignment(probe),
        replan_after_h=args.replan_hours,
        rationale="Balanced shared-scenario controller comparison.",
    )

    rows: list[dict[str, Any]] = []
    scenario_fingerprints: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        shared_scenario = scenario_generator.sample(probe.network, seed=seed)
        scenario_fingerprints[int(seed)] = _scenario_fingerprint(shared_scenario)
        seed_rows = _evaluate_seed(
            args,
            model,
            reward_config,
            goal,
            shared_scenario,
            int(seed),
        )
        _assert_paired_capture(seed_rows, seed)
        rows.extend(seed_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_paths["raw"], rows, RAW_COLUMNS)
    _write_csv(
        output_paths["summary"],
        _summarise(rows),
        _summary_columns(),
    )
    output_paths["metadata"].write_text(
        json.dumps(
            {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "paired_capture_verified": True,
                "scenario_fingerprints": scenario_fingerprints,
                "ppo_config": ppo_config,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Raw metrics: {output_paths['raw']}")
    print(f"Summary metrics: {output_paths['summary']}")
    print(f"Metadata: {output_paths['metadata']}")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid horizons and missing PPO artifacts.

    拒绝无效时域和缺失的 PPO 产物。
    """
    if args.episode_hours <= 0 or args.forecast_context_hours < 0:
        raise ValueError("Episode hours must be positive and context non-negative.")
    if args.planning_horizon_hours <= 0 or args.replan_hours <= 0.0:
        raise ValueError("Planning and replanning horizons must be positive.")
    for name in ("config.json", "ppo_high_level_final.zip"):
        if not (args.ppo_run_dir / name).exists():
            raise FileNotFoundError(args.ppo_run_dir / name)


def _validate_ppo_config(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Ensure the saved policy matches the requested physical task.

    确保已保存策略与请求的物理任务匹配。
    """
    expected = {
        "scenario": args.scenario,
        "episode_hours": args.episode_hours,
        "decision_interval_h": args.replan_hours,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"PPO configuration does not match comparison: {mismatches}")


def _physical_env_config(args: argparse.Namespace) -> CCSEnvConfig:
    """Return one physical configuration shared by every controller.

    返回所有控制器共享的物理配置。
    """
    return CCSEnvConfig(
        episode_hours=args.episode_hours,
        include_goal_obs=False,
        reward_mode="vent_first",
    )


def _build_physical_env(args: argparse.Namespace, scenario: Scenario):
    """Build a fresh environment around one fixed scenario copy.

    围绕一个固定场景副本构建全新环境。
    """
    return build_phase1_env(
        scenario=args.scenario,
        scenario_generator=FixedScenarioGenerator(scenario),
        weather_mode=args.weather_mode,
        config=_physical_env_config(args),
    )


def _evaluate_seed(
    args: argparse.Namespace,
    model,
    reward_config: HighLevelRewardConfig,
    goal: DispatchGoal,
    scenario: Scenario,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate all three controllers against copies of one scenario.

    在同一场景的副本上评估三个控制器。
    """
    rows: list[dict[str, Any]] = []
    standard_controllers = {}
    if "rule" in args.controllers:
        standard_controllers["rule"] = GoalAwareRuleExecutor()
    if "rollout_mpc" in args.controllers:
        standard_controllers["rollout_mpc"] = GoalAwareNativeMpcExecutor(
            planning_horizon_h=args.planning_horizon_hours
        )
    for name, executor in standard_controllers.items():
        metrics = evaluate_executor(
            _build_physical_env(args, scenario),
            executor,
            goal,
            seed=seed,
        )
        row = _standard_metric_row(name, seed, metrics)
        rows.append(row)
        _print_row(row)

    if "ppo" in args.controllers:
        ppo_env = HighLevelDispatchEnv(
            _build_physical_env(args, scenario),
            config=HighLevelEnvConfig(
                decision_interval_h=args.replan_hours,
                event_triggered=True,
                reward=reward_config,
            ),
        )
        ppo_row = _evaluate_ppo(model, ppo_env, seed)
        rows.append(ppo_row)
        _print_row(ppo_row)
    return rows


def _standard_metric_row(controller: str, seed: int, metrics) -> dict[str, Any]:
    """Convert executor metrics to the common comparison schema.

    将执行器指标转换为统一比较结构。
    """
    hard_violations = sum(
        int(count)
        for code, count in metrics.violation_counts.items()
        if code in HARD_VIOLATION_CODES
    )
    return {
        "controller": controller,
        "seed": seed,
        "captured_t": metrics.captured_t,
        "stored_t": metrics.stored_t,
        "vented_t": metrics.vented_t,
        "storage_rate": metrics.storage_rate,
        "vent_rate": metrics.vent_rate,
        "operating_cost_eur": metrics.operating_cost,
        "total_cost_eur": metrics.total_cost,
        "unit_total_cost_eur_per_t": metrics.unit_storage_cost_eur_per_t,
        "episode_reward": metrics.total_reward,
        "wall_clock_seconds": metrics.wall_clock_seconds,
        "hard_violations": hard_violations,
        "violation_counts": json.dumps(metrics.violation_counts, sort_keys=True),
        "decisions": None,
        "mean_decision_interval_h": None,
        "action_counts": "{}",
        "trigger_counts": "{}",
    }


def _evaluate_ppo(model, env: HighLevelDispatchEnv, seed: int) -> dict[str, Any]:
    """Run one deterministic PPO episode on the fixed scenario.

    在固定场景上运行一个确定性 PPO 回合。
    """
    started_at = perf_counter()
    observation = env.reset(seed=seed)
    total_reward = 0.0
    decisions = 0
    elapsed_hours = 0.0
    actions: Counter[str] = Counter()
    triggers: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    done = False
    while not done:
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        decisions += 1
        elapsed_hours += float(info["elapsed_hours"])
        actions[str(info["action_label"])] += 1
        triggers[str(info["decision_trigger"])] += 1
        violations.update(info["violation_counts"])
        done = terminated or truncated

    physical_env = env.env
    captured_t = float(physical_env.cumulative_captured_t)
    stored_t = float(physical_env.cumulative_stored_t)
    vented_t = float(physical_env.ledger.vented_t)
    total_cost = float(physical_env.ledger.total_cost)
    hard_violations = sum(
        int(count)
        for code, count in violations.items()
        if code in HARD_VIOLATION_CODES
    )
    return {
        "controller": "ppo",
        "seed": seed,
        "captured_t": captured_t,
        "stored_t": stored_t,
        "vented_t": vented_t,
        "storage_rate": stored_t / captured_t if captured_t > 1e-9 else 0.0,
        "vent_rate": vented_t / captured_t if captured_t > 1e-9 else 0.0,
        "operating_cost_eur": float(physical_env.ledger.operating_cost),
        "total_cost_eur": total_cost,
        "unit_total_cost_eur_per_t": (
            total_cost / stored_t if stored_t > 1e-9 else None
        ),
        "episode_reward": total_reward,
        "wall_clock_seconds": perf_counter() - started_at,
        "hard_violations": hard_violations,
        "violation_counts": json.dumps(dict(violations), sort_keys=True),
        "decisions": decisions,
        "mean_decision_interval_h": elapsed_hours / max(1, decisions),
        "action_counts": json.dumps(dict(actions), sort_keys=True),
        "trigger_counts": json.dumps(dict(triggers), sort_keys=True),
    }


def _assert_paired_capture(rows: list[dict[str, Any]], seed: int) -> None:
    """Fail if any controller saw a different exogenous capture trajectory.

    若任一控制器看到不同的外生捕集轨迹，则立即失败。
    """
    captured = [float(row["captured_t"]) for row in rows]
    if max(captured) - min(captured) > 1e-6:
        raise AssertionError(
            f"Shared scenario failed for seed {seed}: captured_t={captured}"
        )
    print(
        f"seed={seed} paired_capture_verified={captured[0]:,.3f} t",
        flush=True,
    )


def _scenario_fingerprint(scenario: Scenario) -> dict[str, Any]:
    """Return compact metadata proving the pre-generated horizon and seed.

    返回证明预生成时域与种子的紧凑元数据。
    """
    return {
        "seed": scenario.seed,
        "n_steps": scenario.n_steps,
        "time_step_hours": scenario.time_step_hours,
        "emitter_series_lengths": {
            key: len(values)
            for key, values in scenario.emitter_availability.items()
        },
        "vessel_series_lengths": {
            key: len(values)
            for key, values in scenario.vessel_speed_factor.items()
        },
        "well_series_lengths": {
            key: len(values)
            for key, values in scenario.well_available.items()
        },
    }


def _print_row(row: dict[str, Any]) -> None:
    """Print one complete controller result line immediately.

    立即输出一条完整的控制器结果行。
    """
    print(
        f"seed={row['seed']} controller={row['controller']} "
        f"captured={float(row['captured_t']):,.1f} t "
        f"stored={float(row['stored_t']):,.1f} t "
        f"vented={float(row['vented_t']):,.1f} t "
        f"total_cost=EUR {float(row['total_cost_eur']):,.1f}",
        flush=True,
    )


def _summarise(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute controller means and sample standard deviations.

    计算各控制器均值与样本标准差。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["controller"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for controller, controller_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "controller": controller,
            "runs": len(controller_rows),
        }
        for metric in SUMMARY_METRICS:
            values = [
                float(row[metric])
                for row in controller_rows
                if row[metric] is not None
            ]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else None
            summary[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summaries.append(summary)
    return summaries


def _summary_columns() -> tuple[str, ...]:
    """Return stable summary CSV columns.

    返回稳定的汇总 CSV 列。
    """
    columns = ["controller", "runs"]
    for metric in SUMMARY_METRICS:
        columns.extend((f"{metric}_mean", f"{metric}_std"))
    return tuple(columns)


def _output_paths(output_dir: Path) -> dict[str, Path]:
    """Return the paired-comparison output paths.

    返回配对比较输出路径。
    """
    return {
        "raw": output_dir / "comparison_raw.csv",
        "summary": output_dir / "comparison_summary.csv",
        "metadata": output_dir / "comparison_metadata.json",
    }


def _guard_output_paths(paths: dict[str, Path], overwrite: bool) -> None:
    """Prevent accidental replacement of a completed comparison.

    防止意外覆盖已完成比较。
    """
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output already exists: "
            + ", ".join(str(path) for path in existing)
        )


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    columns: Iterable[str],
) -> None:
    """Write UTF-8 CSV with a stable column order.

    使用稳定列顺序写入 UTF-8 CSV。
    """
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
