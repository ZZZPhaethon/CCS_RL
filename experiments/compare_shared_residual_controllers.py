"""Fairly compare rule, residual PPO, and rollout MPC on shared scenarios.

在共享场景上公平比较规则、残差 PPO 与 rollout MPC。

For each seed, one ``episode + forecast context`` trajectory is generated once
and deep-copied into every controller. All controllers execute only the episode
horizon, and equal captured tonnes are asserted before results are accepted.

每个 seed 只生成一次“执行时域 + 预测上下文”轨迹，再深拷贝给每个控制器。
所有控制器仅执行回合时域，并在接受结果前断言累计捕集量完全一致。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

from algorithms import (
    DispatchGoal,
    GoalAwareNativeMpcExecutor,
    GoalAwareRuleExecutor,
    evaluate_executor,
)
from algorithms.residual_rl.env import ResidualDispatchEnv, ResidualEnvConfig
from algorithms.residual_rl.evaluation import evaluate_seed
from algorithms.rl.reward import HARD_VIOLATION_CODES, HighLevelRewardConfig
from Simulation.control.baselines import balanced_capture_assignment
from Simulation.environment import CCSEnvConfig, build_phase1_env
from Simulation.scenario_generation import Scenario, ScenarioConfig, ScenarioGenerator

from experiments.compare_shared_scenario_controllers import (
    FixedScenarioGenerator,
)


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
    "wall_clock_seconds",
    "hard_violations",
    "decisions",
    "mean_decision_interval_h",
    "interventions_applied",
    "intervention_rate",
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
        "action_counts",
        "trigger_counts",
    }
)


def parse_args() -> argparse.Namespace:
    """Parse strict paired-comparison settings.

    解析严格配对比较设置。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-run-dir", type=Path, required=True)
    parser.add_argument(
        "--residual-model",
        choices=("best", "final"),
        default="best",
    )
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
        choices=("rule", "residual_ppo", "rollout_mpc"),
        default=("rule", "residual_ppo", "rollout_mpc"),
    )
    parser.add_argument(
        "--weather-mode",
        choices=("window", "block"),
        default="window",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the shared-scenario comparison and save CSV/JSON artifacts.

    运行共享场景比较并保存 CSV/JSON 产物。
    """
    args = parse_args()
    _validate_args(args)
    paths = {
        "raw": args.output_dir / "comparison_raw.csv",
        "summary": args.output_dir / "comparison_summary.csv",
        "metadata": args.output_dir / "comparison_metadata.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists: " + ", ".join(map(str, existing))
        )

    run_config = json.loads(
        (args.residual_run_dir / "config.json").read_text(encoding="utf-8")
    )
    _validate_run_config(args, run_config)
    reward = HighLevelRewardConfig(**run_config["high_level_reward"])
    model_stem = (
        "ppo_residual_best_validation"
        if args.residual_model == "best"
        else "ppo_residual_final"
    )
    model = PPO.load(args.residual_run_dir / model_stem, device="cpu")

    generator = ScenarioGenerator(
        ScenarioConfig(
            episode_hours=args.episode_hours + args.forecast_context_hours,
            time_step_hours=1.0,
            weather_process=args.weather_mode,
        )
    )
    probe = build_phase1_env(
        scenario=args.scenario,
        scenario_generator=generator,
        weather_mode=args.weather_mode,
        config=_physical_config(args),
    )
    goal = DispatchGoal(
        emitter_to_vessel=balanced_capture_assignment(probe),
        replan_after_h=args.replan_hours,
        rationale="Shared-scenario residual comparison.",
    )

    rows: list[dict[str, Any]] = []
    fingerprints: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        shared = generator.sample(probe.network, seed=int(seed))
        fingerprints[int(seed)] = _fingerprint(shared)
        seed_rows = _evaluate_seed(args, model, reward, goal, shared, int(seed))
        captured = [float(row["captured_t"]) for row in seed_rows]
        if max(captured) - min(captured) > 1e-6:
            raise AssertionError(
                f"Shared scenario failed for seed {seed}: {captured}"
            )
        print(
            f"seed={seed} paired_capture_verified={captured[0]:,.3f} t",
            flush=True,
        )
        rows.extend(seed_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(paths["raw"], rows, RAW_COLUMNS)
    _write_csv(paths["summary"], _summarise(rows), _summary_columns())
    paths["metadata"].write_text(
        json.dumps(
            {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "paired_capture_verified": True,
                "scenario_fingerprints": fingerprints,
                "residual_run_config": run_config,
                "residual_model_path": str(args.residual_run_dir / model_stem),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Raw metrics: {paths['raw']}")
    print(f"Summary metrics: {paths['summary']}")
    print(f"Metadata: {paths['metadata']}")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    """Validate horizons and saved model artifacts.

    校验时域与已保存模型产物。
    """
    if args.episode_hours <= 0 or args.forecast_context_hours < 0:
        raise ValueError("Invalid episode or forecast-context horizon.")
    if args.replan_hours <= 0 or args.planning_horizon_hours <= 0:
        raise ValueError("Replan and planning horizons must be positive.")
    model_name = (
        "ppo_residual_best_validation.zip"
        if args.residual_model == "best"
        else "ppo_residual_final.zip"
    )
    for name in ("config.json", model_name):
        if not (args.residual_run_dir / name).exists():
            raise FileNotFoundError(args.residual_run_dir / name)


def _validate_run_config(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    """Reject a policy trained for a different observation/action interface.

    拒绝使用针对不同观测/动作接口训练的策略。
    """
    expected = {
        "algorithm": "residual_ppo",
        "scenario": args.scenario,
        "episode_hours": args.episode_hours,
        "forecast_context_hours": args.forecast_context_hours,
        "decision_interval_h": args.replan_hours,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Residual PPO configuration mismatch: {mismatches}"
        )


def _physical_config(args: argparse.Namespace) -> CCSEnvConfig:
    """Return the physical configuration shared by all controllers.

    返回所有控制器共享的物理配置。
    """
    return CCSEnvConfig(
        episode_hours=args.episode_hours,
        include_goal_obs=False,
        reward_mode="vent_first",
    )


def _physical_env(args: argparse.Namespace, scenario: Scenario):
    """Build an independent physical environment for one shared trajectory.

    为一条共享轨迹构建独立物理环境。
    """
    return build_phase1_env(
        scenario=args.scenario,
        scenario_generator=FixedScenarioGenerator(scenario),
        weather_mode=args.weather_mode,
        config=_physical_config(args),
    )


def _evaluate_seed(args, model, reward, goal, scenario, seed):
    """Evaluate selected controllers on copies of one scenario.

    在同一场景的副本上评估所选控制器。
    """
    rows: list[dict[str, Any]] = []
    standard = {}
    if "rule" in args.controllers:
        standard["rule"] = GoalAwareRuleExecutor()
    if "rollout_mpc" in args.controllers:
        standard["rollout_mpc"] = GoalAwareNativeMpcExecutor(
            planning_horizon_h=args.planning_horizon_hours
        )
    for name, executor in standard.items():
        metrics = evaluate_executor(
            _physical_env(args, scenario),
            executor,
            goal,
            seed=seed,
        )
        row = _standard_row(name, seed, metrics)
        rows.append(row)
        _print_row(row)

    if "residual_ppo" in args.controllers:
        residual_env = ResidualDispatchEnv(
            _physical_env(args, scenario),
            config=ResidualEnvConfig(
                decision_interval_h=args.replan_hours,
                event_triggered=True,
                reward=reward,
            ),
        )
        record = evaluate_seed(model, residual_env, seed)
        row = {
            "controller": "residual_ppo",
            "seed": seed,
            "captured_t": record["captured_t"],
            "stored_t": record["stored_t"],
            "vented_t": record["vented_t"],
            "storage_rate": record["storage_rate"],
            "vent_rate": (
                record["vented_t"] / record["captured_t"]
                if record["captured_t"] > 1e-9
                else 0.0
            ),
            "operating_cost_eur": record["operating_cost_eur"],
            "total_cost_eur": record["total_cost_eur"],
            "unit_total_cost_eur_per_t": record[
                "unit_total_cost_eur_per_t"
            ],
            "wall_clock_seconds": record["wall_clock_seconds"],
            "hard_violations": record["hard_violations"],
            "decisions": record["decisions"],
            "mean_decision_interval_h": record[
                "mean_decision_interval_h"
            ],
            "interventions_applied": record["interventions_applied"],
            "intervention_rate": record["intervention_rate"],
            "action_counts": json.dumps(record["actions"], sort_keys=True),
            "trigger_counts": json.dumps(
                record["triggers"],
                sort_keys=True,
            ),
        }
        rows.append(row)
        _print_row(row)
    return rows


def _standard_row(controller: str, seed: int, metrics) -> dict[str, Any]:
    """Convert rule/MPC metrics to the shared schema.

    将规则/MPC 指标转换为共享结构。
    """
    hard = sum(
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
        "wall_clock_seconds": metrics.wall_clock_seconds,
        "hard_violations": hard,
        "decisions": None,
        "mean_decision_interval_h": None,
        "interventions_applied": None,
        "intervention_rate": None,
        "action_counts": "{}",
        "trigger_counts": "{}",
    }


def _fingerprint(scenario: Scenario) -> dict[str, Any]:
    """Return compact proof of the pre-generated shared trajectory.

    返回预生成共享轨迹的紧凑证明。
    """
    return {
        "seed": scenario.seed,
        "n_steps": scenario.n_steps,
        "time_step_hours": scenario.time_step_hours,
        "emitter_series_lengths": {
            key: len(value)
            for key, value in scenario.emitter_availability.items()
        },
        "vessel_series_lengths": {
            key: len(value)
            for key, value in scenario.vessel_speed_factor.items()
        },
        "well_series_lengths": {
            key: len(value)
            for key, value in scenario.well_available.items()
        },
    }


def _print_row(row: dict[str, Any]) -> None:
    """Print one complete result line.

    输出一条完整结果。
    """
    print(
        f"seed={row['seed']} controller={row['controller']} "
        f"captured={float(row['captured_t']):,.1f} t "
        f"stored={float(row['stored_t']):,.1f} t "
        f"vented={float(row['vented_t']):,.1f} t "
        f"total_cost=EUR {float(row['total_cost_eur']):,.1f}",
        flush=True,
    )


def _summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute per-controller means and sample standard deviations.

    计算各控制器的均值与样本标准差。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["controller"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for controller, group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "controller": controller,
            "runs": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = [
                float(row[metric])
                for row in group
                if row[metric] is not None
            ]
            summary[f"{metric}_mean"] = (
                statistics.mean(values) if values else None
            )
            summary[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summaries.append(summary)
    return summaries


def _summary_columns() -> tuple[str, ...]:
    """Return stable summary columns.

    返回稳定的汇总列。
    """
    columns = ["controller", "runs"]
    for metric in SUMMARY_METRICS:
        columns.extend((f"{metric}_mean", f"{metric}_std"))
    return tuple(columns)


def _write_csv(path: Path, rows, columns) -> None:
    """Write one UTF-8 CSV file.

    写入一个 UTF-8 CSV 文件。
    """
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
