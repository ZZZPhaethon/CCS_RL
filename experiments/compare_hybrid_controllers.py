"""Compare rule and native-MPC executors through identical physical rollouts.

在完全一致的物理回放中比较规则与原生 MPC 执行器。

Example / 示例::

    python experiments/compare_hybrid_controllers.py \
        --scenario northern_lights_phase1_3vessels \
        --seeds 1 2 3 4 5 --episode-hours 168

The script reports realised physical outcomes, rather than a controller's
planned objective. It can run before any RL model has been trained.

脚本报告实际物理结果，而不是控制器的计划目标值。它可在训练 RL 前直接运行。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import (  # noqa: E402
    DispatchGoal,
    GoalAwareNativeMpcExecutor,
    GoalAwareRuleExecutor,
    RolloutMetrics,
    RollingMilpExecutor,
    compare_executors,
)
from Simulation.control.baselines import balanced_capture_assignment  # noqa: E402
from Simulation.environment import CCSEnvConfig, build_phase1_env  # noqa: E402


RAW_COLUMNS = (
    "controller",
    "seed",
    "elapsed_hours",
    "captured_t",
    "stored_t",
    "vented_t",
    "operating_cost",
    "total_cost",
    "total_reward",
    "unit_storage_cost_eur_per_t",
    "storage_rate",
    "vent_rate",
    "wall_clock_seconds",
    "violation_count",
    "violation_counts",
)
SUMMARY_METRICS = (
    "captured_t",
    "stored_t",
    "vented_t",
    "operating_cost",
    "total_cost",
    "total_reward",
    "unit_storage_cost_eur_per_t",
    "storage_rate",
    "vent_rate",
    "wall_clock_seconds",
    "violation_count",
)


def parse_args() -> argparse.Namespace:
    """Parse reproducible comparison settings from the command line.

    从命令行解析可复现比较的设置。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="northern_lights_phase1_3vessels",
        help="Fixed scenario identifier. / 固定场景标识符。",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Scenario seeds. / 场景随机种子。",
    )
    parser.add_argument(
        "--episode-hours",
        type=int,
        default=168,
        help="Physical rollout length in hours. / 物理回放时长（小时）。",
    )
    parser.add_argument(
        "--replan-hours",
        type=float,
        default=24.0,
        help="High-level and MPC replan period. / 高层与 MPC 重规划周期。",
    )
    parser.add_argument(
        "--planning-horizon-hours",
        type=int,
        default=72,
        help="Native MPC forecast horizon. / 原生 MPC 预见时域。",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=("rule", "native_mpc", "rolling_milp"),
        default=("rule", "native_mpc"),
        help="Controllers to compare. / 需比较的控制器。",
    )
    parser.add_argument(
        "--milp-time-limit-seconds",
        type=float,
        default=30.0,
        help="Per-replan MILP solve limit. / 每次 MILP 重规划的求解时间上限。",
    )
    parser.add_argument(
        "--milp-solver",
        choices=("cbc", "cplex", "cplex_native"),
        default="cbc",
        help="Rolling MILP solver backend. / 滚动 MILP 求解后端。",
    )
    parser.add_argument(
        "--weather-mode",
        choices=("window", "block"),
        default="window",
        help="Built-in stochastic weather model. / 内置随机天气模型。",
    )
    parser.add_argument(
        "--objective-mode",
        choices=("economic", "vent_first"),
        default="vent_first",
        help=(
            "Shared environment/MILP objective; vent_first is recommended for "
            "storage-first comparison. / 共享的环境/MILP 目标；以封存为主的比较建议使用 vent_first。"
        ),
    )
    parser.add_argument(
        "--vent-first-vent-eur-per-t",
        type=float,
        default=10_000.0,
        help="Venting priority weight for vent_first. / vent_first 模式中的放空优先级权重。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/hybrid_controller_comparison"),
        help="Directory for CSV/JSON results. / 保存 CSV/JSON 结果的目录。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing prior files in output-dir. / 允许覆盖已有输出文件。",
    )
    return parser.parse_args()


def main() -> int:
    """Run all requested seeds and write raw plus aggregated controller metrics.

    运行所有指定种子，并写入原始与汇总控制器指标。
    """
    args = parse_args()
    if args.episode_hours <= 0:
        raise ValueError("episode-hours must be positive.")
    if args.replan_hours <= 0.0:
        raise ValueError("replan-hours must be positive.")

    environment_factory = _environment_factory(args)
    probe = environment_factory()
    goal = DispatchGoal(
        emitter_to_vessel=balanced_capture_assignment(probe),
        replan_after_h=args.replan_hours,
        rationale="Balanced initial vessel assignment for controller comparison.",
    )
    output_paths = _output_paths(args.output_dir)
    _guard_output_paths(output_paths, args.overwrite)
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        executors = _make_executors(args)
        results = compare_executors(
            environment_factory,
            executors,
            goal,
            seed=seed,
        )
        rows.extend(
            _metric_row(name, seed, metrics)
            for name, metrics in results.items()
        )
        _print_seed_summary(seed, results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_paths["raw"], rows, RAW_COLUMNS)
    _write_csv(output_paths["summary"], _summarise(rows), _summary_columns())
    output_paths["metadata"].write_text(
        json.dumps(
            {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "dispatch_goal": {
                    "emitter_to_vessel": dict(goal.emitter_to_vessel),
                    "well_rate_indices": dict(goal.well_rate_indices),
                    "replan_after_h": goal.replan_after_h,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Raw metrics: {output_paths['raw']}")
    print(f"Summary metrics: {output_paths['summary']}")
    print(f"Run metadata: {output_paths['metadata']}")
    return 0


def _environment_factory(args: argparse.Namespace):
    """Return fresh, identically configured environments for fair rollouts.

    返回用于公平回放的、每次均为新建且配置一致的环境工厂。
    """
    config = CCSEnvConfig(
        episode_hours=args.episode_hours,
        reward_mode=args.objective_mode,
        vent_first_vent_eur_per_t=args.vent_first_vent_eur_per_t,
    )

    def factory():
        return build_phase1_env(
            scenario=args.scenario,
            config=config,
            weather_mode=args.weather_mode,
        )

    return factory


def _make_executors(args: argparse.Namespace) -> dict[str, object]:
    """Create fresh controller instances for one seed.

    为一个随机种子创建新的控制器实例。
    """
    executors: dict[str, object] = {}
    for name in args.controllers:
        if name == "rule":
            executors[name] = GoalAwareRuleExecutor()
        elif name == "native_mpc":
            executors[name] = GoalAwareNativeMpcExecutor(
                planning_horizon_h=args.planning_horizon_hours
            )
        elif name == "rolling_milp":
            executors[name] = RollingMilpExecutor(
                planning_horizon_h=args.planning_horizon_hours,
                time_limit_s=args.milp_time_limit_seconds,
                solver=args.milp_solver,
            )
        else:  # pragma: no cover - argparse limits these values.
            raise ValueError(f"Unknown controller: {name}")
    return executors


def _metric_row(
    controller: str,
    seed: int,
    metrics: RolloutMetrics,
) -> dict[str, object]:
    """Flatten one rollout into a serialisable CSV row.

    将单次回放展平为可序列化的 CSV 行。
    """
    return {
        "controller": controller,
        "seed": seed,
        "elapsed_hours": metrics.elapsed_hours,
        "captured_t": metrics.captured_t,
        "stored_t": metrics.stored_t,
        "vented_t": metrics.vented_t,
        "operating_cost": metrics.operating_cost,
        "total_cost": metrics.total_cost,
        "total_reward": metrics.total_reward,
        "unit_storage_cost_eur_per_t": metrics.unit_storage_cost_eur_per_t,
        "storage_rate": metrics.storage_rate,
        "vent_rate": metrics.vent_rate,
        "wall_clock_seconds": metrics.wall_clock_seconds,
        "violation_count": sum(metrics.violation_counts.values()),
        "violation_counts": json.dumps(metrics.violation_counts, sort_keys=True),
    }


def _summarise(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Compute mean and sample standard deviation by controller.

    按控制器计算均值与样本标准差。
    """
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["controller"]), []).append(row)

    summaries: list[dict[str, object]] = []
    for controller, controller_rows in sorted(grouped.items()):
        summary: dict[str, object] = {"controller": controller, "runs": len(controller_rows)}
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in controller_rows if row[metric] is not None]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else None
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def _summary_columns() -> tuple[str, ...]:
    """Return the stable column order for summary CSV output.

    返回汇总 CSV 输出的稳定列顺序。
    """
    columns = ["controller", "runs"]
    for metric in SUMMARY_METRICS:
        columns.extend((f"{metric}_mean", f"{metric}_std"))
    return tuple(columns)


def _output_paths(output_dir: Path) -> dict[str, Path]:
    """Define the three reproducible result paths for one comparison run.

    定义单次比较的三个可复现结果路径。
    """
    return {
        "raw": output_dir / "comparison_raw.csv",
        "summary": output_dir / "comparison_summary.csv",
        "metadata": output_dir / "comparison_metadata.json",
    }


def _guard_output_paths(paths: dict[str, Path], overwrite: bool) -> None:
    """Avoid silently replacing a previous experiment result.

    避免静默覆盖之前的实验结果。
    """
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {joined}. Use --overwrite to replace it.")


def _write_csv(path: Path, rows: Iterable[dict[str, object]], columns: Iterable[str]) -> None:
    """Write UTF-8 CSV rows using a caller-provided stable column order.

    使用调用方指定的稳定列顺序写入 UTF-8 CSV 行。
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _print_seed_summary(seed: int, results: dict[str, RolloutMetrics]) -> None:
    """Print a compact realised-outcome summary while a batch run progresses.

    在批量运行过程中打印精简的实际结果摘要。
    """
    for name, metrics in results.items():
        unit_cost = metrics.unit_storage_cost_eur_per_t
        unit_cost_text = "n/a" if unit_cost is None else f"{unit_cost:,.2f} EUR/t"
        print(
            f"seed={seed} controller={name} stored={metrics.stored_t:,.1f} t "
            f"vented={metrics.vented_t:,.1f} t total_cost={metrics.total_cost:,.1f} "
            f"unit_cost={unit_cost_text} wall={metrics.wall_clock_seconds:.2f} s"
        )


if __name__ == "__main__":
    raise SystemExit(main())
