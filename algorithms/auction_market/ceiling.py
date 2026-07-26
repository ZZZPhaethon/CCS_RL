"""How far is the decentralized market from the centralized optimum?

去中心化市场离中心化最优还有多远?

The auction market beats the balanced rule, and its first-price equilibrium is
efficient *within the market*. This study adds the missing yardstick: the
centralized, look-ahead controllers from :mod:`algorithms.hybrid` (native MPC,
optionally rolling MILP). Every controller is run on the *same* scenario
realisation (same env factory + seed) so realised stored / vented / cost are
directly comparable, and the market's efficiency gap to the ceiling is reported.

拍卖市场胜过平衡规则,且其一价均衡在*市场内*有效率。本研究补上缺失的标尺:
:mod:`algorithms.hybrid` 的中心化、带预见的控制器(原生 MPC,可选滚动 MILP)。
所有控制器都在*同一*场景实现(相同环境工厂 + seed)上运行,使实际的封存/放空/成本
可直接比较,并报告市场相对天花板的效率差距。

Run / 运行:

    python -m algorithms.auction_market.ceiling \
        --scenario northern_lights_phase1_milkrun_imbalanced --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from algorithms import DispatchGoal, GoalAwareNativeMpcExecutor, evaluate_executor
from Simulation.control.baselines import balanced_capture_assignment

from .bidding import AuctionConfig
from .policies import AuctionDispatchPolicy, rule_action
from .runner import build_env, rollout


def _decentralized_metrics(
    policy_name: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Run the rule or auction (per-hour policy) and return realised metrics.

    运行规则或拍卖(逐小时策略)并返回实际指标。
    """
    config = AuctionConfig(
        carbon_price_eur_per_t=args.carbon_price,
        bid_horizon_h=args.bid_horizon_h,
    )
    policy = (
        rule_action
        if policy_name == "rule"
        else AuctionDispatchPolicy(config)
    )
    env = build_env(
        scenario=args.scenario,
        episode_hours=args.episode_hours,
        forecast_context_hours=args.forecast_context_hours,
    )
    env.reset(seed=int(seed))
    started = perf_counter()
    result = rollout(env, policy)
    return {
        "captured_t": result.captured_t,
        "stored_t": result.stored_t,
        "vented_t": result.vented_t,
        "total_cost_eur": result.total_cost_eur,
        "wall_clock_seconds": perf_counter() - started,
    }


def _centralized_metrics(
    controller: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Run a centralized look-ahead executor on the identical scenario.

    在完全相同的场景上运行一个中心化、带预见的执行器。
    """
    probe = build_env(
        scenario=args.scenario,
        episode_hours=args.episode_hours,
        forecast_context_hours=args.forecast_context_hours,
    )
    goal = DispatchGoal(
        emitter_to_vessel=balanced_capture_assignment(probe),
        replan_after_h=args.replan_hours,
        rationale="Balanced initial assignment for the centralized ceiling.",
    )
    if controller == "native_mpc":
        executor = GoalAwareNativeMpcExecutor(
            planning_horizon_h=args.planning_horizon_hours
        )
    elif controller == "rolling_milp":
        from algorithms import RollingMilpExecutor

        executor = RollingMilpExecutor(
            planning_horizon_h=args.planning_horizon_hours,
            time_limit_s=args.milp_time_limit_seconds,
            solver=args.milp_solver,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown centralized controller: {controller}")
    env = build_env(
        scenario=args.scenario,
        episode_hours=args.episode_hours,
        forecast_context_hours=args.forecast_context_hours,
    )
    metrics = evaluate_executor(env, executor, goal, seed=int(seed))
    return {
        "captured_t": metrics.captured_t,
        "stored_t": metrics.stored_t,
        "vented_t": metrics.vented_t,
        "total_cost_eur": metrics.total_cost,
        "wall_clock_seconds": metrics.wall_clock_seconds,
    }


def _controller_metrics(
    controller: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Dispatch to the decentralized or centralized runner.

    分派到去中心化或中心化的运行器。
    """
    if controller in {"rule", "auction"}:
        return _decentralized_metrics(controller, seed, args)
    return _centralized_metrics(controller, seed, args)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run every controller on every seed and summarise the efficiency gap.

    在每个 seed 上运行每个控制器,并汇总效率差距。
    """
    per_seed: dict[str, list[dict[str, float]]] = {c: [] for c in args.controllers}
    for seed in args.seeds:
        captured_ref: float | None = None
        for controller in args.controllers:
            metrics = _controller_metrics(controller, seed, args)
            per_seed[controller].append(metrics)
            if captured_ref is None:
                captured_ref = metrics["captured_t"]
            elif abs(metrics["captured_t"] - captured_ref) > 1.0:
                raise AssertionError(
                    f"seed {seed}: {controller} saw different capture "
                    f"({metrics['captured_t']:.1f} vs {captured_ref:.1f})."
                )
            print(
                f"seed={seed} {controller:<12} stored={metrics['stored_t']:>10.1f} "
                f"vented={metrics['vented_t']:>10.1f} "
                f"cost={metrics['total_cost_eur']:>12.1f} "
                f"wall={metrics['wall_clock_seconds']:.1f}s"
            )

    summary = {
        controller: {
            "stored_t": mean(m["stored_t"] for m in records),
            "vented_t": mean(m["vented_t"] for m in records),
            "total_cost_eur": mean(m["total_cost_eur"] for m in records),
            "wall_clock_seconds": mean(m["wall_clock_seconds"] for m in records),
        }
        for controller, records in per_seed.items()
    }

    report: dict[str, Any] = {
        "scenario": args.scenario,
        "seeds": list(args.seeds),
        "episode_hours": args.episode_hours,
        "planning_horizon_hours": args.planning_horizon_hours,
        "replan_hours": args.replan_hours,
        "summary": summary,
        "per_seed": per_seed,
    }

    # Efficiency gap: how much of the rule->ceiling improvement the market captures.
    # 效率差距:市场捕获了规则->天花板改善的多少。
    ceiling = "rolling_milp" if "rolling_milp" in summary else "native_mpc"
    if {"rule", "auction", ceiling} <= set(summary):
        rule_c = summary["rule"]["total_cost_eur"]
        auc_c = summary["auction"]["total_cost_eur"]
        ceil_c = summary[ceiling]["total_cost_eur"]
        gain_possible = rule_c - ceil_c
        report["ceiling_controller"] = ceiling
        report["market_gap_to_ceiling_eur"] = auc_c - ceil_c
        report["market_gap_to_ceiling_pct"] = (
            (auc_c - ceil_c) / ceil_c * 100.0 if ceil_c > 1e-9 else float("nan")
        )
        report["market_capture_of_possible_gain"] = (
            (rule_c - auc_c) / gain_possible if abs(gain_possible) > 1e-9 else float("nan")
        )
    return report


def _print_report(report: dict[str, Any]) -> None:
    """Print the ceiling comparison and efficiency gap.

    打印天花板对比与效率差距。
    """
    print("\n=== Market vs centralized ceiling / 市场 vs 中心化天花板 ===")
    print(f"scenario: {report['scenario']}   seeds: {report['seeds']}\n")
    print(f"{'controller':<14}{'stored_t':>12}{'vented_t':>12}{'total_cost_eur':>16}{'wall_s':>9}")
    print("-" * 63)
    for controller, values in report["summary"].items():
        print(
            f"{controller:<14}{values['stored_t']:>12.1f}{values['vented_t']:>12.1f}"
            f"{values['total_cost_eur']:>16.1f}{values['wall_clock_seconds']:>9.1f}"
        )
    print("-" * 63)
    if "market_gap_to_ceiling_pct" in report:
        print(
            f"ceiling = {report['ceiling_controller']}   "
            f"market gap to ceiling = {report['market_gap_to_ceiling_pct']:+.2f}% cost   "
            f"market captured {report['market_capture_of_possible_gain'] * 100:.1f}% "
            f"of the rule->ceiling gain"
        )
    print()


def build_parser() -> argparse.ArgumentParser:
    """Build the ceiling command-line parser.

    构建天花板命令行解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="northern_lights_phase1_milkrun_imbalanced",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--episode-hours", type=int, default=360)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["rule", "auction", "native_mpc"],
        choices=["rule", "auction", "native_mpc", "rolling_milp"],
    )
    parser.add_argument("--replan-hours", type=float, default=24.0)
    parser.add_argument("--planning-horizon-hours", type=int, default=72)
    parser.add_argument("--carbon-price", type=float, default=80.0)
    parser.add_argument("--bid-horizon-h", type=float, default=48.0)
    parser.add_argument("--milp-time-limit-seconds", type=float, default=30.0)
    parser.add_argument(
        "--milp-solver",
        choices=("cbc", "cplex", "cplex_native"),
        default="cbc",
    )
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    """Parse arguments, run the study, print and write results.

    解析参数、运行研究、打印并写出结果。
    """
    args = build_parser().parse_args()
    report = run(args)
    _print_report(report)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("output")
        / "auction_market_ceiling"
        / f"{args.scenario}__{args.episode_hours}h__seeds{min(args.seeds)}-{max(args.seeds)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ceiling_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
