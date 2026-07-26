"""Quantify the value of the market against the operational rule.

以运营规则为基准,量化市场的价值。

The research question is *the importance of the market*, not proximity to any
centralized optimizer. The relevant baseline is therefore the balanced
operational rule (the realistic status quo), and the relevant axis is
contention / stress: the tighter and more disturbed the system, the more a
decentralized, urgency-priced market should beat a fixed rule -- using only
local information and near-zero compute.

研究问题是*市场的重要性*,而非与某个中心化优化器的接近程度。因此基准是平衡运营规则
(现实现状),坐标轴是竞争/压力:系统越紧张、扰动越大,一个去中心化、按紧迫度定价的
市场就越应胜过固定规则——而且只用局部信息、近乎零算力。

This script provides two market-centric studies:

  1. stress sweep   -- market vs rule as disturbance intensity rises;
  2. scenario sweep -- market vs rule across scenarios of differing contention.

Run / 运行:

    python -m algorithms.auction_market.value_of_market \
        --scenario northern_lights_phase1_milkrun_imbalanced --seeds 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np

from .bid_policy import SharedLinearBidPolicy
from .bidding import AuctionConfig
from .commitment import AdaptiveCommitmentConfig
from .policies import AuctionDispatchPolicy, rule_action
from .runner import build_env, rollout


def _paired_rule_vs_market(
    scenario: str,
    seed: int,
    hard_probability: float,
    args: argparse.Namespace,
    bid_model: SharedLinearBidPolicy | None,
) -> dict[str, dict[str, float]]:
    """Run rule and market on the identical scenario realisation.

    在完全相同的场景实现上运行规则与市场。
    """
    config = AuctionConfig(
        carbon_price_eur_per_t=args.carbon_price,
        bid_horizon_h=args.bid_horizon_h,
    )
    base = build_env(
        scenario=scenario,
        episode_hours=args.episode_hours,
        forecast_context_hours=args.forecast_context_hours,
        hard_scenario_probability=hard_probability,
    )
    base.reset(seed=int(seed))

    started = perf_counter()
    rule = rollout(deepcopy(base), rule_action)
    rule_wall = perf_counter() - started
    started = perf_counter()
    market = rollout(
        deepcopy(base),
        AuctionDispatchPolicy(
            config,
            bid_model=bid_model,
            commitment_margin=args.commitment_margin,
            defend_floor_t=args.defend_floor_t,
            adaptive_commitment=(
                AdaptiveCommitmentConfig(
                    floor_high_t=args.adaptive_floor_high_t,
                    floor_low_t=args.adaptive_floor_low_t,
                    spread_low=args.adaptive_spread_low,
                    spread_high=args.adaptive_spread_high,
                    fill_low=args.adaptive_fill_low,
                    fill_high=args.adaptive_fill_high,
                )
                if args.adaptive_commitment
                else None
            ),
            protect_horizon_h=args.protect_horizon_h,
        ),
    )
    market_wall = perf_counter() - started

    if abs(rule.captured_t - market.captured_t) > 1.0:
        raise AssertionError(
            f"{scenario} seed {seed}: capture mismatch "
            f"({rule.captured_t:.1f} vs {market.captured_t:.1f})."
        )
    return {
        "rule": {
            "vented_t": rule.vented_t,
            "stored_t": rule.stored_t,
            "total_cost_eur": rule.total_cost_eur,
            "wall_clock_seconds": rule_wall,
        },
        "market": {
            "vented_t": market.vented_t,
            "stored_t": market.stored_t,
            "total_cost_eur": market.total_cost_eur,
            "wall_clock_seconds": market_wall,
        },
    }


def _aggregate_cell(
    scenario: str,
    hard_probability: float,
    args: argparse.Namespace,
    bid_model: SharedLinearBidPolicy | None,
) -> dict[str, float]:
    """Average rule vs market over seeds and compute relative advantages.

    在若干 seed 上平均规则 vs 市场,并计算相对优势。
    """
    rule_vent, rule_cost, rule_wall = [], [], []
    market_vent, market_cost, market_wall = [], [], []
    for seed in args.seeds:
        paired = _paired_rule_vs_market(
            scenario, seed, hard_probability, args, bid_model
        )
        rule_vent.append(paired["rule"]["vented_t"])
        rule_cost.append(paired["rule"]["total_cost_eur"])
        rule_wall.append(paired["rule"]["wall_clock_seconds"])
        market_vent.append(paired["market"]["vented_t"])
        market_cost.append(paired["market"]["total_cost_eur"])
        market_wall.append(paired["market"]["wall_clock_seconds"])

    r_vent, m_vent = mean(rule_vent), mean(market_vent)
    r_cost, m_cost = mean(rule_cost), mean(market_cost)
    return {
        "rule_vented_t": r_vent,
        "market_vented_t": m_vent,
        "vent_reduction_pct": (r_vent - m_vent) / r_vent * 100.0 if r_vent > 1e-9 else 0.0,
        "rule_cost_eur": r_cost,
        "market_cost_eur": m_cost,
        "cost_reduction_pct": (r_cost - m_cost) / r_cost * 100.0 if r_cost > 1e-9 else 0.0,
        "rule_wall_seconds": mean(rule_wall),
        "market_wall_seconds": mean(market_wall),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the stress sweep and (optionally) the scenario sweep.

    运行压力扫描与(可选的)场景扫描。
    """
    bid_model = None
    if args.learned_params:
        params = np.load(args.learned_params)
        bid_model = SharedLinearBidPolicy(params)

    report: dict[str, Any] = {
        "seeds": list(args.seeds),
        "episode_hours": args.episode_hours,
        "learned_params": bool(args.learned_params),
        "stress_sweep": {"scenario": args.scenario, "cells": []},
        "scenario_sweep": {"cells": []},
    }

    for hard_probability in args.hard_probs:
        cell = _aggregate_cell(args.scenario, hard_probability, args, bid_model)
        cell["hard_probability"] = hard_probability
        report["stress_sweep"]["cells"].append(cell)

    for scenario in args.scenarios:
        cell = _aggregate_cell(scenario, 0.0, args, bid_model)
        cell["scenario"] = scenario
        report["scenario_sweep"]["cells"].append(cell)

    return report


def _print_report(report: dict[str, Any]) -> None:
    """Print the two market-centric studies.

    打印两个市场为中心的研究。
    """
    label = "learned" if report["learned_params"] else "myopic"
    print(f"\n=== Value of the market ({label} bidding) / 市场的价值 ===")
    print(f"baseline = balanced operational rule   seeds: {report['seeds']}")

    stress = report["stress_sweep"]
    print(f"\n-- stress sweep on {stress['scenario']} / 压力扫描 --")
    print(f"{'hard_prob':>10}{'rule_vent':>12}{'mkt_vent':>12}{'vent_red%':>11}{'cost_red%':>11}")
    print("-" * 56)
    for cell in stress["cells"]:
        print(
            f"{cell['hard_probability']:>10.2f}{cell['rule_vented_t']:>12.0f}"
            f"{cell['market_vented_t']:>12.0f}{cell['vent_reduction_pct']:>11.1f}"
            f"{cell['cost_reduction_pct']:>11.1f}"
        )

    if report["scenario_sweep"]["cells"]:
        print("\n-- scenario sweep (normal difficulty) / 场景扫描 --")
        print(f"{'scenario':<44}{'vent_red%':>11}{'cost_red%':>11}")
        print("-" * 66)
        for cell in report["scenario_sweep"]["cells"]:
            print(
                f"{cell['scenario']:<44}{cell['vent_reduction_pct']:>11.1f}"
                f"{cell['cost_reduction_pct']:>11.1f}"
            )

    any_cell = report["stress_sweep"]["cells"][0]
    print(
        f"\ncompute: market {any_cell['market_wall_seconds']:.2f}s vs "
        f"rule {any_cell['rule_wall_seconds']:.2f}s per episode "
        f"(market uses only local emitter info; no central planner)."
    )
    print("(positive % = market beats the rule)\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the value-of-market command-line parser.

    构建市场价值命令行解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="northern_lights_phase1_milkrun_imbalanced",
        help="Scenario for the stress sweep. / 压力扫描所用场景。",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=[
            "northern_lights_phase1_3vessels",
            "northern_lights_phase1_milkrun",
            "northern_lights_phase1_milkrun_imbalanced",
            "milk_run_stress",
        ],
        help="Scenarios for the contention sweep. / 竞争扫描所用场景。",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument(
        "--hard-probs",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="Disturbance intensity levels. / 扰动强度水平。",
    )
    parser.add_argument("--carbon-price", type=float, default=80.0)
    parser.add_argument("--bid-horizon-h", type=float, default=48.0)
    parser.add_argument(
        "--commitment-margin",
        type=float,
        default=0.0,
        help=(
            "Margin by which a bid must beat the rule's own destination before "
            "the vessel is reassigned; 0 = purely reactive. / 竞价需超过规则目的地"
            "出价的边际倍数才可改派;0 = 纯反应式。"
        ),
    )
    parser.add_argument(
        "--defend-floor-t",
        type=float,
        default=0.0,
        help=(
            "Tonnes of option value defending the rule's destination, closing "
            "the zero-bid starvation loophole. / 用于防守规则目的地的期权价值吨数,"
            "堵住零出价饿死漏洞。"
        ),
    )
    parser.add_argument(
        "--protect-horizon-h",
        type=float,
        default=0.0,
        help=(
            "Never raid a rule destination that would overflow within this many "
            "hours. / 若规则目的地会在该小时数内溢出,则永不抢走其船。"
        ),
    )
    parser.add_argument(
        "--adaptive-commitment",
        action="store_true",
        help=(
            "Derive the defend floor from the runtime fill spread instead of a "
            "constant. / 用运行时填充极差导出防守下限,而非使用常数。"
        ),
    )
    parser.add_argument("--adaptive-floor-high-t", type=float, default=2000.0)
    parser.add_argument("--adaptive-floor-low-t", type=float, default=50.0)
    parser.add_argument("--adaptive-spread-low", type=float, default=0.10)
    parser.add_argument("--adaptive-spread-high", type=float, default=0.40)
    parser.add_argument("--adaptive-fill-low", type=float, default=0.50)
    parser.add_argument("--adaptive-fill-high", type=float, default=0.85)
    parser.add_argument(
        "--learned-params",
        default=None,
        help="Optional .npy of learned bid weights. / 可选的学习竞价权重 .npy。",
    )
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    """Parse arguments, run the studies, print and write results.

    解析参数、运行研究、打印并写出结果。
    """
    args = build_parser().parse_args()
    report = run(args)
    _print_report(report)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("output")
        / "auction_market_value"
        / f"{args.scenario}__{args.episode_hours}h__seeds{min(args.seeds)}-{max(args.seeds)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "value_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
