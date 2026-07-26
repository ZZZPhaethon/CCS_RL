"""Strategic bidding equilibrium and Price of Anarchy for the shuttle auction.

运力拍卖的策略性竞价均衡与无序代价(Price of Anarchy)。

Each emitter is a self-interested bidder that scales its *true* value (projected
vent x carbon price) by a private *shade* factor before submitting. Under a
first-price rule a winner pays its own bid, so shading is rational; under a
second-price rule truthful bidding (shade = 1) is a dominant strategy and the
allocation is efficient. This module:

  1. learns an approximate pure-strategy Nash equilibrium of the shade factors
     under first-price, by iterated best response (each emitter minimises its own
     cost = vent cost + payments, holding the others fixed);
  2. compares the resulting decentralised market against the efficient truthful
     benchmark and the balanced rule, and reports the Price of Anarchy.

每个排放源是自私竞标者,提交前把*真实*价值(预计放空 x 碳价)乘以一个私有*压价*系数。
一价规则下赢家付自身出价,压价是理性的;二价规则下如实出价(系数=1)是占优策略且分配
有效率。本模块:(1) 用迭代最优反应学习一价下压价系数的近似纯策略纳什均衡(每个排放源在
其他源固定时最小化自身成本 = 放空成本 + 支付);(2) 将去中心化市场与如实基准及平衡规则
对比,报告无序代价。

Run / 运行:

    python -m algorithms.auction_market.equilibrium \
        --scenario northern_lights_phase1_milkrun_imbalanced --seeds 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .bidding import AuctionConfig
from .policies import AuctionDispatchPolicy, rule_action
from .runner import build_env, rollout


def _emitter_ids(args: argparse.Namespace) -> list[str]:
    """Return the scenario's emitter identifiers.

    返回场景的排放源标识符。
    """
    env = build_env(
        scenario=args.scenario,
        episode_hours=args.episode_hours,
        forecast_context_hours=args.forecast_context_hours,
    )
    env.reset(seed=int(args.seeds[0]))
    return list(env.emitter_ids)


def market_outcome(
    shades: dict[str, float],
    payment_rule: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Average one bidding profile over the seeds under a payment rule.

    在给定支付规则下,对若干 seed 平均一个竞价组合的结果。
    """
    config = AuctionConfig(
        carbon_price_eur_per_t=args.carbon_price,
        bid_horizon_h=args.bid_horizon_h,
    )
    social_cost: list[float] = []
    vented: list[float] = []
    payoff: dict[str, list[float]] = {}
    vent_by: dict[str, list[float]] = {}
    pay_by: dict[str, list[float]] = {}
    for seed in args.seeds:
        env = build_env(
            scenario=args.scenario,
            episode_hours=args.episode_hours,
            forecast_context_hours=args.forecast_context_hours,
        )
        env.reset(seed=int(seed))
        policy = AuctionDispatchPolicy(
            config,
            bid_shades=shades,
            payment_rule=payment_rule,
        )
        result = rollout(env, policy)
        social_cost.append(result.total_cost_eur)
        vented.append(result.vented_t)
        for emitter_id in env.emitter_ids:
            vent_t = float(result.per_emitter_vented_t.get(emitter_id, 0.0))
            payment = float(policy.payments.get(emitter_id, 0.0))
            vent_cost = args.carbon_price * vent_t
            payoff.setdefault(emitter_id, []).append(vent_cost + payment)
            vent_by.setdefault(emitter_id, []).append(vent_t)
            pay_by.setdefault(emitter_id, []).append(payment)
    return {
        "social_cost_eur": mean(social_cost),
        "vented_t": mean(vented),
        "emitter_payoff": {e: mean(v) for e, v in payoff.items()},
        "emitter_vented_t": {e: mean(v) for e, v in vent_by.items()},
        "emitter_payment_eur": {e: mean(v) for e, v in pay_by.items()},
    }


def rule_outcome(args: argparse.Namespace) -> dict[str, Any]:
    """Average the balanced rule (no market) over the seeds.

    对若干 seed 平均平衡规则(无市场)的结果。
    """
    social_cost: list[float] = []
    vented: list[float] = []
    payoff: dict[str, list[float]] = {}
    for seed in args.seeds:
        env = build_env(
            scenario=args.scenario,
            episode_hours=args.episode_hours,
            forecast_context_hours=args.forecast_context_hours,
        )
        env.reset(seed=int(seed))
        result = rollout(env, rule_action)
        social_cost.append(result.total_cost_eur)
        vented.append(result.vented_t)
        for emitter_id in env.emitter_ids:
            vent_t = float(result.per_emitter_vented_t.get(emitter_id, 0.0))
            payoff.setdefault(emitter_id, []).append(args.carbon_price * vent_t)
    return {
        "social_cost_eur": mean(social_cost),
        "vented_t": mean(vented),
        "emitter_payoff": {e: mean(v) for e, v in payoff.items()},
    }


def iterated_best_response(
    emitter_ids: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Learn first-price shades by iterated best response toward a Nash point.

    通过迭代最优反应学习一价压价系数,逼近纳什点。
    """
    grid = list(args.shade_grid)
    shades = {emitter_id: 1.0 for emitter_id in emitter_ids}
    history: list[dict[str, Any]] = []
    for round_index in range(args.br_rounds):
        changed = False
        for emitter_id in emitter_ids:
            candidates = sorted(set(grid) | {shades[emitter_id]})
            best_shade = shades[emitter_id]
            best_payoff = float("inf")
            for shade in candidates:
                trial = dict(shades)
                trial[emitter_id] = shade
                payoff = market_outcome(trial, "first_price", args)[
                    "emitter_payoff"
                ][emitter_id]
                if payoff < best_payoff - 1e-6:
                    best_payoff = payoff
                    best_shade = shade
            if abs(best_shade - shades[emitter_id]) > 1e-9:
                shades[emitter_id] = best_shade
                changed = True
            history.append(
                {
                    "round": round_index,
                    "emitter": emitter_id,
                    "shade": best_shade,
                    "payoff_eur": best_payoff,
                }
            )
        if not changed:
            break
    return shades, history


def _print_report(report: dict[str, Any]) -> None:
    """Print the mechanism comparison and Price of Anarchy.

    打印机制对比与无序代价。
    """
    print("\n=== Bidding equilibrium & Price of Anarchy / 竞价均衡与无序代价 ===")
    print(f"scenario: {report['scenario']}   seeds: {report['seeds']}\n")
    print(f"{'mechanism':<32}{'social_cost_eur':>18}{'vented_t':>14}")
    print("-" * 64)
    rows = [
        ("rule (balanced, no market)", report["rule"]),
        ("truthful 2nd-price (efficient)", report["truthful_second_price"]),
        ("truthful 1st-price", report["truthful_first_price"]),
        ("1st-price NASH equilibrium", report["equilibrium"]),
    ]
    for label, outcome in rows:
        print(
            f"{label:<32}{outcome['social_cost_eur']:>18.1f}"
            f"{outcome['vented_t']:>14.1f}"
        )
    print("-" * 64)
    print(
        f"Price of Anarchy (social cost) = "
        f"{report['price_of_anarchy_cost']:.4f}   "
        f"(vented) = {report['price_of_anarchy_vent']:.4f}"
    )
    print("\nlearned first-price shades / 学到的一价压价系数:")
    for emitter_id, shade in report["equilibrium_shades"].items():
        print(f"  {emitter_id:<20} shade = {shade:.3f}")
    print("\nper-emitter payoff (cost, lower=better) / 逐源支付成本:")
    print(f"{'emitter':<20}{'truthful-2p':>16}{'nash-1p':>16}")
    eq = report["equilibrium"]["emitter_payoff"]
    tp = report["truthful_second_price"]["emitter_payoff"]
    for emitter_id in report["equilibrium_shades"]:
        print(
            f"{emitter_id:<20}{tp.get(emitter_id, 0.0):>16.1f}"
            f"{eq.get(emitter_id, 0.0):>16.1f}"
        )
    print()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full equilibrium and Price-of-Anarchy study.

    运行完整的均衡与无序代价研究。
    """
    emitter_ids = _emitter_ids(args)
    rule = rule_outcome(args)
    truthful_2p = market_outcome(
        {e: 1.0 for e in emitter_ids}, "second_price", args
    )
    truthful_1p = market_outcome(
        {e: 1.0 for e in emitter_ids}, "first_price", args
    )
    eq_shades, history = iterated_best_response(emitter_ids, args)
    equilibrium = market_outcome(eq_shades, "first_price", args)

    efficient_cost = truthful_2p["social_cost_eur"]
    efficient_vent = truthful_2p["vented_t"]
    return {
        "scenario": args.scenario,
        "seeds": list(args.seeds),
        "episode_hours": args.episode_hours,
        "carbon_price_eur_per_t": args.carbon_price,
        "bid_horizon_h": args.bid_horizon_h,
        "shade_grid": list(args.shade_grid),
        "rule": rule,
        "truthful_second_price": truthful_2p,
        "truthful_first_price": truthful_1p,
        "equilibrium": equilibrium,
        "equilibrium_shades": eq_shades,
        "best_response_history": history,
        "price_of_anarchy_cost": (
            equilibrium["social_cost_eur"] / efficient_cost
            if efficient_cost > 1e-9
            else float("nan")
        ),
        "price_of_anarchy_vent": (
            equilibrium["vented_t"] / efficient_vent
            if efficient_vent > 1e-9
            else float("nan")
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the equilibrium command-line parser.

    构建均衡命令行解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="northern_lights_phase1_milkrun_imbalanced",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--episode-hours", type=int, default=360)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument("--carbon-price", type=float, default=80.0)
    parser.add_argument("--bid-horizon-h", type=float, default=48.0)
    parser.add_argument(
        "--shade-grid",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.5],
    )
    parser.add_argument("--br-rounds", type=int, default=3)
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
        / "auction_market_equilibrium"
        / f"{args.scenario}__{args.episode_hours}h__seeds{min(args.seeds)}-{max(args.seeds)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "equilibrium_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
