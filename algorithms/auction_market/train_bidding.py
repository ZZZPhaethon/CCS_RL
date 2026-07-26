"""Learn a shared state-conditioned bidding policy to close the ceiling gap.

学习一个共享的、基于状态的竞价策略,以缩小与天花板的差距。

The myopic auction bids the projected vent over a fixed window. This module
learns a shared valuation ``exp(w . local_features)`` that reweights forward-
looking signals (weather, reachability, competitor congestion) so the market's
allocation moves closer to the centralized MPC. Optimisation is by the
cross-entropy method (CEM): derivative-free, deterministic per (params, seed),
and reuses the auction rollout directly -- no gym/PPO machinery required.

近视拍卖按固定窗口的预计放空出价。本模块学习共享估值 ``exp(w . 局部特征)``,重加权
前瞻信号(天气、可达性、对手拥塞),使市场分配更靠近中心化 MPC。优化采用交叉熵方法
(CEM):无梯度、对 (参数, seed) 确定,且直接复用拍卖回放——不需要 gym/PPO。

Run / 运行:

    python -m algorithms.auction_market.train_bidding \
        --scenario northern_lights_phase1_milkrun_imbalanced
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .bid_policy import SharedLinearBidPolicy
from .bidding import AuctionConfig
from .ceiling import _centralized_metrics
from .features import FEATURE_NAMES
from .policies import AuctionDispatchPolicy, rule_action
from .runner import build_env, rollout


def _market_metrics(
    bid_model: SharedLinearBidPolicy | None,
    seeds: list[int],
    args: argparse.Namespace,
) -> dict[str, float]:
    """Average the auction market (given a bid model) over seeds.

    在若干 seed 上平均拍卖市场(给定竞价模型)的结果。
    """
    config = AuctionConfig(
        carbon_price_eur_per_t=args.carbon_price,
        bid_horizon_h=args.bid_horizon_h,
    )
    social_cost: list[float] = []
    vented: list[float] = []
    stored: list[float] = []
    for seed in seeds:
        env = build_env(
            scenario=args.scenario,
            episode_hours=args.episode_hours,
            forecast_context_hours=args.forecast_context_hours,
        )
        env.reset(seed=int(seed))
        policy = AuctionDispatchPolicy(config, bid_model=bid_model)
        result = rollout(env, policy)
        social_cost.append(result.total_cost_eur)
        vented.append(result.vented_t)
        stored.append(result.stored_t)
    return {
        "social_cost_eur": mean(social_cost),
        "vented_t": mean(vented),
        "stored_t": mean(stored),
    }


def _rule_metrics(seeds: list[int], args: argparse.Namespace) -> dict[str, float]:
    """Average the balanced rule (no market) over seeds.

    在若干 seed 上平均平衡规则(无市场)的结果。
    """
    social_cost: list[float] = []
    vented: list[float] = []
    stored: list[float] = []
    for seed in seeds:
        env = build_env(
            scenario=args.scenario,
            episode_hours=args.episode_hours,
            forecast_context_hours=args.forecast_context_hours,
        )
        env.reset(seed=int(seed))
        result = rollout(env, rule_action)
        social_cost.append(result.total_cost_eur)
        vented.append(result.vented_t)
        stored.append(result.stored_t)
    return {
        "social_cost_eur": mean(social_cost),
        "vented_t": mean(vented),
        "stored_t": mean(stored),
    }


def train_cem(args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Optimise the shared bidding weights by the cross-entropy method.

    用交叉熵方法优化共享竞价权重。
    """
    rng = np.random.default_rng(args.cem_seed)
    dim = len(FEATURE_NAMES)
    center = np.zeros(dim)
    std = np.full(dim, args.init_std)
    best_val = float("inf")
    best_params = center.copy()
    history: list[dict[str, Any]] = []

    for iteration in range(args.cem_iters):
        population = [center.copy()] + [
            center + std * rng.standard_normal(dim)
            for _ in range(args.cem_population - 1)
        ]
        costs = [
            _market_metrics(SharedLinearBidPolicy(params), args.train_seeds, args)[
                "social_cost_eur"
            ]
            for params in population
        ]
        order = np.argsort(costs)
        elite = np.array([population[i] for i in order[: args.cem_elite]])
        center = elite.mean(axis=0)
        std = elite.std(axis=0) + args.std_floor

        val_cost = _market_metrics(
            SharedLinearBidPolicy(center), args.val_seeds, args
        )["social_cost_eur"]
        if val_cost < best_val:
            best_val = val_cost
            best_params = center.copy()
        history.append(
            {
                "iteration": iteration,
                "train_best_cost": float(min(costs)),
                "center_val_cost": float(val_cost),
                "best_val_cost": float(best_val),
            }
        )
        print(
            f"[CEM] iter {iteration}: train_best={min(costs):,.0f} "
            f"val_center={val_cost:,.0f} best_val={best_val:,.0f}"
        )
    return best_params, history


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Train the policy and evaluate it against rule, myopic market, and MPC.

    训练策略,并与规则、近视市场、MPC 对比评估。
    """
    best_params, history = train_cem(args)

    rule = _rule_metrics(args.eval_seeds, args)
    myopic = _market_metrics(SharedLinearBidPolicy(), args.eval_seeds, args)
    learned = _market_metrics(
        SharedLinearBidPolicy(best_params), args.eval_seeds, args
    )
    report: dict[str, Any] = {
        "scenario": args.scenario,
        "episode_hours": args.episode_hours,
        "train_seeds": list(args.train_seeds),
        "val_seeds": list(args.val_seeds),
        "eval_seeds": list(args.eval_seeds),
        "learned_params": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, best_params)
        },
        "eval": {
            "rule": rule,
            "myopic_market": myopic,
            "learned_market": learned,
        },
        "cem_history": history,
    }

    # Learned vs myopic is always well defined (independent of any ceiling).
    # 学习 vs 近视始终有定义(与任何天花板无关)。
    myo_c = myopic["social_cost_eur"]
    lrn_c = learned["social_cost_eur"]
    report["learned_vs_myopic_cost_pct"] = (
        (lrn_c - myo_c) / myo_c * 100.0 if myo_c > 1e-9 else float("nan")
    )
    report["learned_vs_myopic_vent_pct"] = (
        (learned["vented_t"] - myopic["vented_t"]) / myopic["vented_t"] * 100.0
        if myopic["vented_t"] > 1e-9
        else float("nan")
    )

    if args.with_mpc:
        mpc_records = [
            _centralized_metrics("native_mpc", seed, args)
            for seed in args.eval_seeds
        ]
        mpc = {
            "social_cost_eur": mean(m["total_cost_eur"] for m in mpc_records),
            "vented_t": mean(m["vented_t"] for m in mpc_records),
            "stored_t": mean(m["stored_t"] for m in mpc_records),
        }
        report["eval"]["native_mpc"] = mpc
        myopic_c = myopic["social_cost_eur"]
        learned_c = learned["social_cost_eur"]
        mpc_c = mpc["social_cost_eur"]
        possible = myopic_c - mpc_c
        report["learned_gap_to_mpc_pct"] = (
            (learned_c - mpc_c) / mpc_c * 100.0 if mpc_c > 1e-9 else float("nan")
        )
        report["myopic_gap_to_mpc_pct"] = (
            (myopic_c - mpc_c) / mpc_c * 100.0 if mpc_c > 1e-9 else float("nan")
        )
        report["gap_closed_by_learning_pct"] = (
            (myopic_c - learned_c) / possible * 100.0
            if abs(possible) > 1e-9
            else float("nan")
        )
    return report


def _print_report(report: dict[str, Any]) -> None:
    """Print evaluation metrics and the closed-gap summary.

    打印评估指标与差距缩小摘要。
    """
    print("\n=== Learned bidding vs baselines / 学习竞价 vs 基线 ===")
    print(f"scenario: {report['scenario']}   eval seeds: {report['eval_seeds']}\n")
    print(f"{'controller':<16}{'stored_t':>12}{'vented_t':>12}{'social_cost_eur':>18}")
    print("-" * 58)
    order = ["rule", "myopic_market", "learned_market", "native_mpc"]
    for name in order:
        if name in report["eval"]:
            values = report["eval"][name]
            print(
                f"{name:<16}{values['stored_t']:>12.1f}{values['vented_t']:>12.1f}"
                f"{values['social_cost_eur']:>18.1f}"
            )
    print("-" * 58)
    print(
        f"learning vs myopic: cost {report['learned_vs_myopic_cost_pct']:+.2f}%   "
        f"vent {report['learned_vs_myopic_vent_pct']:+.2f}%   (negative = better)"
    )
    if "myopic_gap_to_mpc_pct" in report:
        myopic_gap = report["myopic_gap_to_mpc_pct"]
        if myopic_gap <= 0.0:
            print(
                f"note: the market already matches/beats native-MPC on these seeds "
                f"(myopic is {myopic_gap:+.2f}% vs MPC); MPC is a reference, not a ceiling here."
            )
        else:
            print(
                f"myopic gap to MPC = {myopic_gap:+.2f}%   "
                f"learned gap to MPC = {report['learned_gap_to_mpc_pct']:+.2f}%   "
                f"=> learning closed {report['gap_closed_by_learning_pct']:.1f}% of the gap"
            )
    print("\nlearned bid weights / 学到的竞价权重:")
    for name, value in report["learned_params"].items():
        print(f"  {name:<28}{value:+.3f}")
    print()


def build_parser() -> argparse.ArgumentParser:
    """Build the training command-line parser.

    构建训练命令行解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="northern_lights_phase1_milkrun_imbalanced",
    )
    parser.add_argument("--episode-hours", type=int, default=360)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--val-seeds", type=int, nargs="+", default=[3])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--carbon-price", type=float, default=80.0)
    parser.add_argument("--bid-horizon-h", type=float, default=48.0)
    # CEM hyper-parameters / CEM 超参数.
    parser.add_argument("--cem-iters", type=int, default=8)
    parser.add_argument("--cem-population", type=int, default=16)
    parser.add_argument("--cem-elite", type=int, default=4)
    parser.add_argument("--init-std", type=float, default=1.0)
    parser.add_argument("--std-floor", type=float, default=0.05)
    parser.add_argument("--cem-seed", type=int, default=0)
    # Ceiling comparison / 天花板对比.
    parser.add_argument("--with-mpc", action="store_true", default=True)
    parser.add_argument("--no-mpc", dest="with_mpc", action="store_false")
    parser.add_argument("--replan-hours", type=float, default=24.0)
    parser.add_argument("--planning-horizon-hours", type=int, default=72)
    parser.add_argument("--milp-time-limit-seconds", type=float, default=30.0)
    parser.add_argument(
        "--milp-solver",
        choices=("cbc", "cplex", "cplex_native"),
        default="cbc",
    )
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    """Parse arguments, train, evaluate, print and write results.

    解析参数、训练、评估、打印并写出结果。
    """
    args = build_parser().parse_args()
    if args.cem_elite > args.cem_population:
        raise ValueError("cem-elite must not exceed cem-population.")
    report = run(args)
    _print_report(report)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("output")
        / "auction_market_learned"
        / f"{args.scenario}__{args.episode_hours}h"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.save(output_dir / "learned_params.npy", np.array(
        [report["learned_params"][name] for name in FEATURE_NAMES]
    ))
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
