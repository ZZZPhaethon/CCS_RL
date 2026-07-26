"""Does pricing change the allocation? Budget sweep and non-trivial anarchy.

价格会改变分配吗?预算扫描与非平凡无序代价。

Without budgets the auction allocates purely on bid *ranking*, so payments are
transfers and Price of Anarchy is exactly 1. This study switches on the
budgeted uniform-price mechanism and asks two questions:

  1. **Budget sweep** -- as budgets tighten, does the physical outcome change,
     and where does pricing start to bind?
  2. **Price of Anarchy** -- with budgets, does strategic shading now change the
     allocation and cost real efficiency?

无预算时拍卖仅按出价*排序*分配,支付是转移,无序代价恰为 1。本研究启用带预算的统一价格
机制,回答:(1) **预算扫描**——预算收紧时物理结果是否改变、定价从何处开始咬合;
(2) **无序代价**——有预算后,策略性压价是否真的改变分配并造成效率损失?

Run / 运行:

    python -m algorithms.auction_market.budget_study \
        --scenario northern_lights_phase1_milkrun_imbalanced --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .bidding import AuctionConfig
from .budgeted import BudgetConfig, BudgetedAuctionPolicy
from .payoff import decompose_costs
from .policies import rule_action
from .runner import build_env, rollout


def _run_budgeted(
    shades: dict[str, float] | None,
    budget_factor: float | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Average the budgeted market over the seeds.

    在若干 seed 上平均带预算的市场结果。
    """
    config = AuctionConfig(
        carbon_price_eur_per_t=args.carbon_price,
        bid_horizon_h=args.bid_horizon_h,
    )
    budget = BudgetConfig(budget_factor=budget_factor)
    vent, cost, stored = [], [], []
    prices, blocked, spend = [], [], []
    payoff: dict[str, list[float]] = {}
    parties: dict[str, list[float]] = {}
    for seed in args.seeds:
        env = build_env(
            scenario=args.scenario,
            episode_hours=args.episode_hours,
            forecast_context_hours=args.forecast_context_hours,
        )
        env.reset(seed=int(seed))
        policy = BudgetedAuctionPolicy(
            config,
            budget,
            bid_shades=shades,
            commitment_margin=args.commitment_margin,
            defend_floor_t=args.defend_floor_t,
        )
        result = rollout(env, policy)
        vent.append(result.vented_t)
        cost.append(result.total_cost_eur)
        stored.append(result.stored_t)
        positive = [p for p in policy.clearing_prices if p > 0.0]
        prices.append(mean(positive) if positive else 0.0)
        blocked.append(float(policy.budget_blocked_events))
        spend.append(sum(policy.payments.values()))
        # Exact three-party physical cost split from the economic ledger.
        # 来自经济账本的精确三方物理成本分解。
        costs = decompose_costs(result.ledger)
        parties.setdefault("emitter_physical_eur", []).append(costs.emitter_total)
        parties.setdefault("shipping_eur", []).append(costs.shipping_total)
        parties.setdefault("storage_eur", []).append(costs.storage_total)
        for emitter_id in env.emitter_ids:
            vent_t = float(result.per_emitter_vented_t.get(emitter_id, 0.0))
            payment = float(policy.payments.get(emitter_id, 0.0))
            payoff.setdefault(emitter_id, []).append(
                args.carbon_price * vent_t + payment
            )
    return {
        "vented_t": mean(vent),
        "stored_t": mean(stored),
        "social_cost_eur": mean(cost),
        "mean_positive_clearing_price_eur": mean(prices),
        "budget_blocked_events": mean(blocked),
        "total_auction_spend_eur": mean(spend),
        "party_costs": {key: mean(values) for key, values in parties.items()},
        "emitter_payoff": {e: mean(v) for e, v in payoff.items()},
    }


def _run_rule(args: argparse.Namespace) -> dict[str, float]:
    """Average the balanced rule over the seeds. / 在若干 seed 上平均平衡规则。"""
    vent, cost, stored = [], [], []
    for seed in args.seeds:
        env = build_env(
            scenario=args.scenario,
            episode_hours=args.episode_hours,
            forecast_context_hours=args.forecast_context_hours,
        )
        env.reset(seed=int(seed))
        result = rollout(env, rule_action)
        vent.append(result.vented_t)
        cost.append(result.total_cost_eur)
        stored.append(result.stored_t)
    return {
        "vented_t": mean(vent),
        "stored_t": mean(stored),
        "social_cost_eur": mean(cost),
    }


def _best_response_equilibrium(
    emitter_ids: list[str],
    budget_factor: float | None,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Learn shades by iterated best response under the budgeted mechanism.

    在带预算机制下用迭代最优反应学习压价系数。
    """
    shades = {emitter_id: 1.0 for emitter_id in emitter_ids}
    history: list[dict[str, Any]] = []
    # Track the best social cost seen, as a practical social-optimum proxy for
    # the Price of Anarchy denominator.
    # 记录见到的最优社会成本,作为无序代价分母的社会最优近似。
    best_social = {"social_cost_eur": float("inf"), "vented_t": float("inf")}
    for round_index in range(args.br_rounds):
        changed = False
        for emitter_id in emitter_ids:
            best_shade = shades[emitter_id]
            best_payoff = float("inf")
            for shade in args.shade_grid:
                trial = dict(shades)
                trial[emitter_id] = shade
                outcome = _run_budgeted(trial, budget_factor, args)
                if outcome["social_cost_eur"] < best_social["social_cost_eur"]:
                    best_social = {
                        "social_cost_eur": outcome["social_cost_eur"],
                        "vented_t": outcome["vented_t"],
                        "shades": dict(trial),
                    }
                payoff = outcome["emitter_payoff"][emitter_id]
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
    return shades, history, best_social


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the budget sweep and the anarchy study.

    运行预算扫描与无序代价研究。
    """
    probe = build_env(
        scenario=args.scenario,
        episode_hours=args.episode_hours,
        forecast_context_hours=args.forecast_context_hours,
    )
    probe.reset(seed=int(args.seeds[0]))
    emitter_ids = list(probe.emitter_ids)

    report: dict[str, Any] = {
        "scenario": args.scenario,
        "seeds": list(args.seeds),
        "episode_hours": args.episode_hours,
        "rule": _run_rule(args),
        "budget_sweep": [],
    }
    truthful = {emitter_id: 1.0 for emitter_id in emitter_ids}
    for factor in args.budget_factors:
        value = None if factor < 0 else factor
        cell = _run_budgeted(truthful, value, args)
        cell["budget_factor"] = "none" if value is None else value
        report["budget_sweep"].append(cell)

    # Anarchy study at the tightest requested budget. / 在最紧预算下研究无序代价。
    anarchy_factor = args.anarchy_budget_factor
    truthful_outcome = _run_budgeted(truthful, anarchy_factor, args)
    eq_shades, history, best_social = _best_response_equilibrium(
        emitter_ids, anarchy_factor, args
    )
    eq_outcome = _run_budgeted(eq_shades, anarchy_factor, args)
    # PoA is measured against the best social cost found, not against truthful:
    # under budgets truthful bidding is itself inefficient (it drains budgets).
    # 无序代价以搜索到的最优社会成本为分母,而非如实出价:有预算时如实出价本身低效
    # (会耗尽预算)。
    optimum = min(best_social["social_cost_eur"], truthful_outcome["social_cost_eur"])
    optimum_vent = min(best_social["vented_t"], truthful_outcome["vented_t"])
    report["anarchy"] = {
        "budget_factor": anarchy_factor,
        "truthful": truthful_outcome,
        "equilibrium": eq_outcome,
        "best_profile_found": best_social,
        "equilibrium_shades": eq_shades,
        "best_response_history": history,
        "price_of_anarchy_cost": (
            eq_outcome["social_cost_eur"] / optimum
            if optimum > 1e-9
            else float("nan")
        ),
        "price_of_anarchy_vent": (
            eq_outcome["vented_t"] / optimum_vent
            if optimum_vent > 1e-9
            else float("nan")
        ),
        "truthful_inefficiency_vs_best": (
            truthful_outcome["social_cost_eur"] / optimum
            if optimum > 1e-9
            else float("nan")
        ),
    }
    return report


def _print_report(report: dict[str, Any]) -> None:
    """Print the budget sweep and the anarchy result.

    打印预算扫描与无序代价结果。
    """
    print("\n=== Budgeted uniform-price market / 带预算的统一价格市场 ===")
    print(f"scenario: {report['scenario']}   seeds: {report['seeds']}")
    rule = report["rule"]
    print(
        f"\nrule baseline: vented={rule['vented_t']:.0f} t  "
        f"cost={rule['social_cost_eur']:.0f} EUR"
    )
    print(f"\n{'budget':>10}{'vented_t':>12}{'cost_eur':>14}{'clear_price':>14}{'blocked':>10}")
    print("-" * 60)
    for cell in report["budget_sweep"]:
        print(
            f"{str(cell['budget_factor']):>10}{cell['vented_t']:>12.0f}"
            f"{cell['social_cost_eur']:>14.0f}"
            f"{cell['mean_positive_clearing_price_eur']:>14.0f}"
            f"{cell['budget_blocked_events']:>10.1f}"
        )

    anarchy = report["anarchy"]
    print(f"\n-- Price of Anarchy at budget_factor={anarchy['budget_factor']} --")
    print(f"{'profile':<22}{'vented_t':>12}{'social_cost_eur':>18}")
    print("-" * 52)
    for label, key in (
        ("truthful (shade=1)", "truthful"),
        ("nash equilibrium", "equilibrium"),
        ("best profile found", "best_profile_found"),
    ):
        values = anarchy[key]
        print(
            f"{label:<22}{values['vented_t']:>12.0f}"
            f"{values['social_cost_eur']:>18.0f}"
        )
    print("-" * 52)
    print(
        f"PoA vs best profile: cost = {anarchy['price_of_anarchy_cost']:.4f}    "
        f"vent = {anarchy['price_of_anarchy_vent']:.4f}"
    )
    print(
        f"truthful inefficiency vs best = "
        f"{anarchy['truthful_inefficiency_vs_best']:.4f}  "
        f"(>1 means truthful bidding itself wastes budget)"
    )
    print("equilibrium shades:", anarchy["equilibrium_shades"])

    # Three-party split and where the auction money goes.
    # 三方分解,以及拍卖资金的去向。
    truthful = anarchy["truthful"]
    party = truthful["party_costs"]
    print("\n-- three-party physical cost + auction transfers / 三方物理成本与拍卖转移 --")
    print(f"  emitter (conditioning + vent) : {party['emitter_physical_eur']:>14,.0f} EUR")
    print(f"  shipping (fuel + load/unload) : {party['shipping_eur']:>14,.0f} EUR")
    print(f"  storage  (reconditioning)     : {party['storage_eur']:>14,.0f} EUR")
    print(
        f"  auction transfers paid        : {truthful['total_auction_spend_eur']:>14,.0f} EUR"
        "   (transfer, not a physical cost / 转移支付,非物理成本)"
    )
    print("\n  per-emitter total burden (vent cost + payments) / 逐源总负担:")
    for emitter_id, value in sorted(truthful["emitter_payoff"].items()):
        print(f"    {emitter_id:<20}{value:>16,.0f} EUR")
    print()


def build_parser() -> argparse.ArgumentParser:
    """Build the budget-study command-line parser.

    构建预算研究命令行解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="northern_lights_phase1_milkrun_imbalanced",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument("--carbon-price", type=float, default=80.0)
    parser.add_argument("--bid-horizon-h", type=float, default=48.0)
    parser.add_argument("--commitment-margin", type=float, default=10.0)
    parser.add_argument("--defend-floor-t", type=float, default=100.0)
    parser.add_argument(
        "--budget-factors",
        type=float,
        nargs="+",
        default=[-1.0, 1.0, 0.5, 0.25, 0.1, 0.05],
        help="Budget factors to sweep; negative means unlimited. / 预算系数扫描;负值表示无限。",
    )
    parser.add_argument(
        "--anarchy-budget-factor",
        type=float,
        default=0.1,
        help="Budget factor used for the anarchy study. / 无序代价研究所用预算系数。",
    )
    parser.add_argument(
        "--shade-grid",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0],
    )
    parser.add_argument("--br-rounds", type=int, default=2)
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
        / "auction_market_budget"
        / f"{args.scenario}__{args.episode_hours}h__seeds{min(args.seeds)}-{max(args.seeds)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "budget_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
