"""Probe: does urgency-based emitter bidding beat the balanced rule?

探针:基于紧迫度的排放源竞价能否胜过平衡规则?

For each seed it samples one scenario, deep-copies it, and runs both the
balanced rule and the greedy shuttle auction on the *identical* disturbance
trajectory. It then reports, per controller:

  * chain outcomes: stored / vented / storage rate / total cost;
  * the tail-vent CVaR across seeds (the project's own risk metric);
  * the exact emitter / shipping / storage cost decomposition;
  * per-emitter venting, which exposes contention winners and losers.

对每个 seed,采样一个场景、深拷贝,并在*完全相同*的扰动轨迹上分别运行平衡规则与
贪心运力拍卖,随后按控制器报告:链条结果、跨 seed 的尾部放空 CVaR、精确的三方成本分解,
以及暴露争夺中赢家/输家的逐排放源放空。

Run / 运行:

    python -m algorithms.auction_market.probe \
        --scenario northern_lights_phase1_milkrun_imbalanced --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from algorithms.residual_rl.evaluation import validation_metrics

from .bidding import AuctionConfig
from .payoff import decompose_costs, per_emitter_costs, tariff_pnl
from .policies import AuctionDispatchPolicy, rule_action
from .runner import EpisodeResult, build_env, rollout


CONTROLLERS = ("rule", "auction")


def _run_seed(
    seed: int,
    *,
    scenario: str,
    episode_hours: int,
    forecast_context_hours: int,
    hard_scenario_probability: float,
    auction_config: AuctionConfig,
) -> dict[str, EpisodeResult]:
    """Run rule and auction on one identical scenario realisation.

    在同一个场景实现上运行规则与拍卖。
    """
    base = build_env(
        scenario=scenario,
        episode_hours=episode_hours,
        forecast_context_hours=forecast_context_hours,
        hard_scenario_probability=hard_scenario_probability,
    )
    base.reset(seed=int(seed))
    results = {
        "rule": rollout(deepcopy(base), rule_action),
        "auction": rollout(deepcopy(base), AuctionDispatchPolicy(auction_config)),
    }
    captured = {name: res.captured_t for name, res in results.items()}
    if abs(captured["rule"] - captured["auction"]) > 1.0:
        raise AssertionError(
            "Paired check failed: controllers saw different capture "
            f"({captured}); scenarios are not identical."
        )
    return results


def _seed_record(
    seed: int,
    controller: str,
    result: EpisodeResult,
    auction_config: AuctionConfig,
) -> dict[str, Any]:
    """Flatten one controller/seed result into a CSV-friendly record.

    将单个控制器/seed 结果展平为便于 CSV 的记录。
    """
    costs = decompose_costs(result.ledger)
    record: dict[str, Any] = {
        "seed": int(seed),
        "controller": controller,
        "captured_t": result.captured_t,
        "stored_t": result.stored_t,
        "vented_t": result.vented_t,
        "storage_rate": result.storage_rate,
        "operating_cost_eur": result.operating_cost_eur,
        "total_cost_eur": result.total_cost_eur,
        "unit_total_cost_eur_per_t": result.unit_total_cost_eur_per_t,
        "hard_violations": result.hard_violations,
    }
    record.update(costs.as_dict())
    emitters = per_emitter_costs(
        result.per_emitter_captured_t,
        result.per_emitter_vented_t,
        auction_config.carbon_price_eur_per_t,
    )
    for emitter_id, values in emitters.items():
        record[f"vent_t[{emitter_id}]"] = values["vented_t"]
    return record


def _aggregate(records: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    """Average numeric fields across seed records.

    对若干 seed 记录的数值字段求均值。
    """
    return {key: mean(float(r[key]) for r in records) for key in keys}


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the probe and return a structured report.

    执行探针并返回结构化报告。
    """
    auction_config = AuctionConfig(
        carbon_price_eur_per_t=args.carbon_price,
        bid_horizon_h=args.bid_horizon_h,
        reserve_price_eur=args.reserve_price,
    )
    raw: dict[str, list[dict[str, Any]]] = {name: [] for name in CONTROLLERS}
    results_by_seed: dict[int, dict[str, EpisodeResult]] = {}
    for seed in args.seeds:
        results = _run_seed(
            seed,
            scenario=args.scenario,
            episode_hours=args.episode_hours,
            forecast_context_hours=args.forecast_context_hours,
            hard_scenario_probability=args.hard_scenario_probability,
            auction_config=auction_config,
        )
        results_by_seed[int(seed)] = results
        for name in CONTROLLERS:
            raw[name].append(
                _seed_record(seed, name, results[name], auction_config)
            )

    emitter_ids = list(
        next(iter(results_by_seed.values()))["rule"].per_emitter_captured_t
    )
    summary_keys = [
        "captured_t",
        "stored_t",
        "vented_t",
        "storage_rate",
        "operating_cost_eur",
        "total_cost_eur",
        "emitter_total_eur",
        "shipping_total_eur",
        "storage_total_eur",
    ] + [f"vent_t[{eid}]" for eid in emitter_ids]

    report: dict[str, Any] = {
        "scenario": args.scenario,
        "seeds": list(args.seeds),
        "episode_hours": args.episode_hours,
        "auction": {
            "carbon_price_eur_per_t": auction_config.carbon_price_eur_per_t,
            "bid_horizon_h": auction_config.bid_horizon_h,
            "reserve_price_eur": auction_config.reserve_price_eur,
        },
        "summary": {},
        "cvar": {},
        "tariff_pnl": {},
        "raw": raw,
        "emitter_ids": emitter_ids,
    }
    for name in CONTROLLERS:
        report["summary"][name] = _aggregate(raw[name], summary_keys)
        report["cvar"][name] = validation_metrics(
            [
                {
                    "total_cost_eur": r["total_cost_eur"],
                    "vented_t": r["vented_t"],
                    "stored_t": r["stored_t"],
                    "hard_violations": r["hard_violations"],
                }
                for r in raw[name]
            ]
        )
        if args.transport_tariff > 0.0 or args.injection_tariff > 0.0:
            pnl = [
                tariff_pnl(
                    results_by_seed[int(r["seed"])][name].ledger,
                    decompose_costs(results_by_seed[int(r["seed"])][name].ledger),
                    transport_tariff_eur_per_t=args.transport_tariff,
                    injection_tariff_eur_per_t=args.injection_tariff,
                ).as_dict()
                for r in raw[name]
            ]
            report["tariff_pnl"][name] = _aggregate(
                pnl,
                list(pnl[0].keys()),
            )
    return report


def _print_report(report: dict[str, Any]) -> None:
    """Print a compact human-readable comparison.

    打印精简的、可读的对比。
    """
    print(f"\n=== Auction market probe / 拍卖市场探针 ===")
    print(f"scenario: {report['scenario']}   seeds: {report['seeds']}")
    print(
        f"bid = carbon_price({report['auction']['carbon_price_eur_per_t']:.0f}) "
        f"x projected vent over {report['auction']['bid_horizon_h']:.0f} h\n"
    )

    def row(label: str, key: str, fmt: str = "{:>14.1f}") -> str:
        rule = report["summary"]["rule"][key]
        auc = report["summary"]["auction"][key]
        delta = auc - rule
        return (
            f"{label:<26}" + fmt.format(rule) + fmt.format(auc)
            + fmt.format(delta)
        )

    print(f"{'metric (mean)':<26}{'rule':>14}{'auction':>14}{'delta':>14}")
    print("-" * 68)
    print(row("stored_t", "stored_t"))
    print(row("vented_t", "vented_t"))
    print(row("storage_rate", "storage_rate", "{:>14.4f}"))
    print(row("operating_cost_eur", "operating_cost_eur"))
    print(row("total_cost_eur", "total_cost_eur"))
    print("-" * 68)
    print(
        f"{'CVaR vented_t (worst 25%)':<26}"
        f"{report['cvar']['rule']['cvar_vented_t']:>14.1f}"
        f"{report['cvar']['auction']['cvar_vented_t']:>14.1f}"
        f"{report['cvar']['auction']['cvar_vented_t'] - report['cvar']['rule']['cvar_vented_t']:>14.1f}"
    )

    print("\n-- three-party cost split (mean EUR) / 三方成本分解 --")
    for party in ("emitter_total_eur", "shipping_total_eur", "storage_total_eur"):
        print(row(party, party))

    print("\n-- per-emitter vented_t (contention winners/losers) / 逐源放空 --")
    for eid in report["emitter_ids"]:
        print(row(f"  {eid}", f"vent_t[{eid}]"))

    if report["tariff_pnl"]:
        print("\n-- tariff P&L (mean EUR, illustrative) / 资费盈亏(示意) --")
        for name in CONTROLLERS:
            pnl = report["tariff_pnl"][name]
            print(
                f"  {name:<8} shipping_profit={pnl['shipping_profit_eur']:>12.0f}"
                f"  storage_profit={pnl['storage_profit_eur']:>12.0f}"
                f"  emitter_total={pnl['emitter_total_eur']:>12.0f}"
            )
    print()


def _write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    """Write raw, summary, and metadata files.

    写出原始、汇总与元数据文件。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = [r for name in CONTROLLERS for r in report["raw"][name]]
    fieldnames = list(raw_rows[0].keys())
    with (output_dir / "comparison_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    with (output_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = sorted(next(iter(report["summary"].values())).keys())
        writer = csv.writer(handle)
        writer.writerow(["controller", *keys])
        for name in CONTROLLERS:
            writer.writerow([name, *(report["summary"][name][k] for k in keys)])

    metadata = {k: v for k, v in report.items() if k != "raw"}
    (output_dir / "probe_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote results to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Build the probe command-line parser.

    构建探针命令行解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="northern_lights_phase1_milkrun_imbalanced",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument("--hard-scenario-probability", type=float, default=0.0)
    parser.add_argument("--carbon-price", type=float, default=80.0)
    parser.add_argument("--bid-horizon-h", type=float, default=48.0)
    parser.add_argument("--reserve-price", type=float, default=0.0)
    parser.add_argument("--transport-tariff", type=float, default=0.0)
    parser.add_argument("--injection-tariff", type=float, default=0.0)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    """Parse arguments, run the probe, print and write results.

    解析参数、运行探针、打印并写出结果。
    """
    args = build_parser().parse_args()
    report = run_probe(args)
    _print_report(report)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("output")
        / "auction_market_probe"
        / f"{args.scenario}__{args.episode_hours}h__seeds{min(args.seeds)}-{max(args.seeds)}"
    )
    _write_outputs(report, output_dir)


if __name__ == "__main__":
    main()
