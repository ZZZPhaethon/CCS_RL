"""Tune pure-RL ensemble switching thresholds on validation scenarios.

在验证场景上调整纯 RL ensemble 切换阈值。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .ensemble_executor import EnsembleRiskConfig
from .evaluate_ensemble import evaluate_ensemble


def tune_thresholds(
    *,
    seed0_run: Path,
    seed1_run: Path,
    seed2_run: Path,
    output_dir: Path,
    normal_seeds: tuple[int, ...],
    hard_seeds: tuple[int, ...],
    hours_values: tuple[float, ...],
    fill_values: tuple[float, ...],
    score_values: tuple[int, ...],
    forecast_speed_min: float = 0.65,
    worst_hard_vent_penalty_eur_per_t: float = 250.0,
) -> list[dict[str, Any]]:
    """Evaluate a compact threshold grid and return a robust ranking.

    评估紧凑阈值网格并返回鲁棒排序。
    """
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    configurations = [
        EnsembleRiskConfig(
            hours_to_overflow_h=hours,
            fill_ratio=fill,
            forecast_speed_min=forecast_speed_min,
            high_risk_score=score,
        )
        for hours in hours_values
        for fill in fill_values
        for score in score_values
    ]
    for position, risk in enumerate(configurations, start=1):
        label = (
            f"hours{risk.hours_to_overflow_h:g}"
            f"__fill{risk.fill_ratio:g}"
            f"__score{risk.high_risk_score}"
        )
        print(
            f"Ensemble threshold {position}/{len(configurations)} | "
            f"{label}",
            flush=True,
        )
        candidate_dir = output_dir / label
        _normal_records, normal = evaluate_ensemble(
            seed0_run=seed0_run,
            seed1_run=seed1_run,
            seed2_run=seed2_run,
            output_dir=candidate_dir / "normal",
            seeds=normal_seeds,
            hard_scenario_probability=0.0,
            risk_config=risk,
        )
        _hard_records, hard = evaluate_ensemble(
            seed0_run=seed0_run,
            seed1_run=seed1_run,
            seed2_run=seed2_run,
            output_dir=candidate_dir / "hard",
            seeds=hard_seeds,
            hard_scenario_probability=1.0,
            risk_config=risk,
        )
        robust_loss = (
            0.5 * normal["selection_loss"]
            + 0.5 * hard["selection_loss"]
            + worst_hard_vent_penalty_eur_per_t
            * hard["worst_vented_t"]
        )
        rows.append(
            {
                "label": label,
                "hours_to_overflow_h": (
                    risk.hours_to_overflow_h
                ),
                "fill_ratio": risk.fill_ratio,
                "forecast_speed_min": risk.forecast_speed_min,
                "high_risk_score": risk.high_risk_score,
                "normal_stored_t": normal["mean_stored_t"],
                "normal_vented_t": normal["mean_vented_t"],
                "normal_worst_vented_t": normal[
                    "worst_vented_t"
                ],
                "normal_cvar_vented_t": normal["cvar_vented_t"],
                "normal_total_cost_eur": normal[
                    "mean_total_cost_eur"
                ],
                "normal_seed1_switch_rate": normal[
                    "seed1_switch_rate_mean"
                ],
                "hard_stored_t": hard["mean_stored_t"],
                "hard_vented_t": hard["mean_vented_t"],
                "hard_worst_vented_t": hard["worst_vented_t"],
                "hard_cvar_vented_t": hard["cvar_vented_t"],
                "hard_total_cost_eur": hard["mean_total_cost_eur"],
                "hard_seed1_switch_rate": hard[
                    "seed1_switch_rate_mean"
                ],
                "hard_violations": (
                    normal["hard_violations"]
                    + hard["hard_violations"]
                ),
                "robust_selection_loss": robust_loss,
            }
        )
    ranking = sorted(
        rows,
        key=lambda row: (
            float(row["hard_violations"]),
            float(row["robust_selection_loss"]),
        ),
    )
    _write_csv(output_dir / "threshold_ranking.csv", ranking)
    (output_dir / "best.json").write_text(
        json.dumps(
            {
                "best": ranking[0],
                "normal_seeds": list(normal_seeds),
                "hard_seeds": list(hard_seeds),
                "worst_hard_vent_penalty_eur_per_t": (
                    worst_hard_vent_penalty_eur_per_t
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ranking


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write threshold rows to CSV.

    将阈值结果写入 CSV。
    """
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Tune thresholds from a terminal.

    从终端调整阈值。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-run", type=Path, required=True)
    parser.add_argument("--seed1-run", type=Path, required=True)
    parser.add_argument("--seed2-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--normal-seeds",
        type=int,
        nargs="+",
        default=list(range(2_000_001, 2_000_009)),
    )
    parser.add_argument(
        "--hard-seeds",
        type=int,
        nargs="+",
        default=list(range(3_000_001, 3_000_009)),
    )
    parser.add_argument(
        "--hours-values",
        type=float,
        nargs="+",
        default=[72.0, 96.0, 120.0],
    )
    parser.add_argument(
        "--fill-values",
        type=float,
        nargs="+",
        default=[0.80],
    )
    parser.add_argument(
        "--score-values",
        type=int,
        nargs="+",
        default=[2, 3],
    )
    parser.add_argument("--forecast-speed-min", type=float, default=0.65)
    parser.add_argument(
        "--worst-hard-vent-penalty-eur-per-t",
        type=float,
        default=250.0,
    )
    args = parser.parse_args()
    ranking = tune_thresholds(
        seed0_run=args.seed0_run,
        seed1_run=args.seed1_run,
        seed2_run=args.seed2_run,
        output_dir=args.output_dir,
        normal_seeds=tuple(args.normal_seeds),
        hard_seeds=tuple(args.hard_seeds),
        hours_values=tuple(args.hours_values),
        fill_values=tuple(args.fill_values),
        score_values=tuple(args.score_values),
        forecast_speed_min=args.forecast_speed_min,
        worst_hard_vent_penalty_eur_per_t=(
            args.worst_hard_vent_penalty_eur_per_t
        ),
    )
    print(
        "Best ensemble threshold: "
        f"{ranking[0]['label']} | "
        f"robust_loss={ranking[0]['robust_selection_loss']:,.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

