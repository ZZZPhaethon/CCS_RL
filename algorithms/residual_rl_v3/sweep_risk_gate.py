"""Sweep adaptive risk gates using a frozen MaskablePPO policy.

使用冻结的 MaskablePPO 策略扫描 adaptive 风险门控。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO

from algorithms.rl.reward import HighLevelRewardConfig
from algorithms.residual_rl_v2.evaluation import (
    evaluate_seeds,
    validation_metrics,
)
from algorithms.residual_rl_v2.factory import (
    make_masked_residual_native_env,
)

from .factory import make_risk_gated_native_env
from .risk_gate import AdaptiveRiskGateConfig


def sweep_risk_gates(
    *,
    run_dir: Path,
    output_dir: Path,
    model_choice: str = "best",
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    hours_thresholds: tuple[float, ...] = (24.0, 48.0, 72.0, 96.0),
    fill_thresholds: tuple[float, ...] = (0.70, 0.80, 0.90),
    weather_fill_ratio_threshold: float = 0.60,
    weather_speed_threshold: float = 0.65,
    weather_lookahead_h: int = 72,
    cvar_tail_fraction: float = 0.25,
    tail_vent_penalty_eur_per_t: float = 500.0,
    worst_hard_vent_penalty_eur_per_t: float = 250.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate an ungated reference plus every threshold combination.

    评估无门控参考以及所有阈值组合。
    """
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    model_stem = (
        "maskable_residual_v2_best_validation"
        if model_choice == "best"
        else "maskable_residual_v2_final"
    )
    model = MaskablePPO.load(run_dir / model_stem, device="cpu")
    reward = HighLevelRewardConfig(**config["high_level_reward"])
    common = {
        "scenario": str(config["scenario"]),
        "episode_hours": int(config["episode_hours"]),
        "forecast_context_hours": int(
            config["forecast_context_hours"]
        ),
        "decision_interval_h": float(config["decision_interval_h"]),
        "event_triggered": bool(config["event_triggered"]),
        "weather_mode": str(config["weather_mode"]),
        "reward": reward,
    }
    candidates: list[
        tuple[str, AdaptiveRiskGateConfig | None]
    ] = [("ungated", None)]
    for hours in hours_thresholds:
        for fill in fill_thresholds:
            gate = AdaptiveRiskGateConfig(
                hours_to_overflow_threshold_h=float(hours),
                fill_ratio_threshold=float(fill),
                weather_fill_ratio_threshold=float(
                    weather_fill_ratio_threshold
                ),
                weather_speed_threshold=float(weather_speed_threshold),
                weather_lookahead_h=int(weather_lookahead_h),
            )
            candidates.append(
                (
                    f"hours{hours:g}__fill{fill:g}",
                    gate,
                )
            )

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for position, (label, gate) in enumerate(candidates, start=1):
        print(
            f"Risk-gate sweep {position}/{len(candidates)} | {label}",
            flush=True,
        )
        by_difficulty: dict[str, dict[str, float]] = {}
        for difficulty, hard_probability in (
            ("normal", 0.0),
            ("hard", 1.0),
        ):
            if gate is None:
                env = make_masked_residual_native_env(
                    hard_scenario_probability=hard_probability,
                    **common,
                )
            else:
                env = make_risk_gated_native_env(
                    hard_scenario_probability=hard_probability,
                    gate=gate,
                    **common,
                )
            records = evaluate_seeds(model, env, seeds)
            metrics = validation_metrics(
                records,
                cvar_tail_fraction=cvar_tail_fraction,
                tail_vent_penalty_eur_per_t=(
                    tail_vent_penalty_eur_per_t
                ),
            )
            by_difficulty[difficulty] = metrics
            for record in records:
                raw.append(
                    {
                        "gate_label": label,
                        "difficulty": difficulty,
                        "hours_threshold_h": (
                            "" if gate is None
                            else gate.hours_to_overflow_threshold_h
                        ),
                        "fill_threshold": (
                            "" if gate is None
                            else gate.fill_ratio_threshold
                        ),
                        **{
                            key: value
                            for key, value in record.items()
                            if not isinstance(value, dict)
                        },
                        "actions": json.dumps(
                            record["actions"],
                            sort_keys=True,
                        ),
                    }
                )
        normal = by_difficulty["normal"]
        hard = by_difficulty["hard"]
        robust_loss = (
            0.5 * normal["selection_loss"]
            + 0.5 * hard["selection_loss"]
            + worst_hard_vent_penalty_eur_per_t
            * hard["worst_vented_t"]
        )
        rows.append(
            {
                "gate_label": label,
                "hours_threshold_h": (
                    "" if gate is None
                    else gate.hours_to_overflow_threshold_h
                ),
                "fill_threshold": (
                    "" if gate is None else gate.fill_ratio_threshold
                ),
                "normal_stored_t": normal["mean_stored_t"],
                "normal_vented_t": normal["mean_vented_t"],
                "normal_worst_vented_t": normal["worst_vented_t"],
                "normal_cvar_vented_t": normal["cvar_vented_t"],
                "normal_total_cost_eur": normal[
                    "mean_total_cost_eur"
                ],
                "normal_intervention_rate": normal[
                    "mean_effective_intervention_rate"
                ],
                "hard_stored_t": hard["mean_stored_t"],
                "hard_vented_t": hard["mean_vented_t"],
                "hard_worst_vented_t": hard["worst_vented_t"],
                "hard_cvar_vented_t": hard["cvar_vented_t"],
                "hard_total_cost_eur": hard["mean_total_cost_eur"],
                "hard_intervention_rate": hard[
                    "mean_effective_intervention_rate"
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
    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "gate_grid.csv", rows)
    _write_csv(output_dir / "gate_ranking.csv", ranking)
    _write_csv(output_dir / "gate_per_seed.csv", raw)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "model_choice": model_choice,
                "model_path": str(run_dir / model_stem),
                "seeds": list(seeds),
                "hours_thresholds": list(hours_thresholds),
                "fill_thresholds": list(fill_thresholds),
                "weather_gate": {
                    "fill_ratio_threshold": (
                        weather_fill_ratio_threshold
                    ),
                    "speed_threshold": weather_speed_threshold,
                    "lookahead_h": weather_lookahead_h,
                },
                "metric_weights": {
                    "cvar_tail_fraction": cvar_tail_fraction,
                    "tail_vent_penalty_eur_per_t": (
                        tail_vent_penalty_eur_per_t
                    ),
                    "worst_hard_vent_penalty_eur_per_t": (
                        worst_hard_vent_penalty_eur_per_t
                    ),
                },
                "best_gate": ranking[0],
                "gate_configs": [
                    None if gate is None else asdict(gate)
                    for _label, gate in candidates
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows, ranking


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write homogeneous dictionaries to CSV.

    将同构字典写入 CSV。
    """
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the frozen-policy gate sweep.

    运行冻结策略门控扫描。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("best", "final"), default="best")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
    )
    parser.add_argument(
        "--hours-thresholds",
        type=float,
        nargs="+",
        default=[24.0, 48.0, 72.0, 96.0],
    )
    parser.add_argument(
        "--fill-thresholds",
        type=float,
        nargs="+",
        default=[0.70, 0.80, 0.90],
    )
    parser.add_argument(
        "--weather-fill-threshold",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--weather-speed-threshold",
        type=float,
        default=0.65,
    )
    parser.add_argument("--weather-lookahead-h", type=int, default=72)
    parser.add_argument("--cvar-tail-fraction", type=float, default=0.25)
    parser.add_argument(
        "--tail-vent-penalty-eur-per-t",
        type=float,
        default=500.0,
    )
    parser.add_argument(
        "--worst-hard-vent-penalty-eur-per-t",
        type=float,
        default=250.0,
    )
    args = parser.parse_args()
    _rows, ranking = sweep_risk_gates(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        model_choice=args.model,
        seeds=tuple(args.seeds),
        hours_thresholds=tuple(args.hours_thresholds),
        fill_thresholds=tuple(args.fill_thresholds),
        weather_fill_ratio_threshold=args.weather_fill_threshold,
        weather_speed_threshold=args.weather_speed_threshold,
        weather_lookahead_h=args.weather_lookahead_h,
        cvar_tail_fraction=args.cvar_tail_fraction,
        tail_vent_penalty_eur_per_t=(
            args.tail_vent_penalty_eur_per_t
        ),
        worst_hard_vent_penalty_eur_per_t=(
            args.worst_hard_vent_penalty_eur_per_t
        ),
    )
    print(
        "Best risk gate: "
        f"{ranking[0]['gate_label']} | "
        f"robust_loss={ranking[0]['robust_selection_loss']:,.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
