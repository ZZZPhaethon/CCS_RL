"""Screen every feasible residual action along a rule-baseline trajectory.

沿规则基线轨迹筛选每一个可行残差动作。

This is a pre-training action test, not a policy evaluation. At every decision
state visited by the rule, each currently unmasked intervention is executed on
an independent environment copy and compared with a same-duration rule
counterfactual.

这是训练前动作测试，而不是策略评估。在规则轨迹经过的每个决策状态上，
脚本会在独立环境副本中执行每个未被掩码的干预，并与相同时长的规则
反事实比较。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import statistics
from typing import Any

from .factory import make_masked_residual_native_env


RAW_COLUMNS = (
    "seed",
    "decision",
    "time_h",
    "current_trigger",
    "action_index",
    "action_label",
    "elapsed_hours",
    "evaluation_horizon_hours",
    "native_action_changed",
    "changed_native_steps",
    "eligible_vessels",
    "overridden_vessels",
    "incremental_reward",
    "incremental_stored_t",
    "avoided_vent_t",
    "operating_cost_saving_eur",
    "total_cost_saving_eur",
)


def screen_interventions(
    *,
    seeds: tuple[int, ...] = (1, 4),
    scenario: str = "northern_lights_phase1_3vessels",
    episode_hours: int = 720,
    forecast_context_hours: int = 168,
    decision_interval_h: float = 24.0,
    weather_mode: str = "window",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enumerate feasible local interventions on fixed standard scenarios.

    在固定标准场景上枚举可行的局部干预。
    """
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        env = make_masked_residual_native_env(
            scenario=scenario,
            episode_hours=episode_hours,
            forecast_context_hours=forecast_context_hours,
            decision_interval_h=decision_interval_h,
            event_triggered=True,
            weather_mode=weather_mode,
            hard_scenario_probability=0.0,
        )
        env.reset(seed=seed)
        done = False
        decision = 0
        while not done:
            decision += 1
            assert env.env.simulator is not None
            time_h = float(env.env.simulator.state.time_h)
            current_trigger = env.last_decision_trigger
            mask = env.action_masks()
            for action_index in range(1, env.action_count):
                if not bool(mask[action_index]):
                    continue
                candidate = deepcopy(env)
                _obs, reward, terminated, truncated, info = candidate.step(
                    action_index
                )
                total_reward = float(reward)
                total_incremental_stored_t = float(
                    info["incremental_stored_t"]
                )
                total_avoided_vent_t = float(info["avoided_vent_t"])
                total_operating_cost_saving = float(
                    info["operating_cost_saving_eur"]
                )
                total_cost_saving = float(
                    info["total_cost_saving_eur"]
                )
                evaluation_hours = float(info["elapsed_hours"])
                candidate_done = bool(terminated or truncated)
                while not candidate_done:
                    (
                        _obs,
                        future_reward,
                        terminated,
                        truncated,
                        future_info,
                    ) = candidate.step(0)
                    total_reward += float(future_reward)
                    total_incremental_stored_t += float(
                        future_info["incremental_stored_t"]
                    )
                    total_avoided_vent_t += float(
                        future_info["avoided_vent_t"]
                    )
                    total_operating_cost_saving += float(
                        future_info["operating_cost_saving_eur"]
                    )
                    total_cost_saving += float(
                        future_info["total_cost_saving_eur"]
                    )
                    evaluation_hours += float(future_info["elapsed_hours"])
                    candidate_done = bool(terminated or truncated)
                rows.append(
                    {
                        "seed": seed,
                        "decision": decision,
                        "time_h": time_h,
                        "current_trigger": current_trigger,
                        "action_index": action_index,
                        "action_label": info["action_label"],
                        "elapsed_hours": info["elapsed_hours"],
                        "evaluation_horizon_hours": evaluation_hours,
                        "native_action_changed": info[
                            "native_action_changed"
                        ],
                        "changed_native_steps": info[
                            "changed_native_steps"
                        ],
                        "eligible_vessels": json.dumps(
                            info["eligible_vessels_at_decision"],
                            sort_keys=True,
                        ),
                        "overridden_vessels": json.dumps(
                            info["overridden_vessels"],
                            sort_keys=True,
                        ),
                        "incremental_reward": total_reward,
                        "incremental_stored_t": (
                            total_incremental_stored_t
                        ),
                        "avoided_vent_t": total_avoided_vent_t,
                        "operating_cost_saving_eur": (
                            total_operating_cost_saving
                        ),
                        "total_cost_saving_eur": total_cost_saving,
                    }
                )

            _obs, baseline_reward, terminated, truncated, baseline_info = (
                env.step(0)
            )
            if abs(float(baseline_reward)) > 1e-8:
                raise AssertionError(
                    "keep_rule_default must have zero counterfactual reward: "
                    f"seed={seed}, decision={decision}, "
                    f"reward={baseline_reward}"
                )
            if baseline_info["native_action_changed"]:
                raise AssertionError(
                    "keep_rule_default changed the native rule action."
                )
            done = bool(terminated or truncated)
        print(
            f"screened seed={seed} decisions={decision} "
            f"feasible_candidates={sum(row['seed'] == seed for row in rows)}",
            flush=True,
        )
    return rows, _summarise(rows)


def _summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarise feasibility and local improvement by action.

    按动作汇总可行性与局部改善效果。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["action_label"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for label, group in sorted(grouped.items()):
        rewards = [float(row["incremental_reward"]) for row in group]
        avoided = [float(row["avoided_vent_t"]) for row in group]
        stored = [float(row["incremental_stored_t"]) for row in group]
        savings = [float(row["total_cost_saving_eur"]) for row in group]
        summaries.append(
            {
                "action_label": label,
                "feasible_count": len(group),
                "changed_count": sum(
                    bool(row["native_action_changed"]) for row in group
                ),
                "positive_reward_count": sum(value > 1e-9 for value in rewards),
                "vent_improvement_count": sum(
                    value > 1e-9 for value in avoided
                ),
                "storage_improvement_count": sum(
                    value > 1e-9 for value in stored
                ),
                "cost_improvement_count": sum(
                    value > 1e-9 for value in savings
                ),
                "mean_incremental_reward": statistics.mean(rewards),
                "mean_avoided_vent_t": statistics.mean(avoided),
                "max_avoided_vent_t": max(avoided),
                "mean_incremental_stored_t": statistics.mean(stored),
                "max_incremental_stored_t": max(stored),
                "mean_total_cost_saving_eur": statistics.mean(savings),
                "max_total_cost_saving_eur": max(savings),
            }
        )
    return summaries


def _write_csv(path: Path, rows, columns) -> None:
    """Write a UTF-8 CSV file.

    写入 UTF-8 CSV 文件。
    """
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the action screen and save raw plus summary results.

    运行动作筛选，并保存原始及汇总结果。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 4])
    parser.add_argument(
        "--scenario",
        default="northern_lights_phase1_3vessels",
    )
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument("--decision-interval-h", type=float, default=24.0)
    parser.add_argument(
        "--weather-mode",
        choices=("window", "block"),
        default="window",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    rows, summary = screen_interventions(
        seeds=tuple(args.seeds),
        scenario=args.scenario,
        episode_hours=args.episode_hours,
        forecast_context_hours=args.forecast_context_hours,
        decision_interval_h=args.decision_interval_h,
        weather_mode=args.weather_mode,
    )
    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "interventions_raw.csv", rows, RAW_COLUMNS)
    summary_columns = tuple(summary[0]) if summary else ("action_label",)
    _write_csv(
        args.output_dir / "interventions_summary.csv",
        summary,
        summary_columns,
    )
    best = sorted(
        rows,
        key=lambda row: (
            float(row["incremental_reward"]),
            float(row["avoided_vent_t"]),
            float(row["total_cost_saving_eur"]),
        ),
        reverse=True,
    )[:20]
    (args.output_dir / "top_interventions.json").write_text(
        json.dumps(best, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved intervention screen under: {args.output_dir}")


if __name__ == "__main__":
    main()
