"""Compare masked residual PPO v2 with rule and rollout MPC fairly.

在完全共享的场景上公平比较掩码 residual PPO v2、规则和 rollout MPC。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO

from algorithms import (
    DispatchGoal,
    GoalAwareNativeMpcExecutor,
    GoalAwareRuleExecutor,
    evaluate_executor,
)
from algorithms.residual_rl_v2.env import (
    MaskedResidualDispatchEnv,
    MaskedResidualEnvConfig,
)
from algorithms.residual_rl_v2.evaluation import evaluate_seed
from algorithms.rl.reward import HighLevelRewardConfig
from Simulation.control.baselines import balanced_capture_assignment
from Simulation.environment import build_phase1_env
from Simulation.scenario_generation import ScenarioConfig, ScenarioGenerator

from experiments.compare_shared_residual_controllers import (
    RAW_COLUMNS,
    FixedScenarioGenerator,
    _fingerprint,
    _physical_config,
    _print_row,
    _standard_row,
    _summarise,
    _summary_columns,
    _write_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse strict paired-comparison settings.

    解析严格配对比较设置。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("best", "final"),
        default="best",
    )
    parser.add_argument(
        "--scenario",
        default="northern_lights_phase1_3vessels",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--forecast-context-hours", type=int, default=168)
    parser.add_argument("--replan-hours", type=float, default=24.0)
    parser.add_argument("--planning-horizon-hours", type=int, default=168)
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=("rule", "masked_residual_v2", "rollout_mpc"),
        default=("rule", "masked_residual_v2", "rollout_mpc"),
    )
    parser.add_argument(
        "--weather-mode",
        choices=("window", "block"),
        default="window",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Run the paired comparison and save artifacts.

    运行配对比较并保存产物。
    """
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = json.loads(
        (args.run_dir / "config.json").read_text(encoding="utf-8")
    )
    _validate_config(args, config)
    model_stem = (
        "maskable_residual_v2_best_validation"
        if args.model == "best"
        else "maskable_residual_v2_final"
    )
    model = MaskablePPO.load(args.run_dir / model_stem, device="cpu")
    reward = HighLevelRewardConfig(**config["high_level_reward"])

    generator = ScenarioGenerator(
        ScenarioConfig(
            episode_hours=(
                args.episode_hours + args.forecast_context_hours
            ),
            time_step_hours=1.0,
            weather_process=args.weather_mode,
        )
    )
    probe = build_phase1_env(
        scenario=args.scenario,
        scenario_generator=generator,
        weather_mode=args.weather_mode,
        config=_physical_config(args),
    )
    goal = DispatchGoal(
        emitter_to_vessel=balanced_capture_assignment(probe),
        replan_after_h=args.replan_hours,
        rationale="Shared masked-residual-v2 comparison.",
    )

    rows: list[dict[str, Any]] = []
    fingerprints: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        shared = generator.sample(probe.network, seed=int(seed))
        fingerprints[int(seed)] = _fingerprint(shared)
        seed_rows = _evaluate_seed(
            args,
            model,
            reward,
            goal,
            shared,
            int(seed),
        )
        captured = [float(row["captured_t"]) for row in seed_rows]
        if max(captured) - min(captured) > 1e-6:
            raise AssertionError(
                f"Shared scenario failed for seed {seed}: {captured}"
            )
        print(
            f"seed={seed} paired_capture_verified={captured[0]:,.3f} t",
            flush=True,
        )
        rows.extend(seed_rows)

    args.output_dir.mkdir(parents=True)
    _write_csv(
        args.output_dir / "comparison_raw.csv",
        rows,
        RAW_COLUMNS,
    )
    _write_csv(
        args.output_dir / "comparison_summary.csv",
        _summarise(rows),
        _summary_columns(),
    )
    (args.output_dir / "comparison_metadata.json").write_text(
        json.dumps(
            {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "paired_capture_verified": True,
                "scenario_fingerprints": fingerprints,
                "run_config": config,
                "model_path": str(args.run_dir / model_stem),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved comparison under: {args.output_dir}")
    return 0


def _validate_config(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    """Ensure the saved policy matches the requested interface.

    确保已保存策略与请求接口匹配。
    """
    expected = {
        "algorithm": "maskable_residual_ppo_v2",
        "scenario": args.scenario,
        "episode_hours": args.episode_hours,
        "forecast_context_hours": args.forecast_context_hours,
        "decision_interval_h": args.replan_hours,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"v2 configuration mismatch: {mismatches}")


def _build_physical_env(args, scenario):
    """Build one environment around an independent shared-scenario copy.

    使用共享场景的独立副本构建环境。
    """
    return build_phase1_env(
        scenario=args.scenario,
        scenario_generator=FixedScenarioGenerator(scenario),
        weather_mode=args.weather_mode,
        config=_physical_config(args),
    )


def _evaluate_seed(args, model, reward, goal, scenario, seed):
    """Evaluate selected controllers on copies of one trajectory.

    在同一轨迹副本上评估所选控制器。
    """
    rows: list[dict[str, Any]] = []
    standard = {}
    if "rule" in args.controllers:
        standard["rule"] = GoalAwareRuleExecutor()
    if "rollout_mpc" in args.controllers:
        standard["rollout_mpc"] = GoalAwareNativeMpcExecutor(
            planning_horizon_h=args.planning_horizon_hours
        )
    for name, executor in standard.items():
        metrics = evaluate_executor(
            _build_physical_env(args, scenario),
            executor,
            goal,
            seed=seed,
        )
        row = _standard_row(name, seed, metrics)
        rows.append(row)
        _print_row(row)

    if "masked_residual_v2" in args.controllers:
        env = MaskedResidualDispatchEnv(
            _build_physical_env(args, scenario),
            config=MaskedResidualEnvConfig(
                decision_interval_h=args.replan_hours,
                event_triggered=True,
                reward=reward,
            ),
        )
        result = evaluate_seed(model, env, seed)
        vent_rate = (
            result["vented_t"] / result["captured_t"]
            if result["captured_t"] > 1e-9
            else 0.0
        )
        row = {
            "controller": "masked_residual_v2",
            "seed": seed,
            "captured_t": result["captured_t"],
            "stored_t": result["stored_t"],
            "vented_t": result["vented_t"],
            "storage_rate": result["storage_rate"],
            "vent_rate": vent_rate,
            "operating_cost_eur": result["operating_cost_eur"],
            "total_cost_eur": result["total_cost_eur"],
            "unit_total_cost_eur_per_t": result[
                "unit_total_cost_eur_per_t"
            ],
            "wall_clock_seconds": result["wall_clock_seconds"],
            "hard_violations": result["hard_violations"],
            "decisions": result["decisions"],
            "mean_decision_interval_h": result[
                "mean_decision_interval_h"
            ],
            "interventions_applied": result["changed_decisions"],
            "intervention_rate": result[
                "effective_intervention_rate"
            ],
            "action_counts": json.dumps(
                result["actions"],
                sort_keys=True,
            ),
            "trigger_counts": json.dumps(
                result["triggers"],
                sort_keys=True,
            ),
        }
        rows.append(row)
        _print_row(row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
