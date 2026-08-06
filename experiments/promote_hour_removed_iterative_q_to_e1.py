"""Stage the hour-removed Iterative-Q models as the current E1 formal model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = REPO_ROOT / "experiments_results" / "E1"
FORMAL_NAME = (
    "formal_iterative_action_q_g60_p4_seeds_9000031-9000060_run01"
)
ALGORITHM = "iterative_action_q_g60_p4"
DISPLAY_NAME = "Iterative Action-Q G60-P4 (no hour)"
METHOD_CLASS = "reinforcement_learning"
MODEL_SEEDS = (0, 1, 2)
TEST_SEEDS = tuple(range(9_000_031, 9_000_061))
ALGORITHM_ORDER = (
    "fixed_assignment",
    "greedy",
    "ppo_hourly",
    "ppo_high_level",
    "ppo_event_residual",
    ALGORITHM,
    "rolling_milp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=(
            E1_ROOT
            / "iterative_q_state_recursive_ablation_20260730_run01"
            / "drop_hour_of_week"
        ),
    )
    parser.add_argument(
        "--base-comparison",
        type=Path,
        default=E1_ROOT / "algorithms" / "formal_comparison",
    )
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument(
        "--archived-source-root",
        default=(
            "experiments_results/archive/"
            "E1_pre_hour_removed_cleanup_20260730/"
            "iterative_q_state_recursive_ablation_20260730_run01/"
            "drop_hour_of_week"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def float_mean(rows: list[dict[str, str]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    return float(np.mean(values)) if values else None


def outcome_counts(
    rows: list[dict[str, str]],
    field: str = "delta_total_cost_vs_greedy_eur",
) -> tuple[int, int, int]:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    return (
        int(np.count_nonzero(values < -1e-6)),
        int(np.count_nonzero(np.abs(values) <= 1e-6)),
        int(np.count_nonzero(values > 1e-6)),
    )


def evaluation_rows(source_root: Path, model_seed: int) -> list[dict[str, str]]:
    path = (
        source_root
        / f"model_seed_{model_seed}"
        / "eval"
        / f"iqrec_0_s{model_seed}"
        / "evaluation.csv"
    )
    rows = read_csv(path)
    if [int(row["seed"]) for row in rows] != list(TEST_SEEDS):
        raise ValueError(f"{path}: formal seed coverage mismatch")
    return rows


def formal_gate(model_seed: int) -> str:
    return f"formal_iterative_action_q_no_hour_s{model_seed}"


def stage_models(
    staging_root: Path,
    source_root: Path,
    archived_source_root: str,
    evaluations: dict[int, list[dict[str, str]]],
) -> None:
    iterative_root = staging_root / "models" / "iterative_q"
    current_entries: dict[str, dict[str, str]] = {}
    for model_seed in MODEL_SEEDS:
        source = source_root / f"model_seed_{model_seed}"
        checkpoint_source = source / "p4" / "iterative_action_q.pt"
        summary_source = source / "p4" / "summary.json"
        budget_source = source / "budget.json"
        target = iterative_root / f"g60_p4_model_seed_{model_seed}"
        target.mkdir(parents=True, exist_ok=False)
        checkpoint_target = target / "iterative_action_q.pt"
        shutil.copy2(checkpoint_source, checkpoint_target)
        shutil.copy2(summary_source, target / "source_training_summary.json")
        shutil.copy2(budget_source, target / "budget.json")
        shutil.copy2(source / "schedule.txt", target / "source_schedule.txt")
        shutil.copy2(source / "job_ids.txt", target / "source_job_ids.txt")

        checkpoint = torch.load(
            checkpoint_target,
            map_location="cpu",
            weights_only=False,
        )
        metadata = checkpoint["metadata"]
        normalization = checkpoint["normalization"]
        if len(metadata["source_state_feature_names"]) != 94:
            raise ValueError("source observation width must be 94")
        if len(metadata["state_feature_names"]) != 93:
            raise ValueError("hour-removed observation width must be 93")
        if metadata["excluded_state_feature_names"] != ["hour_of_week"]:
            raise ValueError("checkpoint exclusion metadata mismatch")
        if len(normalization["state_mean"]) != 93:
            raise ValueError("checkpoint normalization width mismatch")

        rows = evaluations[model_seed]
        wins, ties, losses = outcome_counts(
            [
                {
                    "delta_total_cost_vs_greedy_eur": row[
                        "delta_total_cost_eur"
                    ]
                }
                for row in rows
            ]
        )
        budget = json.loads(budget_source.read_text(encoding="utf-8"))
        totals = budget["totals"]
        checkpoint_hash = sha256(checkpoint_target)
        checkpoint_rel = (
            "experiments_results/E1/models/iterative_q/"
            f"g60_p4_model_seed_{model_seed}/iterative_action_q.pt"
        )
        formal_rel = (
            "experiments_results/E1/algorithms/"
            f"{FORMAL_NAME}/model_seed_{model_seed}"
        )
        manifest = {
            "kind": "e1_iterative_q_model_manifest",
            "method": "Iterative Action-Q",
            "model_name": "G60-P4-no-hour",
            "model_seed": model_seed,
            "final_stage": "P4",
            "checkpoint": checkpoint_rel,
            "checkpoint_size_bytes": checkpoint_target.stat().st_size,
            "checkpoint_sha256": checkpoint_hash,
            "source_checkpoint": (
                f"{archived_source_root}/model_seed_{model_seed}/"
                "p4/iterative_action_q.pt"
            ),
            "source_training_summary": (
                "experiments_results/E1/models/iterative_q/"
                f"g60_p4_model_seed_{model_seed}/source_training_summary.json"
            ),
            "source_budget": (
                "experiments_results/E1/models/iterative_q/"
                f"g60_p4_model_seed_{model_seed}/budget.json"
            ),
            "training": {
                "observation_input": "shared_future_summary",
                "source_state_features": 94,
                "state_features": 93,
                "excluded_state_features": ["hour_of_week"],
                "forecast_context_hours": 168,
                "policy_iteration_stages": ["P1", "P2", "P3", "P4"],
                "g1_g3_regenerated_for_this_model_seed": True,
                "realized_training_roots": totals["train_roots"],
                "training_simulator_step_calls": totals[
                    "train_simulator_steps"
                ],
                "all_data_simulator_step_calls": totals[
                    "all_data_simulator_steps"
                ],
            },
            "deployment_gate": {
                "required_heads": 4,
                "q_margin_reward_units": 0.4,
                "q_margin_eur": 40_000,
                "maximum_interventions": 12,
                "policy_windows_h": [
                    [108, 155],
                    [156, 203],
                    [204, 251],
                    [252, 299],
                    [300, 347],
                    [348, 395],
                    [396, 443],
                    [444, 491],
                    [492, 539],
                    [540, 587],
                    [588, 635],
                    [636, 680],
                ],
            },
            "formal_test": {
                "directory": formal_rel,
                "seed_range_inclusive": [TEST_SEEDS[0], TEST_SEEDS[-1]],
                "episodes": len(rows),
                "mean_total_cost_eur": float_mean(rows, "total_cost_eur"),
                "mean_delta_vs_greedy_eur": float_mean(
                    rows,
                    "delta_total_cost_eur",
                ),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "mean_vented_t": float_mean(rows, "vented_t"),
                "mean_stored_t": float_mean(rows, "stored_t"),
                "mean_override_events": float_mean(rows, "override_events"),
            },
            "adoption": {
                "status": "current",
                "adopted_on": "2026-07-30",
                "test_accessed_before_adoption": True,
                "note": (
                    "Promoted by user decision after reviewing the paired "
                    "30-seed hour_of_week ablation."
                ),
            },
        }
        write_json(target / "model_manifest.json", manifest)
        (target / "checksums.sha256").write_text(
            f"{checkpoint_hash}  iterative_action_q.pt\n",
            encoding="utf-8",
        )
        (target / "README_ZH.md").write_text(
            "# 当前 E1 Iterative-Q 模型\n\n"
            f"- model seed：{model_seed}\n"
            "- final stage：P4\n"
            "- observation：删除 `hour_of_week`，93 维\n"
            "- G1–G3：按该模型 seed 自洽重新生成\n"
            f"- train simulator calls：{totals['train_simulator_steps']:,}\n"
            f"- checkpoint SHA-256：`{checkpoint_hash}`\n",
            encoding="utf-8",
        )
        current_entries[str(model_seed)] = {
            "checkpoint": checkpoint_rel,
            "checkpoint_sha256": checkpoint_hash,
        }

    primary = current_entries["0"]
    write_json(
        iterative_root / "current.json",
        {
            "kind": "e1_current_iterative_q_model",
            "method": "Iterative Action-Q",
            "selected_model": "G60-P4-no-hour",
            "status": "current",
            "adopted_on": "2026-07-30",
            "model_manifest": (
                "experiments_results/E1/models/iterative_q/"
                "g60_p4_model_seed_0/model_manifest.json"
            ),
            "checkpoint": primary["checkpoint"],
            "checkpoint_sha256": primary["checkpoint_sha256"],
            "model_seeds": list(MODEL_SEEDS),
            "replication_checkpoints": {
                key: value
                for key, value in current_entries.items()
                if key != "0"
            },
            "test_results": (
                "experiments_results/E1/algorithms/" + FORMAL_NAME
            ),
            "observation_schema": {
                "source_state_features": 94,
                "state_features": 93,
                "excluded_state_features": ["hour_of_week"],
            },
            "selection_provenance": {
                "test_accessed_before_adoption": True,
                "test_seed_range_inclusive": [
                    TEST_SEEDS[0],
                    TEST_SEEDS[-1],
                ],
                "adoption_basis": (
                    "User decision after paired 30-seed, three-model-seed "
                    "self-consistent ablation review on 2026-07-30."
                ),
                "holdout_status": (
                    "The 9000031-9000060 range is not an untouched "
                    "model-selection holdout."
                ),
            },
        },
    )


def stage_formal_results(
    staging_root: Path,
    source_root: Path,
    evaluations: dict[int, list[dict[str, str]]],
) -> None:
    formal_root = staging_root / "algorithms" / FORMAL_NAME
    formal_root.mkdir(parents=True, exist_ok=False)
    per_model_rows = []
    all_rows: list[dict[str, str]] = []
    for model_seed in MODEL_SEEDS:
        source = source_root / f"model_seed_{model_seed}"
        source_eval_dir = (
            source / "eval" / f"iqrec_0_s{model_seed}"
        )
        target = formal_root / f"model_seed_{model_seed}"
        target.mkdir(parents=True, exist_ok=False)
        rows = [dict(row) for row in evaluations[model_seed]]
        for row in rows:
            row["gate"] = formal_gate(model_seed)
        write_csv(target / "evaluation.csv", rows)
        all_rows.extend(rows)

        source_summary = json.loads(
            (source_eval_dir / "summary.json").read_text(encoding="utf-8")
        )
        old_gate = next(iter(source_summary["summary"]))
        source_summary["checkpoint"] = (
            "experiments_results/E1/models/iterative_q/"
            f"g60_p4_model_seed_{model_seed}/iterative_action_q.pt"
        )
        source_summary["gates"][0]["name"] = formal_gate(model_seed)
        source_summary["summary"] = {
            formal_gate(model_seed): source_summary["summary"][old_gate]
        }
        source_summary["formal_adoption"] = {
            "adopted_on": "2026-07-30",
            "excluded_state_features": ["hour_of_week"],
        }
        write_json(target / "summary.json", source_summary)

        checkpoint = (
            staging_root
            / "models"
            / "iterative_q"
            / f"g60_p4_model_seed_{model_seed}"
            / "iterative_action_q.pt"
        )
        checkpoint_hash = sha256(checkpoint)
        (target / "source_checkpoint.sha256").write_text(
            checkpoint_hash + "\n",
            encoding="utf-8",
        )
        evaluation_hash = sha256(target / "evaluation.csv")
        summary_hash = sha256(target / "summary.json")
        audit = {
            "model_seed": model_seed,
            "episodes": len(rows),
            "test_seed_range_inclusive": [TEST_SEEDS[0], TEST_SEEDS[-1]],
            "exact_seed_coverage": True,
            "checkpoint_sha256": checkpoint_hash,
            "evaluation_csv_sha256": evaluation_hash,
            "summary_json_sha256": summary_hash,
            "source_state_features": 94,
            "state_features": 93,
            "excluded_state_features": ["hour_of_week"],
            "detailed_episode_cost_fields": True,
            "formal_test_previously_accessed": True,
        }
        write_json(target / "audit.json", audit)
        write_json(
            target / "result_manifest.json",
            {
                "kind": "e1_formal_iterative_q_result_manifest",
                "method": "Iterative Action-Q G60-P4 (no hour)",
                "model_seed": model_seed,
                "evaluation_csv": (
                    "experiments_results/E1/algorithms/"
                    f"{FORMAL_NAME}/model_seed_{model_seed}/evaluation.csv"
                ),
                "checkpoint": (
                    "experiments_results/E1/models/iterative_q/"
                    f"g60_p4_model_seed_{model_seed}/iterative_action_q.pt"
                ),
                "hashes": {
                    "checkpoint": checkpoint_hash,
                    "evaluation_csv": evaluation_hash,
                    "summary_json": summary_hash,
                },
            },
        )
        shutil.copy2(
            source / "budget.json",
            target / "training_budget.json",
        )

        wins, ties, losses = outcome_counts(
            [
                {
                    "delta_total_cost_vs_greedy_eur": row[
                        "delta_total_cost_eur"
                    ]
                }
                for row in rows
            ]
        )
        seed_costs = np.asarray(
            [float(row["total_cost_eur"]) for row in rows],
            dtype=float,
        )
        per_model_rows.append(
            {
                "model_seed": model_seed,
                "episodes": len(rows),
                "mean_total_cost_eur": float(seed_costs.mean()),
                "sd_total_cost_eur": float(seed_costs.std(ddof=1)),
                "mean_delta_vs_greedy_eur": float_mean(
                    rows,
                    "delta_total_cost_eur",
                ),
                "mean_vented_t": float_mean(rows, "vented_t"),
                "mean_stored_t": float_mean(rows, "stored_t"),
                "mean_unit_cost_eur_per_t": float_mean(
                    rows,
                    "unit_cost_eur_per_t",
                ),
                "mean_override_events": float_mean(
                    rows,
                    "override_events",
                ),
                "wins_vs_greedy": wins,
                "ties_vs_greedy": ties,
                "losses_vs_greedy": losses,
            }
        )

        provenance = (
            formal_root / "provenance" / f"model_seed_{model_seed}"
        )
        provenance.mkdir(parents=True, exist_ok=False)
        for name in ("job_ids.txt", "schedule.txt", "budget.json"):
            shutil.copy2(source / name, provenance / name)

    write_csv(formal_root / "analysis" / "per_model_seed.csv", per_model_rows)
    costs = np.asarray(
        [float(row["total_cost_eur"]) for row in all_rows],
        dtype=float,
    )
    seed_means = np.asarray(
        [row["mean_total_cost_eur"] for row in per_model_rows],
        dtype=float,
    )
    greedy_costs = np.asarray(
        [float(row["greedy_total_cost_eur"]) for row in all_rows],
        dtype=float,
    )
    wins, ties, losses = outcome_counts(
        [
            {
                "delta_total_cost_vs_greedy_eur": row[
                    "delta_total_cost_eur"
                ]
            }
            for row in all_rows
        ]
    )
    aggregate = {
        "method": "Iterative Action-Q",
        "model": "G60-P4-no-hour",
        "excluded_state_features": ["hour_of_week"],
        "model_seeds": list(MODEL_SEEDS),
        "test_seed_range_inclusive": [TEST_SEEDS[0], TEST_SEEDS[-1]],
        "records": len(all_rows),
        "mean_total_cost_eur": float(costs.mean()),
        "between_model_seed_sample_sd_total_cost_eur": float(
            seed_means.std(ddof=1)
        ),
        "mean_greedy_total_cost_eur": float(greedy_costs.mean()),
        "mean_delta_vs_greedy_eur": float(
            (costs - greedy_costs).mean()
        ),
        "mean_vented_t": float_mean(all_rows, "vented_t"),
        "mean_stored_t": float_mean(all_rows, "stored_t"),
        "mean_unit_cost_eur_per_t": float_mean(
            all_rows,
            "unit_cost_eur_per_t",
        ),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }
    write_json(formal_root / "analysis" / "aggregate.json", aggregate)
    write_json(
        formal_root / "analysis" / "protocol_audit.json",
        {
            "scenario_protocol": "unified_window_v1",
            "stress_level": "medium",
            "forecast_context_hours": 168,
            "formal_seeds": list(TEST_SEEDS),
            "model_seeds": list(MODEL_SEEDS),
            "paired_across_model_seeds": True,
            "formal_test_previously_accessed": True,
            "observation_schema": {
                "source_state_features": 94,
                "state_features": 93,
                "excluded_state_features": ["hour_of_week"],
            },
        },
    )
    (formal_root / "README_ZH.md").write_text(
        "# E1 正式 Iterative Action-Q\n\n"
        "当前正式模型为 G60-P4-no-hour：删除 `hour_of_week`，保留其余 "
        "93 个状态特征。三个 model seed 均重新生成 G1–G3 并训练至 P4。\n\n"
        f"- 正式场景：{TEST_SEEDS[0]}–{TEST_SEEDS[-1]}（30 个）\n"
        f"- pooled mean total cost：€{aggregate['mean_total_cost_eur']:,.0f}\n"
        f"- pooled mean vented：{aggregate['mean_vented_t']:,.1f} t\n"
        f"- pooled wins/ties/losses vs Greedy：{wins}/{ties}/{losses}\n"
        "- 采用日期：2026-07-30；正式测试已在采用前访问。\n",
        encoding="utf-8",
    )


def comparison_episode_row(
    row: dict[str, str],
    model_seed: int,
) -> dict[str, object]:
    delta = float(row["delta_total_cost_eur"])
    outcome = "win" if delta < -1e-6 else "loss" if delta > 1e-6 else "tie"
    event_count = int(row["event_count"])
    return {
        "algorithm": ALGORITHM,
        "algorithm_display_name": DISPLAY_NAME,
        "method_class": METHOD_CLASS,
        "model_seed": model_seed,
        "test_seed": int(row["seed"]),
        "episode_hours": 720,
        "decision_count": event_count,
        "mean_decision_interval_h": 720.0 / event_count,
        "episode_vessel_fuel_eur": row["episode_vessel_fuel_eur"],
        "episode_conditioning_eur": row["episode_conditioning_eur"],
        "episode_reconditioning_eur": row[
            "episode_reconditioning_eur"
        ],
        "episode_loading_eur": row["episode_loading_eur"],
        "episode_unloading_eur": row["episode_unloading_eur"],
        "episode_operating_cost_eur": row["episode_operating_cost_eur"],
        "episode_vent_penalty_eur": row["episode_vent_penalty_eur"],
        "episode_storage_shortfall_penalty_eur": row[
            "episode_storage_shortfall_penalty_eur"
        ],
        "episode_total_cost_eur": row["episode_total_cost_eur"],
        "terminal_cleanup_operating_cost_eur": row[
            "terminal_cleanup_operating_cost_eur"
        ],
        "operating_cost_eur": row["operating_cost_eur"],
        "total_cost_eur": row["total_cost_eur"],
        "captured_t": "",
        "stored_t": row["stored_t"],
        "vented_t": row["vented_t"],
        "storage_rate": "",
        "loss_rate": "",
        "unit_total_cost_eur_per_t": row["unit_cost_eur_per_t"],
        "greedy_total_cost_eur": row["greedy_total_cost_eur"],
        "delta_total_cost_vs_greedy_eur": row[
            "delta_total_cost_eur"
        ],
        "paired_outcome_vs_greedy": outcome,
        "hard_violations": "",
        "override_events": row["override_events"],
        "proposed_override_events": row["proposed_override_events"],
        "selected_interventions": "",
        "effective_intervention_rate": "",
        "solver_replan_count": "",
        "solver_failure_count": "",
        "solver_timeout_count": "",
        "fallback_used": "",
        "wall_clock_seconds": row.get("wall_clock_seconds", ""),
        "source_file": (
            f"{FORMAL_NAME}/model_seed_{model_seed}/evaluation.csv"
        ),
    }


MEAN_FIELDS = (
    "episode_vessel_fuel_eur",
    "episode_conditioning_eur",
    "episode_reconditioning_eur",
    "episode_loading_eur",
    "episode_unloading_eur",
    "episode_operating_cost_eur",
    "episode_vent_penalty_eur",
    "episode_storage_shortfall_penalty_eur",
    "episode_total_cost_eur",
    "terminal_cleanup_operating_cost_eur",
    "operating_cost_eur",
    "total_cost_eur",
    "captured_t",
    "stored_t",
    "vented_t",
    "storage_rate",
    "loss_rate",
    "unit_total_cost_eur_per_t",
    "delta_total_cost_vs_greedy_eur",
)


def group_mean(rows: list[dict[str, object]], field: str) -> object:
    values = [
        float(row[field])
        for row in rows
        if row.get(field, "") not in ("", None)
    ]
    return float(np.mean(values)) if values else ""


def summarize_model_rows(
    episode_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in episode_rows:
        groups[(str(row["algorithm"]), str(row["model_seed"]))].append(row)
    output = []
    for algorithm in ALGORITHM_ORDER:
        keys = sorted(
            (key for key in groups if key[0] == algorithm),
            key=lambda item: item[1],
        )
        for key in keys:
            rows = groups[key]
            deltas = np.asarray(
                [float(row["delta_total_cost_vs_greedy_eur"]) for row in rows]
            )
            costs = np.asarray(
                [float(row["total_cost_eur"]) for row in rows]
            )
            wins, ties, losses = outcome_counts(
                [
                    {
                        "delta_total_cost_vs_greedy_eur": str(value)
                    }
                    for value in deltas
                ]
            )
            item: dict[str, object] = {
                "algorithm": algorithm,
                "algorithm_display_name": rows[0][
                    "algorithm_display_name"
                ],
                "method_class": rows[0]["method_class"],
                "model_seed": rows[0]["model_seed"],
                "episodes": len(rows),
            }
            for field in MEAN_FIELDS:
                item[f"mean_{field}"] = group_mean(rows, field)
            item["sd_total_cost_eur"] = float(costs.std(ddof=1))
            item["sd_delta_total_cost_vs_greedy_eur"] = float(
                deltas.std(ddof=1)
            )
            item["wins_vs_greedy"] = wins
            item["ties_vs_greedy"] = ties
            item["losses_vs_greedy"] = losses
            output.append(item)
    return output


def summarize_algorithm_rows(
    episode_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in episode_rows:
        groups[str(row["algorithm"])].append(row)
    output = []
    for algorithm in ALGORITHM_ORDER:
        rows = groups[algorithm]
        model_seeds = sorted(
            {
                str(row["model_seed"])
                for row in rows
                if str(row["model_seed"]) != ""
            }
        )
        costs = np.asarray([float(row["total_cost_eur"]) for row in rows])
        deltas = np.asarray(
            [float(row["delta_total_cost_vs_greedy_eur"]) for row in rows]
        )
        wins, ties, losses = outcome_counts(
            [
                {"delta_total_cost_vs_greedy_eur": str(value)}
                for value in deltas
            ]
        )
        item: dict[str, object] = {
            "algorithm": algorithm,
            "algorithm_display_name": rows[0]["algorithm_display_name"],
            "method_class": rows[0]["method_class"],
            "model_instances": len(model_seeds) if model_seeds else 1,
            "trained_model_seed_count": len(model_seeds),
            "episode_records": len(rows),
        }
        for field in MEAN_FIELDS:
            item[f"mean_{field}"] = group_mean(rows, field)
        item["pooled_episode_sd_total_cost_eur"] = float(costs.std(ddof=1))
        if len(model_seeds) > 1:
            seed_means = np.asarray(
                [
                    np.mean(
                        [
                            float(row["total_cost_eur"])
                            for row in rows
                            if str(row["model_seed"]) == model_seed
                        ]
                    )
                    for model_seed in model_seeds
                ]
            )
            item["between_model_seed_sd_mean_total_cost_eur"] = float(
                seed_means.std(ddof=1)
            )
        else:
            item["between_model_seed_sd_mean_total_cost_eur"] = ""
        item["wins_vs_greedy"] = wins
        item["ties_vs_greedy"] = ties
        item["losses_vs_greedy"] = losses
        output.append(item)
    return output


def stage_comparison(
    staging_root: Path,
    base_comparison: Path,
    evaluations: dict[int, list[dict[str, str]]],
) -> None:
    output = staging_root / "algorithms" / "formal_comparison"
    output.mkdir(parents=True, exist_ok=False)
    base_rows: list[dict[str, object]] = [
        dict(row)
        for row in read_csv(base_comparison / "e1_formal_per_episode.csv")
        if row["algorithm"] != ALGORITHM
    ]
    new_rows = [
        comparison_episode_row(row, model_seed)
        for model_seed in MODEL_SEEDS
        for row in evaluations[model_seed]
    ]
    rows = base_rows + new_rows
    order = {name: index for index, name in enumerate(ALGORITHM_ORDER)}
    rows.sort(
        key=lambda row: (
            order[str(row["algorithm"])],
            str(row["model_seed"]),
            int(row["test_seed"]),
        )
    )
    if len(rows) != 450:
        raise ValueError(f"comparison must contain 450 rows, got {len(rows)}")
    write_csv(output / "e1_formal_per_episode.csv", rows)
    write_csv(
        output / "e1_formal_per_model_seed.csv",
        summarize_model_rows(rows),
    )
    write_csv(
        output / "e1_formal_per_algorithm.csv",
        summarize_algorithm_rows(rows),
    )
    shutil.copy2(base_comparison / "schema.json", output / "schema.json")
    audit = json.loads(
        (base_comparison / "audit.json").read_text(encoding="utf-8")
    )
    audit["version"] = "2026-07-30-hour-removed-iterative-q"
    audit["source_directories"] = {
        key: f"algorithms/{value}"
        for key, value in audit["source_directories"].items()
    }
    audit["source_directories"][ALGORITHM] = (
        f"algorithms/{FORMAL_NAME}"
    )
    audit["iterative_q_adoption"] = {
        "model": "G60-P4-no-hour",
        "excluded_state_features": ["hour_of_week"],
        "adopted_on": "2026-07-30",
        "formal_test_previously_accessed": True,
    }
    audit["output_sha256"] = {
        name: sha256(output / name)
        for name in (
            "e1_formal_per_episode.csv",
            "e1_formal_per_model_seed.csv",
            "e1_formal_per_algorithm.csv",
            "schema.json",
        )
    }
    write_json(output / "audit.json", audit)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    base_comparison = args.base_comparison.resolve()
    staging_root = args.staging_root.resolve()
    if staging_root.exists():
        raise FileExistsError(f"refusing existing staging root: {staging_root}")
    staging_root.mkdir(parents=True)
    evaluations = {
        model_seed: evaluation_rows(source_root, model_seed)
        for model_seed in MODEL_SEEDS
    }
    stage_models(
        staging_root,
        source_root,
        args.archived_source_root,
        evaluations,
    )
    stage_formal_results(staging_root, source_root, evaluations)
    stage_comparison(staging_root, base_comparison, evaluations)
    print(f"E1_HOUR_REMOVED_PROMOTION_STAGED root={staging_root}")


if __name__ == "__main__":
    main()
