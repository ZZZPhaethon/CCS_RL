"""Rebuild the canonical E1 comparison with promoted Rolling MILP results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path


EXPECTED_SEEDS = set(range(9_000_031, 9_000_061))
ALGORITHM = "rolling_milp"
DISPLAY_NAME = "Rolling MILP H168-R24-T600s CPLEX 22.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-comparison-dir", type=Path, required=True)
    parser.add_argument("--rolling-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_promoted_rows(
    rolling_root: Path,
    greedy_by_seed: dict[int, float],
    fieldnames: list[str],
) -> list[dict[str, object]]:
    summaries = []
    for path in sorted(rolling_root.glob("seed_*/smoke_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries.append((path, summary["rows"][0]))
    seeds = {int(row["seed"]) for _, row in summaries}
    if len(summaries) != 30 or seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            f"incomplete promoted Rolling results: rows={len(summaries)}, "
            f"missing={sorted(EXPECTED_SEEDS - seeds)}"
        )

    result = []
    for path, source in summaries:
        seed = int(source["seed"])
        if source["controller"] != ALGORITHM or source["run_status"] != "completed":
            raise RuntimeError(f"invalid promoted row in {path}")
        greedy_cost = greedy_by_seed[seed]
        total_cost = float(source["total_cost"])
        difference = total_cost - greedy_cost
        outcome = "tie"
        if difference < -1e-6:
            outcome = "win"
        elif difference > 1e-6:
            outcome = "loss"
        values = {
            "algorithm": ALGORITHM,
            "algorithm_display_name": DISPLAY_NAME,
            "method_class": "optimization",
            "model_seed": "",
            "test_seed": seed,
            "episode_hours": source["horizon_hours"],
            "decision_count": source["controller_decision_calls"],
            "mean_decision_interval_h": (
                float(source["horizon_hours"])
                / float(source["controller_decision_calls"])
            ),
            "episode_vessel_fuel_eur": source["vessel_fuel"],
            "episode_conditioning_eur": source["conditioning"],
            "episode_reconditioning_eur": source["reconditioning"],
            "episode_loading_eur": source["loading"],
            "episode_unloading_eur": source["unloading"],
            "episode_operating_cost_eur": source["episode_operating_cost"],
            "episode_vent_penalty_eur": source["vent_penalty"],
            "episode_storage_shortfall_penalty_eur": source[
                "storage_shortfall_penalty"
            ],
            "episode_total_cost_eur": source["episode_total_cost"],
            "terminal_cleanup_operating_cost_eur": source[
                "terminal_cleanup_operating_cost"
            ],
            "operating_cost_eur": source["operating_cost"],
            "total_cost_eur": total_cost,
            "captured_t": source["captured_t"],
            "stored_t": source["stored_t"],
            "vented_t": source["vented_t"],
            "storage_rate": source["storage_rate"],
            "loss_rate": source["loss_rate"],
            "unit_total_cost_eur_per_t": source["total_cost_per_stored_t"],
            "greedy_total_cost_eur": greedy_cost,
            "delta_total_cost_vs_greedy_eur": difference,
            "paired_outcome_vs_greedy": outcome,
            "hard_violations": "",
            "override_events": "",
            "proposed_override_events": "",
            "selected_interventions": "",
            "effective_intervention_rate": "",
            "solver_replan_count": source["solver_replan_count"],
            "solver_failure_count": source["solver_failure_count"],
            "solver_timeout_count": source["solver_timeout_count"],
            "fallback_used": source["fallback_used"],
            "wall_clock_seconds": source["wall_clock_seconds"],
            "source_file": str(path.relative_to(rolling_root.parent)),
        }
        result.append({field: values.get(field, "") for field in fieldnames})
    return result


def _summary_values(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = (
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
    result = {
        f"mean_{metric}": statistics.fmean(float(row[metric]) for row in rows)
        for metric in metrics
    }
    costs = [float(row["total_cost_eur"]) for row in rows]
    differences = [
        float(row["delta_total_cost_vs_greedy_eur"]) for row in rows
    ]
    result.update(
        {
            "sd_total_cost_eur": statistics.stdev(costs),
            "sd_delta_total_cost_vs_greedy_eur": statistics.stdev(differences),
            "wins_vs_greedy": sum(
                row["paired_outcome_vs_greedy"] == "win" for row in rows
            ),
            "ties_vs_greedy": sum(
                row["paired_outcome_vs_greedy"] == "tie" for row in rows
            ),
            "losses_vs_greedy": sum(
                row["paired_outcome_vs_greedy"] == "loss" for row in rows
            ),
        }
    )
    return result


def _replace_summary_row(
    base_rows: list[dict[str, str]],
    fieldnames: list[str],
    promoted_rows: list[dict[str, object]],
    *,
    algorithm_level: bool,
) -> list[dict[str, object]]:
    values = _summary_values(promoted_rows)
    row = {
        "algorithm": ALGORITHM,
        "algorithm_display_name": DISPLAY_NAME,
        "method_class": "optimization",
        "model_seed": "",
        "episodes": 30,
        "model_instances": 1,
        "trained_model_seed_count": 0,
        "episode_records": 30,
        **values,
    }
    if algorithm_level:
        row["pooled_episode_sd_total_cost_eur"] = values["sd_total_cost_eur"]
        row["between_model_seed_sd_mean_total_cost_eur"] = ""
    return [
        *[item for item in base_rows if item["algorithm"] != ALGORITHM],
        {field: row.get(field, "") for field in fieldnames},
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> Path:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)

    episode_fields, base_episodes = _read_csv(
        args.base_comparison_dir / "e1_formal_per_episode.csv"
    )
    greedy_by_seed = {
        int(row["test_seed"]): float(row["total_cost_eur"])
        for row in base_episodes
        if row["algorithm"] == "greedy"
    }
    if set(greedy_by_seed) != EXPECTED_SEEDS:
        raise RuntimeError("base comparison has incomplete Greedy coverage")
    promoted = _load_promoted_rows(
        args.rolling_root,
        greedy_by_seed,
        episode_fields,
    )
    episodes = [
        *[row for row in base_episodes if row["algorithm"] != ALGORITHM],
        *promoted,
    ]
    _write_csv(
        args.out_dir / "e1_formal_per_episode.csv",
        episode_fields,
        episodes,
    )

    model_fields, base_models = _read_csv(
        args.base_comparison_dir / "e1_formal_per_model_seed.csv"
    )
    _write_csv(
        args.out_dir / "e1_formal_per_model_seed.csv",
        model_fields,
        _replace_summary_row(
            base_models,
            model_fields,
            promoted,
            algorithm_level=False,
        ),
    )
    algorithm_fields, base_algorithms = _read_csv(
        args.base_comparison_dir / "e1_formal_per_algorithm.csv"
    )
    _write_csv(
        args.out_dir / "e1_formal_per_algorithm.csv",
        algorithm_fields,
        _replace_summary_row(
            base_algorithms,
            algorithm_fields,
            promoted,
            algorithm_level=True,
        ),
    )
    shutil.copy2(
        args.base_comparison_dir / "schema.json",
        args.out_dir / "schema.json",
    )

    audit = json.loads(
        (args.base_comparison_dir / "audit.json").read_text(encoding="utf-8")
    )
    audit["version"] = "2026-07-30"
    audit["source_directories"][ALGORITHM] = args.rolling_root.name
    audit["milp_result_promotion"] = {
        "status": "600s_run03_promoted_as_primary",
        "superseded_result_retained": True,
        "single_factor_time_limit_comparison": False,
        "source_hashes_differ_from_superseded_run": True,
    }
    audit["output_sha256"] = {
        name: _sha256(args.out_dir / name)
        for name in (
            "e1_formal_per_episode.csv",
            "e1_formal_per_model_seed.csv",
            "e1_formal_per_algorithm.csv",
            "schema.json",
        )
    }
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    return args.out_dir


if __name__ == "__main__":
    print(run(parse_args()))
