import csv
import json
from pathlib import Path

from experiments.analyze_iterative_q_budget_search import analyze


def _write_evaluation(path: Path, costs: list[float]) -> None:
    path.parent.mkdir(parents=True)
    fields = [
        "gate",
        "seed",
        "total_cost_eur",
        "greedy_total_cost_eur",
        "delta_total_cost_eur",
        "operating_cost_eur",
        "vent_penalty_eur",
        "vented_t",
        "stored_t",
        "unit_cost_eur_per_t",
        "override_events",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for offset, cost in enumerate(costs, start=1):
            writer.writerow(
                {
                    "gate": "test",
                    "seed": 8_100_000 + offset,
                    "total_cost_eur": cost,
                    "greedy_total_cost_eur": 2_000_000.0,
                    "delta_total_cost_eur": cost - 2_000_000.0,
                    "operating_cost_eur": cost,
                    "vent_penalty_eur": 0.0,
                    "vented_t": 0.0,
                    "stored_t": 100_000.0,
                    "unit_cost_eur_per_t": cost / 100_000.0,
                    "override_events": 10,
                }
            )


def _write_run(
    search_root: Path,
    name: str,
    final_stage: str,
    costs: list[float],
) -> Path:
    run_root = search_root / "runs" / name
    run_root.mkdir(parents=True)
    (run_root / "schedule.txt").write_text(
        "\n".join(
            [
                f"config={name}",
                f"final_stage={final_stage}",
                "g0_train_count=120",
                "weighted_train_roots=3600",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_root / "budget.json").write_text(
        json.dumps(
            {
                "train": [{"simulator_step_calls": 3_800_000}],
                "totals": {
                    "train_roots": 3600,
                    "train_simulator_steps": 9_525_119,
                    "train_step_relative_error_pct": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_evaluation(
        run_root / "eval" / name / "evaluation.csv", costs
    )
    return run_root


def test_selection_rejects_large_regression_from_previous_stage(
    tmp_path: Path,
) -> None:
    search_root = tmp_path / "search"
    stable = _write_run(
        search_root, "stable_p2", "p2", [1_800_000.0] * 20
    )
    _write_evaluation(
        stable / "eval" / "stable_p2_p1" / "evaluation.csv",
        [1_850_000.0] * 20,
    )
    unstable = _write_run(
        search_root,
        "unstable_p3",
        "p3",
        [1_900_000.0] + [1_700_000.0] * 19,
    )
    _write_evaluation(
        unstable / "eval" / "unstable_p3_p1" / "evaluation.csv",
        [1_900_000.0] * 20,
    )
    _write_evaluation(
        unstable / "eval" / "unstable_p3_p2" / "evaluation.csv",
        [1_600_000.0] + [1_800_000.0] * 19,
    )

    payload = analyze(search_root)
    by_name = {row["config"]: row for row in payload["results"]}

    assert payload["recommended_by_constrained_mean"] == "stable_p2"
    assert by_name["unstable_p3"]["tail_feasible"]
    assert not by_name["unstable_p3"]["stage_retention_feasible"]
    assert (
        by_name["unstable_p3"]["worst_regression_from_previous_eur"]
        == 300_000.0
    )
