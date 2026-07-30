import csv
from pathlib import Path

from experiments.analyze_iterative_q_router_validation import (
    analyze,
    checkpoint_oracle,
)


def _write_evaluation(
    root: Path,
    name: str,
    costs: list[float],
) -> None:
    path = root / name / "evaluation.csv"
    path.parent.mkdir(parents=True)
    fields = [
        "seed",
        "total_cost_eur",
        "greedy_total_cost_eur",
        "delta_total_cost_eur",
        "vented_t",
        "stored_t",
        "override_events",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for offset, cost in enumerate(costs, start=1):
            writer.writerow(
                {
                    "seed": 8_100_000 + offset,
                    "total_cost_eur": cost,
                    "greedy_total_cost_eur": 2_000_000.0,
                    "delta_total_cost_eur": cost - 2_000_000.0,
                    "vented_t": 0.0,
                    "stored_t": 100_000.0,
                    "override_events": 10,
                }
            )


def test_analysis_keeps_development_and_confirmation_separate(
    tmp_path: Path,
) -> None:
    _write_evaluation(tmp_path, "p4_reference", [1_800_000.0] * 20)
    _write_evaluation(
        tmp_path,
        "router",
        [1_700_000.0] * 10 + [1_850_000.0] * 10,
    )

    payload = analyze(tmp_path)
    rows = {row["router"]: row for row in payload["results"]}

    assert rows["router"]["development_mean_difference_vs_p4_eur"] == -100_000.0
    assert rows["router"]["confirmation_mean_difference_vs_p4_eur"] == 50_000.0
    assert not rows["router"]["confirmation_improves_mean"]
    assert payload["formal_test_accessed"] is False


def test_checkpoint_oracle_reports_headroom_over_p4(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    _write_evaluation(checkpoint_root, "p1", [1_800_000.0] * 20)
    _write_evaluation(checkpoint_root, "p4", [1_900_000.0] * 20)
    for name in ("p1", "p4"):
        source = checkpoint_root / name / "evaluation.csv"
        source.replace(checkpoint_root / f"{name}.csv")
        source.parent.rmdir()

    oracle = checkpoint_oracle(checkpoint_root)

    assert oracle["oracle_mean_cost_eur"] == 1_800_000.0
    assert oracle["oracle_gain_vs_p4_eur"] == 100_000.0
    assert oracle["oracle_winner_counts"] == {"p1": 20, "p4": 0}
