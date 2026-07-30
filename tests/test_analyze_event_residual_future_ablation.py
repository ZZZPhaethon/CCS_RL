import json
from types import SimpleNamespace

import pytest

from experiments.analyze_event_residual_future_ablation import analyze


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_reports_paired_improvement_vs_greedy(tmp_path):
    greedy_records = [
        {"seed": 1, "total_cost_eur": 100.0},
        {"seed": 2, "total_cost_eur": 200.0},
    ]
    _write_json(
        tmp_path / "greedy" / "results.json",
        {"per_seed": greedy_records},
    )
    variant = tmp_path / "summary_168"
    _write_json(
        variant / "config.json",
        {"future_summary_windows_h": [168]},
    )
    _write_json(
        variant
        / "evaluation"
        / "best__hardprob0__seeds_1-2"
        / "results.json",
        {
            "per_seed": [
                {
                    "seed": 1,
                    "total_cost_eur": 90.0,
                    "vented_t": 3.0,
                    "stored_t": 10.0,
                    "selected_interventions": 2,
                },
                {
                    "seed": 2,
                    "total_cost_eur": 180.0,
                    "vented_t": 5.0,
                    "stored_t": 20.0,
                    "selected_interventions": 4,
                },
            ]
        },
    )

    payload = analyze(
        SimpleNamespace(
            run_root=tmp_path,
            variants=["summary_168"],
            bootstrap_samples=100,
            bootstrap_seed=0,
        )
    )

    row = payload["variants"][0]
    assert row["mean_cost_delta_vs_greedy_eur"] == pytest.approx(-15.0)
    assert row["relative_cost_improvement_percent"] == pytest.approx(10.0)
    assert row["paired_relative_improvement_mean_percent"] == pytest.approx(
        10.0
    )
    assert row["wins_vs_greedy"] == 2
    assert (tmp_path / "future_ablation_summary.md").exists()
