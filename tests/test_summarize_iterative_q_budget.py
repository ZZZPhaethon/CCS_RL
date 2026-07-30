import json

import numpy as np

from experiments.summarize_iterative_q_budget import summarize


def _write_stage(root, stage, split, roots, candidates, steps):
    path = root / stage / f"{split}_merged.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "root_count": roots,
        "candidate_count": candidates,
        "scenario_seeds": list(range(max(1, roots // 2))),
        "training_simulator_usage": {
            "simulator_step_calls": steps,
            "simulator_simulated_hours": float(steps),
            "simulator_hour_steps": float(steps),
        },
    }
    np.savez_compressed(
        path, metadata_json=np.asarray(json.dumps(metadata))
    )


def test_summarize_counts_only_contiguous_stages(tmp_path):
    for stage, train_values, validation_values in (
        ("g0", (10, 40, 1000), (2, 8, 200)),
        ("g1", (5, 20, 500), (1, 4, 100)),
    ):
        _write_stage(tmp_path, stage, "train", *train_values)
        _write_stage(
            tmp_path, stage, "validation", *validation_values
        )

    result = summarize(tmp_path, target_train_steps=1600)

    assert result["stages"] == ["g0", "g1"]
    assert result["totals"] == {
        "train_roots": 15,
        "validation_roots": 3,
        "train_candidates": 60,
        "validation_candidates": 12,
        "train_simulator_steps": 1500,
        "validation_simulator_steps": 300,
        "all_data_simulator_steps": 1800,
        "train_step_error": -100,
        "train_step_relative_error_pct": -6.25,
    }
