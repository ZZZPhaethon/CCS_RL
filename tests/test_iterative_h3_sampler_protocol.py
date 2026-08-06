import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "experiments"
    / "protocols"
    / "iterative_h3_sampler_validation_protocol.json"
)


def test_protocol_omits_formal_test_and_locks_controller_validation():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["formal_test_access"] is False
    assert protocol["formal_test_stage_included"] is False
    assert protocol["controller_validation_seeds"] == list(
        range(8100001, 8100021)
    )


def test_protocol_locks_h3_collection_and_three_sampler_variants():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["collection_gate"]["required_heads"] == 3
    assert protocol["collection_gate"]["residual_margin"] == 0.4
    assert protocol["model_seeds"] == [0, 1, 2]
    assert set(protocol["p2_variants"]) == {
        "b_gate_only",
        "c_dedup_balanced",
        "d_dedup_advantage",
    }
    assert (
        protocol["p2_variants"]["c_dedup_balanced"][
            "stage_sampling_temperature"
        ]
        == 0.5
    )
    assert (
        protocol["p2_variants"]["d_dedup_advantage"][
            "root_advantage_weighting"
        ]
        == "stratified"
    )


def test_new_launcher_contains_no_formal_seed_literals():
    launcher = (
        ROOT / "hpc" / "launch_iterative_h3_sampler_p2.sh"
    ).read_text(encoding="utf-8")
    evaluator = (
        ROOT / "hpc" / "submit_iterative_h3_sampler_validation.sh"
    ).read_text(encoding="utf-8")

    assert "9000" not in launcher
    assert "9000" not in evaluator
    assert "--validation-only" in evaluator


def test_recursive_launcher_keeps_two_routes_and_validation_only():
    launcher = (
        ROOT / "hpc" / "launch_iterative_h3_sampler_recursive.sh"
    ).read_text(encoding="utf-8")
    evaluator = (
        ROOT / "hpc" / "submit_iterative_h3_recursive_validation.sh"
    ).read_text(encoding="utf-8")

    assert "9000" not in launcher
    assert "9000" not in evaluator
    assert "SELECTED_VARIANT" in launcher
    assert "--array=0-29%12" in launcher
    assert "--array=0-47%12" in launcher
    assert "--validation-only" in evaluator
