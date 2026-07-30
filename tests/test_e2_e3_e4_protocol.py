import json
from pathlib import Path

from sim.control.event_based.residual_rl_v4.scenario import (
    UNIFIED_WINDOW_STRESS_PROFILES,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "experiments"
    / "protocols"
    / "e2_e3_e4_iterative_q_protocol.json"
)


def test_e2_e3_e4_protocol_locks_current_test_seeds_and_model_seeds():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["formal_test"]["range_inclusive"] == [9000031, 9000060]
    assert protocol["formal_test"]["count"] == 30
    assert protocol["frozen_e1_model"]["name"] == "G60-P4"
    assert protocol["frozen_e1_model"]["model_seeds"] == [0, 1, 2]
    assert protocol["frozen_e1_model"]["required_heads"] == 4
    assert protocol["frozen_e1_model"]["maximum_interventions"] == 12


def test_e4_protocol_matches_implemented_stress_profiles():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    for stress_level, implemented in UNIFIED_WINDOW_STRESS_PROFILES.items():
        locked = protocol["E4"]["profiles"][stress_level]
        for key, value in implemented.items():
            actual = locked[key]
            if isinstance(value, tuple):
                assert tuple(actual) == value
            else:
                assert actual == value


def test_locked_eval_script_contains_exact_30_seed_block():
    script = (
        ROOT / "hpc" / "submit_locked_iterative_q_eval.sh"
    ).read_text(encoding="utf-8")

    for seed in range(9000031, 9000061):
        assert str(seed) in script
    assert "9000030" not in script
    assert "9000061" not in script
