import json
from pathlib import Path

from experiments.gate_sweep_configs import (
    FORMAL_TEST_SEEDS,
    VALIDATION_SEEDS,
    WINDOW_SCHEMES,
    gate_cli_values,
    gate_records,
)
from experiments.evaluate_iterative_action_q import parse_args, parse_gate


def test_gate_sweep_seed_sets_are_locked_and_disjoint():
    assert VALIDATION_SEEDS == tuple(range(8100001, 8100021))
    assert FORMAL_TEST_SEEDS == tuple(range(9000031, 9000061))
    assert set(VALIDATION_SEEDS).isdisjoint(FORMAL_TEST_SEEDS)


def test_window_schemes_cover_the_same_locked_interval():
    assert len(WINDOW_SCHEMES["w12"]) == 12
    assert len(WINDOW_SCHEMES["w24"]) == 24
    assert len(WINDOW_SCHEMES["w48"]) == 48
    for name in ("w12", "w24", "w48"):
        windows = WINDOW_SCHEMES[name]
        assert windows[0][0] == 108
        assert windows[-1][1] == 680
        assert all(
            previous[1] + 1 == current[0]
            for previous, current in zip(windows, windows[1:])
        )


def test_gate_grid_is_unique_and_parseable():
    records = gate_records()
    values = gate_cli_values()
    assert len(records) == 31
    assert len({record["name"] for record in records}) == len(records)
    assert len(set(values)) == len(values)
    assert [parse_gate(value)["name"] for value in values] == [
        record["name"] for record in records
    ]
    assert any(
        record["required_heads"] == 3
        and record["margin"] == 0.0
        for record in records
    )
    assert any(
        record["window_scheme"] == "global"
        and record["max_overrides"] == 48
        for record in records
    )


def test_all_gate_values_survive_one_nargs_cli_option():
    args = parse_args(
        [
            "--checkpoint",
            "unused.pt",
            "--out-dir",
            "unused",
            "--eval-seeds",
            *[str(seed) for seed in VALIDATION_SEEDS],
            "--validation-only",
            "--gates",
            *gate_cli_values(),
        ]
    )
    assert len(args.gates) == 31
    assert [gate["name"] for gate in args.gates] == [
        record["name"] for record in gate_records()
    ]


def test_followup_protocol_uses_validation_only_for_gate_selection():
    protocol_path = (
        Path(__file__).parents[1]
        / "experiments"
        / "protocols"
        / "e2_followup_one_shot_stress_gate_sweep_protocol.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["one_shot_e4"]["formal_test_range_inclusive"] == [
        9000031,
        9000060,
    ]
    assert protocol["gate_sweep"]["validation_range_inclusive"] == [
        8100001,
        8100020,
    ]
    assert protocol["gate_sweep"]["formal_test_prohibited"] is True
