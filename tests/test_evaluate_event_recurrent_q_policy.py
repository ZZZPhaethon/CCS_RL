import argparse

import numpy as np
import pytest

from experiments.evaluate_event_recurrent_q_policy import (
    parse_args,
    parse_gate,
    select_safe_action,
)


def test_safe_gate_accepts_consensus_override_with_positive_margin():
    q = np.zeros((5, 4))
    q[:, 2] = [0.4, 0.5, 0.6, 0.7, 0.1]
    action, info = select_safe_action(
        q, np.ones(4, dtype=bool), 3, required_heads=4, margin=0.2
    )
    assert action == 2
    assert info["agreement"] == 5
    assert info["positive_heads"] == 4


def test_safe_gate_falls_back_when_heads_disagree():
    q = np.zeros((5, 4))
    for head in range(5):
        q[head, head % 3] = 1.0
    action, _info = select_safe_action(
        q, np.ones(4, dtype=bool), 3, required_heads=4, margin=0.0
    )
    assert action == 3


def test_safe_gate_can_reject_high_uncertainty_advantage():
    q = np.zeros((5, 4))
    q[:, 2] = [1.0, 1.0, 1.0, 1.0, -1.0]
    accepted, _info = select_safe_action(
        q,
        np.ones(4, dtype=bool),
        3,
        required_heads=4,
        margin=0.0,
        uncertainty_beta=0.0,
    )
    rejected, info = select_safe_action(
        q,
        np.ones(4, dtype=bool),
        3,
        required_heads=4,
        margin=0.0,
        uncertainty_beta=1.0,
    )
    assert accepted == 2
    assert rejected == 3
    assert info["lower_confidence_advantage"] < 0.0


def test_safe_gate_never_selects_illegal_action():
    q = np.zeros((5, 4))
    q[:, 1] = 100.0
    action, _info = select_safe_action(
        q, np.asarray([True, False, True, True]), 3, required_heads=1, margin=0.0
    )
    assert action != 1


def test_gate_parser_supports_optional_diagnostic_override_budget():
    assert parse_gate("strict:5:0.25:2")["max_overrides"] == 2
    assert parse_gate("strict:5:0.25")["max_overrides"] is None
    windowed = parse_gate("window:4:0.1:1:60:100")
    assert windowed["max_overrides"] == 1
    assert windowed["min_hour"] == 60.0
    assert windowed["max_hour"] == 100.0
    multi = parse_gate("multi:4:0.1:3:100-150,200-250,300-350")
    assert multi["max_overrides"] == 3
    assert multi["windows"] == [[100.0, 150.0], [200.0, 250.0], [300.0, 350.0]]
    uncertain = parse_gate("uncertain:4:0.1:3:100-150,200-250:1.25")
    assert uncertain["windows"] == [[100.0, 150.0], [200.0, 250.0]]
    assert uncertain["uncertainty_beta"] == 1.25
    with pytest.raises(argparse.ArgumentTypeError):
        parse_gate("invalid")


def test_policy_eval_parser_supports_resetting_recurrent_history():
    args = parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--out-dir",
            "out",
            "--reset-recurrent-state",
        ]
    )
    assert args.reset_recurrent_state is True
