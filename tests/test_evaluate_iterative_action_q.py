import argparse

import numpy as np
import pytest

from experiments.evaluate_iterative_action_q import (
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


def test_gate_parser_supports_windows_and_override_budget():
    gate = parse_gate("window:4:0.1:8:108-179,180-251")
    assert gate["max_overrides"] == 8
    assert gate["windows"] == [[108.0, 179.0], [180.0, 251.0]]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_gate("invalid")


def test_policy_eval_parser_has_no_recurrent_or_eta_options():
    args = parse_args(["--checkpoint", "model.pt", "--out-dir", "out"])
    assert args.checkpoint == "model.pt"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--out-dir",
                "out",
                "--reset-recurrent-state",
            ]
        )
