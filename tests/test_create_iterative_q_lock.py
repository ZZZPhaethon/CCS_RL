import argparse

import pytest

from experiments.create_iterative_q_lock import parse_args, parse_windows_h


def test_parse_windows_h_supports_k_sweep_layout():
    assert parse_windows_h("108-251,252-395,396-539,540-680") == [
        [108, 251],
        [252, 395],
        [396, 539],
        [540, 680],
    ]


def test_parse_windows_h_rejects_overlaps():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_windows_h("108-251,251-395")


def test_lock_parser_rejects_override_budget_above_window_count():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--out-path",
                "lock.json",
                "--protocol-id",
                "test",
                "--residual-margin",
                "0.1",
                "--economic-margin-eur",
                "10000",
                "--max-overrides",
                "5",
                "--windows-h",
                "108-251,252-395,396-539,540-680",
            ]
        )


def _required_lock_args():
    return [
        "--checkpoint",
        "model.pt",
        "--out-path",
        "lock.json",
        "--protocol-id",
        "test",
        "--residual-margin",
        "0.4",
        "--economic-margin-eur",
        "0",
    ]


def test_lock_parser_accepts_three_required_heads():
    args = parse_args([*_required_lock_args(), "--required-heads", "3"])

    assert args.required_heads == 3


def test_lock_parser_defaults_to_four_required_heads():
    args = parse_args(_required_lock_args())

    assert args.required_heads == 4


def test_lock_parser_rejects_nonpositive_required_heads():
    with pytest.raises(SystemExit):
        parse_args([*_required_lock_args(), "--required-heads", "0"])
