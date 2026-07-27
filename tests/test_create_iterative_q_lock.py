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
