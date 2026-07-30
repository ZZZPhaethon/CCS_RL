import argparse

import numpy as np
import pytest

from experiments.evaluate_iterative_q_checkpoint_router import (
    parse_args,
    parse_checkpoint,
    parse_router,
    select_routed_action,
)


def _q(candidate: int, advantage: float) -> np.ndarray:
    values = np.zeros((5, 3), dtype=np.float64)
    values[:, candidate] = advantage
    return values


def test_checkpoint_and_router_parsers() -> None:
    assert parse_checkpoint("p4=model.pt") == ("p4", "model.pt")
    assert parse_router("all:confidence:p1,p4:4:0.4:1.0") == {
        "name": "all",
        "mode": "confidence",
        "checkpoints": ["p1", "p4"],
        "required_heads": 4,
        "margin": 0.4,
        "uncertainty_beta": 1.0,
    }
    with pytest.raises(argparse.ArgumentTypeError):
        parse_router("bad:unknown:p1:4:0.4:0")


def test_confidence_router_selects_checkpoint_with_larger_advantage() -> None:
    action, decision = select_routed_action(
        {"p1": _q(1, 0.8), "p4": _q(2, 1.2)},
        np.ones(3, dtype=bool),
        0,
        parse_router("router:confidence:p1,p4:4:0.4:0"),
    )
    assert action == 2
    assert decision["chosen_checkpoint"] == "p4"


def test_confidence_router_follows_when_no_checkpoint_passes_gate() -> None:
    action, decision = select_routed_action(
        {"p1": _q(1, 0.3), "p4": _q(2, 0.2)},
        np.ones(3, dtype=bool),
        0,
        parse_router("router:confidence:p1,p4:4:0.4:0"),
    )
    assert action == 0
    assert decision["chosen_checkpoint"] is None


def test_pooled_router_combines_checkpoint_heads() -> None:
    action, decision = select_routed_action(
        {"p3": _q(1, 0.8), "p4": _q(1, 1.0)},
        np.ones(3, dtype=bool),
        0,
        parse_router("router:pooled:p3,p4:8:0.4:0"),
    )
    assert action == 1
    assert decision["agreement"] == 10
    assert decision["chosen_checkpoint"] == "pooled"


def test_router_cli_rejects_unknown_checkpoint() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--checkpoints",
                "p4=model.pt",
                "--routers",
                "bad:confidence:p3,p4:4:0.4:0",
                "--out-dir",
                "out",
            ]
        )
