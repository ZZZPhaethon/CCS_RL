import argparse
from types import SimpleNamespace

import numpy as np
import pytest

import experiments.evaluate_iterative_action_q as evaluation
from experiments.evaluate_iterative_action_q import (
    apply_state_mean_ablation,
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


def test_policy_eval_parser_accepts_state_mean_ablation():
    args = parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--out-dir",
            "out",
            "--state-mean-ablation",
            "hour_of_week",
            "episode_progress",
        ]
    )
    assert args.state_mean_ablation == ["hour_of_week", "episode_progress"]


def test_apply_state_mean_ablation_copies_and_replaces_selected_values():
    observation = {
        "state": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "forecast": np.ones((2, 2), dtype=np.float32),
    }

    updated = apply_state_mean_ablation(observation, ((0, 0.5), (2, 0.7)))

    assert updated is not observation
    assert updated["state"] == pytest.approx([0.5, 0.2, 0.7])
    assert observation["state"] == pytest.approx([0.1, 0.2, 0.3])
    assert updated["forecast"] is observation["forecast"]


def test_tensor_observation_projects_full_state_to_checkpoint_schema():
    model = SimpleNamespace(
        source_state_feature_names=("hour_of_week", "fill", "episode_progress"),
        state_feature_names=("fill",),
    )
    observation = {
        "state": np.asarray([0.25, 0.75, 0.5], dtype=np.float32),
    }

    tensor = evaluation._tensor_observation(
        observation,
        evaluation.torch.device("cpu"),
        model,
    )

    assert tensor.shape == (1, 1, 1)
    assert tensor.item() == pytest.approx(0.75)


def test_validation_only_parser_rejects_formal_test_seed():
    args = parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--out-dir",
            "out",
            "--eval-seeds",
            "8100001",
            "8100020",
            "--validation-only",
        ]
    )
    assert args.validation_only
    with pytest.raises(ValueError, match="validation-only evaluation rejected"):
        parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--out-dir",
                "out",
                "--eval-seeds",
                "9000031",
                "--validation-only",
            ]
        )


def test_metrics_exports_episode_cost_breakdown(monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "_terminal_cleanup_cost_for_state",
        lambda _env, _params: 29.0,
    )
    ledger = SimpleNamespace(
        vessel_fuel=2.0,
        conditioning=3.0,
        reconditioning=5.0,
        loading=7.0,
        unloading=11.0,
        operating_cost=28.0,
        vent_penalty=13.0,
        storage_shortfall_penalty=17.0,
        total_cost=58.0,
        vented_t=19.0,
        stored_t=23.0,
    )
    env = SimpleNamespace(
        ledger=ledger,
        cost_model=SimpleNamespace(parameters=object()),
    )

    metrics = evaluation._metrics(env)

    assert metrics["episode_vessel_fuel_eur"] == 2.0
    assert metrics["episode_conditioning_eur"] == 3.0
    assert metrics["episode_reconditioning_eur"] == 5.0
    assert metrics["episode_loading_eur"] == 7.0
    assert metrics["episode_unloading_eur"] == 11.0
    assert metrics["episode_operating_cost_eur"] == 28.0
    assert metrics["episode_vent_penalty_eur"] == 13.0
    assert metrics["episode_storage_shortfall_penalty_eur"] == 17.0
    assert metrics["terminal_cleanup_operating_cost_eur"] == 29.0
    assert metrics["operating_cost_eur"] == 57.0
    assert metrics["total_cost_eur"] == 87.0
