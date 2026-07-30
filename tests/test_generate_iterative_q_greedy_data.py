import json

import numpy as np

from experiments.generate_iterative_q_greedy_data import (
    generate_dataset,
    parse_args,
    select_dense_actions,
    select_root_fractions,
)
from sim.control.event_based.rl.observation_encoder import (
    FUTURE_SUMMARY_REPRESENTATION_ID,
)


class _Residual:
    follow_indices = np.asarray([2, 2, 2], dtype=np.int64)


class _Wrapper:
    residual_env = _Residual()
    _joint_action_array = np.asarray(
        [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
    )

    def follow_action(self):
        return 26

    def action_masks(self):
        return np.ones(27, dtype=bool)


def test_dense_action_selection_keeps_all_single_and_limits_joint_overrides():
    actions = select_dense_actions(_Wrapper(), np.random.default_rng(2), 4, 2)
    residuals = _Wrapper._joint_action_array[actions]
    counts = (residuals != _Residual.follow_indices).sum(axis=1)
    assert (counts == 1).sum() == 6
    assert (counts == 2).sum() == 4
    assert (counts == 3).sum() == 2
    assert len(actions) == len(np.unique(actions))


def test_root_fraction_budget_rotates_omitted_fixed_root_across_seeds(tmp_path):
    args = parse_args(
        [
            "--out-path",
            str(tmp_path / "unused.npz"),
            "--split",
            "train",
            "--seeds",
            "1",
            "--root-fractions",
            "0.1",
            "0.2",
            "0.3",
            "--roots-per-seed",
            "2",
        ]
    )
    assert select_root_fractions(args, 1) == [(1, 0.2), (2, 0.3)]
    assert select_root_fractions(args, 2) == [(0, 0.1), (2, 0.3)]


def test_small_dense_dataset_has_paired_actions_and_aligned_returns(tmp_path):
    out_path = tmp_path / "dense.npz"
    args = parse_args(
        [
            "--out-path",
            str(out_path),
            "--split",
            "train",
            "--seeds",
            "81",
            "--root-fractions",
            "0.2",
            "--max-two-vessel-actions",
            "1",
            "--max-three-vessel-actions",
            "0",
            "--episode-hours",
            "48",
        ]
    )
    summary = generate_dataset(args)
    assert summary["candidates"] > 1
    assert summary["simulator_step_calls"] > args.episode_hours
    assert summary["simulator_hour_steps"] == summary["simulator_step_calls"]
    with np.load(out_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        assert metadata["uses_mpc"] is False
        assert (
            metadata["future_summary_representation_id"]
            == FUTURE_SUMMARY_REPRESENTATION_ID
        )
        assert metadata["future_summary_windows_h"] == [168]
        assert (
            metadata["training_simulator_usage"]["simulator_step_calls"]
            == summary["simulator_step_calls"]
        )
        assert len(np.unique(data["root_time_h"])) == 1
        assert len(np.unique(data["actions"][:, 0])) == len(data["actions"])
        assert np.all(data["baseline_terminal_cleanup_operating_cost_eur"] >= 0.0)
        assert np.all(data["candidate_terminal_cleanup_operating_cost_eur"] >= 0.0)
        expected = 1e-5 * (
            data["baseline_total_cost_eur"] - data["candidate_total_cost_eur"]
        )
        np.testing.assert_allclose(data["residual_return"], expected, atol=2e-5)
