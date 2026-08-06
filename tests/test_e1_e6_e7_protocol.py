from types import SimpleNamespace

import numpy as np
import pytest

from experiments import iterative_q_data_common as common
from experiments.aggregate_e1_cost_timing import _paired_reductions
from experiments.aggregate_e7_temporal_generalization import (
    _hierarchical_ratio_ci,
)
from experiments.run_e1_online_timing import _normalize
from experiments.run_e6_mechanism import _action_label
from experiments.run_e7_temporal_generalization import (
    ANNUAL_SCENARIO_HOURS,
    HORIZONS,
    expanded_policy_windows,
    global_episode_progress,
    q_policy_windows,
    receding_episode_progress,
    with_global_progress,
    with_receding_progress,
)


def test_e1_normalizes_absolute_time_without_time_percentage():
    rows = _normalize(
        "greedy",
        None,
        [
            {
                "seed": 9000031,
                "wall_clock_seconds": 1.25,
                "total_cost": 100.0,
                "vented_t": 2.0,
                "stored_t": 8.0,
            }
        ],
    )

    assert rows[0]["episode_wall_time_s"] == 1.25
    assert all("percent" not in key for key in rows[0])


def test_e1_cost_reduction_is_paired_against_fixed_assignment():
    rows = [
        {
            "algorithm": "fixed_assignment",
            "model_seed": "",
            "test_seed": "1",
            "total_cost_eur": "100",
        },
        {
            "algorithm": "fixed_assignment",
            "model_seed": "",
            "test_seed": "2",
            "total_cost_eur": "200",
        },
        {
            "algorithm": "greedy",
            "model_seed": "",
            "test_seed": "1",
            "total_cost_eur": "80",
        },
        {
            "algorithm": "greedy",
            "model_seed": "",
            "test_seed": "2",
            "total_cost_eur": "150",
        },
    ]

    values, by_model = _paired_reductions(rows, "greedy")

    assert values.tolist() == [20.0, 25.0]
    assert by_model["not_applicable"] == {1: 20.0, 2: 25.0}


def test_e6_native_action_labels_match_environment_schema():
    emitters = ("brevik", "celsio", "yara_sluiskil")

    assert _action_label(0, emitters) == "Wait"
    assert _action_label(1, emitters) == "Terminal"
    assert _action_label(2, emitters) == "Brevik"
    assert _action_label(4, emitters) == "Yara Sluiskil"


def test_e7_receding_windows_repeat_every_720_hours():
    windows_720 = expanded_policy_windows(720)
    windows_2160 = expanded_policy_windows(2160)
    windows_4320 = expanded_policy_windows(4320)
    windows_8760 = expanded_policy_windows(8760)

    assert len(windows_720) == 12
    assert len(windows_2160) == 36
    assert windows_2160[12] == (828, 875)
    assert len(windows_4320) == 72
    assert windows_4320[-1] == (4236, 4280)
    assert len(windows_8760) == 145
    assert windows_8760[-1] == (8748, 8759)
    assert HORIZONS == (720, 2160, 4320, 8760)


def test_e7_receding_progress_restarts_without_changing_other_features():
    observation = {
        "state": np.asarray([3.0, 4.0, 0.9], dtype=np.float32),
        "future": np.asarray([5.0], dtype=np.float32),
    }

    updated = with_receding_progress(observation, 900)

    np.testing.assert_allclose(updated["state"], [3.0, 4.0, 0.25])
    np.testing.assert_allclose(observation["state"], [3.0, 4.0, 0.9])
    assert updated["future"] is observation["future"]
    assert receding_episode_progress(720) == 0.0
    assert receding_episode_progress(8760) == pytest.approx(1 / 6)


def test_e7_direct_global_differs_only_in_episode_progress():
    observation = {
        "state": np.asarray([3.0, 4.0, 0.9], dtype=np.float32),
        "future": np.asarray([5.0], dtype=np.float32),
    }

    direct = with_global_progress(observation, 900, 2160)
    receding = with_receding_progress(observation, 900)

    np.testing.assert_allclose(direct["state"], [3.0, 4.0, 900 / 2160])
    np.testing.assert_allclose(receding["state"], [3.0, 4.0, 0.25])
    assert direct["future"] is receding["future"]
    assert global_episode_progress(8760, 8760) == 1.0
    assert q_policy_windows("iterative_q_direct", 2160) == q_policy_windows(
        "iterative_q_receding",
        2160,
    )


def test_e7_scenario_must_cover_execution_and_forecast_context():
    args = SimpleNamespace(
        scenario_protocol="unified_window_v1",
        hard_scenario_probability=0.5,
        forecast_context_hours=168,
        scenario_episode_hours=720,
        episode_hours=720,
        stress_level="medium",
        reward_scale=0.00001,
    )

    with pytest.raises(ValueError, match="execution horizon plus forecast"):
        common.make_native_env(args)
    assert ANNUAL_SCENARIO_HOURS == 8928


def test_e7_paired_percentage_uses_ratio_of_means():
    numerator = {"0": {1: 10.0, 2: 0.0}}
    denominator = {"0": {1: 100.0, 2: 10.0}}

    point, low, high = _hierarchical_ratio_ci(
        numerator,
        denominator,
        samples=200,
        rng=np.random.default_rng(7),
    )

    assert point == pytest.approx(100.0 * 5.0 / 55.0)
    assert low <= point <= high
