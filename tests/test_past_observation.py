import numpy as np
import pytest

from sim.environment.past import (
    PastObservationBuffer,
    demonstration_past_observation,
)


def test_demonstration_history_is_causal_padded_and_seed_local():
    states = np.arange(12, dtype=np.float32).reshape(4, 3)
    actions = np.asarray([[0, 1], [1, 2], [1, 0], [0, 3]], dtype=np.int64)
    seeds = np.asarray([5, 5, 9, 9], dtype=np.int64)
    hours = np.asarray([0, 1, 0, 1], dtype=np.int64)

    past = demonstration_past_observation(
        states,
        actions,
        seeds,
        hours,
        [2, 4],
        history_hours=2,
    )

    assert past.shape == (4, 2, 6)
    np.testing.assert_array_equal(past[0], 0.0)
    np.testing.assert_array_equal(past[2], 0.0)
    np.testing.assert_array_equal(past[1, 0], 0.0)
    np.testing.assert_array_equal(past[1, 1, :3], states[0])
    np.testing.assert_allclose(past[1, 1, 3:5], [0.0, 1.0 / 3.0])
    assert past[1, 1, -1] == 1.0
    np.testing.assert_array_equal(past[3, 1, :3], states[2])


def test_zero_history_control_preserves_shape_without_information():
    past = demonstration_past_observation(
        np.ones((2, 3), dtype=np.float32),
        np.asarray([[0], [1]], dtype=np.int64),
        np.asarray([4, 4]),
        np.asarray([0, 1]),
        [2],
        history_hours=3,
        zero=True,
    )

    assert past.shape == (2, 3, 5)
    np.testing.assert_array_equal(past, 0.0)


def test_history_rejects_noncontiguous_rows():
    with pytest.raises(ValueError, match="contiguous"):
        demonstration_past_observation(
            np.ones((2, 3), dtype=np.float32),
            np.asarray([[0], [1]], dtype=np.int64),
            np.asarray([4, 4]),
            np.asarray([0, 2]),
            [2],
        )


def test_live_buffer_matches_offline_row_encoding():
    buffer = PastObservationBuffer(3, [2, 4], hours=2)
    buffer.append(np.asarray([1.0, 2.0, 3.0]), np.asarray([1, 3]))

    observation = buffer.observation()

    np.testing.assert_array_equal(observation[0], 0.0)
    np.testing.assert_array_equal(observation[1], [1, 2, 3, 1, 1, 1])
