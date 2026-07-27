"""Causal rolling-history observations for forecast policies."""

from __future__ import annotations

from collections import deque

import numpy as np


PAST_HOURS = 24


def normalized_action(
    action: np.ndarray,
    action_dimensions: tuple[int, ...] | list[int],
) -> np.ndarray:
    """Map one MultiDiscrete action to [0, 1] without changing its ordering."""

    values = np.asarray(action, dtype=np.float32)
    dimensions = np.asarray(action_dimensions, dtype=np.float32)
    if values.shape != dimensions.shape:
        raise ValueError(
            "action and action_dimensions must have the same one-dimensional shape"
        )
    if np.any(dimensions <= 1.0):
        raise ValueError("all action dimensions must be greater than one")
    if np.any(values < 0.0) or np.any(values >= dimensions):
        raise ValueError("action contains an out-of-range choice")
    return values / (dimensions - 1.0)


class PastObservationBuffer:
    """Keep the preceding completed state/action rows, oldest to newest."""

    def __init__(
        self,
        state_size: int,
        action_dimensions: tuple[int, ...] | list[int],
        hours: int = PAST_HOURS,
    ) -> None:
        if state_size <= 0:
            raise ValueError("state_size must be positive")
        if hours <= 0:
            raise ValueError("hours must be positive")
        self.state_size = int(state_size)
        self.action_dimensions = tuple(int(value) for value in action_dimensions)
        self.hours = int(hours)
        self.row_size = self.state_size + len(self.action_dimensions) + 1
        self._rows: deque[np.ndarray] = deque(maxlen=self.hours)

    def reset(self) -> None:
        self._rows.clear()

    def append(self, state: np.ndarray, action: np.ndarray) -> None:
        state_values = np.asarray(state, dtype=np.float32)
        if state_values.shape != (self.state_size,) or not np.all(
            np.isfinite(state_values)
        ):
            raise ValueError("past state has an invalid shape or non-finite values")
        row = np.concatenate(
            (
                state_values,
                normalized_action(action, self.action_dimensions),
                np.ones(1, dtype=np.float32),
            )
        ).astype(np.float32, copy=False)
        self._rows.append(row)

    def observation(self, *, zero: bool = False) -> np.ndarray:
        result = np.zeros((self.hours, self.row_size), dtype=np.float32)
        if not zero and self._rows:
            rows = np.asarray(self._rows, dtype=np.float32)
            result[-len(rows) :] = rows
        return result


def demonstration_past_observation(
    states: np.ndarray,
    actions: np.ndarray,
    seeds: np.ndarray,
    hours: np.ndarray,
    action_dimensions: tuple[int, ...] | list[int],
    *,
    history_hours: int = PAST_HOURS,
    zero: bool = False,
) -> np.ndarray:
    """Build strictly pre-decision histories for ordered demonstration rows."""

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions)
    seeds = np.asarray(seeds)
    hours = np.asarray(hours)
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError("states and actions must be rank-two arrays")
    if seeds.ndim != 1 or hours.ndim != 1:
        raise ValueError("seeds and hours must be rank-one arrays")
    if len({len(states), len(actions), len(seeds), len(hours)}) != 1:
        raise ValueError("demonstration history inputs are misaligned")

    buffer = PastObservationBuffer(
        states.shape[1],
        action_dimensions,
        hours=history_hours,
    )
    result = np.zeros(
        (len(states), history_hours, buffer.row_size),
        dtype=np.float32,
    )
    previous_seed = None
    previous_hour = None
    for index, (seed, hour) in enumerate(zip(seeds.tolist(), hours.tolist())):
        seed = int(seed)
        hour = int(hour)
        if seed != previous_seed:
            if hour != 0:
                raise ValueError(f"seed {seed} history must begin at hour 0")
            buffer.reset()
        elif hour != int(previous_hour) + 1:
            raise ValueError(f"seed {seed} history hours must be contiguous")
        result[index] = buffer.observation(zero=zero)
        buffer.append(states[index], actions[index])
        previous_seed = seed
        previous_hour = hour
    return result
