"""Replay-validated MPC demonstration collection and cache I/O."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from ..environment.forecast import (
    current_state_observation,
    future_forecast_observation,
)
from ..environment.gym_adapter import flat_action_from_native, flat_action_mask
from .native_mpc import RollingNativeMpcController
from .replay import ReplayExpectation, ReplaySnapshot, replay_native_actions


@dataclass(frozen=True)
class MpcDemonstrationBatch:
    state: np.ndarray
    forecast: np.ndarray
    actions: np.ndarray
    masks: np.ndarray
    seeds: np.ndarray
    hours: np.ndarray
    metadata: dict[str, object]

    def observations(self, variant: Literal["state", "flat", "tcn"]):
        if variant == "state":
            return self.state
        if variant == "flat":
            return np.concatenate(
                (self.state, self.forecast.reshape(len(self.state), -1)),
                axis=1,
            ).astype(np.float32, copy=False)
        if variant == "tcn":
            return {"state": self.state, "forecast": self.forecast}
        raise ValueError(f"unknown demonstration observation variant: {variant}")


def save_demonstrations(batch: MpcDemonstrationBatch, path) -> None:
    """Write a demonstration batch as a compressed, pickle-free NPZ cache."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(
        batch.metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    np.savez_compressed(
        destination,
        state=batch.state,
        forecast=batch.forecast,
        actions=batch.actions,
        masks=batch.masks,
        seeds=batch.seeds,
        hours=batch.hours,
        metadata_json=np.asarray(metadata_json),
    )


def load_demonstrations(path, expected_metadata: dict[str, object]) -> MpcDemonstrationBatch:
    """Load, schema-check, and canonicalize a demonstration cache."""

    required = {
        "state",
        "forecast",
        "actions",
        "masks",
        "seeds",
        "hours",
        "metadata_json",
    }
    try:
        with np.load(Path(path), allow_pickle=False) as cache:
            missing = required - set(cache.files)
            if missing:
                raise ValueError(f"demonstration cache missing fields: {sorted(missing)}")
            arrays = {
                name: np.asarray(cache[name])
                for name in required - {"metadata_json"}
            }
            metadata_array = np.asarray(cache["metadata_json"])
    except (OSError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("demonstration cache"):
            raise
        raise ValueError(f"invalid demonstration cache: {error}") from error

    if metadata_array.ndim != 0 or metadata_array.dtype.kind not in {"U", "S"}:
        raise ValueError("invalid demonstration cache metadata JSON")
    try:
        metadata = json.loads(str(metadata_array.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid demonstration cache metadata JSON: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError("invalid demonstration cache metadata: expected an object")
    for name, expected_value in expected_metadata.items():
        if name not in metadata:
            raise ValueError(f"metadata mismatch for {name!r}: field is missing")
        if metadata[name] != expected_value:
            raise ValueError(
                f"metadata mismatch for {name!r}: "
                f"expected {expected_value!r}, actual {metadata[name]!r}"
            )

    state = _float_array("state", arrays["state"], rank=2)
    forecast = _float_array("forecast", arrays["forecast"], rank=3)
    if forecast.shape[1:] != (168, 9):
        raise ValueError(
            f"invalid forecast shape: expected [N, 168, 9], actual {forecast.shape}"
        )
    actions = _integer_array("actions", arrays["actions"], rank=2)
    masks = _mask_array(arrays["masks"])
    seeds = _integer_array("seeds", arrays["seeds"], rank=1)
    hours = _integer_array("hours", arrays["hours"], rank=1)

    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(forecast)):
        raise ValueError("state and forecast observations must contain only finite values")
    leading = {
        "state": state.shape[0],
        "forecast": forecast.shape[0],
        "actions": actions.shape[0],
        "masks": masks.shape[0],
        "seeds": seeds.shape[0],
        "hours": hours.shape[0],
    }
    if len(set(leading.values())) != 1:
        raise ValueError(f"misaligned leading dimensions: {leading}")

    return MpcDemonstrationBatch(
        state=state,
        forecast=forecast,
        actions=actions,
        masks=masks,
        seeds=seeds,
        hours=hours,
        metadata=metadata,
    )


def collect_mpc_demonstrations(
    env_factory: Callable,
    seeds,
    episode_hours: int = 720,
) -> MpcDemonstrationBatch:
    """Collect exact native-MPC actions and validate each complete trace twice."""

    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("seeds must contain at least one value")
    if episode_hours <= 0:
        raise ValueError("episode_hours must be positive")

    states: list[np.ndarray] = []
    forecasts: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    recorded_seeds: list[int] = []
    hours: list[int] = []

    for seed in seed_values:
        env = env_factory(demonstration=True)
        env.reset(seed=seed)
        replay_env = copy.deepcopy(env)
        controller = RollingNativeMpcController(
            env,
            replan_every=24,
            planning_horizon_h=168,
        )
        native_actions: list[dict[str, list[int]]] = []

        for hour in range(int(episode_hours)):
            states.append(np.asarray(current_state_observation(env), dtype=np.float32))
            forecasts.append(
                np.asarray(future_forecast_observation(env), dtype=np.float32)
            )
            masks.append(
                flat_action_mask(
                    env.vessel_action_mask(),
                    env.well_rate_action_mask(),
                )
            )

            native_action = controller(env)
            if not controller.last_trace_replay_is_exact:
                details = _mismatch_details(controller.last_trace_replay_mismatches)
                raise RuntimeError(
                    f"MPC candidate replay mismatch at seed={seed}, hour={hour}: {details}"
                )
            actions.append(flat_action_from_native(env, native_action))
            native_actions.append(copy.deepcopy(native_action))
            recorded_seeds.append(seed)
            hours.append(hour)

            _observation, _reward, terminated, truncated, _info = env.step(native_action)
            if (terminated or truncated) and hour + 1 < episode_hours:
                raise RuntimeError(
                    f"seed={seed}, hour={hour}: premature episode completion; "
                    f"expected {episode_hours} rows"
                )

        initial_replay = replay_native_actions(
            replay_env,
            native_actions,
            horizon_h=episode_hours,
        )
        if not initial_replay.is_executable:
            raise RuntimeError(
                f"full-trace replay is not executable for seed={seed}: "
                f"{_mismatch_details(initial_replay.mismatches)}"
            )
        expectation = _expectation_from_snapshot(initial_replay.actual)
        exact_replay = replay_native_actions(
            replay_env,
            native_actions,
            horizon_h=episode_hours,
            expected=expectation,
        )
        if not exact_replay.is_exact:
            raise RuntimeError(
                f"full-trace replay is not exact for seed={seed}: "
                f"{_mismatch_details(exact_replay.mismatches)}"
            )

    return MpcDemonstrationBatch(
        state=np.asarray(states, dtype=np.float32),
        forecast=np.asarray(forecasts, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        masks=np.asarray(masks, dtype=bool),
        seeds=np.asarray(recorded_seeds, dtype=np.int64),
        hours=np.asarray(hours, dtype=np.int64),
        metadata=env_factory.metadata(),
    )


def _float_array(name: str, value: np.ndarray, *, rank: int) -> np.ndarray:
    if value.ndim != rank:
        raise ValueError(f"invalid {name} rank: expected {rank}, actual {value.ndim}")
    if value.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"invalid {name} dtype: {value.dtype}")
    return value.astype(np.float32, copy=False)


def _integer_array(name: str, value: np.ndarray, *, rank: int) -> np.ndarray:
    if value.ndim != rank:
        raise ValueError(f"invalid {name} rank: expected {rank}, actual {value.ndim}")
    if value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"invalid {name} dtype: {value.dtype}")
    return value.astype(np.int64, copy=False)


def _mask_array(value: np.ndarray) -> np.ndarray:
    if value.ndim != 2:
        raise ValueError(f"invalid masks rank: expected 2, actual {value.ndim}")
    if value.dtype.kind == "b":
        return value.astype(bool, copy=False)
    if value.dtype.kind not in {"i", "u"} or not np.all((value == 0) | (value == 1)):
        raise ValueError(f"invalid masks dtype or values: {value.dtype}")
    return value.astype(bool)


def _expectation_from_snapshot(snapshot: ReplaySnapshot) -> ReplayExpectation:
    names = frozenset(field.name for field in fields(ReplaySnapshot))
    values = {name: getattr(snapshot, name) for name in names}
    return ReplayExpectation(required_fields=names, **values)


def _mismatch_details(mismatches) -> str:
    return "; ".join(str(mismatch) for mismatch in mismatches) or "no mismatch details"
