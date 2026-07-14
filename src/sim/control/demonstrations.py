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
    replan_phase_observation,
)
from ..environment.gym_adapter import flat_action_from_native, flat_action_mask
from ..environment.vessel_mode import (
    vessel_operation_mode_observation,
    vessel_sailing_destination_observation,
)
from .native_mpc import RollingNativeMpcController, native_mpc_candidate_names
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
    operation_modes: np.ndarray | None = None
    vessel_destinations: np.ndarray | None = None
    plan_candidates: np.ndarray | None = None
    candidate_names: tuple[str, ...] | None = None
    plan_context: np.ndarray | None = None

    def observations(
        self,
        variant: Literal[
            "state",
            "flat",
            "tcn",
            "state_mode",
            "tcn_mode",
            "tcn_mode_destination",
            "gnn_mode_destination",
            "larger_mlp_mode_destination",
            "edge_gnn_mode_destination",
            "fixed_scale_larger_mlp_mode_destination",
            "fixed_scale_edge_gnn_mode_destination",
            "stable_tcn_mode_destination",
            "fixed_scale_tcn_mode_destination",
            "fixed_scale_tcn_mode_destination_replan_phase",
            "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate",
            "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context",
        ],
    ):
        if variant == "state":
            return self.state
        if variant == "flat":
            return np.concatenate(
                (self.state, self.forecast.reshape(len(self.state), -1)),
                axis=1,
            ).astype(np.float32, copy=False)
        if variant == "tcn":
            return {"state": self.state, "forecast": self.forecast}
        if variant in {
            "state_mode",
            "tcn_mode",
            "tcn_mode_destination",
            "gnn_mode_destination",
            "larger_mlp_mode_destination",
            "edge_gnn_mode_destination",
            "fixed_scale_larger_mlp_mode_destination",
            "fixed_scale_edge_gnn_mode_destination",
            "stable_tcn_mode_destination",
            "fixed_scale_tcn_mode_destination",
            "fixed_scale_tcn_mode_destination_replan_phase",
            "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate",
            "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context",
        }:
            if self.operation_modes is None:
                raise ValueError(
                    f"operation mode observations are required for variant {variant!r}"
                )
            enriched = np.concatenate(
                (self.state, self.operation_modes.reshape(len(self.state), -1)),
                axis=1,
            ).astype(np.float32, copy=False)
            if variant == "state_mode":
                return enriched
            if variant in {
                "tcn_mode_destination",
                "gnn_mode_destination",
                "larger_mlp_mode_destination",
                "edge_gnn_mode_destination",
                "fixed_scale_larger_mlp_mode_destination",
                "fixed_scale_edge_gnn_mode_destination",
                "stable_tcn_mode_destination",
                "fixed_scale_tcn_mode_destination",
                "fixed_scale_tcn_mode_destination_replan_phase",
                "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate",
                "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context",
            }:
                if self.vessel_destinations is None:
                    raise ValueError(
                        "vessel destination observations are required for variant "
                        f"{variant!r}"
                    )
                enriched = np.concatenate(
                    (
                        enriched,
                        self.vessel_destinations.reshape(len(self.state), -1),
                    ),
                    axis=1,
                ).astype(np.float32, copy=False)
            if variant in {
                "fixed_scale_tcn_mode_destination_replan_phase",
                "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate",
                "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context",
            }:
                phase = np.asarray(
                    [replan_phase_observation(hour) for hour in self.hours],
                    dtype=np.float32,
                )
                enriched = np.concatenate((enriched, phase), axis=1).astype(
                    np.float32,
                    copy=False,
                )
            if variant == "fixed_scale_tcn_mode_destination_replan_phase_oracle_candidate":
                if self.plan_candidates is None or self.candidate_names is None:
                    raise ValueError("oracle candidate observations require plan candidates")
                candidates = np.eye(len(self.candidate_names), dtype=np.float32)[
                    self.plan_candidates
                ]
                enriched = np.concatenate((enriched, candidates), axis=1).astype(
                    np.float32,
                    copy=False,
                )
            if variant == "fixed_scale_tcn_mode_destination_replan_phase_learned_plan_context":
                if (
                    self.plan_context is None
                    or self.plan_context.ndim != 2
                    or self.plan_context.shape != (len(self.state), 8)
                ):
                    raise ValueError("learned plan-context observations require [N, 8] context")
                enriched = np.concatenate((enriched, self.plan_context), axis=1).astype(
                    np.float32,
                    copy=False,
                )
            return {"state": enriched, "forecast": self.forecast}
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
    payload = {
        "state": batch.state,
        "forecast": batch.forecast,
        "actions": batch.actions,
        "masks": batch.masks,
        "seeds": batch.seeds,
        "hours": batch.hours,
        "metadata_json": np.asarray(metadata_json),
    }
    if batch.operation_modes is not None:
        payload["operation_modes"] = batch.operation_modes
    if batch.vessel_destinations is not None:
        payload["vessel_destinations"] = batch.vessel_destinations
    if batch.plan_candidates is not None:
        if batch.candidate_names is None:
            raise ValueError("candidate names are required with plan candidates")
        payload["plan_candidates"] = batch.plan_candidates
        payload["candidate_names"] = np.asarray(batch.candidate_names)
    np.savez_compressed(
        destination,
        **payload,
    )


def load_demonstrations(
    path,
    expected_metadata: dict[str, object] | None,
) -> MpcDemonstrationBatch:
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
            operation_modes_array = (
                np.asarray(cache["operation_modes"])
                if "operation_modes" in cache.files
                else None
            )
            vessel_destinations_array = (
                np.asarray(cache["vessel_destinations"])
                if "vessel_destinations" in cache.files
                else None
            )
            plan_candidates_array = (
                np.asarray(cache["plan_candidates"])
                if "plan_candidates" in cache.files
                else None
            )
            candidate_names_array = (
                np.asarray(cache["candidate_names"])
                if "candidate_names" in cache.files
                else None
            )
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
    if expected_metadata is not None:
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
    operation_modes = _operation_mode_array(operation_modes_array)
    vessel_destinations = _vessel_destination_array(vessel_destinations_array)
    if (plan_candidates_array is None) != (candidate_names_array is None):
        raise ValueError(
            "invalid demonstration cache: plan candidates and candidate names must coexist"
        )
    plan_candidates = (
        _integer_array("plan_candidates", plan_candidates_array, rank=1)
        if plan_candidates_array is not None
        else None
    )
    candidate_names = None
    if candidate_names_array is not None:
        if candidate_names_array.ndim != 1 or candidate_names_array.dtype.kind not in {
            "U",
            "S",
        }:
            raise ValueError("invalid candidate_names array")
        candidate_names = tuple(str(value) for value in candidate_names_array.tolist())
        if not candidate_names or len(set(candidate_names)) != len(candidate_names):
            raise ValueError("candidate_names must be non-empty and unique")
        if np.any(plan_candidates < 0) or np.any(plan_candidates >= len(candidate_names)):
            raise ValueError("plan_candidates contain an out-of-range candidate index")

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
    if operation_modes is not None:
        leading["operation_modes"] = operation_modes.shape[0]
    if vessel_destinations is not None:
        leading["vessel_destinations"] = vessel_destinations.shape[0]
    if plan_candidates is not None:
        leading["plan_candidates"] = plan_candidates.shape[0]
    if len(set(leading.values())) != 1:
        raise ValueError(f"misaligned leading dimensions: {leading}")
    if (
        operation_modes is not None
        and vessel_destinations is not None
        and operation_modes.shape[1] != vessel_destinations.shape[1]
    ):
        raise ValueError(
            "operation mode and vessel destination observations disagree on vessel count"
        )

    return MpcDemonstrationBatch(
        state=state,
        forecast=forecast,
        actions=actions,
        masks=masks,
        seeds=seeds,
        hours=hours,
        metadata=metadata,
        operation_modes=operation_modes,
        vessel_destinations=vessel_destinations,
        plan_candidates=plan_candidates,
        candidate_names=candidate_names,
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
    operation_modes: list[np.ndarray] = []
    vessel_destinations: list[np.ndarray] = []
    plan_candidates: list[int] = []
    candidate_names: tuple[str, ...] | None = None

    for seed in seed_values:
        env = env_factory(demonstration=True)
        env.reset(seed=seed)
        seed_candidate_names = native_mpc_candidate_names(env)
        if candidate_names is None:
            candidate_names = seed_candidate_names
        elif candidate_names != seed_candidate_names:
            raise RuntimeError("MPC candidate names changed across demonstration seeds")
        candidate_index = {
            name: index for index, name in enumerate(seed_candidate_names)
        }
        replay_env = copy.deepcopy(env)
        controller = RollingNativeMpcController(
            env,
            replan_every=24,
            planning_horizon_h=168,
        )
        native_actions: list[dict[str, list[int]]] = []

        for hour in range(int(episode_hours)):
            states.append(np.asarray(current_state_observation(env), dtype=np.float32))
            operation_modes.append(
                np.asarray(vessel_operation_mode_observation(env), dtype=np.float32).reshape(
                    len(env.vessel_ids), 5
                )
            )
            vessel_destinations.append(
                np.asarray(
                    vessel_sailing_destination_observation(env),
                    dtype=np.float32,
                ).reshape(
                    len(env.vessel_ids),
                    len(env.terminal_ids) + len(env.emitter_ids),
                )
            )
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
            try:
                plan_candidates.append(candidate_index[controller.last_candidate_name])
            except KeyError as error:
                raise RuntimeError(
                    f"unknown MPC candidate at seed={seed}, hour={hour}: "
                    f"{controller.last_candidate_name!r}"
                ) from error
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
        operation_modes=np.asarray(operation_modes, dtype=np.float32),
        vessel_destinations=np.asarray(vessel_destinations, dtype=np.float32),
        plan_candidates=np.asarray(plan_candidates, dtype=np.int64),
        candidate_names=candidate_names,
    )


def merge_demonstration_shards(
    shards: list[MpcDemonstrationBatch],
    *,
    expected_seeds,
    episode_hours: int,
) -> MpcDemonstrationBatch:
    """Validate and merge complete demonstration shards in seed/hour order."""

    if not shards:
        raise ValueError("at least one demonstration shard is required")
    expected = {int(seed) for seed in expected_seeds}
    if not expected:
        raise ValueError("expected_seeds must not be empty")
    if episode_hours <= 0:
        raise ValueError("episode_hours must be positive")
    metadata = shards[0].metadata
    for shard in shards[1:]:
        if shard.metadata != metadata:
            raise ValueError("demonstration shard metadata mismatch")

    seeds = np.concatenate([shard.seeds for shard in shards])
    hours = np.concatenate([shard.hours for shard in shards])
    pairs = list(zip(seeds.tolist(), hours.tolist()))
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate demonstration seed/hour rows across shards")
    actual = {int(seed) for seed in seeds.tolist()}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"missing or unexpected demonstration seeds: missing={missing}, unexpected={unexpected}"
        )
    expected_hours = list(range(int(episode_hours)))
    for seed in sorted(expected):
        seed_hours = sorted(int(hour) for row_seed, hour in pairs if int(row_seed) == seed)
        if seed_hours != expected_hours:
            raise ValueError(
                f"seed {seed} does not contain complete hours 0..{episode_hours - 1}: {seed_hours}"
            )

    operation_presence = [shard.operation_modes is not None for shard in shards]
    if len(set(operation_presence)) != 1:
        raise ValueError("demonstration shards disagree on operation mode schema")
    destination_presence = [shard.vessel_destinations is not None for shard in shards]
    if len(set(destination_presence)) != 1:
        raise ValueError("demonstration shards disagree on vessel destination schema")
    candidate_presence = [shard.plan_candidates is not None for shard in shards]
    if len(set(candidate_presence)) != 1:
        raise ValueError("demonstration shards disagree on plan candidate schema")
    candidate_names = shards[0].candidate_names
    if candidate_presence[0]:
        if candidate_names is None or any(
            shard.candidate_names != candidate_names for shard in shards
        ):
            raise ValueError("demonstration shards disagree on candidate names")
    order = np.lexsort((hours, seeds))

    def merged(name: str):
        return np.concatenate([getattr(shard, name) for shard in shards], axis=0)[order]

    return MpcDemonstrationBatch(
        state=merged("state"),
        forecast=merged("forecast"),
        actions=merged("actions"),
        masks=merged("masks"),
        seeds=seeds[order].astype(np.int64, copy=False),
        hours=hours[order].astype(np.int64, copy=False),
        metadata=metadata,
        operation_modes=(merged("operation_modes") if operation_presence[0] else None),
        vessel_destinations=(
            merged("vessel_destinations") if destination_presence[0] else None
        ),
        plan_candidates=(merged("plan_candidates") if candidate_presence[0] else None),
        candidate_names=candidate_names if candidate_presence[0] else None,
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


def _operation_mode_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    modes = _float_array("operation_modes", value, rank=3)
    if modes.shape[2] != 5:
        raise ValueError(
            f"invalid operation_modes shape: expected final dimension 5, actual {modes.shape}"
        )
    if not np.all(np.isfinite(modes)):
        raise ValueError("operation mode observations must contain only finite values")
    if not np.all((modes == 0.0) | (modes == 1.0)) or not np.all(
        modes.sum(axis=2) == 1.0
    ):
        raise ValueError("operation mode observations must be one-hot")
    return modes


def _vessel_destination_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    destinations = _float_array("vessel_destinations", value, rank=3)
    if destinations.shape[2] <= 0:
        raise ValueError("invalid vessel_destinations shape: no destination slots")
    if not np.all(np.isfinite(destinations)):
        raise ValueError("vessel destination observations must contain only finite values")
    if not np.all((destinations == 0.0) | (destinations == 1.0)) or not np.all(
        destinations.sum(axis=2) <= 1.0
    ):
        raise ValueError("vessel destination observations must be zero-or-one-hot")
    return destinations


def _expectation_from_snapshot(snapshot: ReplaySnapshot) -> ReplayExpectation:
    names = frozenset(field.name for field in fields(ReplaySnapshot))
    values = {name: getattr(snapshot, name) for name in names}
    return ReplayExpectation(required_fields=names, **values)


def _mismatch_details(mismatches) -> str:
    return "; ".join(str(mismatch) for mismatch in mismatches) or "no mismatch details"
