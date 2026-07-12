"""Mode-conditioned diagnostics for vessel WAIT and dispatch decisions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..environment.vessel_mode import VESSEL_OPERATION_MODES, vessel_operation_modes
from .imitation import _index_observations, _observation_count, _tensor_observations


def masked_vessel_action_probabilities(
    model,
    observations,
    masks,
    vessel_count: int,
    batch_size: int = 2048,
):
    """Return one masked categorical probability array per vessel dimension."""

    import torch

    policy = model.policy
    count = _observation_count(observations)
    masks = np.asarray(masks, dtype=bool)
    if len(masks) != count:
        raise ValueError("observations and masks must share a leading dimension")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    chunks: list[list[np.ndarray]] = [[] for _ in range(vessel_count)]
    policy.set_training_mode(False)
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        batch_observations = _index_observations(observations, slice(start, stop))
        observation_tensors = _tensor_observations(batch_observations, policy.device)
        mask_tensors = torch.as_tensor(masks[start:stop], device=policy.device)
        with torch.no_grad():
            features = policy.extract_features(observation_tensors)
            if policy.share_features_extractor:
                latent_pi, _latent_vf = policy.mlp_extractor(features)
            else:
                policy_features, _value_features = features
                latent_pi = policy.mlp_extractor.forward_actor(policy_features)
            distribution = policy._get_action_dist_from_latent(latent_pi)
            distribution.apply_masking(mask_tensors)
            if not hasattr(distribution, "distributions"):
                raise TypeError("vessel diagnostics require a MultiCategorical policy")
            if len(distribution.distributions) < vessel_count:
                raise ValueError("policy has fewer action dimensions than vessels")
            for vessel_index, categorical in enumerate(
                distribution.distributions[:vessel_count]
            ):
                chunks[vessel_index].append(categorical.probs.detach().cpu().numpy())
    return [np.concatenate(values, axis=0) for values in chunks]


def _mode_labels(operation_modes: np.ndarray, vessel_count: int) -> np.ndarray:
    values = np.asarray(operation_modes)
    if values.ndim != 3 or values.shape[1:] != (vessel_count, len(VESSEL_OPERATION_MODES)):
        raise ValueError(
            "operation_modes must have shape "
            f"[N, {vessel_count}, {len(VESSEL_OPERATION_MODES)}]"
        )
    if not np.all((values == 0) | (values == 1)) or not np.all(values.sum(axis=2) == 1):
        raise ValueError("operation_modes must contain one-hot rows")
    return np.asarray(VESSEL_OPERATION_MODES, dtype=object)[values.argmax(axis=2)]


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def demonstration_mode_diagnostics(
    model,
    observations,
    actions,
    masks,
    operation_modes,
    vessel_count: int,
) -> list[dict[str, object]]:
    """Summarize BC/teacher agreement without mixing forced and voluntary WAIT."""

    actions = np.asarray(actions, dtype=np.int64)
    masks = np.asarray(masks, dtype=bool)
    labels = _mode_labels(operation_modes, vessel_count)
    probabilities = masked_vessel_action_probabilities(
        model, observations, masks, vessel_count
    )
    if actions.ndim != 2 or actions.shape[1] < vessel_count:
        raise ValueError("actions must contain one column per vessel")
    if len(actions) != len(labels) or len(masks) != len(labels):
        raise ValueError("diagnostic arrays must share a leading dimension")

    records: list[dict[str, object]] = []
    offset = 0
    for vessel_index, vessel_probabilities in enumerate(probabilities):
        action_count = vessel_probabilities.shape[1]
        vessel_masks = masks[:, offset : offset + action_count]
        offset += action_count
        if vessel_masks.shape != vessel_probabilities.shape:
            raise ValueError("action mask does not match vessel action dimensions")
        expected = actions[:, vessel_index]
        predicted = vessel_probabilities.argmax(axis=1)
        for row_index in range(len(actions)):
            sorted_probabilities = np.sort(vessel_probabilities[row_index])
            records.append(
                {
                    "vessel_index": vessel_index,
                    "mode": str(labels[row_index, vessel_index]),
                    "forced_wait": bool(
                        vessel_masks[row_index, 0]
                        and vessel_masks[row_index].sum() == 1
                    ),
                    "expected": int(expected[row_index]),
                    "predicted": int(predicted[row_index]),
                    "wait_probability": float(vessel_probabilities[row_index, 0]),
                    "argmax_margin": float(
                        sorted_probabilities[-1] - sorted_probabilities[-2]
                    ),
                }
            )

    rows = []
    vessel_groups: Iterable[int | str] = [*range(vessel_count), "all"]
    for vessel in vessel_groups:
        vessel_records = (
            records
            if vessel == "all"
            else [record for record in records if record["vessel_index"] == vessel]
        )
        for mode in (*VESSEL_OPERATION_MODES, "all"):
            selected = (
                vessel_records
                if mode == "all"
                else [record for record in vessel_records if record["mode"] == mode]
            )
            if not selected:
                continue
            forced = sum(bool(record["forced_wait"]) for record in selected)
            voluntary_wait = [
                record
                for record in selected
                if record["expected"] == 0 and not record["forced_wait"]
            ]
            dispatch = [record for record in selected if record["expected"] != 0]
            predicted_dispatch = sum(record["predicted"] != 0 for record in dispatch)
            active = [record for record in selected if not record["forced_wait"]]
            predicted_dispatch_records = [
                record for record in active if record["predicted"] != 0
            ]
            correct_predicted_dispatch = sum(
                record["expected"] != 0 for record in predicted_dispatch_records
            )
            correct_destination = sum(
                record["predicted"] == record["expected"] for record in dispatch
            )
            wait_probabilities = np.asarray(
                [record["wait_probability"] for record in selected], dtype=np.float64
            )
            rows.append(
                {
                    "vessel": vessel,
                    "mode": mode,
                    "count": len(selected),
                    "forced_wait_count": forced,
                    "voluntary_wait_count": len(voluntary_wait),
                    "dispatch_count": len(dispatch),
                    "voluntary_wait_accuracy": _safe_rate(
                        sum(record["predicted"] == 0 for record in voluntary_wait),
                        len(voluntary_wait),
                    ),
                    "dispatch_recall": _safe_rate(predicted_dispatch, len(dispatch)),
                    "dispatch_precision": _safe_rate(
                        correct_predicted_dispatch,
                        len(predicted_dispatch_records),
                    ),
                    "wait_specificity": _safe_rate(
                        sum(record["predicted"] == 0 for record in voluntary_wait),
                        len(voluntary_wait),
                    ),
                    "conditional_destination_accuracy": _safe_rate(
                        correct_destination, len(dispatch)
                    ),
                    "mean_argmax_margin": float(
                        np.mean([record["argmax_margin"] for record in active])
                    )
                    if active
                    else float("nan"),
                    "mean_wait_probability": float(wait_probabilities.mean()),
                    "mean_dispatch_probability": float((1.0 - wait_probabilities).mean()),
                }
            )
    return rows


@dataclass
class VesselRolloutDiagnostics:
    """Accumulate pre-action vessel behavior during one rollout."""

    _records: list[dict[str, object]] = field(default_factory=list)
    _current_streak: dict[tuple[int, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _longest_streak: dict[tuple[int, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def observe(self, env, vessel_actions, wait_probabilities=None) -> None:
        actions = [int(action) for action in vessel_actions]
        if len(actions) != len(env.vessel_ids):
            raise ValueError("vessel_actions must contain one action per vessel")
        probabilities = (
            [float("nan")] * len(actions)
            if wait_probabilities is None
            else [float(value) for value in wait_probabilities]
        )
        if len(probabilities) != len(actions):
            raise ValueError("wait_probabilities must contain one value per vessel")

        modes = vessel_operation_modes(env)
        masks = env.vessel_action_mask()
        state = env.simulator.state
        time_h = float(state.time_h)
        for vessel_index, (vessel_id, action, mode, mask, wait_probability) in enumerate(
            zip(env.vessel_ids, actions, modes, masks, probabilities)
        ):
            vessel_state = env.simulator.vessel_states[vessel_id]
            berth = vessel_state["berth"]
            berthed = vessel_state["mode"] == "berthed"
            destination = env._vessel_action_destination(vessel_id, action)
            dispatch = bool(berthed and destination is not None and destination != berth)
            cargo_t = float(state.entity_inventory_t.get(vessel_id, 0.0))
            capacity_t = float(env.network.entities[vessel_id].capacity_t)
            partial = dispatch and cargo_t < capacity_t - 1e-9
            milk_run = (
                dispatch
                and berth in env.emitter_ids
                and destination in env.emitter_ids
            )
            key = (vessel_index, mode)
            if berthed and not dispatch:
                self._current_streak[key] += 1
                self._longest_streak[key] = max(
                    self._longest_streak[key], self._current_streak[key]
                )
            else:
                self._current_streak[key] = 0
            self._records.append(
                {
                    "vessel_index": vessel_index,
                    "mode": mode,
                    "forced_wait": bool(mask[0] and sum(bool(value) for value in mask) == 1),
                    "wait": action == 0,
                    "dispatch": dispatch,
                    "partial": partial,
                    "milk_run": milk_run,
                    "time_h": time_h,
                    "wait_probability": wait_probability,
                }
            )

    def rows(self, **identity) -> list[dict[str, object]]:
        if not self._records:
            return []
        vessel_count = 1 + max(int(row["vessel_index"]) for row in self._records)
        rows = []
        for vessel in (*range(vessel_count), "all"):
            vessel_records = (
                self._records
                if vessel == "all"
                else [row for row in self._records if row["vessel_index"] == vessel]
            )
            for mode in (*VESSEL_OPERATION_MODES, "all"):
                selected = (
                    vessel_records
                    if mode == "all"
                    else [row for row in vessel_records if row["mode"] == mode]
                )
                if not selected:
                    continue
                dispatch_times = [
                    float(row["time_h"]) for row in selected if row["dispatch"]
                ]
                streak_values = [
                    value
                    for (index, streak_mode), value in self._longest_streak.items()
                    if (vessel == "all" or index == vessel)
                    and (mode == "all" or streak_mode == mode)
                ]
                wait_probabilities = np.asarray(
                    [float(row["wait_probability"]) for row in selected], dtype=np.float64
                )
                row = {
                    **identity,
                    "vessel": vessel,
                    "mode": mode,
                    "count": len(selected),
                    "forced_wait_count": sum(bool(item["forced_wait"]) for item in selected),
                    "wait_count": sum(bool(item["wait"]) for item in selected),
                    "dispatch_count": sum(bool(item["dispatch"]) for item in selected),
                    "partial_load_departure_count": sum(
                        bool(item["partial"]) for item in selected
                    ),
                    "milk_run_departure_count": sum(
                        bool(item["milk_run"]) for item in selected
                    ),
                    "first_dispatch_hour": min(dispatch_times) if dispatch_times else float("nan"),
                    "longest_berthed_no_dispatch_streak": max(streak_values, default=0),
                    "mean_wait_probability": (
                        float(np.nanmean(wait_probabilities))
                        if not np.isnan(wait_probabilities).all()
                        else float("nan")
                    ),
                }
                rows.append(row)
        return rows
