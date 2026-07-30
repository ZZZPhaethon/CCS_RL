"""Evaluate checkpoint-level routers for Iterative-Q policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

if __package__:
    from experiments import evaluate_iterative_action_q as base
else:  # pragma: no cover
    import evaluate_iterative_action_q as base


ROUTER_MODES = ("confidence", "pooled")


def parse_checkpoint(value: str) -> tuple[str, str]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "checkpoint must be NAME=PATH"
        ) from error
    if not name or not path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    return name, path


def parse_router(value: str) -> dict[str, object]:
    """Parse NAME:MODE:CHECKPOINTS:HEADS:MARGIN:BETA."""

    try:
        name, mode, checkpoint_text, heads, margin, beta = value.split(":")
        checkpoints = checkpoint_text.split(",")
        required_heads = int(heads)
        margin = float(margin)
        uncertainty_beta = float(beta)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "router must be NAME:MODE:CHECKPOINTS:HEADS:MARGIN:BETA"
        ) from error
    if (
        not name
        or mode not in ROUTER_MODES
        or not checkpoints
        or any(not checkpoint for checkpoint in checkpoints)
        or len(checkpoints) != len(set(checkpoints))
        or required_heads <= 0
        or margin < 0.0
        or uncertainty_beta < 0.0
    ):
        raise argparse.ArgumentTypeError("invalid router values")
    return {
        "name": name,
        "mode": mode,
        "checkpoints": checkpoints,
        "required_heads": required_heads,
        "margin": margin,
        "uncertainty_beta": uncertainty_beta,
    }


def parse_windows(value: str) -> list[list[float]]:
    try:
        windows = [
            [float(piece) for piece in window.split("-", 1)]
            for window in value.split(",")
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "windows must be START-END comma-separated pairs"
        ) from error
    if (
        not windows
        or any(len(window) != 2 for window in windows)
        or any(not 0.0 <= start <= end for start, end in windows)
    ):
        raise argparse.ArgumentTypeError("invalid policy windows")
    return windows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=parse_checkpoint,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--routers",
        type=parse_router,
        nargs="+",
        required=True,
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=list(range(2000, 2030)),
    )
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--reward-scale", type=float, default=1e-5)
    parser.add_argument("--max-overrides", type=int, default=12)
    parser.add_argument(
        "--policy-windows",
        type=parse_windows,
        default=parse_windows(
            "108-155,156-203,204-251,252-299,300-347,348-395,"
            "396-443,444-491,492-539,540-587,588-635,636-680"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Reject every seed outside the locked validation range.",
    )
    parser.add_argument(
        "--seed-manifest",
        default="experiments/protocols/unified_window_v1_seed_manifest.json",
    )
    base.common.add_scenario_protocol_arguments(parser)
    args = parser.parse_args(argv)
    if (
        args.episode_hours <= 0
        or args.reward_scale <= 0.0
        or args.max_overrides <= 0
    ):
        parser.error(
            "episode hours, reward scale, and max overrides must be positive"
        )
    checkpoint_names = [name for name, _path in args.checkpoints]
    if len(checkpoint_names) != len(set(checkpoint_names)):
        parser.error("checkpoint names must be unique")
    missing = sorted(
        {
            checkpoint
            for router in args.routers
            for checkpoint in router["checkpoints"]
            if checkpoint not in checkpoint_names
        }
    )
    if missing:
        parser.error(f"router references unknown checkpoints: {missing}")
    if args.validation_only:
        base.validate_controller_validation_seeds(args)
    return args


def select_routed_action(
    expected_q: dict[str, np.ndarray],
    legal_mask: np.ndarray,
    follow_index: int,
    router: dict[str, object],
) -> tuple[int, dict[str, object]]:
    """Route one decision without using rollout outcomes or labels."""

    selected = {
        name: expected_q[name]
        for name in router["checkpoints"]
    }
    if router["mode"] == "pooled":
        pooled = np.concatenate(list(selected.values()), axis=0)
        action, decision = base.select_safe_action(
            pooled,
            legal_mask,
            follow_index,
            required_heads=int(router["required_heads"]),
            margin=float(router["margin"]),
            uncertainty_beta=float(router["uncertainty_beta"]),
        )
        return action, {
            **decision,
            "chosen_checkpoint": "pooled",
            "checkpoint_actions": {},
        }

    checkpoint_actions = {}
    candidates = []
    for order, (name, q_values) in enumerate(selected.items()):
        action, decision = base.select_safe_action(
            q_values,
            legal_mask,
            follow_index,
            required_heads=int(router["required_heads"]),
            margin=float(router["margin"]),
            uncertainty_beta=float(router["uncertainty_beta"]),
        )
        checkpoint_actions[name] = int(action)
        if action != int(follow_index):
            candidates.append(
                (
                    float(decision["lower_confidence_advantage"]),
                    order,
                    name,
                    int(action),
                    decision,
                )
            )
    if not candidates:
        return int(follow_index), {
            "candidate": int(follow_index),
            "agreement": 0,
            "positive_heads": 0,
            "ensemble_advantage": 0.0,
            "advantage_std": 0.0,
            "lower_confidence_advantage": 0.0,
            "chosen_checkpoint": None,
            "checkpoint_actions": checkpoint_actions,
        }
    _score, _order, name, action, decision = max(candidates)
    return action, {
        **decision,
        "chosen_checkpoint": name,
        "checkpoint_actions": checkpoint_actions,
    }


def _metadata_signature(metadata: dict[str, object]) -> tuple[object, ...]:
    return (
        metadata["observation_variant"],
        int(metadata["follow_action_index"]),
        tuple(metadata["state_feature_names"]),
        tuple(tuple(row) for row in metadata["joint_actions"]),
        tuple(metadata.get("future_feature_names", ())),
        metadata.get("future_summary_representation_id"),
    )


def _load_models(args, device):
    models = {}
    metadata_by_name = {}
    for name, checkpoint in args.checkpoints:
        model, metadata = base._load_model(
            SimpleNamespace(checkpoint=checkpoint),
            device,
        )
        models[name] = model
        metadata_by_name[name] = metadata
    signatures = {
        name: _metadata_signature(metadata)
        for name, metadata in metadata_by_name.items()
    }
    reference = next(iter(signatures.values()))
    incompatible = [
        name for name, signature in signatures.items()
        if signature != reference
    ]
    if incompatible:
        raise ValueError(
            f"incompatible checkpoint metadata: {incompatible}"
        )
    return models, metadata_by_name


def evaluate_router(
    args,
    models,
    metadata,
    router,
    baselines,
    device,
    *,
    event_env_factory=None,
):
    rows = []
    variant = str(metadata["observation_variant"])
    follow_index = int(metadata["follow_action_index"])
    event_env_factory = event_env_factory or (
        lambda: base.make_event_env(args, variant)
    )
    for seed in args.eval_seeds:
        wrapper = event_env_factory()
        observation, _info = wrapper.reset_native_seed(int(seed))
        done = False
        event_count = 0
        override_events = 0
        proposed_override_events = 0
        early_departures = 0
        emitter_to_emitter_legs = 0
        agreement_sum = 0
        checkpoint_switches = 0
        selected_counts = {
            checkpoint: 0 for checkpoint in router["checkpoints"]
        }
        previous_checkpoint = None
        used_windows = set()
        while not done:
            expected_q = {
                name: base.expected_q_for_observation(
                    models[name],
                    observation,
                    wrapper.env,
                    device,
                )
                for name in router["checkpoints"]
            }
            action, decision = select_routed_action(
                expected_q,
                wrapper.action_masks(),
                follow_index,
                router,
            )
            proposed_override_events += int(action != follow_index)
            chosen_checkpoint = decision["chosen_checkpoint"]
            if chosen_checkpoint in selected_counts and action != follow_index:
                selected_counts[chosen_checkpoint] += 1
                if (
                    previous_checkpoint is not None
                    and chosen_checkpoint != previous_checkpoint
                ):
                    checkpoint_switches += 1
                previous_checkpoint = chosen_checkpoint

            active_window = next(
                (
                    index
                    for index, (start, end) in enumerate(args.policy_windows)
                    if float(start) <= float(wrapper.env.t) <= float(end)
                ),
                None,
            )
            if (
                action != follow_index
                and (active_window is None or active_window in used_windows)
            ):
                action = follow_index
            if (
                action != follow_index
                and override_events >= int(args.max_overrides)
            ):
                action = follow_index
            if action != follow_index:
                used_windows.add(active_window)

            agreement_sum += int(decision["agreement"])
            observation, _reward, terminated, truncated, info = wrapper.step(
                action
            )
            event_count += 1
            override_events += int(action != follow_index)
            early_departures += int(
                info["residual_early_terminal_departures"]
            )
            emitter_to_emitter_legs += int(
                info["residual_emitter_to_emitter_legs"]
            )
            done = bool(terminated or truncated)

        actual = base._metrics(wrapper.env)
        baseline = baselines[int(seed)]
        row = {
            "gate": router["name"],
            "seed": int(seed),
            "event_count": event_count,
            "override_events": override_events,
            "proposed_override_events": proposed_override_events,
            "mean_head_agreement": agreement_sum / max(1, event_count),
            "checkpoint_switches": checkpoint_switches,
            "selected_checkpoint_counts": json.dumps(
                selected_counts, sort_keys=True
            ),
            "early_departures": early_departures,
            "emitter_to_emitter_legs": emitter_to_emitter_legs,
        }
        for key, value in actual.items():
            row[key] = value
            row[f"greedy_{key}"] = baseline[key]
            row[f"delta_{key}"] = value - baseline[key]
        rows.append(row)
    return rows


def run(args):
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-empty output directory: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    models, metadata_by_name = _load_models(args, device)
    metadata = next(iter(metadata_by_name.values()))
    variant = str(metadata["observation_variant"])
    baselines = {
        int(seed): base.greedy_metrics(args, variant, int(seed))
        for seed in args.eval_seeds
    }
    rows = []
    for router in args.routers:
        rows.extend(
            evaluate_router(
                args,
                models,
                metadata,
                router,
                baselines,
                device,
            )
        )
    with (out_dir / "evaluation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "kind": "iterative_q_checkpoint_router_validation",
        "checkpoints": {
            name: checkpoint for name, checkpoint in args.checkpoints
        },
        "eval_seeds": [int(seed) for seed in args.eval_seeds],
        "validation_only": bool(args.validation_only),
        "seed_manifest": (
            str(args.seed_manifest) if args.validation_only else None
        ),
        "max_overrides": int(args.max_overrides),
        "policy_windows": args.policy_windows,
        "routers": args.routers,
        "summary": base._summary(rows),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
