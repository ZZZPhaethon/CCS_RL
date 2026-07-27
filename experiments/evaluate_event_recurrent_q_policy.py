"""Roll out a pretrained recurrent residual Q policy with ensemble safety gates."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import compare_forecast_encoders_rl as compare

from sim.control.baselines import greedy_shuttle_policy
from sim.control.recurrent_distributional_q import (
    RecurrentBootstrappedQuantileQ,
    StatelessStructuredActionQuantileQ,
    StructuredActionRecurrentQuantileQ,
)
from sim.environment.event_residual_gym import EventJointResidualGymEnv


DEFAULT_GATES = (
    "strict4_margin10k:4:0.10",
    "strict4_margin0:4:0.0",
    "majority3_margin10k:3:0.10",
    "ensemble_margin10k:1:0.10",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(2000, 2030)))
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--reward-scale", type=float, default=1e-5)
    parser.add_argument("--gates", nargs="+", default=list(DEFAULT_GATES))
    parser.add_argument("--reset-recurrent-state", action="store_true")
    parser.add_argument(
        "--eta-feature-indices",
        type=int,
        nargs="*",
        default=None,
        help="Keep only these ETA summary feature indices during evaluation.",
    )
    parser.add_argument(
        "--eta-residual-scale",
        type=float,
        default=1.0,
        help="Multiply the learned ETA future correction by this factor.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if (
        args.episode_hours <= 0
        or args.reward_scale <= 0.0
        or args.eta_residual_scale < 0.0
    ):
        parser.error(
            "episode hours and reward scale must be positive, and ETA scale nonnegative"
        )
    if args.eta_feature_indices is not None and any(
        index < 0 or index >= 20 for index in args.eta_feature_indices
    ):
        parser.error("ETA feature indices must be between 0 and 19")
    args.gates = [parse_gate(value) for value in args.gates]
    return args


def parse_gate(value: str) -> dict[str, object]:
    try:
        parts = value.split(":")
        if len(parts) not in (3, 4, 5, 6):
            raise ValueError
        name, agreement, margin = parts[:3]
        agreement = int(agreement)
        margin = float(margin)
        max_overrides = int(parts[3]) if len(parts) >= 4 else None
        min_hour = None
        max_hour = None
        windows = None
        uncertainty_beta = 0.0
        if len(parts) == 5:
            windows = []
            for window_text in parts[4].split(","):
                start_text, end_text = window_text.split("-", 1)
                windows.append([float(start_text), float(end_text)])
        if len(parts) == 6:
            if "-" in parts[4] or "," in parts[4]:
                windows = []
                for window_text in parts[4].split(","):
                    start_text, end_text = window_text.split("-", 1)
                    windows.append([float(start_text), float(end_text)])
                uncertainty_beta = float(parts[5])
            else:
                min_hour = float(parts[4])
                max_hour = float(parts[5])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "gate must be NAME:HEADS:MARGIN[:MAX_OVERRIDES] or "
            "NAME:HEADS:MARGIN:MAX_OVERRIDES:WINDOWS or "
            "NAME:HEADS:MARGIN:MAX_OVERRIDES:WINDOWS:UNCERTAINTY_BETA or "
            "NAME:HEADS:MARGIN:MAX_OVERRIDES:MIN_HOUR:MAX_HOUR"
        ) from error
    if (
        not name
        or agreement <= 0
        or margin < 0.0
        or (max_overrides is not None and max_overrides <= 0)
        or uncertainty_beta < 0.0
        or (min_hour is not None and not 0.0 <= min_hour < max_hour)
        or (
            windows is not None
            and (
                not windows
                or any(not 0.0 <= start <= end for start, end in windows)
            )
        )
    ):
        raise argparse.ArgumentTypeError("invalid gate values")
    return {
        "name": name,
        "required_heads": agreement,
        "margin": margin,
        "max_overrides": max_overrides,
        "min_hour": min_hour,
        "max_hour": max_hour,
        "windows": windows,
        "uncertainty_beta": uncertainty_beta,
    }


def _compare_args(args, variant: str):
    return compare.parse_args(
        [
            "train",
            "--variant",
            variant,
            "--demo-cache",
            "unused-recurrent-q-eval.npz",
            "--timesteps",
            "0",
            "--bc-only",
            "--episode-hours",
            str(args.episode_hours),
            "--device",
            str(args.device),
        ]
    )


def _make_native(args, variant):
    native = compare.make_experiment_env(
        _compare_args(args, variant), demonstration=False
    )
    native.config.reward_scale = float(args.reward_scale)
    return native


def make_event_env(args, variant):
    return EventJointResidualGymEnv(
        _make_native(args, variant),
        variant,
        include_episode_progress=True,
        greedy_control_variate=True,
        hourly_gamma=1.0,
    )


def _metrics(env):
    stored = float(env.ledger.stored_t)
    return {
        "total_cost_eur": float(env.ledger.total_cost),
        "operating_cost_eur": float(env.ledger.operating_cost),
        "vent_penalty_eur": float(env.ledger.vent_penalty),
        "vented_t": float(env.ledger.vented_t),
        "stored_t": stored,
        "unit_cost_eur_per_t": (
            float(env.ledger.total_cost) / stored if stored > 1e-9 else np.nan
        ),
    }


def greedy_metrics(args, variant, seed):
    env = _make_native(args, variant)
    env.reset(seed=int(seed))
    while env.t < env.n_steps:
        env.step(greedy_shuttle_policy(env))
    return _metrics(env)


def select_safe_action(
    expected_q: np.ndarray,
    legal_mask: np.ndarray,
    follow_index: int,
    *,
    required_heads: int,
    margin: float,
    uncertainty_beta: float = 0.0,
) -> tuple[int, dict[str, float | int]]:
    """Require head agreement and positive advantage over FOLLOW."""

    masked = np.where(legal_mask[None, :], expected_q, -np.inf)
    ensemble = masked.mean(axis=0)
    candidate = int(np.argmax(ensemble))
    if candidate == int(follow_index):
        return int(follow_index), {
            "candidate": candidate,
            "agreement": int((masked.argmax(axis=1) == candidate).sum()),
            "positive_heads": 0,
            "ensemble_advantage": 0.0,
            "advantage_std": 0.0,
            "lower_confidence_advantage": 0.0,
        }
    head_best = masked.argmax(axis=1)
    advantages = expected_q[:, candidate] - expected_q[:, int(follow_index)]
    agreement = int((head_best == candidate).sum())
    positive_heads = int((advantages > float(margin)).sum())
    ensemble_advantage = float(advantages.mean())
    advantage_std = float(advantages.std())
    lower_confidence_advantage = float(
        ensemble_advantage - float(uncertainty_beta) * advantage_std
    )
    accepted = (
        agreement >= int(required_heads)
        and positive_heads >= int(required_heads)
        and lower_confidence_advantage > float(margin)
    )
    return (candidate if accepted else int(follow_index)), {
        "candidate": candidate,
        "agreement": agreement,
        "positive_heads": positive_heads,
        "ensemble_advantage": ensemble_advantage,
        "advantage_std": advantage_std,
        "lower_confidence_advantage": lower_confidence_advantage,
    }


def _load_model(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    configuration = checkpoint["configuration"]
    normalization = checkpoint["normalization"]
    q_head = configuration.get("q_head", "direct")
    if q_head == "stateless_structured":
        observation_input = configuration.get("observation_input", "state_only")
        model = StatelessStructuredActionQuantileQ(
            metadata["state_feature_names"],
            (168, len(metadata["forecast_channel_names"])),
            metadata["joint_actions"],
            **normalization,
            heads=int(configuration["heads"]),
            quantiles=int(configuration["quantiles"]),
            prior_scale=float(configuration["prior_scale"]),
            action_embedding_size=int(configuration["action_embedding_size"]),
            action_feature_size=int(configuration["action_feature_size"]),
            forecast_encoder=(
                "state_only"
                if observation_input == "state_only"
                else configuration.get("forecast_encoder", "eta_aligned")
            ),
            forecast_channel_names=metadata["forecast_channel_names"],
            episode_hours=int(
                metadata.get("episode_hours", getattr(args, "episode_hours", 720))
            ),
        ).to(device)
    elif q_head == "structured":
        model = StructuredActionRecurrentQuantileQ(
            metadata["state_feature_names"],
            (168, len(metadata["forecast_channel_names"])),
            metadata["joint_actions"],
            **normalization,
            heads=int(configuration["heads"]),
            quantiles=int(configuration["quantiles"]),
            prior_scale=float(configuration["prior_scale"]),
            action_embedding_size=int(configuration["action_embedding_size"]),
            action_feature_size=int(configuration["action_feature_size"]),
            forecast_encoder=configuration.get("forecast_encoder", "tcn"),
            forecast_channel_names=metadata["forecast_channel_names"],
            episode_hours=int(
                metadata.get("episode_hours", getattr(args, "episode_hours", 720))
            ),
        ).to(device)
    else:
        model = RecurrentBootstrappedQuantileQ(
            metadata["state_feature_names"],
            (168, len(metadata["forecast_channel_names"])),
            len(metadata["joint_actions"]),
            **normalization,
            heads=int(configuration["heads"]),
            quantiles=int(configuration["quantiles"]),
            prior_scale=float(configuration["prior_scale"]),
            forecast_encoder=configuration.get("forecast_encoder", "tcn"),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    eta_feature_indices = getattr(args, "eta_feature_indices", None)
    if eta_feature_indices is not None:
        eta_residual = getattr(model, "eta_aligned_residual", None)
        if eta_residual is None:
            raise ValueError("--eta-feature-indices requires an ETA-aligned checkpoint")
        eta_residual.summary_feature_mask.zero_()
        eta_residual.summary_feature_mask[eta_feature_indices] = 1.0
    eta_residual_scale = float(getattr(args, "eta_residual_scale", 1.0))
    if eta_residual_scale != 1.0:
        eta_residual = getattr(model, "eta_aligned_residual", None)
        if eta_residual is None:
            raise ValueError("--eta-residual-scale requires an ETA-aligned checkpoint")
        with torch.no_grad():
            eta_residual.residual_head[-1].weight.mul_(eta_residual_scale)
            eta_residual.residual_head[-1].bias.mul_(eta_residual_scale)
    model.observation_input = configuration.get("observation_input", "state_future")
    model.eval()
    return model, metadata


def _tensor_observation(observation, device):
    state = torch.as_tensor(observation["state"], dtype=torch.float32, device=device)
    forecast = torch.as_tensor(
        observation["forecast"], dtype=torch.float32, device=device
    )
    return state[None, None], forecast[None, None]


def evaluate_gate(args, model, metadata, gate, baselines, device):
    rows = []
    variant = str(metadata["observation_variant"])
    follow_index = int(metadata["follow_action_index"])
    for seed in args.eval_seeds:
        wrapper = make_event_env(args, variant)
        observation, _info = wrapper.reset_native_seed(int(seed))
        stateless = bool(getattr(model, "is_stateless", False))
        if not stateless:
            hidden = model.initial_hidden(1, device)
            previous_action = torch.full(
                (1, 1), -1, dtype=torch.long, device=device
            )
            previous_reward = torch.zeros(
                (1, 1), dtype=torch.float32, device=device
            )
            previous_duration = torch.zeros(
                (1, 1), dtype=torch.float32, device=device
            )
        done = False
        event_count = 0
        override_events = 0
        proposed_override_events = 0
        early_departures = 0
        emitter_to_emitter_legs = 0
        agreement_sum = 0
        used_windows = set()
        while not done:
            if args.reset_recurrent_state and not stateless:
                hidden = model.initial_hidden(1, device)
                previous_action = torch.full(
                    (1, 1), -1, dtype=torch.long, device=device
                )
                previous_reward = torch.zeros(
                    (1, 1), dtype=torch.float32, device=device
                )
                previous_duration = torch.zeros(
                    (1, 1), dtype=torch.float32, device=device
                )
            states, forecasts = _tensor_observation(observation, device)
            if (
                not stateless
                and getattr(model, "observation_input", "state_future") == "state_only"
            ):
                forecasts = model.forecast_mean.reshape(1, 1, 1, -1).expand_as(
                    forecasts
                )
            with torch.no_grad():
                if stateless:
                    q = model(states, forecasts)
                else:
                    q, hidden = model(
                        states,
                        forecasts,
                        previous_action,
                        previous_reward,
                        previous_duration,
                        hidden,
                    )
            expected_q = (
                q[0, 0].mean(dim=-1).cpu().numpy() * float(model.return_scale)
            )
            action, decision = select_safe_action(
                expected_q,
                wrapper.action_masks(),
                follow_index,
                required_heads=int(gate["required_heads"]),
                margin=float(gate["margin"]),
                uncertainty_beta=float(gate.get("uncertainty_beta", 0.0)),
            )
            active_window = None
            if gate.get("windows") is not None:
                active_window = next(
                    (
                        index
                        for index, (start, end) in enumerate(gate["windows"])
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
                gate.get("min_hour") is not None
                and not float(gate["min_hour"])
                <= float(wrapper.env.t)
                <= float(gate["max_hour"])
                and action != follow_index
            ):
                action = follow_index
            if (
                gate.get("max_overrides") is not None
                and override_events >= int(gate["max_overrides"])
                and action != follow_index
            ):
                action = follow_index
            if action != follow_index and active_window is not None:
                used_windows.add(active_window)
            proposed_override_events += int(decision["candidate"] != follow_index)
            agreement_sum += int(decision["agreement"])
            observation, reward, terminated, truncated, info = wrapper.step(action)
            event_count += 1
            override_events += int(action != follow_index)
            early_departures += int(info["residual_early_terminal_departures"])
            emitter_to_emitter_legs += int(info["residual_emitter_to_emitter_legs"])
            if not stateless:
                previous_action = torch.as_tensor(
                    [[action]], dtype=torch.long, device=device
                )
                previous_reward = torch.as_tensor(
                    [[float(reward)]], dtype=torch.float32, device=device
                )
                previous_duration = torch.as_tensor(
                    [[float(info["event_duration_h"])]],
                    dtype=torch.float32,
                    device=device,
                )
            done = bool(terminated or truncated)

        actual = _metrics(wrapper.env)
        baseline = baselines[int(seed)]
        row = {
            "gate": gate["name"],
            "seed": int(seed),
            "event_count": event_count,
            "override_events": override_events,
            "proposed_override_events": proposed_override_events,
            "mean_head_agreement": agreement_sum / max(1, event_count),
            "early_departures": early_departures,
            "emitter_to_emitter_legs": emitter_to_emitter_legs,
        }
        for key, value in actual.items():
            row[key] = value
            row[f"greedy_{key}"] = baseline[key]
            row[f"delta_{key}"] = value - baseline[key]
        rows.append(row)
    return rows


def _summary(rows):
    result = {}
    for gate in sorted({row["gate"] for row in rows}):
        selected = [row for row in rows if row["gate"] == gate]
        delta = np.asarray([row["delta_total_cost_eur"] for row in selected])
        bootstrap_rng = np.random.default_rng(0)
        bootstrap_means = delta[
            bootstrap_rng.integers(0, len(delta), size=(10_000, len(delta)))
        ].mean(axis=1)
        result[gate] = {
            "episodes": len(selected),
            "mean_total_cost_eur": float(np.mean([row["total_cost_eur"] for row in selected])),
            "mean_greedy_total_cost_eur": float(
                np.mean([row["greedy_total_cost_eur"] for row in selected])
            ),
            "mean_delta_total_cost_eur": float(delta.mean()),
            "median_delta_total_cost_eur": float(np.median(delta)),
            "mean_delta_95pct_ci_eur": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
            "wins": int((delta < -1e-6).sum()),
            "ties": int((np.abs(delta) <= 1e-6).sum()),
            "losses": int((delta > 1e-6).sum()),
            "mean_vented_t": float(np.mean([row["vented_t"] for row in selected])),
            "mean_stored_t": float(np.mean([row["stored_t"] for row in selected])),
            "mean_unit_cost_eur_per_t": float(
                np.mean([row["unit_cost_eur_per_t"] for row in selected])
            ),
            "mean_override_events": float(
                np.mean([row["override_events"] for row in selected])
            ),
            "mean_proposed_override_events": float(
                np.mean([row["proposed_override_events"] for row in selected])
            ),
            "early_departures": int(sum(row["early_departures"] for row in selected)),
            "emitter_to_emitter_legs": int(
                sum(row["emitter_to_emitter_legs"] for row in selected)
            ),
        }
    return result


def run(args):
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model, metadata = _load_model(args, device)
    variant = str(metadata["observation_variant"])
    baselines = {
        int(seed): greedy_metrics(args, variant, int(seed)) for seed in args.eval_seeds
    }
    rows = []
    for gate in args.gates:
        rows.extend(evaluate_gate(args, model, metadata, gate, baselines, device))
    with (out_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "kind": "recurrent_distributional_q_policy_evaluation",
        "checkpoint": str(args.checkpoint),
        "eval_seeds": [int(seed) for seed in args.eval_seeds],
        "gates": args.gates,
        "reset_recurrent_state": bool(args.reset_recurrent_state),
        "eta_feature_indices": getattr(args, "eta_feature_indices", None),
        "eta_residual_scale": float(getattr(args, "eta_residual_scale", 1.0)),
        "summary": _summary(rows),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
