"""Evaluate iterative Q policies with ensemble safety gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

if __package__:
    from scripts import compare_forecast_encoders_rl as compare
else:  # pragma: no cover
    import compare_forecast_encoders_rl as compare

from sim.control.baselines import greedy_shuttle_policy
from sim.control.cplex_milp import _terminal_cleanup_cost_for_state
from sim.control.event_based.rl.observation_encoder import (
    FUTURE_SUMMARY_REPRESENTATION_ID,
    future_summary_observation,
)
from sim.control.iterative_action_q import (
    IterativeActionQuantileQ,
    IterativeForecastActionQuantileQ,
    IterativeFutureActionQuantileQ,
    IterativeResidualFutureActionQuantileQ,
)
from sim.environment.event_residual_gym import EventJointResidualGymEnv
from sim.environment.forecast import (
    masked_forecast_band_summary_observation,
    masked_forecast_summary_observation,
    masked_future_forecast_observation,
)

from experiments import iterative_q_data_common as common


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
    parser.add_argument(
        "--future-ablation",
        choices=("none", "mean"),
        default="none",
    )
    parser.add_argument(
        "--state-mean-ablation",
        nargs="*",
        default=[],
        metavar="FEATURE",
        help=(
            "Replace selected state features with their checkpoint training means "
            "before Q inference."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Require every evaluation seed to belong to the locked controller-"
            "validation range and reject formal-test seeds."
        ),
    )
    parser.add_argument(
        "--seed-manifest",
        default="experiments/protocols/unified_window_v1_seed_manifest.json",
    )
    common.add_scenario_protocol_arguments(parser)
    args = parser.parse_args(argv)
    if args.episode_hours <= 0 or args.reward_scale <= 0.0:
        parser.error("episode hours and reward scale must be positive")
    args.gates = [parse_gate(value) for value in args.gates]
    if args.validation_only:
        validate_controller_validation_seeds(args)
    return args


def validate_controller_validation_seeds(args) -> None:
    manifest = json.loads(
        Path(args.seed_manifest).read_text(encoding="utf-8")
    )
    validation_start, validation_end = manifest["controller_validation"][
        "range_inclusive"
    ]
    formal_start, formal_end = manifest["formal_test"]["range_inclusive"]
    invalid = [
        int(seed)
        for seed in args.eval_seeds
        if not int(validation_start) <= int(seed) <= int(validation_end)
    ]
    formal = [
        int(seed)
        for seed in args.eval_seeds
        if int(formal_start) <= int(seed) <= int(formal_end)
    ]
    if invalid or formal:
        raise ValueError(
            "validation-only evaluation rejected seeds outside the locked "
            f"controller-validation range: invalid={invalid}, formal={formal}"
        )


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
            "unused-iterative-q-eval.npz",
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
    if str(getattr(args, "scenario_protocol", "q_original")) != "q_original":
        return common.make_native_env(
            SimpleNamespace(**vars(args), variant=variant)
        )
    native = compare.make_experiment_env(
        _compare_args(args, variant), demonstration=False
    )
    native.config.reward_scale = float(args.reward_scale)
    return native


def make_event_env(args, variant, *, greedy_control_variate=True):
    return EventJointResidualGymEnv(
        _make_native(args, variant),
        variant,
        include_episode_progress=True,
        greedy_control_variate=greedy_control_variate,
        hourly_gamma=1.0,
    )


def _metrics(env):
    stored = float(env.ledger.stored_t)
    episode_vessel_fuel_eur = float(env.ledger.vessel_fuel)
    episode_conditioning_eur = float(env.ledger.conditioning)
    episode_reconditioning_eur = float(env.ledger.reconditioning)
    episode_loading_eur = float(env.ledger.loading)
    episode_unloading_eur = float(env.ledger.unloading)
    episode_operating_cost_eur = float(env.ledger.operating_cost)
    episode_vent_penalty_eur = float(env.ledger.vent_penalty)
    episode_storage_shortfall_penalty_eur = float(
        env.ledger.storage_shortfall_penalty
    )
    episode_total_cost_eur = float(env.ledger.total_cost)
    terminal_cleanup_operating_cost_eur = float(
        _terminal_cleanup_cost_for_state(env, env.cost_model.parameters)
    )
    operating_cost_eur = (
        episode_operating_cost_eur + terminal_cleanup_operating_cost_eur
    )
    total_cost_eur = (
        episode_total_cost_eur + terminal_cleanup_operating_cost_eur
    )
    return {
        "episode_vessel_fuel_eur": episode_vessel_fuel_eur,
        "episode_conditioning_eur": episode_conditioning_eur,
        "episode_reconditioning_eur": episode_reconditioning_eur,
        "episode_loading_eur": episode_loading_eur,
        "episode_unloading_eur": episode_unloading_eur,
        "episode_operating_cost_eur": episode_operating_cost_eur,
        "episode_vent_penalty_eur": episode_vent_penalty_eur,
        "episode_storage_shortfall_penalty_eur": (
            episode_storage_shortfall_penalty_eur
        ),
        "episode_total_cost_eur": episode_total_cost_eur,
        "terminal_cleanup_operating_cost_eur": (
            terminal_cleanup_operating_cost_eur
        ),
        "total_cost_eur": total_cost_eur,
        "operating_cost_eur": operating_cost_eur,
        "vent_penalty_eur": float(env.ledger.vent_penalty),
        "vented_t": float(env.ledger.vented_t),
        "stored_t": stored,
        "unit_cost_eur_per_t": (
            total_cost_eur / stored if stored > 1e-9 else np.nan
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
    model_arguments = {
        "state_mean": normalization["state_mean"],
        "state_std": normalization["state_std"],
        "return_scale": float(normalization["return_scale"]),
        "heads": int(configuration["heads"]),
        "quantiles": int(configuration["quantiles"]),
        "prior_scale": float(configuration["prior_scale"]),
        "action_embedding_size": int(configuration["action_embedding_size"]),
        "action_feature_size": int(configuration["action_feature_size"]),
    }
    q_head = configuration.get("q_head")
    if q_head in (
        "iterative_action_q_future_v4_24_72",
        "iterative_action_q_future_summary",
    ):
        model = IterativeFutureActionQuantileQ(
            metadata["state_feature_names"],
            metadata["future_feature_names"],
            metadata["joint_actions"],
            future_mean=normalization["future_mean"],
            future_std=normalization["future_std"],
            **model_arguments,
        ).to(device)
        if "future_summary_windows_h" in metadata:
            model.forecast_summary_windows_h = tuple(
                int(value) for value in metadata["future_summary_windows_h"]
            )
        if q_head == "iterative_action_q_future_summary":
            if "future_summary_bands_h" in metadata:
                model.forecast_summary_bands_h = tuple(
                    (int(start), int(end))
                    for start, end in metadata["future_summary_bands_h"]
                )
        model.future_summary_representation_id = metadata.get(
            "future_summary_representation_id"
        )
    elif q_head == "iterative_action_q_future_residual_summary":
        model = IterativeResidualFutureActionQuantileQ(
            metadata["state_feature_names"],
            metadata["future_feature_names"],
            metadata["joint_actions"],
            future_mean=normalization["future_mean"],
            future_std=normalization["future_std"],
            future_residual_scale_limit=float(
                configuration["future_residual_scale_limit"]
            ),
            future_dropout=float(configuration["future_dropout"]),
            **model_arguments,
        ).to(device)
        model.forecast_summary_windows_h = tuple(
            int(value) for value in metadata["future_summary_windows_h"]
        )
        model.future_summary_representation_id = metadata.get(
            "future_summary_representation_id"
        )
    elif q_head == "iterative_action_q_future_168":
        model = IterativeForecastActionQuantileQ(
            metadata["state_feature_names"],
            metadata["forecast_feature_names"],
            metadata["joint_actions"],
            forecast_mean=normalization["forecast_mean"],
            forecast_std=normalization["forecast_std"],
            forecast_horizon_h=int(metadata["forecast_horizon_h"]),
            forecast_encoder=str(configuration["forecast_encoder"]),
            **model_arguments,
        ).to(device)
    elif q_head == "iterative_action_q":
        model = IterativeActionQuantileQ(
            metadata["state_feature_names"],
            metadata["joint_actions"],
            **model_arguments,
        ).to(device)
    else:
        raise ValueError("checkpoint is not an iterative Q model")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.state_feature_names = tuple(metadata["state_feature_names"])
    model.source_state_feature_names = tuple(
        metadata.get("source_state_feature_names", metadata["state_feature_names"])
    )
    model.eval()
    return model, metadata


def _tensor_observation(observation, device, model):
    state = torch.as_tensor(observation["state"], dtype=torch.float32, device=device)
    source_names = tuple(model.source_state_feature_names)
    model_names = tuple(model.state_feature_names)
    if state.shape[-1] == len(source_names):
        indices = [source_names.index(name) for name in model_names]
        state = state.index_select(
            -1,
            torch.as_tensor(indices, dtype=torch.long, device=device),
        )
    elif state.shape[-1] != len(model_names):
        raise ValueError(
            "observation state width does not match checkpoint source or model schema"
        )
    return state[None, None]


def state_mean_ablation_spec(model, metadata, feature_names):
    model_names = list(metadata["state_feature_names"])
    source_names = list(
        metadata.get("source_state_feature_names", metadata["state_feature_names"])
    )
    missing = [name for name in feature_names if name not in model_names]
    if missing:
        raise ValueError(f"unknown state-mean ablation features: {missing}")
    state_mean = model.state_mean.detach().cpu().numpy()
    return tuple(
        (source_names.index(name), float(state_mean[model_names.index(name)]))
        for name in dict.fromkeys(feature_names)
    )


def apply_state_mean_ablation(observation, ablation_spec):
    if not ablation_spec:
        return observation
    updated = dict(observation)
    state = np.asarray(observation["state"], dtype=np.float32).copy()
    for index, value in ablation_spec:
        state[int(index)] = float(value)
    updated["state"] = state
    return updated


def expected_q_for_observation(
    model,
    observation,
    env,
    device,
    future_ablation="none",
) -> np.ndarray:
    states = _tensor_observation(observation, device, model)
    with torch.no_grad():
        if isinstance(
            model,
            (
                IterativeFutureActionQuantileQ,
                IterativeResidualFutureActionQuantileQ,
            ),
        ):
            if future_ablation == "mean":
                summary_values = model.future_mean.detach().cpu().numpy()
            else:
                windows_h = getattr(model, "forecast_summary_windows_h", None)
                bands_h = getattr(model, "forecast_summary_bands_h", None)
                representation_id = getattr(
                    model,
                    "future_summary_representation_id",
                    None,
                )
                if representation_id == FUTURE_SUMMARY_REPRESENTATION_ID:
                    summary_values = future_summary_observation(
                        env,
                        windows_h,
                    )
                elif bands_h is not None:
                    summary_values = masked_forecast_band_summary_observation(
                        env, bands_h
                    )
                elif windows_h is not None:
                    summary_values = masked_forecast_summary_observation(
                        env, windows_h
                    )
                else:
                    summary_values = future_summary_observation(env)
            summary = torch.as_tensor(
                summary_values, dtype=torch.float32, device=device
            )[None, None]
            q = model(states, summary)
        elif isinstance(model, IterativeForecastActionQuantileQ):
            forecast = torch.as_tensor(
                masked_future_forecast_observation(
                    env, horizon_h=model.forecast_horizon_h
                ),
                dtype=torch.float32,
                device=device,
            )[None, None]
            q = model(states, forecast)
        else:
            q = model(states)
    return q[0, 0].mean(dim=-1).cpu().numpy() * float(model.return_scale)


def evaluate_gate(
    args,
    model,
    metadata,
    gate,
    baselines,
    device,
    *,
    event_env_factory=None,
):
    rows = []
    variant = str(metadata["observation_variant"])
    follow_index = int(metadata["follow_action_index"])
    event_env_factory = event_env_factory or (
        lambda: make_event_env(args, variant)
    )
    ablation_spec = state_mean_ablation_spec(
        model,
        metadata,
        args.state_mean_ablation,
    )
    for seed in args.eval_seeds:
        wrapper = event_env_factory()
        started_at = perf_counter()
        observation, _info = wrapper.reset_native_seed(int(seed))
        done = False
        event_count = 0
        override_events = 0
        proposed_override_events = 0
        early_departures = 0
        emitter_to_emitter_legs = 0
        agreement_sum = 0
        used_windows = set()
        while not done:
            policy_observation = apply_state_mean_ablation(
                observation,
                ablation_spec,
            )
            expected_q = expected_q_for_observation(
                model,
                policy_observation,
                wrapper.env,
                device,
                args.future_ablation,
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
            observation, _reward, terminated, truncated, info = wrapper.step(action)
            event_count += 1
            override_events += int(action != follow_index)
            early_departures += int(info["residual_early_terminal_departures"])
            emitter_to_emitter_legs += int(info["residual_emitter_to_emitter_legs"])
            done = bool(terminated or truncated)

        actual = _metrics(wrapper.env)
        baseline = baselines[int(seed)]
        row = {
            "gate": gate["name"],
            "seed": int(seed),
            "wall_clock_seconds": perf_counter() - started_at,
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
            "mean_wall_clock_seconds": float(
                np.mean([row["wall_clock_seconds"] for row in selected])
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
        "kind": "iterative_action_q_policy_evaluation",
        "checkpoint": str(args.checkpoint),
        "eval_seeds": [int(seed) for seed in args.eval_seeds],
        "scenario_protocol": str(args.scenario_protocol),
        "stress_level": str(args.stress_level),
        "forecast_context_hours": int(args.forecast_context_hours),
        "state_mean_ablation": list(args.state_mean_ablation),
        "validation_only": bool(args.validation_only),
        "seed_manifest": (
            str(args.seed_manifest) if args.validation_only else None
        ),
        "gates": args.gates,
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
