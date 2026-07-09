"""Compare old economic RL vs vent-first RL after greedy BC warm-start.

Both models use the same scenario, seeds, teacher, and partial-dispatch action
mask. The reported "actual" cost is operating cost plus venting penalty only.

Usage:
    set PYTHONPATH=src
    python scripts/compare_reward_modes_bc.py --timesteps 100000 --episode-hours 720
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import replace
from pathlib import Path

import numpy as np

from sim.control.baselines import greedy_shuttle_policy, idle_policy
from sim.control.imitation import bc_pretrain, make_kickstart_callback
from sim.environment.gym_adapter import CCSGymEnv, make_ppo_policy
from sim.metrics import run_episode
from sim.train import make_native_env


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _actual_total_cost(metric) -> float:
    return metric.operating_cost + metric.vent_penalty


def _actual_cost_per_stored_t(metric) -> float:
    if metric.stored_t <= 0.0:
        return float("nan")
    return _actual_total_cost(metric) / metric.stored_t


def apply_yara_buffer_capacity_override(env, args) -> None:
    capacity = getattr(args, "yara_buffer_capacity", None)
    if capacity is None:
        return
    emitter_id = "yara_sluiskil"
    try:
        emitter = env.network.entities[emitter_id]
    except KeyError as exc:
        raise ValueError(f"Cannot set Yara buffer: {emitter_id!r} is not in this scenario.") from exc
    env.network.entities[emitter_id] = replace(emitter, buffer_capacity_t=float(capacity))


def yara_buffer_tag(args) -> str:
    capacity = getattr(args, "yara_buffer_capacity", None)
    if capacity is None:
        return ""
    return f"_yara{capacity:g}"


def bc_tag(args) -> str:
    return f"_bc{args.bc_episodes}w{args.nonwait_weight:g}"


def disturbance_tag(args) -> str:
    capture = getattr(args, "capture_noise_std", 0.30)
    inventory = getattr(args, "initial_inventory_fill_max", 0.5)
    wave_multiplier = getattr(args, "leg_wave_slowdown_multiplier", 1.0)
    wave_floor = getattr(args, "leg_wave_speed_factor_floor", 0.0)
    if capture == 0.30 and inventory == 0.5 and wave_multiplier == 1.0 and wave_floor == 0.0:
        return ""
    return f"_cap{capture:g}_inv{inventory:g}_wave{wave_multiplier:g}_floor{wave_floor:g}"


def summarize(policy: str, metrics) -> dict[str, float | str]:
    return {
        "policy": policy,
        "storage_rate": _mean([m.storage_rate for m in metrics]),
        "loss_rate": _mean([m.loss_rate for m in metrics]),
        "stored_t": _mean([m.stored_t for m in metrics]),
        "vented_t": _mean([m.vented_t for m in metrics]),
        "operating_cost": _mean([m.operating_cost for m in metrics]),
        "vent_penalty": _mean([m.vent_penalty for m in metrics]),
        "actual_total_cost": _mean([_actual_total_cost(m) for m in metrics]),
        "actual_cost_per_stored_t": _mean([_actual_cost_per_stored_t(m) for m in metrics]),
    }


def make_env(args, reward_mode: str):
    env = make_native_env(
        episode_hours=args.episode_hours,
        warm_start=True,
        injection_reward_eur_per_t=args.injection_reward_eur_per_t,
        store_reward_eur_per_t=args.store_reward,
        vent_penalty_weight=args.vent_weight,
        operating_cost_weight=args.operating_cost_weight,
        reward_mode=reward_mode,
        vent_first_vent_eur_per_t=args.vent_first_vent_eur_per_t,
        overflow_risk_eur_per_t=args.overflow_risk_eur_per_t,
        overflow_risk_lookahead_h=args.overflow_risk_lookahead_h,
        carbon_price_eur_per_t=args.carbon_price,
        enforce_full_load_dispatch=args.enforce_full_load_dispatch,
        scenario=args.scenario,
        weather_mode=args.weather_mode,
        include_weather_obs=args.weather_obs,
        wave_height_nc_paths=args.wave_height_nc_paths,
        lstm_prediction_csv=args.lstm_prediction_csv,
        capture_noise_std=args.capture_noise_std,
        initial_inventory_fill_max=args.initial_inventory_fill_max,
        leg_wave_slowdown_multiplier=args.leg_wave_slowdown_multiplier,
        leg_wave_speed_factor_floor=args.leg_wave_speed_factor_floor,
    )
    apply_yara_buffer_capacity_override(env, args)
    return env


def pretrain_one(args, reward_mode: str):
    from sb3_contrib import MaskablePPO

    native_env = make_env(args, reward_mode)
    gym_env = CCSGymEnv(native_env)
    model = MaskablePPO(
        "MlpPolicy",
        gym_env,
        seed=args.seed,
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        verbose=args.verbose,
    )
    print(
        f"[{dt.datetime.now():%H:%M:%S}] {reward_mode}: BC greedy "
        f"({args.bc_episodes} episodes, {args.bc_epochs} epochs)",
        flush=True,
    )
    demo_obs, demo_acts, demo_masks, demo_weights = bc_pretrain(
        model,
        gym_env,
        greedy_shuttle_policy,
        n_episodes=args.bc_episodes,
        epochs=args.bc_epochs,
        batch_size=args.bc_batch_size,
        lr=args.bc_lr,
        seed0=args.bc_seed0,
        nonwait_weight=args.nonwait_weight,
    )
    return model, (demo_obs, demo_acts, demo_masks, demo_weights)


def fine_tune_one(args, reward_mode: str, model, demo_data) -> None:
    demo_obs, demo_acts, demo_masks, demo_weights = demo_data
    if args.timesteps > 0:
        callback = None
        if args.kickstart_coef > 0.0:
            callback = make_kickstart_callback(
                demo_obs,
                demo_acts,
                demo_masks,
                demo_weights,
                total_timesteps=args.timesteps,
                coef0=args.kickstart_coef,
                batch_size=args.bc_batch_size,
                lr=args.bc_lr,
            )
        print(
            f"[{dt.datetime.now():%H:%M:%S}] {reward_mode}: PPO fine-tune "
            f"({args.timesteps} steps)",
            flush=True,
        )
        model.learn(
            total_timesteps=args.timesteps,
            progress_bar=args.progress_bar,
            callback=callback,
        )


def eval_model(args, reward_mode: str, model, label_suffix: str = ""):
    rows = []
    label = f"{reward_mode}_{label_suffix}" if label_suffix else reward_mode
    entries = [
        (f"{label}_stochastic", make_ppo_policy(model, deterministic=False)),
        (f"{label}_deterministic", make_ppo_policy(model, deterministic=True)),
    ]
    for name, policy in entries:
        metrics = []
        for seed in args.eval_seeds:
            env = make_env(args, reward_mode)
            metrics.append(run_episode(env, policy, seed=seed))
        row = summarize(name, metrics)
        rows.append(row)
        print_row(row)
    return rows


def eval_baselines(args):
    rows = []
    for name, policy in [
        ("idle", idle_policy),
        ("greedy_teacher", greedy_shuttle_policy),
    ]:
        metrics = []
        for seed in args.eval_seeds:
            env = make_env(args, "economic")
            metrics.append(run_episode(env, policy, seed=seed))
        row = summarize(name, metrics)
        rows.append(row)
        print_row(row)
    return rows


def print_row(row: dict[str, float | str]) -> None:
    print(
        f"{row['policy']:24s} "
        f"storage={row['storage_rate']:6.1%}  loss={row['loss_rate']:6.1%}  "
        f"stored={row['stored_t']:9,.0f}t  vented={row['vented_t']:8,.0f}t  "
        f"actual={row['actual_total_cost']:11,.0f}  "
        f"actual/t={row['actual_cost_per_stored_t']:7,.1f}",
        flush=True,
    )


def write_outputs(args, rows, model_paths: dict[str, Path]) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    kick_tag = f"_kick{args.kickstart_coef:g}" if args.kickstart_coef > 0.0 else ""
    buffer_tag = yara_buffer_tag(args)
    demo_tag = bc_tag(args)
    stress_tag = disturbance_tag(args)
    stem = (
        f"reward_mode_compare_{args.scenario}_{args.episode_hours}h"
        f"_ts{args.timesteps}{kick_tag}{buffer_tag}{demo_tag}{stress_tag}_{stamp}"
    )
    csv_path = out / f"{stem}.csv"
    md_path = out / f"{stem}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = [
        f"# Reward Mode Comparison - {args.scenario}",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"episode_hours={args.episode_hours}, timesteps={args.timesteps}, seeds={args.eval_seeds}",
        f"kickstart_coef={args.kickstart_coef}",
        f"bc_episodes={args.bc_episodes}, bc_epochs={args.bc_epochs}, nonwait_weight={args.nonwait_weight:g}",
        f"yara_buffer_capacity={args.yara_buffer_capacity}",
        f"capture_noise_std={args.capture_noise_std:g}",
        f"initial_inventory_fill_max={args.initial_inventory_fill_max:g}",
        f"leg_wave_slowdown_multiplier={args.leg_wave_slowdown_multiplier:g}",
        f"leg_wave_speed_factor_floor={args.leg_wave_speed_factor_floor:g}",
        f"weather_mode={args.weather_mode}",
        f"reward_modes={args.reward_modes}",
        f"partial_dispatch={not args.enforce_full_load_dispatch}",
        "",
        "Actual cost = operating_cost + vent_penalty.",
        "",
        "| policy | storage_rate | loss_rate | stored_t | vented_t | operating_cost | vent_penalty | actual_total_cost | actual_cost_per_stored_t |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['policy']} | {row['storage_rate']:.1%} | {row['loss_rate']:.1%} | "
            f"{row['stored_t']:,.0f} | {row['vented_t']:,.0f} | "
            f"{row['operating_cost']:,.0f} | {row['vent_penalty']:,.0f} | "
            f"{row['actual_total_cost']:,.0f} | {row['actual_cost_per_stored_t']:,.1f} |"
        )
    md += ["", "## Models", ""]
    for mode, path in model_paths.items():
        md.append(f"- {mode}: `{path}`")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote: {md_path}", flush=True)
    print(f"wrote: {csv_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="northern_lights_phase1_3vessels")
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--bc-episodes", type=int, default=30)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bc-seed0", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--nonwait-weight", type=float, default=10.0)
    parser.add_argument("--kickstart-coef", type=float, default=0.0)
    parser.add_argument("--injection-reward-eur-per-t", type=float, default=80.0)
    parser.add_argument("--store-reward", type=float, default=None)
    parser.add_argument("--vent-weight", type=float, default=1.0)
    parser.add_argument("--operating-cost-weight", type=float, default=1.0)
    parser.add_argument("--carbon-price", type=float, default=80.0)
    parser.add_argument("--vent-first-vent-eur-per-t", type=float, default=10_000.0)
    parser.add_argument("--overflow-risk-eur-per-t", type=float, default=100.0)
    parser.add_argument("--overflow-risk-lookahead-h", type=float, default=24.0)
    parser.add_argument("--enforce-full-load-dispatch", action="store_true")
    parser.add_argument("--yara-buffer-capacity", type=float, default=None)
    parser.add_argument("--capture-noise-std", type=float, default=0.30)
    parser.add_argument("--initial-inventory-fill-max", type=float, default=0.5)
    parser.add_argument("--leg-wave-slowdown-multiplier", type=float, default=1.0)
    parser.add_argument("--leg-wave-speed-factor-floor", type=float, default=0.0)
    parser.add_argument(
        "--weather-mode",
        choices=["window", "leg_wave_climatology", "wave_height_netcdf", "lstm_forecast"],
        default="window",
    )
    parser.add_argument("--weather-obs", action="store_true")
    parser.add_argument("--wave-height-nc-paths", nargs="+", default=None)
    parser.add_argument("--lstm-prediction-csv", default=None)
    parser.add_argument(
        "--reward-modes",
        nargs="+",
        choices=["economic", "vent_first"],
        default=["economic", "vent_first"],
        help="reward modes to train/evaluate",
    )
    parser.add_argument("--out-dir", default="output/rl_ppo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = eval_baselines(args)
    model_paths = {}
    kick_tag = f"_kick{args.kickstart_coef:g}" if args.kickstart_coef > 0.0 else ""
    buffer_tag = yara_buffer_tag(args)
    demo_tag = bc_tag(args)
    stress_tag = disturbance_tag(args)
    for reward_mode in args.reward_modes:
        model, demo_data = pretrain_one(args, reward_mode)
        print(f"[{dt.datetime.now():%H:%M:%S}] evaluating {reward_mode} after BC", flush=True)
        rows.extend(eval_model(args, reward_mode, model, label_suffix="bc"))
        if args.timesteps > 0:
            fine_tune_one(args, reward_mode, model, demo_data)
        model_path = out / (
            f"ppo_{reward_mode}_{args.scenario}_{args.episode_hours}h"
            f"_ts{args.timesteps}{kick_tag}{buffer_tag}{demo_tag}{stress_tag}.zip"
        )
        model.save(str(model_path))
        model_paths[reward_mode] = model_path
        if args.timesteps > 0:
            print(f"[{dt.datetime.now():%H:%M:%S}] evaluating {reward_mode}", flush=True)
            rows.extend(eval_model(args, reward_mode, model))

    write_outputs(args, rows, model_paths)


if __name__ == "__main__":
    main()
