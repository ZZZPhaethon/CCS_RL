"""Evaluate a saved PPO model against idle/greedy baselines and write results to
output/rl_ppo/. Reports both stochastic and deterministic PPO because an argmax
(deterministic) action can collapse a still-stochastic dispatch policy to WAIT.

Usage (ccs-rlllm-gpu env, from repo root):
    set PYTHONPATH=src
    python scripts/eval_ppo_model.py output/rl_ppo/ppo_phase1_720h_inj80_ts200000.zip
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from sim.train import make_native_env
from sim.environment.gym_adapter import flat_action_mask, native_action_from_flat
from sim.metrics import run_episode
from sim.control.baselines import idle_policy, greedy_shuttle_policy


def ppo_policy(model, deterministic):
    def policy(env):
        obs = np.asarray(env._observation(), dtype=np.float32)
        masks = flat_action_mask(env.vessel_action_mask(), env.well_rate_action_mask())
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
        return native_action_from_flat(env, action)
    return policy


def mean_metric(metrics, name):
    values = [getattr(m, name) for m in metrics if getattr(m, name) is not None]
    return float(np.mean(values)) if values else float("nan")


def summarize_policy(name, metrics):
    return {
        "policy": name,
        "storage_rate": round(mean_metric(metrics, "storage_rate"), 4),
        "loss_rate": round(mean_metric(metrics, "loss_rate"), 4),
        "stored_t": round(mean_metric(metrics, "stored_t"), 2),
        "vented_t": round(mean_metric(metrics, "vented_t"), 2),
        "operating_cost": round(mean_metric(metrics, "operating_cost"), 2),
        "vent_penalty": round(mean_metric(metrics, "vent_penalty"), 2),
        "total_cost": round(mean_metric(metrics, "total_cost"), 2),
        "cost_per_stored_t": round(mean_metric(metrics, "cost_per_stored_t"), 4),
        "total_cost_per_stored_t": round(mean_metric(metrics, "total_cost_per_stored_t"), 4),
    }


def print_row(row):
    print(
        f"{row['policy']:20s} storage={row['storage_rate']:6.1%}  "
        f"loss={row['loss_rate']:6.1%}  stored={row['stored_t']:9,.0f}t  "
        f"vented={row['vented_t']:8,.0f}t  total={row['total_cost']:11,.0f}  "
        f"total/t={row['total_cost_per_stored_t']:7,.1f}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episode-hours", type=int, default=720)
    p.add_argument("--warm-start", action="store_true", default=True)
    p.add_argument("--weather-obs", action="store_true",
                   help="use the weather-observation env for models trained with --weather-obs")
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    args = p.parse_args()

    model = MaskablePPO.load(args.model)

    entries = [
        ("idle", idle_policy),
        ("greedy_shuttle", greedy_shuttle_policy),
        ("ppo_stochastic", ppo_policy(model, deterministic=False)),
        ("ppo_deterministic", ppo_policy(model, deterministic=True)),
    ]

    rows = []
    for name, policy in entries:
        metrics = []
        for s in args.seeds:
            env = make_native_env(
                episode_hours=args.episode_hours,
                warm_start=args.warm_start,
                include_weather_obs=args.weather_obs,
            )
            m = run_episode(env, policy, seed=s)
            metrics.append(m)
        row = summarize_policy(name, metrics)
        rows.append(row)
        print_row(row)

    out = Path("output/rl_ppo")
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.model).stem
    with (out / f"eval_corrected_{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    md = [f"# Corrected PPO evaluation — {stem}", "",
          f"Generated: {dt.datetime.now().isoformat(timespec='seconds')} | "
          f"ep={args.episode_hours}h, warm_start={args.warm_start}, seeds={args.seeds}", "",
          "Both PPO variants are reported: a still-stochastic dispatch policy scores far",
          "higher when sampled than under argmax (deterministic), which can collapse to WAIT.", "",
          "| policy | storage_rate | loss_rate | stored_t | vented_t | operating_cost | vent_penalty | total_cost | cost_per_stored_t | total_cost_per_stored_t |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| {r['policy']} | {r['storage_rate']:.1%} | {r['loss_rate']:.1%} | "
            f"{r['stored_t']:,.0f} | {r['vented_t']:,.0f} | "
            f"{r['operating_cost']:,.0f} | {r['vent_penalty']:,.0f} | "
            f"{r['total_cost']:,.0f} | {r['cost_per_stored_t']:,.1f} | "
            f"{r['total_cost_per_stored_t']:,.1f} |"
        )
    (out / f"eval_corrected_{stem}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote: {out / ('eval_corrected_' + stem + '.md')}")
    print(f"wrote: {out / ('eval_corrected_' + stem + '.csv')}")


if __name__ == "__main__":
    main()
