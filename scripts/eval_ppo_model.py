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
from sim.environment.gym_adapter import native_action_from_flat
from sim.metrics import run_episode
from sim.control.baselines import idle_policy, greedy_shuttle_policy


def ppo_policy(model, deterministic):
    def policy(env):
        obs = np.asarray(env._observation(), dtype=np.float32)
        action, _ = model.predict(obs, deterministic=deterministic)
        return native_action_from_flat(env, action)
    return policy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episode-hours", type=int, default=720)
    p.add_argument("--warm-start", action="store_true", default=True)
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
        srs, losses = [], []
        for s in args.seeds:
            env = make_native_env(episode_hours=args.episode_hours, warm_start=args.warm_start)
            m = run_episode(env, policy, seed=s)
            srs.append(m.storage_rate); losses.append(m.loss_rate)
        row = {"policy": name, "storage_rate": round(float(np.mean(srs)), 4),
               "loss_rate": round(float(np.mean(losses)), 4)}
        rows.append(row)
        print(f"{name:20s} storage={row['storage_rate']:6.1%}  loss={row['loss_rate']:6.1%}")

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
          "| policy | storage_rate | loss_rate |", "|---|---|---|"]
    for r in rows:
        md.append(f"| {r['policy']} | {r['storage_rate']:.1%} | {r['loss_rate']:.1%} |")
    (out / f"eval_corrected_{stem}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote: {out / ('eval_corrected_' + stem + '.md')}")
    print(f"wrote: {out / ('eval_corrected_' + stem + '.csv')}")


if __name__ == "__main__":
    main()
