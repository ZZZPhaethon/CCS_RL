"""Measure the achievable ceiling: rolling-MILP (near-optimal, receding-horizon)
vs greedy on the same Phase 1 env, with cost metrics. Tells us how much room
there is above greedy for RL/LLM to chase.

Usage (ccs-rlllm-gpu env, from repo root):
    set PYTHONPATH=src
    python scripts/eval_milp_ceiling.py --seeds 101 102 --episode-hours 720
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from sim.train import make_native_env
from sim.metrics import run_episode
from sim.control.baselines import idle_policy, greedy_shuttle_policy
from sim.control.rolling_milp import RollingMilpController


def mean(metrics, name):
    vals = [getattr(m, name) for m in metrics if getattr(m, name) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode-hours", type=int, default=720)
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 102])
    p.add_argument("--replan-every", type=int, default=24)
    p.add_argument("--planning-horizon", type=int, default=168)
    p.add_argument("--time-limit-s", type=float, default=15.0)
    p.add_argument("--free-dispatch", action="store_true",
                   help="disable the full-load dispatch constraint (measure the true optimum)")
    args = p.parse_args()

    def make():
        return make_native_env(
            episode_hours=args.episode_hours, warm_start=True,
            enforce_full_load_dispatch=not args.free_dispatch,
        )

    entries = [
        ("idle", lambda env: idle_policy),
        ("greedy_shuttle", lambda env: greedy_shuttle_policy),
        ("rolling_milp", lambda env: RollingMilpController(
            env, replan_every=args.replan_every,
            planning_horizon_h=args.planning_horizon, time_limit_s=args.time_limit_s)),
    ]

    rows = []
    for name, make_policy in entries:
        print(f"[{dt.datetime.now():%H:%M:%S}] evaluating {name} ...", flush=True)
        metrics = []
        for s in args.seeds:
            env = make()
            metrics.append(run_episode(env, make_policy(env), seed=s))
        row = {
            "policy": name,
            "storage_rate": mean(metrics, "storage_rate"),
            "loss_rate": mean(metrics, "loss_rate"),
            "stored_t": mean(metrics, "stored_t"),
            "vented_t": mean(metrics, "vented_t"),
            "total_cost": mean(metrics, "total_cost"),
            "total_cost_per_stored_t": mean(metrics, "total_cost_per_stored_t"),
        }
        rows.append(row)
        print(f"  {name:16s} storage={row['storage_rate']:6.1%}  loss={row['loss_rate']:6.1%}  "
              f"total/t={row['total_cost_per_stored_t']:7,.1f}", flush=True)

    out = Path("output/rl_ppo"); out.mkdir(parents=True, exist_ok=True)
    md = [f"# MILP ceiling — Phase 1 {args.episode_hours}h, seeds={args.seeds}", "",
          f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}", "",
          "| policy | storage% | loss% | stored t | vented t | total EUR | total/t |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['policy']} | {r['storage_rate']:.1%} | {r['loss_rate']:.1%} | "
                  f"{r['stored_t']:,.0f} | {r['vented_t']:,.0f} | {r['total_cost']:,.0f} | "
                  f"{r['total_cost_per_stored_t']:,.1f} |")
    (out / "milp_ceiling.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[{dt.datetime.now():%H:%M:%S}] wrote {out / 'milp_ceiling.md'}", flush=True)


if __name__ == "__main__":
    main()
