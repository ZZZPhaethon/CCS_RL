"""One-shot PPO training runner: train with the aligned reward, save the model,
evaluate against baselines, and write results under output/rl_ppo/.

Usage (from repo root, ccs-rlllm-gpu env):
    set PYTHONPATH=src
    python scripts/train_ppo_run.py --timesteps 200000 --episode-hours 720 \
        --injection-reward-eur-per-t 80
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from sim.train import train_ppo, compare, _format_comparison


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--episode-hours", type=int, default=720)
    p.add_argument("--injection-reward-eur-per-t", type=float, default=80.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    p.add_argument("--progress-bar", action="store_true", help="show a tqdm/rich progress bar")
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    args = p.parse_args()

    out = Path("output/rl_ppo")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"phase1_{args.episode_hours}h_inj{args.injection_reward_eur_per_t:.0f}_ts{args.timesteps}"

    print(f"[{dt.datetime.now():%H:%M:%S}] training {tag} ...", flush=True)
    model = train_ppo(
        total_timesteps=args.timesteps,
        seed=args.seed,
        episode_hours=args.episode_hours,
        injection_reward_eur_per_t=args.injection_reward_eur_per_t,
        n_steps=args.n_steps,
        device=args.device,
        progress_bar=args.progress_bar,
        verbose=1,
    )

    model_path = out / f"ppo_{tag}.zip"
    model.save(str(model_path))
    print(f"[{dt.datetime.now():%H:%M:%S}] saved model -> {model_path}", flush=True)

    print(f"[{dt.datetime.now():%H:%M:%S}] evaluating vs baselines ...", flush=True)
    rows = compare(model, seeds=args.eval_seeds, episode_hours=args.episode_hours)
    table = _format_comparison(rows)
    print(table, flush=True)

    (out / f"ppo_vs_baselines_{tag}.md").write_text(
        f"# PPO vs baselines — {tag}\n\n"
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n\n"
        f"Config: timesteps={args.timesteps}, episode_hours={args.episode_hours}, "
        f"injection_reward_eur_per_t={args.injection_reward_eur_per_t}, "
        f"n_steps={args.n_steps}, seed={args.seed}, eval_seeds={args.eval_seeds}\n\n"
        f"```\n{table}\n```\n",
        encoding="utf-8",
    )
    (out / f"ppo_vs_baselines_{tag}.json").write_text(
        json.dumps(rows, indent=2, default=float), encoding="utf-8"
    )
    print(f"[{dt.datetime.now():%H:%M:%S}] DONE. results in {out}", flush=True)


if __name__ == "__main__":
    main()
