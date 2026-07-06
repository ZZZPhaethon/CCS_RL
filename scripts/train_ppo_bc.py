"""BC warm-start + PPO fine-tune runner.

1. Build the aligned Phase 1 env (injection reward on, long horizon).
2. Behavior-clone the policy from greedy_shuttle demonstrations.
3. Evaluate the BC-only policy.
4. PPO fine-tune, evaluate again, save model + results to output/rl_ppo/.

Usage (ccs-rlllm-gpu env, from repo root):
    set PYTHONPATH=src
    python scripts/train_ppo_bc.py --bc-episodes 20 --bc-epochs 10 --timesteps 100000
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from sim.train import make_native_env
from sim.environment.gym_adapter import CCSGymEnv, make_ppo_policy
from sim.control.baselines import idle_policy, greedy_shuttle_policy
from sim.control.imitation import bc_pretrain, make_kickstart_callback
from sim.metrics import run_episode


def eval_policies(model, episode_hours, seeds, include_weather_obs=False):
    entries = [
        ("idle", idle_policy),
        ("greedy_shuttle", greedy_shuttle_policy),
        ("ppo_stochastic", make_ppo_policy(model, deterministic=False)),
        ("ppo_deterministic", make_ppo_policy(model, deterministic=True)),
    ]
    lines = []
    for name, policy in entries:
        srs, losses = [], []
        for s in seeds:
            env = make_native_env(episode_hours=episode_hours, warm_start=True,
                                  include_weather_obs=include_weather_obs)
            m = run_episode(env, policy, seed=s)
            srs.append(m.storage_rate); losses.append(m.loss_rate)
        line = f"{name:20s} storage={np.mean(srs):6.1%}  loss={np.mean(losses):6.1%}"
        print(line, flush=True)
        lines.append(line)
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode-hours", type=int, default=720)
    p.add_argument("--injection-reward-eur-per-t", type=float, default=80.0)
    p.add_argument("--bc-episodes", type=int, default=20)
    p.add_argument("--bc-epochs", type=int, default=10)
    p.add_argument("--nonwait-weight", type=float, default=10.0,
                   help="loss up-weight for dispatch (non-WAIT) steps in BC")
    p.add_argument("--weather-obs", action="store_true",
                   help="expose per-leg wave weather + seasonality in the observation")
    p.add_argument("--kickstart-coef", type=float, default=0.0,
                   help="initial weight of the decaying BC anchor during PPO fine-tune (0 = off)")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--progress-bar", action="store_true", help="show a tqdm/rich progress bar")
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    args = p.parse_args()

    from sb3_contrib import MaskablePPO

    out = Path("output/rl_ppo"); out.mkdir(parents=True, exist_ok=True)
    weather_tag = "_weather" if args.weather_obs else ""
    tag = f"bc_phase1_{args.episode_hours}h_inj{args.injection_reward_eur_per_t:.0f}{weather_tag}_ts{args.timesteps}"
    report = []

    native_env = make_native_env(
        episode_hours=args.episode_hours, warm_start=True,
        injection_reward_eur_per_t=args.injection_reward_eur_per_t,
        include_weather_obs=args.weather_obs,
    )
    gym_env = CCSGymEnv(native_env)
    model = MaskablePPO(
        "MlpPolicy", gym_env, seed=args.seed, gamma=0.999,
        n_steps=args.n_steps, batch_size=64, learning_rate=3e-4,
        device=args.device, verbose=1,
    )
    print(f"[{dt.datetime.now():%H:%M:%S}] policy device = {model.policy.device}", flush=True)

    print(f"[{dt.datetime.now():%H:%M:%S}] === BC pretrain (greedy, {args.bc_episodes} eps) ===", flush=True)
    demo_obs, demo_acts, demo_masks, demo_weights = bc_pretrain(
        model, gym_env, greedy_shuttle_policy,
        n_episodes=args.bc_episodes, epochs=args.bc_epochs,
        nonwait_weight=args.nonwait_weight,
    )

    print(f"[{dt.datetime.now():%H:%M:%S}] === eval AFTER BC (before PPO) ===", flush=True)
    report.append("## After BC (before PPO)")
    report += eval_policies(model, args.episode_hours, args.eval_seeds,
                            include_weather_obs=args.weather_obs)

    if args.timesteps > 0:
        callback = None
        if args.kickstart_coef > 0:
            print(f"[{dt.datetime.now():%H:%M:%S}] kickstart anchor on, coef0={args.kickstart_coef}", flush=True)
            callback = make_kickstart_callback(
                demo_obs, demo_acts, demo_masks, demo_weights,
                total_timesteps=args.timesteps, coef0=args.kickstart_coef, verbose=0,
            )
        print(f"[{dt.datetime.now():%H:%M:%S}] === PPO fine-tune ({args.timesteps} steps) ===", flush=True)
        model.learn(total_timesteps=args.timesteps, progress_bar=args.progress_bar, callback=callback)
        print(f"[{dt.datetime.now():%H:%M:%S}] === eval AFTER PPO fine-tune ===", flush=True)
        report.append("\n## After PPO fine-tune")
        report += eval_policies(model, args.episode_hours, args.eval_seeds,
                            include_weather_obs=args.weather_obs)

    model_path = out / f"ppo_{tag}.zip"
    model.save(str(model_path))
    (out / f"results_{tag}.md").write_text(
        f"# BC warm-start + PPO — {tag}\n\n"
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n\n"
        f"bc_episodes={args.bc_episodes}, bc_epochs={args.bc_epochs}, "
        f"timesteps={args.timesteps}, injection_reward={args.injection_reward_eur_per_t}\n\n"
        "```\n" + "\n".join(report) + "\n```\n",
        encoding="utf-8",
    )
    print(f"[{dt.datetime.now():%H:%M:%S}] DONE. model -> {model_path}", flush=True)


if __name__ == "__main__":
    main()
