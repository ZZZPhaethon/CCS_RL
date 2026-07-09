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
from sim.control.baselines import idle_policy, greedy_shuttle_policy, make_cluster_shuttle_policy
from sim.control.imitation import bc_pretrain, make_kickstart_callback
from sim.metrics import run_episode


def _mean_metric(metrics, name):
    values = [getattr(m, name) for m in metrics if getattr(m, name) is not None]
    return float(np.mean(values)) if values else float("nan")


def _format_policy_metrics(name, metrics) -> str:
    storage = _mean_metric(metrics, "storage_rate")
    loss = _mean_metric(metrics, "loss_rate")
    stored_t = _mean_metric(metrics, "stored_t")
    vented_t = _mean_metric(metrics, "vented_t")
    operating_cost = _mean_metric(metrics, "operating_cost")
    vent_penalty = _mean_metric(metrics, "vent_penalty")
    total_cost = _mean_metric(metrics, "total_cost")
    actual_total_cost = operating_cost + vent_penalty
    cost_per_stored_t = _mean_metric(metrics, "cost_per_stored_t")
    stored_t_for_unit = stored_t
    actual_cost_per_stored_t = (
        actual_total_cost / stored_t_for_unit if stored_t_for_unit > 0 else float("nan")
    )
    return (
        f"{name:20s} storage={storage:6.1%}  loss={loss:6.1%}  "
        f"stored={stored_t:9,.0f}t  vented={vented_t:8,.0f}t  "
        f"op_cost={operating_cost:11,.0f}  vent_pen={vent_penalty:10,.0f}  "
        f"total={actual_total_cost:11,.0f}  "
        f"op/t={cost_per_stored_t:6,.1f}  actual/t={actual_cost_per_stored_t:7,.1f}"
    )


def eval_policies(model, episode_hours, seeds, include_weather_obs=False, scenario="northern_lights_phase1"):
    entries = [
        ("idle", lambda env: idle_policy),
        ("greedy_shuttle", lambda env: greedy_shuttle_policy),
        ("cluster_balanced", lambda env: make_cluster_shuttle_policy(env)),
        ("ppo_stochastic", lambda env: make_ppo_policy(model, deterministic=False)),
        ("ppo_deterministic", lambda env: make_ppo_policy(model, deterministic=True)),
    ]
    lines = []
    for name, make_policy in entries:
        metrics = []
        for s in seeds:
            env = make_native_env(episode_hours=episode_hours, warm_start=True,
                                  include_weather_obs=include_weather_obs, scenario=scenario)
            m = run_episode(env, make_policy(env), seed=s)
            metrics.append(m)
        line = _format_policy_metrics(name, metrics)
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
    p.add_argument("--store-reward", type=float, default=None,
                   help="EUR/t credit for stored CO2 in the reward (default: injection-reward)")
    p.add_argument("--vent-weight", type=float, default=1.0,
                   help="multiplier on the vent penalty in the reward")
    p.add_argument("--operating-cost-weight", type=float, default=1.0,
                   help="multiplier on operating cost in the reward (raise to reward efficiency)")
    p.add_argument("--reward-mode", type=str, default="economic", choices=["economic", "vent_first"],
                   help="economic keeps the old stored-credit reward; vent_first prioritizes vent minimization")
    p.add_argument("--vent-first-vent-eur-per-t", type=float, default=10_000.0,
                   help="EUR-equivalent penalty per vented tonne for reward-mode=vent_first")
    p.add_argument("--overflow-risk-eur-per-t", type=float, default=100.0,
                   help="EUR-equivalent dense penalty per tonne of projected emitter overflow risk")
    p.add_argument("--overflow-risk-lookahead-h", type=float, default=24.0,
                   help="lookahead window for projected emitter overflow risk")
    p.add_argument("--carbon-price", type=float, default=None,
                   help="symmetric carbon price: sets both vent tax and stored-CO2 credit")
    p.add_argument("--enforce-full-load-dispatch", action="store_true",
                   help="restore the old hard mask that only lets full vessels sail to the terminal")
    p.add_argument("--scenario", type=str, default="northern_lights_phase1",
                   help="fixed-scenario id to train on (e.g. northern_lights_phase1_milkrun_imbalanced)")
    p.add_argument("--teacher", type=str, default="greedy", choices=["greedy", "cluster"],
                   help="BC demonstrator: greedy_shuttle or load-balanced cluster")
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
    kick_tag = f"_kick{args.kickstart_coef:g}" if args.kickstart_coef > 0 else ""
    if args.carbon_price is not None:
        store_v = args.store_reward if args.store_reward is not None else args.carbon_price
        rew_tag = f"_{args.reward_mode}_cp{args.carbon_price:g}v{args.vent_weight:g}c{args.operating_cost_weight:g}"
    else:
        store_v = args.store_reward if args.store_reward is not None else args.injection_reward_eur_per_t
        rew_tag = f"_{args.reward_mode}_s{store_v:g}v{args.vent_weight:g}c{args.operating_cost_weight:g}"
    if args.reward_mode == "vent_first":
        rew_tag += f"_vf{args.vent_first_vent_eur_per_t:g}r{args.overflow_risk_eur_per_t:g}h{args.overflow_risk_lookahead_h:g}"
    dispatch_tag = "_fullmask" if args.enforce_full_load_dispatch else "_partial"
    scen_tag = args.scenario.replace("northern_lights_", "")
    tag = (f"bc_{scen_tag}_{args.teacher}_{args.episode_hours}h{weather_tag}{kick_tag}{rew_tag}{dispatch_tag}_ts{args.timesteps}")
    report = []

    native_env = make_native_env(
        episode_hours=args.episode_hours, warm_start=True,
        injection_reward_eur_per_t=args.injection_reward_eur_per_t,
        include_weather_obs=args.weather_obs,
        store_reward_eur_per_t=args.store_reward,
        vent_penalty_weight=args.vent_weight,
        operating_cost_weight=args.operating_cost_weight,
        reward_mode=args.reward_mode,
        vent_first_vent_eur_per_t=args.vent_first_vent_eur_per_t,
        overflow_risk_eur_per_t=args.overflow_risk_eur_per_t,
        overflow_risk_lookahead_h=args.overflow_risk_lookahead_h,
        carbon_price_eur_per_t=args.carbon_price,
        enforce_full_load_dispatch=args.enforce_full_load_dispatch,
        scenario=args.scenario,
    )
    gym_env = CCSGymEnv(native_env)
    model = MaskablePPO(
        "MlpPolicy", gym_env, seed=args.seed, gamma=0.999,
        n_steps=args.n_steps, batch_size=64, learning_rate=3e-4,
        device=args.device, verbose=1,
    )
    print(f"[{dt.datetime.now():%H:%M:%S}] policy device = {model.policy.device}", flush=True)

    if args.teacher == "cluster":
        teacher = make_cluster_shuttle_policy(native_env)
    else:
        teacher = greedy_shuttle_policy
    print(f"[{dt.datetime.now():%H:%M:%S}] === BC pretrain (teacher={args.teacher}, {args.bc_episodes} eps) ===", flush=True)
    demo_obs, demo_acts, demo_masks, demo_weights = bc_pretrain(
        model, gym_env, teacher,
        n_episodes=args.bc_episodes, epochs=args.bc_epochs,
        nonwait_weight=args.nonwait_weight,
    )

    print(f"[{dt.datetime.now():%H:%M:%S}] === eval AFTER BC (before PPO) ===", flush=True)
    report.append("## After BC (before PPO)")
    report += eval_policies(model, args.episode_hours, args.eval_seeds,
                            include_weather_obs=args.weather_obs, scenario=args.scenario)

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
                            include_weather_obs=args.weather_obs, scenario=args.scenario)

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
