"""Residual (gated-override) RL over a heuristic base controller.

Instead of learning a full policy from scratch, the base controller (the
load-balanced cluster shuttle) proposes an action every step, and the RL policy
learns only a *correction*: for each vessel it either FOLLOWS the base action or
OVERRIDES it with a different destination. Because FOLLOW is always available and
reproduces the base, the policy can always fall back to base performance and only
overrides where it helps - giving stability (>= base, no drift) and letting it
learn corrective reallocation under disturbances where the rigid base fails.

Discrete-action adaptation of residual RL: a_final = FOLLOW ? a_base : a_override.

Results are written to output/rl_ppo/residual_*.md/.csv.

Usage (ccs-rlllm-gpu env, from repo root):
    set PYTHONPATH=src
    python scripts/train_residual_rl.py --timesteps 150000 --load-shift
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import statistics
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("requires gymnasium") from exc

from sim.network_scenarios import build_fixed_scenario_demo, _load_fixed_scenario_data
from sim.environment.env import CCSEnv, CCSEnvConfig
from sim.environment.factories import _scenario_locations
from sim.environment.gym_adapter import flat_action_mask
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from sim.scenario_generation.load_shift import LoadShiftScenarioGenerator, LoadShiftConfig
from sim.economics import CostModel, EconomicParameters
from sim.control.baselines import (
    greedy_shuttle_policy, make_cluster_shuttle_policy, balanced_capture_assignment,
)
from sim.control.imitation import behavior_clone, make_kickstart_callback
from sim.metrics import run_episode

CARBON = 80.0


def build_env(scenario, episode_hours, load_shift, outage_rate):
    net, _ = build_fixed_scenario_demo(scenario)
    loc = _scenario_locations(_load_fixed_scenario_data(scenario))
    scen_cfg = ScenarioConfig(episode_hours=episode_hours, capture_outage_rate_per_week=outage_rate)
    if load_shift:
        gen = LoadShiftScenarioGenerator(config=scen_cfg,
              load_shift=LoadShiftConfig(phase_hours=120, hot_level=1.0, cold_level=0.15, hot_count=2))
    else:
        gen = ScenarioGenerator(config=scen_cfg)
    return CCSEnv(net, loc, scenario_generator=gen,
                  cost_model=CostModel(EconomicParameters(carbon_price_eur_per_t=CARBON)),
                  config=CCSEnvConfig(episode_hours=episode_hours, store_reward_eur_per_t=CARBON,
                                      enforce_full_load_dispatch=False))


class ResidualGymEnv(gym.Env):
    """Gated-override residual wrapper. Vessel action = FOLLOW base or override."""
    metadata = {"render_modes": []}

    def __init__(self, env: CCSEnv, base_factory=None):
        super().__init__()
        self.env = env
        # base_factory(env) -> policy(env); defaults to the load-balanced cluster
        self.base = (base_factory or (lambda e: make_cluster_shuttle_policy(e)))(env)
        self.nv = len(env.vessel_ids)
        self.nw = len(env.well_ids)
        self.vac = env.vessel_action_count
        self.FOLLOW = self.vac  # extra action index = "follow the base"
        vessel_dims = [self.vac + 1] * self.nv
        self.action_space = spaces.MultiDiscrete(vessel_dims + list(env.well_rate_action_dims))
        obs_size = env.observation_size + self.nv * self.vac  # + base vessel action one-hot
        self.observation_space = spaces.Box(-10.0, 10.0, (obs_size,), dtype=np.float32)
        self._base_v = None
        self._base_w = None

    def _refresh_base(self):
        a = self.base(self.env)
        self._base_v = a["vessels"]
        self._base_w = a["wells"]

    def _obs(self):
        obs = list(self.env._observation())
        for bv in self._base_v:
            oh = [0.0] * self.vac
            if 0 <= bv < self.vac:
                oh[bv] = 1.0
            obs += oh
        return np.asarray(obs, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.env.reset(seed=episode_seed)
        self._refresh_base()
        return self._obs(), {}

    def _native(self, flat):
        flat = np.asarray(flat).reshape(-1)
        vessels = []
        for i in range(self.nv):
            c = int(flat[i])
            vessels.append(int(self._base_v[i]) if c == self.FOLLOW else c)
        wells = [int(x) for x in flat[self.nv:self.nv + self.nw]]
        return {"vessels": vessels, "wells": wells}

    def step(self, action):
        _o, r, term, trunc, info = self.env.step(self._native(action))
        self._refresh_base()
        return self._obs(), float(r), term, trunc, info

    def action_masks(self):
        vm = self.env.vessel_action_mask()
        wm = self.env.well_rate_action_mask()
        flat = []
        for i in range(self.nv):
            flat += list(vm[i]) + [True]  # override options + FOLLOW (always legal)
        for i in range(self.nw):
            flat += list(wm[i])
        return np.asarray(flat, dtype=bool)

    def _to_array(self, obs):
        return np.asarray(obs, dtype=np.float32)

    def follow_action(self):
        return np.array([self.FOLLOW] * self.nv + [int(w) for w in self._base_w], dtype=np.int64)


def collect_follow_demos(wrapper, n_eps):
    obs_r, act_r, mask_r = [], [], []
    for i in range(n_eps):
        o, _ = wrapper.reset(seed=i)
        done = False
        while not done:
            a = wrapper.follow_action()
            obs_r.append(o); act_r.append(a); mask_r.append(wrapper.action_masks())
            o, _r, term, trunc, _i = wrapper.step(a)
            done = term or trunc
    return (np.asarray(obs_r, np.float32), np.asarray(act_r, np.int64), np.asarray(mask_r, bool))


def _metrics(env):
    stored = env.cumulative_stored_t
    return {
        "storage_rate": env.storage_rate(),
        "loss_rate": env.loss_rate(),
        "total_cost": env.ledger.total_cost,
        "cost_per_t": (env.ledger.total_cost / stored) if stored > 1e-6 else float("nan"),
    }


def eval_residual(env_factory, model, seeds, base_factory=None):
    srs, lo, cpt = [], [], []
    for s in seeds:
        wrapper = ResidualGymEnv(env_factory(), base_factory)
        o, _ = wrapper.reset(seed=s)
        done = False
        while not done:
            a, _ = model.predict(o, deterministic=True, action_masks=wrapper.action_masks())
            o, _r, term, trunc, _i = wrapper.step(a)
            done = term or trunc
        m = _metrics(wrapper.env)
        srs.append(m["storage_rate"]); lo.append(m["loss_rate"]); cpt.append(m["cost_per_t"])
    return statistics.mean(srs), statistics.mean(lo), statistics.mean(cpt)


def eval_native(env_factory, policy_factory, seeds):
    srs, lo, cpt = [], [], []
    for s in seeds:
        env = env_factory()
        run_episode(env, policy_factory(env), seed=s)
        m = _metrics(env)
        srs.append(m["storage_rate"]); lo.append(m["loss_rate"]); cpt.append(m["cost_per_t"])
    return statistics.mean(srs), statistics.mean(lo), statistics.mean(cpt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="northern_lights_phase1_milkrun")
    p.add_argument("--episode-hours", type=int, default=720)
    p.add_argument("--load-shift", action="store_true", help="rotating capture hot spot")
    p.add_argument("--outage-rate", type=float, default=0.5)
    p.add_argument("--bc-episodes", type=int, default=20)
    p.add_argument("--bc-epochs", type=int, default=15)
    p.add_argument("--timesteps", type=int, default=150000)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--kickstart-coef", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--base", type=str, default="cluster", choices=["cluster", "greedy"],
                   help="base controller the residual policy corrects")
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 102, 103])
    args = p.parse_args()

    from sb3_contrib import MaskablePPO

    def env_factory():
        return build_env(args.scenario, args.episode_hours, args.load_shift, args.outage_rate)

    if args.base == "greedy":
        base_factory = lambda e: greedy_shuttle_policy
    else:
        base_factory = lambda e: make_cluster_shuttle_policy(e)

    gym_env = ResidualGymEnv(env_factory(), base_factory)
    model = MaskablePPO("MlpPolicy", gym_env, seed=args.seed, gamma=0.999,
                        n_steps=args.n_steps, batch_size=64, learning_rate=3e-4,
                        device=args.device, verbose=1)
    print(f"[{dt.datetime.now():%H:%M:%S}] device={model.policy.device}, "
          f"obs={gym_env.observation_space.shape}, act={gym_env.action_space.nvec}", flush=True)

    print(f"[{dt.datetime.now():%H:%M:%S}] === BC: clone 'always FOLLOW base' ===", flush=True)
    obs, acts, masks = collect_follow_demos(gym_env, args.bc_episodes)
    behavior_clone(model, obs, acts, masks=masks, epochs=args.bc_epochs)

    if args.timesteps > 0:
        cb = make_kickstart_callback(obs, acts, masks, None, total_timesteps=args.timesteps,
                                     coef0=args.kickstart_coef) if args.kickstart_coef > 0 else None
        print(f"[{dt.datetime.now():%H:%M:%S}] === PPO fine-tune ({args.timesteps}) ===", flush=True)
        model.learn(total_timesteps=args.timesteps, callback=cb)

    out = Path("output/rl_ppo"); out.mkdir(parents=True, exist_ok=True)
    tag = f"residual_{args.base}base_{args.scenario.replace('northern_lights_','')}"
    tag += ("_loadshift" if args.load_shift else "") + f"_ts{args.timesteps}"
    model.save(str(out / f"{tag}.zip"))

    print(f"[{dt.datetime.now():%H:%M:%S}] === EVAL ===", flush=True)
    rows = []
    rows.append(("greedy", *eval_native(env_factory, lambda e: greedy_shuttle_policy, args.eval_seeds)))
    rows.append(("cluster_base", *eval_native(env_factory, lambda e: make_cluster_shuttle_policy(e), args.eval_seeds)))
    rows.append((f"residual_rl({args.base})", *eval_residual(env_factory, model, args.eval_seeds, base_factory)))

    print(f"{'policy':16s} {'storage':>8s} {'vent':>7s} {'cost/t':>8s}")
    md = [f"# Residual RL over cluster base - {tag}", "",
          f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
          f"scenario={args.scenario}, load_shift={args.load_shift}, outage_rate={args.outage_rate}, "
          f"episode_hours={args.episode_hours}, timesteps={args.timesteps}", "",
          "| policy | storage | vent | cost/t |", "|---|---|---|---|"]
    for name, sr, lo, cpt in rows:
        print(f"{name:16s} {sr:7.1%} {lo:6.1%} {cpt:8.1f}", flush=True)
        md.append(f"| {name} | {sr:.1%} | {lo:.1%} | {cpt:.1f} |")
    (out / f"{tag}.md").write_text("\n".join(md), encoding="utf-8")
    with (out / f"{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["policy", "storage_rate", "loss_rate", "cost_per_t"])
        for r in rows:
            w.writerow(r)
    print(f"[{dt.datetime.now():%H:%M:%S}] DONE -> {out/(tag+'.md')}", flush=True)


if __name__ == "__main__":
    main()
