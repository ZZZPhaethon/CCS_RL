"""Goal-conditioned RL across randomized layouts, for zero-shot generalization.

Each episode samples a random milk-run layout (different capture-rate imbalance),
sets the per-vessel goal (a balanced assignment) in the observation, and trains
the policy to EXECUTE that goal. At test time a held-out layout is used: the goal
is set from the same heuristic (or an LLM) and the policy runs zero-shot - no
retraining - while greedy/cluster are the non-learned references.

Usage (ccs-rlllm-gpu env, from repo root):
    set PYTHONPATH=src
    python scripts/train_goal_conditioned.py --bc-episodes 40 --timesteps 200000
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import statistics
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("requires gymnasium") from exc

from sim.network_scenarios import _build_network_from_scenario_data, _load_fixed_scenario_data
from sim.environment.env import CCSEnv, CCSEnvConfig
from sim.environment.factories import _scenario_locations
from sim.environment.gym_adapter import flat_action_mask, native_action_from_flat, make_ppo_policy
from sim.scenario_generation import ScenarioGenerator, ScenarioConfig
from sim.economics import CostModel, EconomicParameters
from sim.control.baselines import (
    greedy_shuttle_policy, make_cluster_shuttle_policy, balanced_capture_assignment,
)
from sim.control.imitation import collect_demonstrations, decision_step_weights, behavior_clone, make_kickstart_callback
from sim.metrics import run_episode

BASE = "northern_lights_phase1_milkrun_imbalanced"
# capture-rate multipliers on (brevik, celsio, yara_sluiskil)
TRAIN_LAYOUTS = [(2.5, 2.5, 0.4), (0.4, 2.5, 2.5), (2.5, 0.4, 2.5), (1.6, 1.0, 1.6)]
TEST_LAYOUT = (0.5, 2.5, 1.3)  # held out - never trained on
EP_H = 720
CARBON = 80.0
# Disturbance strength (set from CLI). Strong, long capture outages make a rigid
# static assignment idle a vessel whose emitter is offline, so a policy that can
# reallocate should win. Set outage_rate=0 to reproduce the near-static regime.
DISTURB = {"outage_rate": 0.5, "outage_hours": 12.0}


def make_layout_data(factors):
    data = copy.deepcopy(_load_fixed_scenario_data(BASE))
    # BASE already scaled brevik/celsio x2.5, yara x0.4 relative to phase1; undo then apply
    undo = {"brevik": 2.5, "celsio": 2.5, "yara_sluiskil": 0.4}
    order = ["brevik", "celsio", "yara_sluiskil"]
    fmap = dict(zip(order, factors))
    for e in data["emitters"]:
        eid = e["entity_id"]
        if eid in fmap:
            f = fmap[eid] / undo[eid]
            for k in ("annual_target_export_tpy", "nominal_capture_tph", "max_production_tph"):
                if k in e:
                    e[k] = e[k] * f
    return data


def build_env(factors, include_goal=True):
    data = make_layout_data(factors)
    network, _ = _build_network_from_scenario_data(data)
    loc = _scenario_locations(data)
    scen_cfg = ScenarioConfig(
        episode_hours=EP_H,
        capture_outage_rate_per_week=DISTURB["outage_rate"],
        capture_outage_mean_hours=DISTURB["outage_hours"],
    )
    return CCSEnv(
        network, loc,
        scenario_generator=ScenarioGenerator(config=scen_cfg),
        cost_model=CostModel(EconomicParameters(carbon_price_eur_per_t=CARBON)),
        config=CCSEnvConfig(episode_hours=EP_H, include_goal_obs=include_goal,
                            store_reward_eur_per_t=CARBON),
    )


class MultiLayoutGymEnv(gym.Env):
    """Samples a random layout per episode and sets its balanced goal."""
    metadata = {"render_modes": []}

    def __init__(self, envs):
        super().__init__()
        self.envs = envs
        self.env = envs[0]
        self.action_space = spaces.MultiDiscrete(self.env.vessel_action_dims + self.env.well_rate_action_dims)
        self.observation_space = spaces.Box(-10.0, 10.0, (self.env.observation_size,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env = self.envs[int(self.np_random.integers(0, len(self.envs)))]
        self.env.set_goal_assignment(balanced_capture_assignment(self.env))
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        obs = self.env.reset(seed=episode_seed)
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(native_action_from_flat(self.env, action))
        return np.asarray(obs, dtype=np.float32), float(r), term, trunc, info

    def action_masks(self):
        return flat_action_mask(self.env.vessel_action_mask(), self.env.well_rate_action_mask())

    def _to_array(self, obs):
        return np.asarray(obs, dtype=np.float32)


def cluster_teacher(env):
    """Cluster shuttle using the env's current (goal) assignment."""
    return make_cluster_shuttle_policy(env, env.goal_assignment or None)(env)


def eval_on(factors, model, seeds):
    """Evaluate greedy / cluster / goal-RL on a layout (goal set from heuristic)."""
    def make():
        e = build_env(factors, include_goal=True)
        e.set_goal_assignment(balanced_capture_assignment(e))
        return e
    rows = {}
    for name, mk in [("greedy", lambda e: greedy_shuttle_policy),
                     ("cluster", lambda e: make_cluster_shuttle_policy(e, e.goal_assignment or None)),
                     ("goal_rl", lambda e: make_ppo_policy(model, deterministic=True))]:
        srs, lo = [], []
        for s in seeds:
            e = make()
            e.set_goal_assignment(balanced_capture_assignment(e))
            m = run_episode(e, mk(e), seed=s)
            srs.append(m.storage_rate); lo.append(m.loss_rate)
        rows[name] = (statistics.mean(srs), statistics.mean(lo))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bc-episodes", type=int, default=40)
    p.add_argument("--bc-epochs", type=int, default=20)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--kickstart-coef", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 102, 103])
    p.add_argument("--outage-rate", type=float, default=0.5,
                   help="capture outages per week per emitter (raise for a dynamic stress test)")
    p.add_argument("--outage-hours", type=float, default=12.0, help="mean outage duration")
    args = p.parse_args()
    DISTURB["outage_rate"] = args.outage_rate
    DISTURB["outage_hours"] = args.outage_hours
    print(f"disturbance: outage_rate={args.outage_rate}/wk, mean={args.outage_hours}h", flush=True)

    from sb3_contrib import MaskablePPO

    train_envs = [build_env(f, include_goal=True) for f in TRAIN_LAYOUTS]
    gym_env = MultiLayoutGymEnv(train_envs)
    model = MaskablePPO("MlpPolicy", gym_env, seed=args.seed, gamma=0.999,
                        n_steps=args.n_steps, batch_size=64, learning_rate=3e-4,
                        device=args.device, verbose=1)
    print(f"[{dt.datetime.now():%H:%M:%S}] device={model.policy.device}, obs={gym_env.observation_space.shape}", flush=True)

    print(f"[{dt.datetime.now():%H:%M:%S}] === BC across layouts (cluster teacher) ===", flush=True)
    obs, acts, masks = collect_demonstrations(gym_env, cluster_teacher, args.bc_episodes)
    weights = decision_step_weights(acts, len(train_envs[0].vessel_ids), nonwait_weight=10.0)
    print(f"[bc] {len(obs)} pairs; {int((weights>1).sum())} dispatch steps", flush=True)
    behavior_clone(model, obs, acts, masks=masks, weights=weights, epochs=args.bc_epochs)

    if args.timesteps > 0:
        cb = make_kickstart_callback(obs, acts, masks, weights, total_timesteps=args.timesteps,
                                     coef0=args.kickstart_coef) if args.kickstart_coef > 0 else None
        print(f"[{dt.datetime.now():%H:%M:%S}] === PPO fine-tune ({args.timesteps}) ===", flush=True)
        model.learn(total_timesteps=args.timesteps, callback=cb)

    out = Path("output/rl_ppo"); out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "goal_conditioned_milkrun.zip"))

    report = ["# Goal-conditioned RL - zero-shot generalization", "",
              f"train layouts (brevik,celsio,yara mult): {TRAIN_LAYOUTS}",
              f"HELD-OUT test layout: {TEST_LAYOUT}", "",
              "| layout | greedy | cluster | goal_rl |", "|---|---|---|---|"]
    print(f"\n[{dt.datetime.now():%H:%M:%S}] === EVAL ===", flush=True)
    for label, f in [("test(HELD-OUT)", TEST_LAYOUT)] + [(f"train{i}", f) for i, f in enumerate(TRAIN_LAYOUTS)]:
        r = eval_on(f, model, args.eval_seeds)
        line = (f"| {label} {f} | {r['greedy'][0]:.1%} | {r['cluster'][0]:.1%} | {r['goal_rl'][0]:.1%} |")
        print(line.replace("|", " "), flush=True)
        report.append(line)
    (out / "goal_conditioned_generalization.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[{dt.datetime.now():%H:%M:%S}] DONE -> {out/'goal_conditioned_generalization.md'}", flush=True)


if __name__ == "__main__":
    main()
