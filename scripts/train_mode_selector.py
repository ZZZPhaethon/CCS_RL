"""Distil the native MPC by learning its MODE SELECTION, not its raw actions.

The RollingNativeMpcController is not an optimisation MPC: every 24 h it rolls out
a small library of candidate heuristics (greedy, forecast-urgency, and each
dedicated vessel->emitter assignment) over a 168 h horizon and executes whichever
minimises (vent, end-unstored, cost). So its "policy" is really a low-dimensional,
forecast-conditioned CHOICE among ~N interpretable modes.

Cloning its raw per-step actions suffers covariate shift over 720 h. Instead we
distil the CHOICE: collect (state+forecast-summary -> chosen mode) at each replan
from the MPC, train a classifier, and deploy a policy that re-selects a mode every
24 h and runs that mode's deterministic heuristic (self-consistent, no drift).

Results -> output/rl_forecast/mode_selector_*.md/.csv

Usage (ccs-rlllm-gpu env, from repo root):
    set PYTHONPATH=src
    python scripts/train_mode_selector.py --train-seeds 1-15 --eval-seeds 101-105
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import statistics
from pathlib import Path

import numpy as np

from sim.network_scenarios import build_fixed_scenario_demo, fixed_scenario_locations
from sim.environment import CCSEnv, CCSEnvConfig
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from sim.economics import CostModel, EconomicParameters
from sim.environment.forecast import (
    current_state_observation, future_forecast_observation, forecast_channel_names,
)
from sim.control.native_mpc import (
    RollingNativeMpcController, _forecast_urgency_policy, _make_dedicated_policy,
    _dedicated_assignments,
)
from sim.control.baselines import greedy_shuttle_policy
from sim.metrics import run_episode

SCENARIO_ID = "northern_lights_phase1_3vessels"
REPLAN_EVERY = 24
HORIZON_H = 168


def make_env(episode_hours=720):
    network, _ = build_fixed_scenario_demo(SCENARIO_ID)
    cfg = ScenarioConfig(
        episode_hours=episode_hours,
        weather_process="block",
        weather_update_hours=24,
        weather_update_speed_factor_range=(0.75, 1.00),
        capture_noise_std=0.10,
        capture_outage_rate_per_week=0.0,
        capture_high_output_rate_per_week=0.5,
        capture_high_output_mean_hours=48.0,
        capture_high_output_multiplier_range=(1.25, 1.75),
        well_maintenance_rate_per_week=0.0,
        randomize_initial_inventory=True,
    )
    return CCSEnv(
        network, fixed_scenario_locations(SCENARIO_ID),
        scenario_generator=ScenarioGenerator(config=cfg),
        cost_model=CostModel(EconomicParameters()),
        config=CCSEnvConfig(episode_hours=episode_hours, reward_mode="vent_first"),
    )


def mode_list(env):
    names = ["greedy", "forecast_urgency"]
    for a in _dedicated_assignments(env):
        names.append("dedicated:" + ",".join(a[v] for v in env.vessel_ids))
    return names


def build_mode_policy(env, name):
    if name == "greedy":
        return greedy_shuttle_policy
    if name == "forecast_urgency":
        return _forecast_urgency_policy
    if name.startswith("dedicated:"):
        emitters = name[len("dedicated:"):].split(",")
        return _make_dedicated_policy(dict(zip(env.vessel_ids, emitters)))
    raise ValueError(f"unknown mode: {name}")


def features(env):
    state = current_state_observation(env)
    dt_h = env.network.time_step_hours
    now = int(round(env.simulator.state.time_h / dt_h))
    h = max(1, min(HORIZON_H, env.n_steps - now - 1))
    try:
        fc = np.asarray(future_forecast_observation(env, h), dtype=np.float32)
    except Exception:
        fc = np.zeros((1, len(forecast_channel_names(env))), dtype=np.float32)
    near = fc[: min(24, len(fc))]
    summ = np.concatenate([fc.mean(0), fc.min(0), fc.max(0), near.mean(0), near.min(0)])
    return np.concatenate([np.asarray(state, dtype=np.float32), summ]).astype(np.float32)


def collect_labels(seeds):
    X, y = [], []
    modes = midx = None
    for s in seeds:
        env = make_env(); env.reset(seed=s)
        mpc = RollingNativeMpcController(env, replan_every=REPLAN_EVERY, planning_horizon_h=HORIZON_H)
        if modes is None:
            modes = mode_list(env); midx = {m: i for i, m in enumerate(modes)}
        prev = mpc._plan_origin_h
        done = False
        while not done:
            feats = features(env)
            action = mpc(env)
            if mpc._plan_origin_h != prev:  # a replan happened at this state
                if mpc.last_candidate_name in midx:
                    X.append(feats); y.append(midx[mpc.last_candidate_name])
                prev = mpc._plan_origin_h
            _o, _r, term, trunc, _i = env.step(action)
            done = term or trunc
        print(f"  seed {s}: {len(y)} labels so far", flush=True)
    return np.asarray(X), np.asarray(y), modes


class ModeSelectorPolicy:
    def __init__(self, env, clf, modes):
        self.clf = clf
        self.modes = modes
        self._cur = None
        self._next = -1e9

    def __call__(self, env):
        t = env.simulator.state.time_h
        if self._cur is None or t >= self._next:
            idx = int(self.clf.predict(features(env).reshape(1, -1))[0])
            self._cur = build_mode_policy(env, self.modes[idx])
            self._next = t + REPLAN_EVERY
        return self._cur(env)


def _m(metrics, a):
    vs = [getattr(x, a) for x in metrics if getattr(x, a) is not None]
    return statistics.mean(vs) if vs else float("nan")


def score(name, policy_factory, seeds):
    ms = []
    for s in seeds:
        env = make_env()
        ms.append(run_episode(env, policy_factory(env), seed=s))
    return {
        "policy": name,
        "vent_t": _m(ms, "vented_t"),
        "storage": _m(ms, "storage_rate"),
        "cost_per_t": _m(ms, "total_cost_per_stored_t"),
    }


def _parse_seeds(spec):
    if "-" in spec and "," not in spec:
        a, b = spec.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-seeds", default="1-15")
    p.add_argument("--eval-seeds", default="101-105")
    args = p.parse_args()
    train_seeds = _parse_seeds(args.train_seeds)
    eval_seeds = _parse_seeds(args.eval_seeds)

    from sklearn.ensemble import RandomForestClassifier

    print(f"[{dt.datetime.now():%H:%M:%S}] collecting MPC mode labels ({len(train_seeds)} seeds)...", flush=True)
    X, y, modes = collect_labels(train_seeds)
    print(f"[{dt.datetime.now():%H:%M:%S}] {len(y)} labels, {len(modes)} modes, feat_dim={X.shape[1]}", flush=True)
    counts = {modes[i]: int((y == i).sum()) for i in range(len(modes))}
    print("  mode counts:", counts, flush=True)

    clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=0, n_jobs=-1)
    clf.fit(X, y)
    print(f"[{dt.datetime.now():%H:%M:%S}] train accuracy = {clf.score(X, y):.3f}", flush=True)

    print(f"[{dt.datetime.now():%H:%M:%S}] evaluating on {len(eval_seeds)} held-out seeds...", flush=True)
    rows = [
        score("greedy", lambda e: greedy_shuttle_policy, eval_seeds),
        score("mpc", lambda e: RollingNativeMpcController(e, REPLAN_EVERY, HORIZON_H), eval_seeds),
        score("mode_selector", lambda e: ModeSelectorPolicy(e, clf, modes), eval_seeds),
    ]

    out = Path("output/rl_forecast"); out.mkdir(parents=True, exist_ok=True)
    tag = f"mode_selector_3vessels_{len(train_seeds)}train_{len(eval_seeds)}eval"
    print(f"\n{'policy':16s} {'vent_t':>9s} {'storage':>8s} {'cost/t':>8s}")
    md = [f"# Mode-selector distillation of native MPC — {tag}", "",
          f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
          f"train seeds={train_seeds}, eval seeds={eval_seeds}, modes={len(modes)}, "
          f"train_acc={clf.score(X, y):.3f}", "",
          "| policy | vent t | storage | cost/t |", "|---|---:|---:|---:|"]
    for r in rows:
        print(f"{r['policy']:16s} {r['vent_t']:9,.0f} {r['storage']:7.1%} {r['cost_per_t']:8,.1f}", flush=True)
        md.append(f"| {r['policy']} | {r['vent_t']:,.0f} | {r['storage']:.1%} | {r['cost_per_t']:,.1f} |")
    (out / f"{tag}.md").write_text("\n".join(md), encoding="utf-8")
    with (out / f"{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[{dt.datetime.now():%H:%M:%S}] DONE -> {out/(tag+'.md')}", flush=True)


if __name__ == "__main__":
    main()
