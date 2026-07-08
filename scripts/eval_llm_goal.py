"""Bring the LLM into the goal-conditioned RL loop AT EVAL TIME.

For each layout, get the high-level goal from two sources - the deterministic
balanced-capture heuristic and the LLM (Ollama) - and run the trained goal-
conditioned policy (and the cluster executor) under each goal. Shows (a) whether
the LLM's assignment differs from the heuristic, and (b) whether the goal source
changes the outcome. This is the minimal test of LLM-in-the-loop value.

Usage (ccs-rlllm-gpu env, Ollama running, from repo root):
    set PYTHONPATH=src
    python scripts/eval_llm_goal.py --model-zip output/rl_ppo/goal_conditioned_milkrun.zip
"""
from __future__ import annotations

import argparse
import statistics

# scripts/ is on sys.path[0] when run directly, so these import cleanly
from train_goal_conditioned import build_env, TRAIN_LAYOUTS, TEST_LAYOUT, DISTURB
from llm_planner import llm_assignment

from sim.environment.gym_adapter import make_ppo_policy
from sim.control.baselines import balanced_capture_assignment, make_cluster_shuttle_policy
from sim.metrics import run_episode


def score(factors, goal, policy_factory, seeds):
    srs = []
    for s in seeds:
        env = build_env(factors, include_goal=True)
        env.set_goal_assignment(goal)
        m = run_episode(env, policy_factory(env), seed=s)
        srs.append(m.storage_rate)
    return statistics.mean(srs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-zip", default="output/rl_ppo/goal_conditioned_milkrun.zip")
    p.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    p.add_argument("--outage-rate", type=float, default=0.5)
    p.add_argument("--outage-hours", type=float, default=12.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    args = p.parse_args()
    DISTURB["outage_rate"] = args.outage_rate
    DISTURB["outage_hours"] = args.outage_hours

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model_zip)

    layouts = [("test", TEST_LAYOUT)] + [(f"train{i}", f) for i, f in enumerate(TRAIN_LAYOUTS)]
    print(f"model={args.model_zip}  llm={args.llm_model}  outage={args.outage_rate}/wk\n")
    header = f"{'layout':16s} {'goals agree?':12s} {'gRL(heur)':10s} {'gRL(llm)':10s} {'clu(heur)':10s} {'clu(llm)':10s}"
    print(header); print("-" * len(header))
    for label, f in layouts:
        env = build_env(f, include_goal=True); env.reset(seed=args.seeds[0])
        heur = balanced_capture_assignment(env)
        llm = llm_assignment(env, args.llm_model, verbose=False)
        agree = all(heur.get(e) == llm.get(e) for e in env.emitter_ids)
        grl_h = score(f, heur, lambda e: make_ppo_policy(model, deterministic=True), args.seeds)
        grl_l = score(f, llm, lambda e: make_ppo_policy(model, deterministic=True), args.seeds)
        clu_h = score(f, heur, lambda e: make_cluster_shuttle_policy(e, heur), args.seeds)
        clu_l = score(f, llm, lambda e: make_cluster_shuttle_policy(e, llm), args.seeds)
        print(f"{label:16s} {'YES' if agree else 'NO -> diff':12s} "
              f"{grl_h:9.1%} {grl_l:9.1%} {clu_h:9.1%} {clu_l:9.1%}", flush=True)
        if not agree:
            print(f"    heuristic: {heur}")
            print(f"    llm      : {llm}")


if __name__ == "__main__":
    main()
