"""Hierarchical step 1: LLM high-level planner + deterministic cluster executor.

Once per episode the LLM reads the network layout (emitter positions/capture
rates, vessel home ports) and outputs a vessel->emitter assignment (the strategic
decision). A deterministic cluster policy then executes per-step within that
assignment. This isolates the LLM's strategic-reasoning value; the low level is
the proven cluster rule (~77% on milk-run). Compares:

    greedy  vs  cluster(geographic)  vs  llm_planner(LLM assignment + cluster)

Usage (ccs-rlllm-gpu env, Ollama running):
    set PYTHONPATH=src
    python scripts/llm_planner.py --model qwen2.5:7b-instruct --seeds 101 102
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import urllib.request

from sim.network_scenarios import build_fixed_scenario_demo, _load_fixed_scenario_data
from sim.environment.env import CCSEnv, CCSEnvConfig, VESSEL_WAIT, VESSEL_GO_TERMINAL
from sim.environment.factories import _scenario_locations
from sim.scenario_generation import ScenarioGenerator, ScenarioConfig
from sim.control.baselines import greedy_shuttle_policy, idle_policy
from sim.metrics import run_episode

_EPS = 1e-9
OLLAMA_URL = "http://localhost:11434/api/generate"


def ollama_generate(model: str, prompt: str, timeout: float = 60.0) -> str:
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "num_predict": 400},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"]


def _supply(env, emitter_id):
    e = env.network.entities[emitter_id]
    st = env.simulator.state
    avail = st.emitter_availability.get(emitter_id, e.availability)
    return st.entity_inventory_t.get(emitter_id, 0.0) + e.nominal_capture_tph * max(0.0, avail)


def geographic_assignment(env):
    homes = {vid: env._routes[vid]["origin"] for vid in env.vessel_ids}

    def d2(a, b):
        (y1, x1), (y2, x2) = env.locations[a], env.locations[b]
        return (y1 - y2) ** 2 + (x1 - x2) ** 2

    return {e: min(env.vessel_ids, key=lambda v: d2(e, homes[v])) for e in env.emitter_ids}


def llm_assignment(env, model: str, verbose: bool = True):
    """Ask the LLM to partition emitters among vessels. Returns {emitter: vessel}."""
    lines = [
        "You are the strategic planner for a ship-based CO2 storage network.",
        "Assign every emitter to exactly one vessel that will collect its CO2. Group",
        "geographically close emitters under one vessel; give a far/isolated emitter its",
        "own dedicated vessel; balance capture load so no buffer is left to overflow.",
        "",
        "Vessels (home port and coordinates lat,lon):",
    ]
    for vid in env.vessel_ids:
        home = env._routes[vid]["origin"]
        lat, lon = env.locations[home]
        lines.append(f"  - {vid}: home={home} ({lat:.2f},{lon:.2f})")
    lines.append("Emitters (coordinates lat,lon and capture rate):")
    for e in env.emitter_ids:
        lat, lon = env.locations[e]
        rate = env.network.entities[e].nominal_capture_tph
        lines.append(f"  - {e}: ({lat:.2f},{lon:.2f}), {rate:.0f} t/h")
    lines += [
        "",
        'Respond with ONLY a JSON object mapping each vessel id to its list of emitter ids, '
        'covering every emitter exactly once. Example: {"vessel_x": ["e1","e2"], "vessel_y": ["e3"]}',
    ]
    raw = ollama_generate(model, "\n".join(lines))
    if verbose:
        print("  [llm raw]:", raw.strip().replace("\n", " ")[:300], flush=True)
    assign = {}
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0))
        for vid, emitters in data.items():
            if vid not in env.vessel_ids:
                continue
            for e in emitters:
                if e in env.emitter_ids:
                    assign[e] = vid
    except Exception as exc:
        print(f"  [llm parse failed: {exc}] -> falling back to geographic", flush=True)
    # any emitter the LLM missed -> geographic fallback
    geo = geographic_assignment(env)
    for e in env.emitter_ids:
        assign.setdefault(e, geo[e])
    return assign


def make_cluster_policy(env, assign):
    def policy(env):
        st = env.simulator.state
        acts = []
        for i, vid in enumerate(env.vessel_ids):
            mask = env.vessel_action_mask()[i]
            cargo = st.entity_inventory_t.get(vid, 0.0)
            vessel = env.network.entities[vid]
            berth = st.vessel_berths.get(vid)
            mine = [e for e in env.emitter_ids if assign.get(e) == vid]
            if berth in env.terminal_ids and cargo > _EPS:
                acts.append(VESSEL_WAIT); continue
            if mask[VESSEL_GO_TERMINAL] and cargo >= vessel.capacity_t - _EPS:
                acts.append(VESSEL_GO_TERMINAL); continue
            if berth in mine and cargo < vessel.capacity_t - _EPS and _supply(env, str(berth)) > _EPS:
                acts.append(VESSEL_WAIT); continue
            best = None
            for e in mine:
                a = env.vessel_go_emitter_action(e)
                if not mask[a]:
                    continue
                sc = _supply(env, e)
                if best is None or sc > best[0]:
                    best = (sc, a)
            if best is not None:
                acts.append(best[1]); continue
            acts.append(VESSEL_GO_TERMINAL if (mask[VESSEL_GO_TERMINAL] and cargo > _EPS) else VESSEL_WAIT)
        return {"vessels": acts,
                "wells": [env.highest_feasible_well_rate_index(w) for w in env.well_ids]}
    return policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:7b-instruct")
    p.add_argument("--scenario", default="northern_lights_phase1_milkrun")
    p.add_argument("--episode-hours", type=int, default=720)
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 102])
    args = p.parse_args()

    def make_env():
        network, _ = build_fixed_scenario_demo(args.scenario)
        loc = _scenario_locations(_load_fixed_scenario_data(args.scenario))
        return CCSEnv(network, loc,
                      scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=args.episode_hours)),
                      config=CCSEnvConfig(episode_hours=args.episode_hours))

    # compute assignments once (layout is static)
    env0 = make_env(); env0.reset(seed=args.seeds[0])
    geo = geographic_assignment(env0)
    print(f"=== planner assignments ({args.scenario}) ===")
    print("  geographic:", geo)
    llm_assign = llm_assignment(env0, args.model)
    print("  llm       :", llm_assign)

    def score(name, make_policy):
        srs, losses = [], []
        for s in args.seeds:
            env = make_env()
            m = run_episode(env, make_policy(env), seed=s)
            srs.append(m.storage_rate); losses.append(m.loss_rate)
        print(f"{name:24s} storage={statistics.mean(srs):6.1%}  loss(vent)={statistics.mean(losses):6.1%}", flush=True)

    print(f"\n=== {args.scenario} {args.episode_hours}h, seeds={args.seeds} ===")
    score("greedy_shuttle", lambda env: greedy_shuttle_policy)
    score("cluster_geographic", lambda env: make_cluster_policy(env, geo))
    score(f"llm_planner({args.model.split(':')[0]})", lambda env: make_cluster_policy(env, llm_assign))


if __name__ == "__main__":
    main()
