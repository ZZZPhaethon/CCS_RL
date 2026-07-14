"""LLM routing policy PoC: at each real dispatch decision, describe the state in
text, ask a local Ollama model (Qwen/Llama) which destination to pick, parse the
answer back into an env action. Non-decision steps use cheap rules so the LLM is
only queried when it matters. Compares LLM vs greedy vs cluster on the milk-run
scenario to test whether LLM reasoning beats the greedy teacher.

Usage (ccs-rlllm-gpu env, from repo root, Ollama running):
    set PYTHONPATH=src
    python experiments/llm_router.py --model qwen2.5:7b-instruct --seeds 101 --episode-hours 720
"""
from __future__ import annotations

import argparse
import json
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


def ollama_generate(model: str, prompt: str, timeout: float = 30.0) -> str:
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "num_predict": 8},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"].strip().lower()


def _supply(env, emitter_id):
    e = env.network.entities[emitter_id]
    st = env.simulator.state
    avail = st.emitter_availability.get(emitter_id, e.availability)
    return st.entity_inventory_t.get(emitter_id, 0.0) + e.nominal_capture_tph * max(0.0, avail)


def _describe(env, vessel_id, legal_emitters, can_terminal):
    st = env.simulator.state
    vessel = env.network.entities[vessel_id]
    cargo = st.entity_inventory_t.get(vessel_id, 0.0)
    berth = st.vessel_berths.get(vessel_id)
    lines = [
        f"You dispatch CO2 vessel '{vessel_id}' in a ship-based carbon-storage network.",
        f"The vessel is at '{berth}', carrying {cargo/vessel.capacity_t:.0%} of its {vessel.capacity_t:.0f} t capacity.",
        "Emitters capture CO2 into buffers; if a buffer overflows it is vented (lost). "
        "Full vessels should deliver to the terminal. Pick the destination that avoids venting and minimizes wasted sailing.",
        "Emitter status (buffer fill / capture rate):",
    ]
    for e in env.emitter_ids:
        em = env.network.entities[e]
        fill = st.entity_inventory_t.get(e, 0.0) / max(1.0, em.buffer_capacity_t)
        lines.append(f"  - {e}: buffer {fill:.0%} full, capturing {em.nominal_capture_tph:.0f} t/h")
    options = list(legal_emitters) + (["terminal"] if can_terminal else [])
    lines.append(f"Legal next destinations: {', '.join(options)}.")
    lines.append("Answer with ONLY one destination name from that list, nothing else.")
    return "\n".join(lines)


def make_llm_policy(env, model: str, verbose: bool = False):
    stats = {"calls": 0}

    def policy(env):
        st = env.simulator.state
        acts = []
        for i, vid in enumerate(env.vessel_ids):
            mask = env.vessel_action_mask()[i]
            cargo = st.entity_inventory_t.get(vid, 0.0)
            vessel = env.network.entities[vid]
            berth = st.vessel_berths.get(vid)
            # cheap rules for non-decisions
            if berth in env.terminal_ids and cargo > _EPS:
                acts.append(VESSEL_WAIT); continue
            if mask[VESSEL_GO_TERMINAL] and cargo >= vessel.capacity_t - _EPS:
                acts.append(VESSEL_GO_TERMINAL); continue
            if berth in env.emitter_ids and cargo < vessel.capacity_t - _EPS and _supply(env, str(berth)) > _EPS:
                acts.append(VESSEL_WAIT); continue
            legal_emitters = [e for e in env.emitter_ids if mask[env.vessel_go_emitter_action(e)]]
            can_terminal = bool(mask[VESSEL_GO_TERMINAL] and cargo > _EPS)
            if not legal_emitters and not can_terminal:
                acts.append(VESSEL_WAIT); continue
            # genuine decision -> ask the LLM
            stats["calls"] += 1
            try:
                ans = ollama_generate(model, _describe(env, vid, legal_emitters, can_terminal))
            except Exception:
                ans = ""
            chosen = None
            for e in legal_emitters:
                if e in ans:
                    chosen = env.vessel_go_emitter_action(e); break
            if chosen is None and can_terminal and "terminal" in ans:
                chosen = VESSEL_GO_TERMINAL
            if chosen is None:  # fallback: best-supply legal emitter, else terminal/wait
                best = None
                for e in legal_emitters:
                    sc = _supply(env, e)
                    if best is None or sc > best[0]:
                        best = (sc, env.vessel_go_emitter_action(e))
                chosen = best[1] if best else (VESSEL_GO_TERMINAL if can_terminal else VESSEL_WAIT)
            if verbose:
                print(f"    [llm] {vid} -> {ans!r} => action {chosen}", flush=True)
            acts.append(chosen)
        return {"vessels": acts,
                "wells": [env.highest_feasible_well_rate_index(w) for w in env.well_ids]}
    policy.stats = stats
    return policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:7b-instruct")
    p.add_argument("--scenario", default="northern_lights_phase1_milkrun")
    p.add_argument("--episode-hours", type=int, default=720)
    p.add_argument("--seeds", type=int, nargs="+", default=[101])
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    def make_env():
        network, _ = build_fixed_scenario_demo(args.scenario)
        loc = _scenario_locations(_load_fixed_scenario_data(args.scenario))
        return CCSEnv(network, loc,
                      scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=args.episode_hours)),
                      config=CCSEnvConfig(episode_hours=args.episode_hours))

    def score(name, make_policy):
        srs, losses = [], []
        for s in args.seeds:
            env = make_env()
            m = run_episode(env, make_policy(env), seed=s)
            srs.append(m.storage_rate); losses.append(m.loss_rate)
        print(f"{name:22s} storage={statistics.mean(srs):6.1%}  loss(vent)={statistics.mean(losses):6.1%}", flush=True)

    print(f"=== {args.scenario} {args.episode_hours}h, seeds={args.seeds}, model={args.model} ===", flush=True)
    score("greedy_shuttle", lambda env: greedy_shuttle_policy)
    llm = None
    def mk(env):
        nonlocal llm
        llm = make_llm_policy(env, args.model, verbose=args.verbose)
        return llm
    score(f"llm({args.model.split(':')[0]})", mk)
    if llm is not None:
        print(f"  (LLM was queried {llm.stats['calls']} times)", flush=True)


if __name__ == "__main__":
    main()
