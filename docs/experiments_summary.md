# CCS RL + LLM — Experiments Summary

End-to-end record of the RL / LLM experiments on the ship-based CCS dispatch
problem. All storage rates are `stored / captured` over 720 h episodes (unless
noted), averaged over eval seeds. "vent" = share of captured CO2 lost to venting.

---

## Phase 1 — RL on the original Northern Lights Phase 1 problem

**Problem:** 4 vessels, 3 emitters, one terminal. Starting point: a trained PPO
reached only ~2.3 % storage (collapsed to idling).

**Diagnosis & fixes**
- Root cause: training horizon too short + reward rewarded idling (delayed,
  sparse venting penalty). Added a dense per-step injection reward.
- Restored full-load / unload-before-leave vessel action masks; passed masks at
  eval; decision-step-weighted behaviour cloning (BC) from greedy.
- Kickstarting (decaying BC anchor) to stop PPO drifting away from the BC policy.
- Realigned the reward to a symmetric carbon price (store credit = vent tax).

**Result**

| policy | storage | notes |
|---|---|---|
| original PPO | ~2.3 % | collapsed to idle |
| BC-only (clone greedy) | 81–88 % | matches greedy |
| BC + kickstart PPO (carbon price 80) | 87 % | ties greedy |
| greedy (reference) | ~88 % | near-optimal, vents <1 % |

**Conclusion:** on this single-route problem **greedy is near-optimal** (vents
<1 %); RL can match it but not beat it. There is almost no headroom above a good
static policy here.

---

## Phase 2 — Milk-run scenarios (make greedy suboptimal)

To create headroom, tightened the fleet: 2 vessels serve 3 geographically spread
emitters (`northern_lights_phase1_milkrun`). A hand-written cluster policy (one
vessel per region) beat greedy, confirming greedy is now suboptimal.

| scenario | greedy | cluster (balanced) |
|---|---|---|
| milk-run (balanced) | 74.7 % | 76.7 % |
| milk-run **imbalanced** (brevik/celsio heavy, yara light) | 57.6 % | 63.4 % |

A **load-balanced** assignment (split the two heavy emitters across vessels)
beats the naive geographic grouping (49.6 %) — routing headroom is real.

---

## Phase 3 — LLM as a planner (local Qwen/Llama via Ollama, RTX 5080)

| approach | result | takeaway |
|---|---|---|
| **naive per-step LLM router** | 45.5 % vs greedy 76 % | LLMs are weak at precise multi-vessel numeric coordination |
| **hierarchical: LLM assignment + deterministic executor** | 76.7 % (balanced), beats greedy | LLM belongs at the strategic layer |
| — with weak prompt on imbalanced | 49.6 % (picked geographic) | 7B ignored load balance |
| — with load-balancing prompt (qwen & llama) | 59.9 %, beats greedy 57.6 % | prompt engineering unlocked it |
| **dynamic re-planning** (re-query every 168 h) | 56.6 % < static 59.9 % | reactive reassignment thrashes; keep a stable strategy |

**Conclusion:** the LLM adds value as a *stable high-level planner*, not a
per-step controller and not a micro-manager.

---

## Phase 4 — Goal-conditioned RL + zero-shot generalization

Added a goal channel to the observation (per-vessel emitter one-hot). Trained a
goal-conditioned policy across 4 random capture-imbalance layouts (goal = the
balanced-capture heuristic), then evaluated on a **held-out** layout never seen.

| layout | greedy | cluster | goal_RL |
|---|---|---|---|
| **test (HELD-OUT)** | 52.9 % | 56.4 % | **56.4 %** |
| train0 | 55.2 % | 63.0 % | 63.0 % |
| train1 | 34.0 % | 38.4 % | 38.4 % |
| train3 | 52.6 % | 54.4 % | 54.4 % |

**Conclusion:** **zero-shot generalization works** — the learned policy transfers
to a new layout, matching the per-layout heuristic and beating greedy, with no
retraining. But it **matches, not beats** the heuristic (static is near-optimal
on stationary layouts).

---

## Phase 5 — Dynamic stress (strong capture outages)

Tried strong outages (2.5/week, 48 h) to see if a reactive policy beats a rigid
one. **Backfired:** capture outages *reduce* the CO2 load, so with only 2 vessels
the problem got *easier* (storage rose to ~75 %) and no dynamic advantage emerged.

| layout | greedy | cluster | goal_RL |
|---|---|---|---|
| test (HELD-OUT) | 75.8 % | 74.4 % | 74.4 % |

**Lesson:** to reward dynamic reallocation the disturbance must keep the load
high while shifting *where* it is (load-shifting capture, or downstream capacity
outages), not remove capture. Not yet built.

---

## Phase 6 — LLM in the RL loop (goal source: heuristic vs LLM)

At eval, set the goal-conditioned policy's goal from the heuristic **and** from
the LLM, on varied layouts (outage 0.5/wk).

| layout | goals agree? | gRL(heur) | gRL(llm) | cluster(heur) | cluster(llm) |
|---|---|---|---|---|---|
| test | no | 57.2 % | 54.9 % | **58.4 %** | 38.8 % |
| train0 | no | 62.8 % | 44.5 % | **62.8 %** | 59.9 % |
| train1 | no | 37.6 % | 37.6 % | **40.8 %** | 25.1 % |
| train2 | no | 35.6 % | 38.6 % | 43.0 % | 44.2 % |
| train3 | no | 52.5 % | 53.5 % | 52.5 % | 52.8 % |

**Two findings**
1. The **7B LLM is a worse goal-provider than the balanced-capture heuristic**
   on this formalized problem (cluster(heur) ≥ cluster(llm) on most layouts).
   Putting the LLM in the loop *hurts* here.
2. **goal_RL tolerates a bad goal far better than the rigid cluster** (e.g. test:
   gRL(llm) 54.9 % vs cluster(llm) 38.8 %) — the RL can deviate from a bad
   assignment; the rigid executor cannot. This robustness is the RL's real value.

---

## Overall conclusions

- On **formalized, stationary** CCS dispatch problems, a good **heuristic / static
  assignment is near-optimal**; MILP is the ceiling. RL and LLM can **match** it
  but not beat it on raw storage %.
- **RL's real value** is a *robust, goal-conditioned executor* that (a) generalizes
  zero-shot to new layouts and (b) tolerates imperfect goals better than rigid
  execution.
- **LLM's real value** is a *stable high-level planner* — but only when it beats the
  heuristic, which the 7B model does **not** on these formalized layouts. It would
  need a stronger model or a problem with factors the heuristic ignores
  (un-formalizable constraints, natural-language rules, novel topologies).
- To make learning/LLM **clearly beat** the heuristics, the problem must become
  genuinely **dynamic** (load-shifting disturbances, more emitters than vessels)
  where static plans break — not yet demonstrated.

## Open next steps

1. A load-shifting / downstream-capacity disturbance where dynamic > static.
2. A stronger LLM (e.g. qwen2.5:14b) as goal-provider vs the heuristic.
3. A layout class where the balanced-capture heuristic is provably suboptimal, to
   expose real LLM planning value.
4. Fix / time-budget the rolling MILP to quantify the true ceiling.
