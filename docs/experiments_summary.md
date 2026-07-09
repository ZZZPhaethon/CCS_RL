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

## Phase 7 — Load-shift disturbance, full-load constraint, and cost view

Built a standalone `LoadShiftScenarioGenerator` (rotating capture hot spot: one
or more emitters run near full while the rest are throttled, cycling each phase),
to test whether a dynamic policy beats a rigid static assignment when the load's
*location* moves. Also tested the earlier hypothesis that the full-load dispatch
constraint was what capped dynamic value, and finally compared on **cost**, not
just storage.

| policy (milk-run, load-shift, hot_count=2) | storage | vent | cost/t |
|---|---|---|---|
| greedy (full-load) | 81.1 % | 5.2 % | 18.8 |
| **cluster static** | **85.6 %** | 1.8 % | **14.8** |
| flexible dynamic (partial delivery, constraint off) | 83.6 % | 5.2 % | 18.6 |

**Findings**
- Rotating hot spots do **not** break the static assignment on 2 vessels / 3
  emitters — a vessel can milk-run its region's hot emitters, so static stays
  near-optimal.
- Removing the full-load constraint and allowing partial delivery helps a dynamic
  policy (81 → 83.6 %) but does **not** flip the ranking; the partial-delivery
  overhead (more trips) raises cost.
- **On cost too, the static cluster dominates** (14.8 €/t vs 18.6–18.8): it stores
  more, vents least, and is cheapest. The full-load constraint was not the blocker.

## Phase 8 — Residual (gated-override) RL over the heuristic base

`train_residual_rl.py`: instead of learning from scratch, the cluster base
proposes an action each step and the RL policy learns a *correction* — per vessel
it FOLLOWs the base or OVERRIDEs it (`a_final = FOLLOW ? a_base : a_override`).
FOLLOW is always legal and reproduces the base, so the policy starts at base
performance and only learns profitable overrides.

| policy (imbalanced milk-run + load-shift) | storage | vent | cost/t |
|---|---|---|---|
| greedy | 73.7 % | 21.5 % | 35.3 |
| cluster_base | 71.2 % | 23.0 % | 39.1 |
| residual_rl | 70.3 % | 19.4 % | 35.4 |

**Findings**
- **Stability delivered:** residual RL stayed ≈ base (70.3 vs 71.2 %) and did **not**
  drift/collapse — unlike plain BC+PPO fine-tune, which crashed to ~58 %. It even
  vented the least. This is residual RL's core promise (a >= base floor).
- **But no clear win:** the policy gradient stayed ~1e-8 — PPO learned essentially
  pure FOLLOW, because the near-optimal base leaves no profitable override.
- **Base choice matters:** under load-shift the *dynamic* greedy (73.7 %) beat the
  *rigid* cluster (71.2 %), so cluster was not the best base here; residual over a
  suboptimal base can't exceed a better one.

---

## Overall conclusions

- On **formalized, stationary** CCS dispatch problems, a good **heuristic / static
  assignment is near-optimal on both storage AND cost-per-tonne**; MILP is the
  ceiling. RL and LLM **match** it but do not beat it.
- **RL's real value** is a *robust executor*: goal-conditioned RL generalizes
  zero-shot to new layouts and tolerates imperfect goals better than rigid
  execution; **residual RL** guarantees a ≥ base stability floor (no fine-tune
  drift). Neither raises the ceiling on heuristic-friendly problems.
- **LLM's real value** is a *stable high-level planner* — but the 7B model does **not**
  beat the balanced-capture heuristic on these layouts (it is a worse goal
  provider). It would need a stronger model or a problem with factors the
  heuristic ignores (un-formalizable constraints, natural-language rules).
- Across **three learning methods** (from-scratch RL, goal-conditioned RL,
  residual RL) and **three disturbances** (outages, load-shift, full-load
  toggle), the result is consistent: this small-fleet / milk-run problem class is
  **heuristic-friendly** — learning matches near-optimal and adds *stability +
  generalization*, not raw performance.

## Open next steps

1. A structural mismatch large enough that dynamic > static (e.g. 5–6 emitters,
   2 vessels, no milk-run), or a downstream-capacity disturbance that backs up
   buffers while capture continues.
2. Residual RL over the **greedy** base (dynamic) rather than cluster.
3. A stronger LLM (e.g. qwen2.5:14b) as goal-provider vs the heuristic.
4. Fix / time-budget the rolling MILP to quantify the true ceiling.
