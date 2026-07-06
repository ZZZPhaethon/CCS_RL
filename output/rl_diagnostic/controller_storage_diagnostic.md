# RL storage-rate diagnostic

Generated: 2026-07-06T16:10:49  |  Phase 1 env, seed=1, no leg-wave weather.

Reference point: the trained RL policy reaches storage_rate ~= 2.3%.

| horizon | controller | storage_rate | loss_rate (vented) | vented_t | net |
|---|---|---|---|---|---|
| 168h | idle | 8.7% | 0.0% | 0 | -230,368 |
| 168h | greedy_shuttle | 43.3% | 0.0% | 0 | -336,457 |
| 720h | idle | 2.0% | 57.8% | 69,245 | -5,776,134 |
| 720h | greedy_shuttle | 86.4% | 0.0% | 0 | -1,525,189 |

## Reading

- Env is solvable: greedy_shuttle reaches 86.4% storage with 0 venting at 720h.
- RL (2.3%) is worse than idle (8.7% @168h / 2.0% @720h) => it collapsed to a
  counter-productive/idle-like policy, not an unsolvable environment.
- Reward misalignment at short horizon: at 168h greedy's net (-336k) is WORSE than
  idle (-230k) because buffers have not overflowed yet, so activity only adds cost.
  Only at 720h (buffers overflow -> idle vents 57.7%) does net reward storage.
- Training horizon is episode_hours=168 and storage_shortfall_eur_per_t=0, so the
  training objective effectively rewards idling. Fix reward/horizon before adding a
  teacher.