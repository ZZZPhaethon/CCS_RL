# Corrected PPO evaluation — ppo_bc_phase1_720h_inj80_ts150000

Generated: 2026-07-06T18:12:03 | ep=720h, warm_start=True, seeds=[101, 102, 103]

Both PPO variants are reported: a still-stochastic dispatch policy scores far
higher when sampled than under argmax (deterministic), which can collapse to WAIT.

| policy | storage_rate | loss_rate |
|---|---|---|
| idle | 3.0% | 55.8% |
| greedy_shuttle | 80.8% | 1.1% |
| ppo_stochastic | 58.4% | 22.1% |
| ppo_deterministic | 3.0% | 55.8% |