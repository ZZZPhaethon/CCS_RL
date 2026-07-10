# Native MPC vs Greedy: Preliminary Results

## Scope

This note summarizes the 720 h, three-vessel comparison between the greedy
shuttle baseline and the native-action rolling MPC controller.

- Scenario: `northern_lights_phase1_3vessels`.
- Seeds: 1-10.
- Horizon: 720 h.
- MPC replan interval: 24 h.
- MPC planning horizon: 168 h.
- MPC objective: minimize forecast vented CO2, then end-of-window un-stored
  inventory, then operating cost.
- Yara Sluiskil buffer: **15,000 t**. Brevik and Celsio buffers remain 7,500 t.

The controller emits the same native discrete vessel and well-rate actions as
the RL environment. Loading, terminal unloading, sailing, injection, costs,
and venting are recomputed by the environment during replay.

## Disturbance Configuration

- Weather: global 24 h block process; one shared vessel-speed factor per block,
  sampled from [0.75, 1.00].
- Capture noise standard deviation: 0.10.
- High-output event rate: 0.5 events per week.
- High-output duration mean: 48 h.
- High-output capture multiplier: [1.25, 1.75].
- Capture outages and well maintenance: disabled.
- Initial inventory: randomized.

## Aggregate Results

| Metric | Greedy | Native MPC | Difference (MPC - Greedy) |
|---|---:|---:|---:|
| Mean vented CO2 | 7,877.7 t | 1,576.3 t | -6,301.4 t |
| Total vented CO2 | 78,776.9 t | 15,763.4 t | -63,013.5 t |
| Weighted vent reduction | - | - | 79.99% |
| Mean per-seed vent reduction | - | - | 82.36% |
| Mean total cost | EUR 2.124 m | EUR 1.712 m | -EUR 0.412 m |
| Mean unit total cost | EUR 20.07/t | EUR 15.39/t | -EUR 4.68/t |

The results show substantial replay-executable scheduling headroom over the
greedy baseline. The effect is heterogeneous: per-seed vent reduction ranges
from 52.97% to 100.00%.

## Per-Seed Results

| Seed | Greedy vent t | MPC vent t | Vent reduction | Greedy total cost | MPC total cost | Cost saving | Greedy EUR/t | MPC EUR/t | Replay |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 2,570.0 | 295.2 | 88.51% | EUR 1.689 m | EUR 1.614 m | EUR 0.075 m | 14.72 | 14.45 | True |
| 2 | 8,004.6 | 447.6 | 94.41% | EUR 2.114 m | EUR 1.625 m | EUR 0.489 m | 21.17 | 14.67 | True |
| 3 | 7,319.4 | 0.0 | 100.00% | EUR 1.992 m | EUR 1.555 m | EUR 0.438 m | 20.12 | 14.15 | True |
| 4 | 8,351.6 | 187.3 | 97.76% | EUR 2.277 m | EUR 1.697 m | EUR 0.580 m | 20.18 | 14.86 | True |
| 5 | 9,162.8 | 2,992.8 | 67.34% | EUR 2.220 m | EUR 1.815 m | EUR 0.405 m | 19.90 | 16.23 | True |
| 6 | 10,697.7 | 5,031.1 | 52.97% | EUR 2.450 m | EUR 1.949 m | EUR 0.501 m | 21.74 | 17.47 | True |
| 7 | 7,136.2 | 1,858.6 | 73.96% | EUR 2.043 m | EUR 1.719 m | EUR 0.324 m | 20.79 | 16.21 | True |
| 8 | 7,889.8 | 1,147.7 | 85.45% | EUR 2.121 m | EUR 1.678 m | EUR 0.443 m | 20.50 | 14.93 | True |
| 9 | 7,313.2 | 0.0 | 100.00% | EUR 1.997 m | EUR 1.521 m | EUR 0.476 m | 20.13 | 14.24 | True |
| 10 | 10,331.7 | 3,803.1 | 63.19% | EUR 2.340 m | EUR 1.952 m | EUR 0.388 m | 21.50 | 16.70 | True |

## Replay Validation

The 10-seed batch was generated with the prior action-trace replay check: each
row recorded `replay_ok=True`, and the exported JSON trace has exactly 720
native actions per controller.

The current shared replay validator is stricter. It verifies, on a separately
reset same-seed environment, every action mask and the full end-to-end replay:
mass-flow KPIs, cost components, reward/objective, overflow-risk accumulation,
per-hour injection, final entity inventory, and vessel berths. Seed 4 was
rerun under this validator and passed with both greedy and MPC traces
executable and exact, with no mismatches.

Consequently, the 10-seed table is suitable as a preliminary comparison; the
remaining nine seeds should be rerun with the shared validator before being
used as the final strict-replay result table.

## Source Artifacts

- Scenario definition: `scenarios/northern_lights_phase1_3vessels.json`.
- Ten-seed comparison: `output/rolling_native_mpc_yara15k_block_720h_10seeds/by_seed.csv`.
- Ten-seed configuration: `output/rolling_native_mpc_yara15k_block_720h_10seeds/run_config.json`.
- Strict shared-replay confirmation: `output/post_edit_shared_replay_seed4/by_seed.csv`.
