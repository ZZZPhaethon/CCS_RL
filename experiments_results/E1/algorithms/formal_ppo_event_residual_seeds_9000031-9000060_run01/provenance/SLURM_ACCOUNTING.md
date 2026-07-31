# E1 PPO formal-test provenance

## Scope

- Algorithms: Centralized Maskable PPO and Event-Residual PPO
- Model seeds: 0, 1, 2
- Model selection: fixed best-validation checkpoint; no test-based checkpoint selection
- Test seeds: 9000031–9000060
- Scenario: 720 h execution plus 168 h read-only forecast context
- Future summary: one 168 h window
- Cost definition: `total_cost_eur = episode_total_cost_eur + terminal_cleanup_operating_cost_eur`
- Paired baseline: existing E1 Greedy results for the same 30 seeds

## SLURM accounting

All tasks ran on Borg partition `root`, QoS `short`, node `rootrunner`.

| Job/task | Role | State | Exit | Elapsed | Total CPU | Max RSS | CPUs | Memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 34178 | Validation-seed smoke | COMPLETED | 0:0 | 00:00:10 | 00:08.893 | 438944K | 4 | 16G |
| 34179_0 | Centralized, model seed 0 | COMPLETED | 0:0 | 00:00:46 | 00:43.562 | 439448K | 4 | 24G |
| 34179_1 | Centralized, model seed 1 | COMPLETED | 0:0 | 00:00:45 | 00:43.329 | 439512K | 4 | 24G |
| 34179_2 | Centralized, model seed 2 | COMPLETED | 0:0 | 00:00:45 | 00:44.180 | 438724K | 4 | 24G |
| 34179_3 | Event-Residual, model seed 0 | COMPLETED | 0:0 | 00:01:13 | 01:12.156 | 439732K | 4 | 24G |
| 34179_4 | Event-Residual, model seed 1 | COMPLETED | 0:0 | 00:01:13 | 01:11.476 | 440084K | 4 | 24G |
| 34179_5 | Event-Residual, model seed 2 | COMPLETED | 0:0 | 00:01:14 | 01:12.622 | 440328K | 4 | 24G |

Smoke ran from 2026-07-29 14:40:50 to 14:41:00. The formal array ran from 14:41:33 to 14:42:47, Borg scheduler local time. All error logs are empty.

## Audit

- Six task-level `audit.json` files each contain exactly 30 formal episodes.
- The aggregate contains 180 PPO records and 30 unique paired Greedy records.
- All 210 records satisfy the cleanup cost identity within 1e-6 EUR.
- Every PPO configuration has 720 execution hours, 168 read-only forecast hours, and one 168 h future-summary window.
- Each task used the model seed's copied best-validation checkpoint.
- Formal outputs were written atomically from task-specific staging directories.

The Centralized evaluator originally emits an additional file whose name enumerates all 30 seeds. Windows cannot represent that full path under the local result root. Each task therefore also produced short canonical files, `results.json` and `results.csv`. Remote SHA-256 checks confirmed that every canonical file is byte-identical to its long-name evaluator copy; no result data were omitted locally.

## Code hashes

```text
84db6ccbfc8cf47d2d21535a261fc9e4fda66e774238ba08b748888e7193bbee  hpc/submit_e1_ppo_formal_smoke.sh
e05a0fe96decad91aec79a1eb9d840e7ce48b1b125a2cc97e8d175746bd3ddf0  hpc/submit_e1_ppo_formal_array.sh
c11eec400bbd8c0e7fa6fdf7134e16c12d1b0733e6e74f5ac728ed6baff738f6  experiments/aggregate_e1_ppo_formal.py
8bf66253e3ec390ea8738d9d927c33898ccc2d18e879fe4b5f0fa4e00638b768  src/sim/control/event_based/rl/evaluate_high_level_ppo.py
cd702afb3bb9175d85e115f9b435336a035d12724f4eba65cb01c210dbb82fc6  src/sim/control/event_based/residual_rl_v4/evaluate_ppo.py
a3141226ea698900b1fb9d2afca834a28f6ad085eb4e982bac2f591e9c9234a5  src/sim/control/event_based/residual_rl_v2/evaluation.py
```

## Checkpoint hashes

| Algorithm | Model seed | SHA-256 |
|---|---:|---|
| Centralized Maskable PPO | 0 | `68db6aec014832f9dd055c9a4ba29d79e0457a79da23b297d1b3023f084638e9` |
| Centralized Maskable PPO | 1 | `912373efc52dd1044a8c1ce1cc484bd974504ebd6c64df2ee58d4b7c1503f910` |
| Centralized Maskable PPO | 2 | `535b26bbdb56c0dc35e74db424020aaeb972f9488d2c7ff09df2197da93ebf43` |
| Event-Residual PPO | 0 | `3622a1dcceb3b7a128ec99537a8bcf9ff21d5be189d04fd908f4c2187c63def9` |
| Event-Residual PPO | 1 | `eabf796444a16715e0daed0ed6cf9642011e23b88c6fdd4d6d80cd35955c18eb` |
| Event-Residual PPO | 2 | `bcb6aab6943ae18d4ee0c5b1090951ac12064e6986950f9dde2a8d58d4afeff9` |
