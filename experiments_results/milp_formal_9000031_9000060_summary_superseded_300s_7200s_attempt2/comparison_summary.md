# Formal Greedy and MILP comparison

Seeds: 9000031–9000060 (n=30), protocol: unified_window_v1, 720 h execution, terminal cleanup included.

| Method | Role | Total cost, mean ± SD | Stored t, mean ± SD | Vented t, mean ± SD | Wall time/seed, mean |
|---|---|---:|---:|---:|---:|
| greedy | online baseline | 2,254,211.56 ± 419,091.88 | 102,984.18 ± 9,772.21 | 7,296.77 ± 5,925.33 | 1.07 s |
| rolling_milp | online controller | 2,213,972.12 ± 384,975.57 | 103,676.77 ± 9,495.64 | 6,572.17 ± 5,449.69 | 6,566.73 s |
| full_milp | offline perfect-information reference | 2,254,197.73 ± 419,094.66 | 102,984.18 ± 9,772.21 | 7,296.77 ± 5,925.33 | 7,307.29 s |

| Paired comparison vs Greedy | Mean cost improvement | 95% bootstrap CI | Lower/equal/higher cost seeds |
|---|---:|---:|---:|
| Rolling MILP | 0.941% | [-2.900%, 4.979%] | 12/8/10 |
| Full MILP | 0.001% | [0.000%, 0.002%] | 1/29/0 |

Rolling MILP vs Greedy is the valid online paired comparison. Full MILP used the complete future trajectory and all 30 runs ended with time-limited feasible incumbents rather than proven optima, so its comparison is descriptive only.
