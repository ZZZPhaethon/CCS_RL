# Formal Greedy and MILP comparison

Seeds: 9000031–9000060 (n=30), protocol: unified_window_v1, 720 h execution, terminal cleanup included.
Primary MILP budgets: Rolling 600 s/replan; Full MILP 18,000 s/seed.

| Method | Role | Total cost, mean ± SD | Stored t, mean ± SD | Vented t, mean ± SD | Wall time/seed, mean |
|---|---|---:|---:|---:|---:|
| greedy | online baseline | 2,254,211.56 ± 419,091.88 | 102,984.18 ± 9,772.21 | 7,296.77 ± 5,925.33 | 1.07 s |
| rolling_milp | online controller | 2,089,728.46 ± 289,555.43 | 105,909.61 ± 8,556.67 | 4,663.19 ± 4,082.69 | 12,312.73 s |
| full_milp | offline perfect-information reference | 2,254,197.73 ± 419,094.66 | 102,984.18 ± 9,772.21 | 7,296.77 ± 5,925.33 | 18,108.82 s |

| Paired comparison vs Greedy | Mean cost improvement | 95% bootstrap CI | Lower/equal/higher cost seeds |
|---|---:|---:|---:|
| Rolling MILP | 5.519% | [0.590%, 10.914%] | 16/5/9 |
| Full MILP | 0.001% | [0.000%, 0.002%] | 1/29/0 |

Rolling MILP vs Greedy is the valid online paired comparison. Full MILP used the complete future trajectory and all 30 runs ended with time-limited feasible incumbents rather than proven optima, so its comparison is descriptive only.
