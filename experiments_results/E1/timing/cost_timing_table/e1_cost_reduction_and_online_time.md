| Method | Mean total cost (EUR) | Cost reduction vs Fixed-Assignment (95% CI) | Online solution time, mean / median / P95 (s) |
|---|---:|---:|---:|
| Fixed-Assignment Heuristic | 2,586,942 | 0.00% [0.00, 0.00] | 0.981 / 1.031 / 1.442 |
| Greedy | 2,254,212 | 11.76% [5.72, 17.48] | 1.085 / 1.058 / 1.561 |
| Hourly Centralized Maskable PPO | 5,659,632 | -121.74% [-135.14, -109.00] | 1.292 / 1.267 / 1.581 |
| High-level Centralized Maskable PPO | 2,187,244 | 14.68% [10.90, 18.16] | 1.325 / 1.370 / 1.802 |
| Event-Residual PPO | 2,239,850 | 12.98% [8.62, 17.09] | 2.265 / 2.346 / 3.141 |
| Iterative Action-Q G60-P4 (no hour) | 1,864,718 | 26.88% [23.71, 29.87] | 2.751 / 2.767 / 3.315 |
| Rolling MILP H168-R24-T600s CPLEX 22.2 | 2,089,728 | 17.66% [11.81, 23.10] | 12,312.734 / 12,263.061 / 14,869.391 |

Cost reduction is paired by test seed against Fixed-Assignment. Positive values indicate lower cost. Time is reported in absolute seconds and is not converted to a percentage. Online solution time covers scenario reset through the completed 720 h closed-loop rollout and terminal-cleanup calculation; it excludes training and checkpoint loading.
