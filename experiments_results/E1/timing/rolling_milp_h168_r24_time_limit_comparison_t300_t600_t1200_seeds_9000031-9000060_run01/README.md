# Rolling MILP H168-R24 time-limit comparison

Paired seeds 9000031-9000060; four deterministic CPLEX threads.
T300 is superseded, T600 is the formal result, and T1200 is a time-budget ablation.
T300 used different runner and solver source hashes, so comparisons involving T300 are descriptive rather than single-factor time-limit effects. T600 versus T1200 is the clean time-limit comparison.

## Episode-level means

| Metric | T300 | T600 | T1200 |
|---|---:|---:|---:|
| Total cost (EUR) | 2.21397e+06 | 2.08973e+06 | 2.09601e+06 |
| Operating cost (EUR) | 1.6882e+06 | 1.71667e+06 | 1.71237e+06 |
| Terminal cleanup cost (EUR) | 226006 | 223281 | 219940 |
| Stored CO2 (t) | 103677 | 105910 | 105554 |
| Vented CO2 (t) | 6572.17 | 4663.19 | 4795.54 |
| Captured CO2 (t) | 126283 | 126283 | 126283 |
| Wall-clock time (s) | 6566.73 | 12312.7 | 24230.9 |
| Solver time (s) | 6304.75 | 12050.9 | 23972.4 |

## Solver proof status

| Limit | Optimal | Integer feasible | Optimal rate | 30/30 optimal seeds | Mean gap | P95 gap |
|---:|---:|---:|---:|---:|---:|---:|
| 300s | 297 | 603 | 33.00% | 0 | 12.64% | 39.56% |
| 600s | 329 | 571 | 36.56% | 0 | 10.28% | 35.80% |
| 1200s | 332 | 568 | 36.89% | 0 | 9.98% | 34.11% |

All three runs have 900/900 valid solver results and 900/900 valid execution replays.
