| Method | Horizon | Mean total cost (EUR) | Cost / 720 h (EUR) | Cost reduction vs Fixed (95% CI) | Vent / 720 h (t) | Storage rate |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-Assignment | 30 d | 2,683,391 | 2,683,391 | 0.00% [0.00, 0.00] | 13,295.5 | 0.752 |
| Greedy | 30 d | 2,217,769 | 2,217,769 | 15.52% [9.10, 21.20] | 6,622.6 | 0.822 |
| Iterative-Q (direct-global) | 30 d | 1,924,572 | 1,924,572 | 27.31% [24.26, 30.02] | 1,540.5 | 0.866 |
| Iterative-Q (receding-cyclic) | 30 d | 1,924,572 | 1,924,572 | 27.31% [24.26, 30.02] | 1,540.5 | 0.866 |
| Fixed-Assignment | 90 d | 9,122,089 | 3,040,696 | 0.00% [0.00, 0.00] | 19,806.0 | 0.797 |
| Greedy | 90 d | 7,093,741 | 2,364,580 | 22.11% [18.95, 25.15] | 9,662.0 | 0.882 |
| Iterative-Q (direct-global) | 90 d | 6,035,319 | 2,011,773 | 33.80% [32.17, 35.60] | 3,588.1 | 0.929 |
| Iterative-Q (receding-cyclic) | 90 d | 5,998,074 | 1,999,358 | 34.17% [32.05, 35.95] | 3,472.7 | 0.928 |
| Fixed-Assignment | 180 d | 18,449,088 | 3,074,848 | 0.00% [0.00, 0.00] | 20,694.8 | 0.817 |
| Greedy | 180 d | 14,381,961 | 2,396,993 | 22.06% [20.00, 24.12] | 10,393.5 | 0.898 |
| Iterative-Q (direct-global) | 180 d | 12,441,390 | 2,073,565 | 32.57% [30.97, 34.13] | 4,707.2 | 0.944 |
| Iterative-Q (receding-cyclic) | 180 d | 12,403,952 | 2,067,325 | 32.80% [31.18, 34.27] | 4,670.1 | 0.944 |
| Fixed-Assignment | 365 d | 37,866,089 | 3,112,281 | 0.00% [0.00, 0.00] | 21,376.9 | 0.824 |
| Greedy | 365 d | 29,176,067 | 2,398,033 | 22.93% [21.49, 24.36] | 10,522.4 | 0.909 |
| Iterative-Q (direct-global) | 365 d | 25,542,383 | 2,099,374 | 32.51% [30.68, 34.43] | 5,159.3 | 0.950 |
| Iterative-Q (receding-cyclic) | 365 d | 25,518,463 | 2,097,408 | 32.58% [30.65, 34.77] | 5,152.0 | 0.950 |

All costs and venting are normalized to 720 h. Direct-global and receding-cyclic use identical repeated policy windows; their only controller difference is episode progress t / H versus (t mod 720) / 720. Rolling MILP is intentionally excluded from this initial E7 run.
