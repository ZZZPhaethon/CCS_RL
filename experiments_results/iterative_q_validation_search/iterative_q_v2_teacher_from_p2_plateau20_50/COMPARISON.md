# Full recursive Plateau-20/50 comparison

## Conclusion

The best checkpoint in the new recursive chain is Plateau chain P4 at EUR 1,803,443, saving 13.05% versus Greedy.
The final P4 costs EUR 1,803,443. It is EUR 12,568 more expensive than P4-only hard teacher on the mean.

## New Plateau recursive chain

| Stage | Mean total cost EUR | Delta vs Greedy EUR | Saving | W/T/L | Vent t | Stored t | Overrides |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plateau chain P1 | 1,944,340 | -129,668 | 6.25% | 12/0/8 | 1,937.0 | 107,417.6 | 11.80 |
| Plateau chain P2 | 1,907,778 | -166,230 | 8.01% | 16/0/4 | 1,772.4 | 108,719.7 | 11.45 |
| Plateau chain P3 | 1,827,517 | -246,491 | 11.88% | 19/0/1 | 679.8 | 109,877.7 | 11.40 |
| Plateau chain P4 | 1,803,443 | -270,566 | 13.05% | 18/0/2 | 499.5 | 110,340.0 | 10.60 |

## Reference models

| Model | Mean total cost EUR | Delta vs Greedy EUR | Saving | W/T/L | Vent t | Stored t | Overrides |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hard chain P1 | 1,944,340 | -129,668 | 6.25% | 12/0/8 | 1,937.0 | 107,417.6 | 11.80 |
| Hard chain P2 | 1,862,985 | -211,024 | 10.17% | 14/0/6 | 1,168.9 | 110,177.3 | 11.70 |
| Hard chain P3 | 1,818,153 | -255,856 | 12.34% | 16/0/4 | 562.2 | 110,000.9 | 11.20 |
| Hard chain P4 | 1,838,824 | -235,184 | 11.34% | 18/0/2 | 1,085.0 | 110,062.7 | 10.60 |
| P4-only hard | 1,790,875 | -283,134 | 13.65% | 18/0/2 | 330.3 | 110,919.5 | 8.70 |
| P4-only Plateau-20/50 | 1,793,611 | -280,398 | 13.52% | 18/0/2 | 267.2 | 109,872.9 | 10.50 |

## Changes along the new recursive chain

Difference = left - right; negative means the left model is cheaper.

| Comparison | Mean difference EUR | Bootstrap 95% CI EUR | Left W/T/L |
|---|---:|---:|---:|
| Plateau chain P2 - Plateau chain P1 | -36,562 | [-116,099, 40,724] | 11/0/9 |
| Plateau chain P3 - Plateau chain P2 | -80,261 | [-162,952, -5,679] | 12/0/8 |
| Plateau chain P4 - Plateau chain P3 | -24,074 | [-72,077, 15,124] | 14/0/6 |

## Plateau versus hard recursive chain at the same stage

Difference = left - right; negative means the left model is cheaper.

| Comparison | Mean difference EUR | Bootstrap 95% CI EUR | Left W/T/L |
|---|---:|---:|---:|
| Plateau chain P1 - Hard chain P1 | 0 | [0, 0] | 0/20/0 |
| Plateau chain P2 - Hard chain P2 | 44,793 | [-4,128, 96,185] | 9/0/11 |
| Plateau chain P3 - Hard chain P3 | 9,364 | [-30,435, 63,003] | 10/1/9 |
| Plateau chain P4 - Hard chain P4 | -35,381 | [-111,467, 34,810] | 14/0/6 |

## Final and best-checkpoint comparisons

Difference = left - right; negative means the left model is cheaper.

| Comparison | Mean difference EUR | Bootstrap 95% CI EUR | Left W/T/L |
|---|---:|---:|---:|
| Plateau chain P4 - Hard chain P4 | -35,381 | [-111,467, 34,810] | 14/0/6 |
| Plateau chain P4 - P4-only hard | 12,568 | [-18,821, 47,785] | 13/0/7 |
| Plateau chain P4 - P4-only Plateau-20/50 | 9,832 | [-22,813, 43,359] | 10/0/10 |
| Plateau chain P4 - Hard chain P3 | -14,710 | [-43,735, 15,984] | 14/0/6 |

## Tail-seed regressions

- Plateau chain P4 versus Hard chain P4: 8100011: +EUR 378,465, 8100010: +EUR 83,970, 8100017: +EUR 83,028, 8100001: +EUR 49,402, 8100013: +EUR 22,833
- Plateau chain P4 versus P4-only hard: 8100011: +EUR 229,379, 8100009: +EUR 111,366, 8100020: +EUR 100,137, 8100001: +EUR 93,822, 8100010: +EUR 77,975
- Plateau chain P4 versus P4-only Plateau-20/50: 8100011: +EUR 196,535, 8100010: +EUR 105,557, 8100009: +EUR 93,515, 8100001: +EUR 82,198, 8100020: +EUR 81,871
- Plateau chain P4 versus Hard chain P3: 8100010: +EUR 121,332, 8100009: +EUR 113,857, 8100011: +EUR 93,665, 8100004: +EUR 48,197, 8100001: +EUR 44,005

## Protocol verification

- SLURM jobs 34138-34153 and all array tasks completed with exit code 0:0.
- All 38 matching stderr files were empty; no traceback or OOM was found.
- Validation seeds were exactly 8100001-8100020; formal-test remained inaccessible.
- P1 previous-policy anchor was 0 and FOLLOW calibration was 0.5.
- P2-P4 used anchor coefficient 1.0, Plateau-20/50 weighting, and model seed 0.
- Paired bootstrap used 200,000 draws with seed 20260729.
