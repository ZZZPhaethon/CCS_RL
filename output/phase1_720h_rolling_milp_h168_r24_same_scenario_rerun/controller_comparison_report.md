# Same-Scenario Controller Comparison

All episode controllers are evaluated on identical disturbance trajectories for each seed.
The static MILP row is a separate perfect-foresight benchmark, not an online episode controller.

## Static MILP Benchmark

- horizon: 720 h; static MILP solve skipped.

## Episode Controller Summary

| Controller | Hours mean | Solve s mean | Stored t mean | Vented t mean | Shortfall penalty mean | Total cost mean | Op cost mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| rolling_milp | 720.0 | 899.7 | 93,255.3 | 211.1 | 1,774,658 | 3,294,986 | 1,503,444 |
