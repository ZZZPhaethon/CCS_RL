# Primary formal MILP results: seeds 9000031-9000060

This directory contains analysis-ready tables derived from the promoted
extended-budget run03: 600 s per Rolling replan and 18,000 s per Full MILP
seed, with four deterministic CPLEX threads per process.

- `rolling_milp_per_seed.csv`: 30 Rolling MILP episode rows.
- `full_milp_per_seed.csv`: 30 time-limited Full MILP rows.
- `rolling_replan_diagnostics.csv`: all 900 Rolling replans.
- `full_solver_diagnostics.csv`: Full MILP solve and MIP-start diagnostics.
- `paired_replay_metrics.csv`: same-seed descriptive replay metrics.
- `summary.json`: completion, solver, gap, warm-start and metric summaries.
- `comparison_summary.json`: paired comparisons against Greedy.
- `comparison_summary.md`: concise human-readable Greedy comparison.

Rolling MILP is an online controller. Full MILP uses perfect information and is
an offline reference, so its paired table is descriptive rather than a direct
online-controller ranking. A completed Full MILP row means that a valid
incumbent was replayed; it does not necessarily mean optimality was proven.

The configuration and job lock is stored in the sibling directory
`../milp_extended_600s_18000s_9000031_9000060_run03_lock/`.

The complete per-seed source artifacts, including action trajectories, remain in
`../E1/formal_rolling_milp_h168_r24_t600s_cplex222_seeds_9000031-9000060_run03/` and
`../E5/formal_full_milp_h720_t18000s_cplex222_seeds_9000031-9000060_run03/`.

The superseded 300 s/7,200 s artifacts are retained for provenance. Their
runner and solver source hashes differ from run03, so the old/new comparison
must not be interpreted as a single-factor time-limit experiment.
