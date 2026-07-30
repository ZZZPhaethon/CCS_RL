# Formal MILP results: seeds 9000031-9000060

This directory contains analysis-ready tables for Greedy and the formal MILP
experiments.

- `greedy_per_seed.csv`: 30 same-protocol Greedy episode rows.
- `fixed_assignment_per_seed.csv`: 30 same-protocol fixed-assignment
  rule-based episode rows.
- `rolling_milp_per_seed.csv`: 30 Rolling MILP episode rows.
- `full_milp_per_seed.csv`: 30 time-limited Full MILP rows.
- `fixed_assignment_vs_greedy_paired.csv`: same-seed online baseline
  comparison.
- `fixed_assignment_summary.json`: fixed-assignment metric and paired
  comparison statistics.
- `rolling_vs_greedy_paired.csv`: valid same-seed online comparison.
- `full_vs_greedy_descriptive.csv`: offline perfect-information comparison.
- `comparison_summary.json`: machine-readable three-method statistics.
- `comparison_summary.md`: compact paper-facing comparison table.
- `rolling_warmstart_ablation_first_replan.csv`: three-seed paired
  Greedy-versus-no-warm-start diagnostics for the first 168 h replan.
- `rolling_warmstart_ablation_first_replan.json`: configuration, provenance,
  and machine-readable conclusion for the warm-start ablation.
- `rolling_replan_diagnostics.csv`: all 900 Rolling replans.
- `full_solver_diagnostics.csv`: Full MILP solve and MIP-start diagnostics.
- `paired_replay_metrics.csv`: same-seed descriptive replay metrics.
- `summary.json`: completion, solver, gap, warm-start and metric summaries.
- `slurm_accounting.csv`: SLURM elapsed time and peak-memory records.
- `source_artifact_manifest_sha256.csv`: hashes for all 242 copied source files.
- `greedy_source_artifact_manifest_sha256.csv`: hashes for all 120 Greedy
  source files; each hash matched the corresponding HPC artifact.
- `fixed_assignment_source_artifact_manifest_sha256.csv`: hashes for all 90
  locally generated fixed-assignment source files.
- `rolling_warmstart_ablation_none_source_manifest_sha256.csv`: hashes for the
  12 copied No-warm-start ablation source files; all matched the HPC artifacts.

Rolling MILP is an online controller. Full MILP uses perfect information and is
an offline reference, so its paired table is descriptive rather than a direct
online-controller ranking. A completed Full MILP row means that a valid
incumbent was replayed; it does not necessarily mean optimality was proven.

The immutable configuration and job lock is stored in the sibling directory
`../formal_milp_9000031_9000060_attempt2_lock/`.

The complete per-seed source artifacts, including action trajectories, remain in
`../E1/formal_greedy_seeds_9000031-9000060_run01/`,
`../E1/formal_fixed_assignment_seeds_9000031-9000060_run01/`,
`../E1/ablation_rolling_milp_warmstart_cplex222_t300s_n3_run01/none/`,
`../E1/formal_rolling_milp_h168_r24_t300s_cplex222_seeds_9000031-9000060_run02/` and
`../E5/full_milp_formal_9000031_9000060_cplex222_7200s_attempt2/`.
