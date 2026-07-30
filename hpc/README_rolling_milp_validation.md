# Rolling MILP validation on Borg

This configuration runs the locked 168 h planning / 24 h execution Rolling
MILP controller-validation comparison on Borg:

- seeds `8100001` through `8100020`;
- paired 30 s and 300 s limits per replan;
- 720 h execution and 168 h forecast context;
- deterministic CPLEX mode configured by the controller;
- four CPLEX threads per process;
- Greedy plus complete-cleanup warm starts;
- no fallback.

The 40 runs are represented by one SLURM array. Adjacent task IDs are the
30 s and 300 s runs for the same seed. The launcher defaults to 20 concurrent
tasks, using at most 80 CPUs under the Borg `intermediate` QoS 96-CPU limit.

## Prerequisite: unrestricted Linux CPLEX

Borg does not currently expose a CPLEX module or executable. Install an
x86-64 Linux build of CPLEX Optimization Studio under your scratch directory
and set `CPLEX_BIN` to its executable, for example:

```bash
export CPLEX_BIN=/scratch_root/hx721/software/CPLEX_Studio2220/cplex/bin/x86-64_linux/cplex
```

The no-cost Community Edition is not sufficient: it is limited to 1,000
variables and constraints, while the 168 h model is much larger. The Borg
installation uses CPLEX 22.2.0. Rerun all 20 validation seeds on Borg and do
not mix those results with the existing local CPLEX 12.10 runs.

IBM's Linux installation instructions are:
<https://www.ibm.com/docs/en/icos/22.1.1?topic=2211-installing-cplex-optimization-studio>

IBM Academic Initiative access is described at:
<https://www.ibm.com/products/ilog-cplex-optimization-studio/pricing>

## Submit

Run these commands from the synced Borg project directory:

```bash
export CPLEX_BIN=/scratch_root/hx721/software/CPLEX_Studio2220/cplex/bin/x86-64_linux/cplex
export ARRAY_CONCURRENCY=20
export RUN_LABEL=hpc_cplex_validation_v1

DRY_RUN=1 bash hpc/launch_rolling_milp_validation.sh
bash hpc/launch_rolling_milp_validation.sh
```

The launcher first submits a 24 h execution smoke test. The 40-task validation
array starts only if that check confirms that CPLEX can solve the full-size
168 h model and that replay is valid.

For an initial concurrency/license check, use
`ARRAY_CONCURRENCY=4`. After the first four tasks run successfully, the array
throttle can be raised without resubmitting:

```bash
scontrol update JobId=<array_job_id> ArrayTaskThrottle=20
```

## Monitor and retrieve

```bash
squeue -j <check_job_id>,<array_job_id> -o "%.18i %.50j %.2t %.10M %.10l %R"
sacct -j <array_job_id> --format=JobID,JobName,Elapsed,State,ExitCode
```

Each task writes its solver and replay summary under:

```text
experiments_results/E1/rolling_milp_budget_validation_<RUN_LABEL>/<limit>s/seed_<seed>/
```

Do not submit formal-test seeds until the 30 s versus 300 s choice is locked
from these controller-validation results.
