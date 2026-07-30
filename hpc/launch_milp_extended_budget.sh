#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_milp_validation_20260728}"
CPLEX_BIN="${CPLEX_BIN:-/scratch_root/hx721/software/CPLEX_Studio2220/cplex/bin/x86-64_linux/cplex}"
ROLLING_CONCURRENCY="${ROLLING_CONCURRENCY:-9}"
FULL_CONCURRENCY="${FULL_CONCURRENCY:-15}"
RUN_LABEL="${RUN_LABEL:-run01}"
ROLLING_ROOT="$PROJECT_DIR/experiments_results/E1/rolling_milp_extended_h168_r24_t600s_cplex222_seeds_9000031-9000060_${RUN_LABEL}"
FULL_ROOT="$PROJECT_DIR/experiments_results/E5/full_milp_extended_h720_t18000s_cplex222_seeds_9000031-9000060_${RUN_LABEL}"
LOCK_DIR="$PROJECT_DIR/experiments_results/milp_extended_600s_18000s_9000031_9000060_${RUN_LABEL}_lock"
LOCK_PATH="$LOCK_DIR/configuration_lock.txt"
JOB_MANIFEST="$LOCK_DIR/job_manifest.txt"
SUBMIT_SCRIPT="$PROJECT_DIR/hpc/submit_milp_extended_budget_array.sh"

if [[ "$ROLLING_CONCURRENCY" != "9" || "$FULL_CONCURRENCY" != "15" ]]; then
  echo "Extended-budget concurrency is locked to Rolling=9 and Full=15." >&2
  exit 2
fi
if [[ ! "$RUN_LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_LABEL must contain only letters, digits, dots, underscores, or hyphens." >&2
  exit 2
fi
if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "PROJECT_DIR does not exist: $PROJECT_DIR" >&2
  exit 2
fi
if [[ ! -x "$CPLEX_BIN" ]]; then
  echo "CPLEX_BIN is not executable: $CPLEX_BIN" >&2
  exit 2
fi
if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "Submission script does not exist: $SUBMIT_SCRIPT" >&2
  exit 2
fi
if [[ -e "$ROLLING_ROOT" || -e "$FULL_ROOT" || -e "$LOCK_DIR" ]]; then
  echo "Refusing repeated extended-budget run: output or lock already exists." >&2
  exit 3
fi

cd "$PROJECT_DIR"
mkdir -p logs
mkdir -p experiments_results

python3 - <<'PY'
import json
from pathlib import Path

protocol = json.loads(
    Path("experiments/protocols/unified_window_v1_paper_protocol.json").read_text(
        encoding="utf-8"
    )
)
manifest = json.loads(
    Path("experiments/protocols/unified_window_v1_seed_manifest.json").read_text(
        encoding="utf-8"
    )
)
rolling = protocol["milp_compute_protocol"]["rolling_milp"]

assert protocol["protocol_version"] == 6
assert protocol["test_set_revision"]["active_range_inclusive"] == [9000031, 9000060]
assert rolling["planning_horizon_hours"] == 168
assert rolling["replan_interval_hours"] == 24
assert rolling["fallback"] == "none"
assert protocol["milp_compute_protocol"]["full_horizon_milp"]["fallback"] == "none"
assert protocol["milp_compute_protocol"]["solver_threads_per_process"] == 4
assert manifest["formal_test"]["range_inclusive"] == [9000031, 9000060]
assert manifest["formal_test"]["count"] == 30
print("extended_budget_static_configuration_check=passed")
PY

mkdir "$LOCK_DIR"
{
  printf 'run_type=extended_budget_sensitivity\n'
  printf 'base_protocol=unified_window_v1_version_6\n'
  printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'formal_test_seeds=9000031-9000060\n'
  printf 'formal_test_count=30\n'
  printf 'stress_level=Medium\n'
  printf 'cplex=%s\n' "$CPLEX_BIN"
  printf 'cplex_parallel_mode=deterministic\n'
  printf 'solver_threads_per_process=4\n'
  printf 'rolling_horizon_hours=168\n'
  printf 'rolling_replan_hours=24\n'
  printf 'rolling_time_limit_seconds_per_replan=600\n'
  printf 'rolling_concurrency=9\n'
  printf 'rolling_memory_per_task=8G\n'
  printf 'full_horizon_hours=720\n'
  printf 'full_time_limit_seconds_per_seed=18000\n'
  printf 'full_concurrency=15\n'
  printf 'full_memory_per_task=32G\n'
  printf 'maximum_concurrent_tasks=24\n'
  printf 'maximum_requested_cpus=96\n'
  printf 'maximum_requested_memory=552G\n'
  printf 'slurm_time_limit=06:30:00\n'
  printf 'warm_start=greedy_with_complete_cleanup\n'
  printf 'fallback=none\n'
  printf 'runner_sha256=%s\n' "$(sha256sum experiments/smoke_test_paper_controllers.py | awk '{print $1}')"
  printf 'rolling_solver_sha256=%s\n' "$(sha256sum src/sim/control/rolling_milp.py | awk '{print $1}')"
  printf 'full_solver_sha256=%s\n' "$(sha256sum src/sim/control/cplex_milp.py | awk '{print $1}')"
  printf 'protocol_sha256=%s\n' "$(sha256sum experiments/protocols/unified_window_v1_paper_protocol.json | awk '{print $1}')"
  printf 'seed_manifest_sha256=%s\n' "$(sha256sum experiments/protocols/unified_window_v1_seed_manifest.json | awk '{print $1}')"
  printf 'submit_script_sha256=%s\n' "$(sha256sum hpc/submit_milp_extended_budget_array.sh | awk '{print $1}')"
  printf 'launcher_sha256=%s\n' "$(sha256sum hpc/launch_milp_extended_budget.sh | awk '{print $1}')"
} > "$LOCK_PATH"

ROLLING_PURPOSE="extended_budget_sensitivity_E1_rolling_milp_cplex222_600s"
FULL_PURPOSE="extended_budget_sensitivity_E5_full_milp_cplex222_18000s"

ROLLING_JOB_ID=$(sbatch --parsable \
  --job-name=rolling_milp_600s \
  --array="0-29%$ROLLING_CONCURRENCY" \
  --mem=8G \
  -o "logs/rolling_milp_600s-%A_%a.out" \
  -e "logs/rolling_milp_600s-%A_%a.err" \
  --export=ALL,MODE=rolling,PROJECT_DIR="$PROJECT_DIR",CPLEX_BIN="$CPLEX_BIN",RESULT_ROOT="$ROLLING_ROOT",PURPOSE="$ROLLING_PURPOSE" \
  "$SUBMIT_SCRIPT")

FULL_JOB_ID=$(sbatch --parsable \
  --job-name=full_milp_5h \
  --array="0-29%$FULL_CONCURRENCY" \
  --mem=32G \
  -o "logs/full_milp_5h-%A_%a.out" \
  -e "logs/full_milp_5h-%A_%a.err" \
  --export=ALL,MODE=full,PROJECT_DIR="$PROJECT_DIR",CPLEX_BIN="$CPLEX_BIN",RESULT_ROOT="$FULL_ROOT",PURPOSE="$FULL_PURPOSE" \
  "$SUBMIT_SCRIPT")

{
  printf 'rolling_array_job_id=%s\n' "$ROLLING_JOB_ID"
  printf 'full_array_job_id=%s\n' "$FULL_JOB_ID"
  printf 'rolling_array=0-29%%%s\n' "$ROLLING_CONCURRENCY"
  printf 'full_array=0-29%%%s\n' "$FULL_CONCURRENCY"
  printf 'rolling_result_root=%s\n' "$ROLLING_ROOT"
  printf 'full_result_root=%s\n' "$FULL_ROOT"
} | tee "$JOB_MANIFEST"
