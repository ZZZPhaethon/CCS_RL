#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_milp_validation_20260728}"
CPLEX_BIN="${CPLEX_BIN:-/scratch_root/hx721/software/CPLEX_Studio2220/cplex/bin/x86-64_linux/cplex}"
ROLLING_CONCURRENCY="${ROLLING_CONCURRENCY:-10}"
FULL_CONCURRENCY="${FULL_CONCURRENCY:-8}"
ROLLING_ROOT="$PROJECT_DIR/experiments_results/E1/algorithms/formal_rolling_milp_h168_r24_t300s_cplex222_seeds_9000031-9000060_run02"
FULL_ROOT="$PROJECT_DIR/experiments_results/E5/full_milp_formal_9000031_9000060_cplex222_7200s_attempt2"
LOCK_DIR="$PROJECT_DIR/experiments_results/formal_milp_9000031_9000060_attempt2_lock"
LOCK_PATH="$LOCK_DIR/configuration_lock.txt"
JOB_MANIFEST="$LOCK_DIR/job_manifest.txt"

if [[ "$ROLLING_CONCURRENCY" != "10" || "$FULL_CONCURRENCY" != "8" ]]; then
  echo "Formal concurrency is locked to Rolling=10 and Full=8." >&2
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
if [[ -e "$ROLLING_ROOT" || -e "$FULL_ROOT" || -e "$LOCK_DIR" ]]; then
  echo "Refusing repeated formal-test access: output or lock already exists." >&2
  exit 3
fi

cd "$PROJECT_DIR"
mkdir -p logs

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
full = protocol["milp_compute_protocol"]["full_horizon_milp"]

assert protocol["protocol_version"] == 6
assert protocol["test_set_revision"]["active_range_inclusive"] == [9000031, 9000060]
assert rolling["planning_horizon_hours"] == 168
assert rolling["replan_interval_hours"] == 24
assert rolling["formal_time_limit_seconds_per_replan"] == 300
assert rolling["fallback"] == "none"
assert full["time_limit_seconds_per_seed"] == 7200
assert full["fallback"] == "none"
assert protocol["milp_compute_protocol"]["solver_threads_per_process"] == 4
assert manifest["formal_test"]["range_inclusive"] == [9000031, 9000060]
assert manifest["formal_test"]["count"] == 30
print("static_formal_configuration_check=passed")
PY

mkdir "$LOCK_DIR"
{
  printf 'access=authorized_one_shot_formal_test\n'
  printf 'attempt=2\n'
  printf 'supersedes_aborted_rolling_job_id=33494\n'
  printf 'supersedes_aborted_full_job_id=33495\n'
  printf 'attempt_1_abort_reason=warm_start_roundoff_exceeded_variable_upper_bound_before_solver_start\n'
  printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'formal_test_seeds=9000031-9000060\n'
  printf 'formal_test_count=30\n'
  printf 'stress_level=Medium\n'
  printf 'cplex=%s\n' "$CPLEX_BIN"
  printf 'cplex_parallel_mode=deterministic\n'
  printf 'solver_threads_per_process=4\n'
  printf 'rolling_horizon_hours=168\n'
  printf 'rolling_replan_hours=24\n'
  printf 'rolling_time_limit_seconds_per_replan=300\n'
  printf 'rolling_concurrency=10\n'
  printf 'rolling_memory_per_task=24G\n'
  printf 'full_horizon_hours=720\n'
  printf 'full_time_limit_seconds_per_seed=7200\n'
  printf 'full_concurrency=8\n'
  printf 'full_memory_per_task=64G\n'
  printf 'warm_start=greedy_with_complete_cleanup\n'
  printf 'fallback=none\n'
  printf 'runner_sha256=%s\n' "$(sha256sum experiments/smoke_test_paper_controllers.py | awk '{print $1}')"
  printf 'rolling_solver_sha256=%s\n' "$(sha256sum src/sim/control/rolling_milp.py | awk '{print $1}')"
  printf 'full_solver_sha256=%s\n' "$(sha256sum src/sim/control/cplex_milp.py | awk '{print $1}')"
  printf 'protocol_sha256=%s\n' "$(sha256sum experiments/protocols/unified_window_v1_paper_protocol.json | awk '{print $1}')"
  printf 'seed_manifest_sha256=%s\n' "$(sha256sum experiments/protocols/unified_window_v1_seed_manifest.json | awk '{print $1}')"
  printf 'rolling_sbatch_sha256=%s\n' "$(sha256sum hpc/submit_rolling_milp_formal_array.sh | awk '{print $1}')"
  printf 'full_sbatch_sha256=%s\n' "$(sha256sum hpc/submit_full_milp_formal_array.sh | awk '{print $1}')"
} > "$LOCK_PATH"

ROLLING_JOB_ID=$(sbatch --parsable \
  --array="0-29%$ROLLING_CONCURRENCY" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CPLEX_BIN="$CPLEX_BIN",RESULT_ROOT="$ROLLING_ROOT" \
  hpc/submit_rolling_milp_formal_array.sh)
FULL_JOB_ID=$(sbatch --parsable \
  --array="0-29%$FULL_CONCURRENCY" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CPLEX_BIN="$CPLEX_BIN",RESULT_ROOT="$FULL_ROOT" \
  hpc/submit_full_milp_formal_array.sh)

{
  printf 'rolling_array_job_id=%s\n' "$ROLLING_JOB_ID"
  printf 'full_array_job_id=%s\n' "$FULL_JOB_ID"
  printf 'rolling_array=0-29%%%s\n' "$ROLLING_CONCURRENCY"
  printf 'full_array=0-29%%%s\n' "$FULL_CONCURRENCY"
  printf 'rolling_result_root=%s\n' "$ROLLING_ROOT"
  printf 'full_result_root=%s\n' "$FULL_ROOT"
} | tee "$JOB_MANIFEST"
