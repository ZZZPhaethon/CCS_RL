#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_milp_validation_20260728}"
CPLEX_BIN="${CPLEX_BIN:-/scratch_root/hx721/software/CPLEX_Studio2220/cplex/bin/x86-64_linux/cplex}"
CONCURRENCY="${CONCURRENCY:-24}"
RUN_LABEL="${RUN_LABEL:-run01}"
RESULT_ROOT="$PROJECT_DIR/experiments_results/E1/ablation_rolling_milp_h168_r24_t1200s_cplex222_seeds_9000031-9000060_${RUN_LABEL}"
LOCK_DIR="$PROJECT_DIR/experiments_results/rolling_milp_h168_r24_t1200s_9000031_9000060_${RUN_LABEL}_lock"
LOCK_PATH="$LOCK_DIR/configuration_lock.txt"
JOB_MANIFEST="$LOCK_DIR/job_manifest.txt"
SUBMIT_SCRIPT="$PROJECT_DIR/hpc/submit_rolling_milp_h168_r24_t1200s_array.sh"

if [[ "$CONCURRENCY" != "24" ]]; then
  echo "H168-T1200 concurrency is locked to 24 tasks." >&2
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
if [[ -e "$RESULT_ROOT" || -e "$LOCK_DIR" ]]; then
  echo "Refusing repeated H168-T1200 run: output or lock already exists." >&2
  exit 3
fi

cd "$PROJECT_DIR"
mkdir -p logs
mkdir -p experiments_results

EXPECTED_RUNNER_SHA="d99b76527be8a2efad0480d30cb0c908e7c4f45ad519e762d524af84133b3973"
EXPECTED_ROLLING_SHA="ac63929e8072361be7f953375a2e5965784af95209e15145029b31070f610425"
EXPECTED_CPLEX_SHA="aa7c74c2289663a6db1717396ad235cbb214a96f67d6c809df4c7e6d4985b89a"

[[ "$(sha256sum experiments/smoke_test_paper_controllers.py | awk '{print $1}')" == "$EXPECTED_RUNNER_SHA" ]]
[[ "$(sha256sum src/sim/control/rolling_milp.py | awk '{print $1}')" == "$EXPECTED_ROLLING_SHA" ]]
[[ "$(sha256sum src/sim/control/cplex_milp.py | awk '{print $1}')" == "$EXPECTED_CPLEX_SHA" ]]

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

assert protocol["protocol_version"] == 6
assert protocol["test_set_revision"]["active_range_inclusive"] == [9000031, 9000060]
assert manifest["formal_test"]["range_inclusive"] == [9000031, 9000060]
assert manifest["formal_test"]["count"] == 30
print("h168_r24_t1200s_static_configuration_check=passed")
PY

mkdir "$LOCK_DIR"
{
  printf 'run_type=rolling_time_budget_ablation\n'
  printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'formal_test_seeds=9000031-9000060\n'
  printf 'formal_test_count=30\n'
  printf 'stress_level=Medium\n'
  printf 'cplex=%s\n' "$CPLEX_BIN"
  printf 'cplex_parallel_mode=deterministic\n'
  printf 'solver_threads_per_process=4\n'
  printf 'online_episode_hours=720\n'
  printf 'forecast_context_hours=168\n'
  printf 'rolling_horizon_hours=168\n'
  printf 'rolling_replan_hours=24\n'
  printf 'rolling_time_limit_seconds_per_replan=1200\n'
  printf 'rolling_concurrency=24\n'
  printf 'rolling_memory_per_task=8G\n'
  printf 'maximum_requested_cpus=96\n'
  printf 'maximum_requested_memory=192G\n'
  printf 'slurm_time_limit=12:00:00\n'
  printf 'warm_start=greedy_with_complete_cleanup\n'
  printf 'fallback=none\n'
  printf 'interpretation=time_budget_ablation_not_replacement_for_h168_t600s_formal_result\n'
  printf 'runner_sha256=%s\n' "$EXPECTED_RUNNER_SHA"
  printf 'rolling_solver_sha256=%s\n' "$EXPECTED_ROLLING_SHA"
  printf 'full_solver_sha256=%s\n' "$EXPECTED_CPLEX_SHA"
  printf 'protocol_sha256=%s\n' "$(sha256sum experiments/protocols/unified_window_v1_paper_protocol.json | awk '{print $1}')"
  printf 'seed_manifest_sha256=%s\n' "$(sha256sum experiments/protocols/unified_window_v1_seed_manifest.json | awk '{print $1}')"
  printf 'submit_script_sha256=%s\n' "$(sha256sum hpc/submit_rolling_milp_h168_r24_t1200s_array.sh | awk '{print $1}')"
  printf 'launcher_sha256=%s\n' "$(sha256sum hpc/launch_rolling_milp_h168_r24_t1200s.sh | awk '{print $1}')"
} > "$LOCK_PATH"

PURPOSE="time_budget_ablation_E1_rolling_milp_h168_r24_cplex222_1200s"
JOB_ID=$(sbatch --parsable \
  --job-name=rolling_h168_t1200 \
  --array="0-29%$CONCURRENCY" \
  --mem=8G \
  -o "logs/rolling_h168_t1200-%A_%a.out" \
  -e "logs/rolling_h168_t1200-%A_%a.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CPLEX_BIN="$CPLEX_BIN",RESULT_ROOT="$RESULT_ROOT",PURPOSE="$PURPOSE" \
  "$SUBMIT_SCRIPT")

{
  printf 'array_job_id=%s\n' "$JOB_ID"
  printf 'array=0-29%%%s\n' "$CONCURRENCY"
  printf 'result_root=%s\n' "$RESULT_ROOT"
} | tee "$JOB_MANIFEST"
