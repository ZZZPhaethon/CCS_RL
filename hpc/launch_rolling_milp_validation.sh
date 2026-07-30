#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-mas-ccus}"
CPLEX_BIN="${CPLEX_BIN:-$(command -v cplex || true)}"
RUN_LABEL="${RUN_LABEL:-hpc_cplex_validation_v1}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-20}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$CPLEX_BIN" || ! -x "$CPLEX_BIN" ]]; then
  echo "Set CPLEX_BIN to the Linux CPLEX executable before launching." >&2
  exit 2
fi
if [[ ! "$RUN_LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_LABEL must contain only letters, digits, dots, underscores, or hyphens." >&2
  exit 2
fi
if [[ ! "$ARRAY_CONCURRENCY" =~ ^[0-9]+$ ]] ||
   (( ARRAY_CONCURRENCY < 1 || ARRAY_CONCURRENCY > 24 )); then
  echo "ARRAY_CONCURRENCY must be between 1 and 24." >&2
  echo "The intermediate QoS permits 96 CPUs and each task reserves four." >&2
  exit 2
fi

mkdir -p "$PROJECT_DIR/logs"
SBATCH_EXPORT="ALL,PROJECT_DIR=$PROJECT_DIR,CONDA_ENV=$CONDA_ENV,CPLEX_BIN=$CPLEX_BIN,RUN_LABEL=$RUN_LABEL"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'sbatch --export=%q %q\n' \
    "$SBATCH_EXPORT" \
    "$SCRIPT_DIR/submit_rolling_milp_cplex_check.sh"
  printf 'sbatch --dependency=afterok:<check_job_id> --array=%q --export=%q %q\n' \
    "0-39%$ARRAY_CONCURRENCY" \
    "$SBATCH_EXPORT" \
    "$SCRIPT_DIR/submit_rolling_milp_validation_array.sh"
  exit 0
fi

CHECK_JOB_ID="$(
  sbatch --parsable \
    --export="$SBATCH_EXPORT" \
    "$SCRIPT_DIR/submit_rolling_milp_cplex_check.sh"
)"
ARRAY_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:$CHECK_JOB_ID" \
    --array="0-39%$ARRAY_CONCURRENCY" \
    --export="$SBATCH_EXPORT" \
    "$SCRIPT_DIR/submit_rolling_milp_validation_array.sh"
)"

echo "cplex_check_job_id=$CHECK_JOB_ID"
echo "validation_array_job_id=$ARRAY_JOB_ID"
echo "array_tasks=40"
echo "array_concurrency=$ARRAY_CONCURRENCY"
echo "solver_threads_per_task=4"
echo "maximum_requested_cpus=$((ARRAY_CONCURRENCY * 4))"
