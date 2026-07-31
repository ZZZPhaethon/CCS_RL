#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_e1_20260728}"
SOURCE_ROOT="${SOURCE_ROOT:-experiments_results/E1/hourly_ppo_gpu_20260728/centralized_maskable_ppo}"
RESULT_ROOT="${RESULT_ROOT:-experiments_results/E1/formal_hourly_centralized_maskable_ppo_seeds_9000031-9000060_run02}"

cd "$PROJECT_DIR"
mkdir -p logs
if [[ -e "$RESULT_ROOT" ]]; then
  printf 'Refusing output collision: %s\n' "$RESULT_ROOT" >&2
  exit 2
fi
mkdir -p "$RESULT_ROOT"

smoke_job=$(
  sbatch --parsable \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",SOURCE_ROOT="$SOURCE_ROOT",RESULT_ROOT="$RESULT_ROOT" \
    hpc/submit_e1_hourly_ppo_formal_smoke.sh
)
formal_job=$(
  sbatch --parsable \
    --dependency=afterok:"$smoke_job" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",SOURCE_ROOT="$SOURCE_ROOT",RESULT_ROOT="$RESULT_ROOT" \
    hpc/submit_e1_hourly_ppo_formal_array.sh
)

{
  printf 'smoke_job=%s\n' "$smoke_job"
  printf 'formal_array_job=%s\n' "$formal_job"
} > "$RESULT_ROOT/job_ids.txt"
printf 'smoke_job=%s\nformal_array_job=%s\n' \
  "$smoke_job" "$formal_job"
