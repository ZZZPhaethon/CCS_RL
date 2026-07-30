#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_e1_20260728}"
RESULT_ROOT="${RESULT_ROOT:-experiments_results/E1/matched_learning_algorithms_20260728}"

env_job=$(
  sbatch --parsable \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR" \
    hpc/submit_e1_learning_env_check.sh
)
central_job=$(
  sbatch --parsable \
    --dependency=afterok:"$env_job" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RESULT_ROOT="$RESULT_ROOT" \
    hpc/submit_e1_centralized_ppo_array.sh
)
event_job=$(
  sbatch --parsable \
    --dependency=afterok:"$env_job" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RESULT_ROOT="$RESULT_ROOT" \
    hpc/submit_e1_event_residual_ppo_array.sh
)

mkdir -p "$RESULT_ROOT"
{
  printf 'environment_check=%s\n' "$env_job"
  printf 'centralized_maskable_ppo_array=%s\n' "$central_job"
  printf 'event_residual_ppo_array=%s\n' "$event_job"
} > "$RESULT_ROOT/job_ids.txt"
printf '%s\n' \
  "environment_check=$env_job" \
  "centralized_maskable_ppo_array=$central_job" \
  "event_residual_ppo_array=$event_job"
