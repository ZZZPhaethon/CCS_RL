#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
RUN_ROOT="${RUN_ROOT:-output/unified_window12/event_residual_future_ablation_20260727}"
cd "$PROJECT_DIR"
mkdir -p logs

train_job="$(
  sbatch --parsable \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT" \
    hpc/submit_event_residual_future_ablation.sh
)"
report_job="$(
  sbatch --parsable \
    --dependency=afterok:"$train_job" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT" \
    hpc/submit_event_residual_future_report.sh
)"

printf 'training_array_job=%s\n' "$train_job"
printf 'report_job=%s\n' "$report_job"
printf 'run_root=%s\n' "$RUN_ROOT"
