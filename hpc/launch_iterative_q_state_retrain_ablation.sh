#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
DATA_ROOT="${DATA_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"
OUT_ROOT="${OUT_ROOT:-experiments_results/E1/iterative_q_state_retrain_ablation_20260730_run01}"
MODEL_SEED="${MODEL_SEED:-0}"

cd "$PROJECT_DIR"
if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing existing ablation root: $OUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT" logs

common_export="ALL,PROJECT_DIR=$PROJECT_DIR,DATA_ROOT=$DATA_ROOT,OUT_ROOT=$OUT_ROOT,MODEL_SEED=$MODEL_SEED"
env_job=$(sbatch --parsable \
  --export="$common_export" \
  hpc/submit_iterative_q_state_retrain_env_check.sh)
env_job="${env_job%%;*}"
train_job=$(sbatch --parsable \
  --dependency=afterok:"$env_job" \
  --export="$common_export" \
  hpc/submit_iterative_q_state_retrain.sh)
train_job="${train_job%%;*}"
eval_job=$(sbatch --parsable \
  --dependency=afterok:"$train_job" \
  --export="$common_export" \
  hpc/submit_iterative_q_state_retrain_eval.sh)
eval_job="${eval_job%%;*}"

{
  printf 'env_check=%s\n' "$env_job"
  printf 'train=%s\n' "$train_job"
  printf 'eval=%s\n' "$eval_job"
} | tee "$OUT_ROOT/job_ids.txt"

{
  printf 'purpose=compare_retrained_state_feature_ablations\n'
  printf 'baseline=G60-P4 model seed 0\n'
  printf 'conditions=drop_hour_of_week,drop_all_three\n'
  printf 'drop_hour_of_week=hour_of_week\n'
  printf 'drop_all_three=hour_of_week,in_transit_fill,episode_progress\n'
  printf 'data_root=%s\n' "$DATA_ROOT"
  printf 'model_seed=%s\n' "$MODEL_SEED"
  printf 'formal_eval_seeds=9000031-9000060\n'
  printf 'formal_test_previously_accessed=true\n'
} > "$OUT_ROOT/protocol_lock.txt"
