#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
DATA_ROOT="${DATA_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"
OUT_ROOT="${OUT_ROOT:-experiments_results/E1/iterative_q_state_retrain_fixed_roots_seeds_1_2_20260730_run01}"
MODEL_SEEDS="${MODEL_SEEDS:-1 2}"

cd "$PROJECT_DIR"
if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing existing ablation root: $OUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT" logs

env_export="ALL,PROJECT_DIR=$PROJECT_DIR,DATA_ROOT=$DATA_ROOT,OUT_ROOT=$OUT_ROOT"
env_job=$(sbatch --parsable \
  --export="$env_export" \
  hpc/submit_iterative_q_state_retrain_env_check.sh)
env_job="${env_job%%;*}"

{
  printf 'env_check=%s\n' "$env_job"
} > "$OUT_ROOT/job_ids.txt"

for model_seed in $MODEL_SEEDS; do
  common_export="ALL,PROJECT_DIR=$PROJECT_DIR,DATA_ROOT=$DATA_ROOT,OUT_ROOT=$OUT_ROOT,MODEL_SEED=$model_seed"
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
    printf 'seed_%s_train=%s\n' "$model_seed" "$train_job"
    printf 'seed_%s_eval=%s\n' "$model_seed" "$eval_job"
  } >> "$OUT_ROOT/job_ids.txt"
done

{
  printf 'purpose=fixed-root model-seed replication for state-feature deletion\n'
  printf 'conditions=drop_hour_of_week,drop_all_three\n'
  printf 'drop_hour_of_week=hour_of_week\n'
  printf 'drop_all_three=hour_of_week,in_transit_fill,episode_progress\n'
  printf 'data_root=%s\n' "$DATA_ROOT"
  printf 'root_policy_chain=original G60-P4 G0-G3; roots held fixed\n'
  printf 'model_seeds=%s\n' "$MODEL_SEEDS"
  printf 'formal_eval_seeds=9000031-9000060\n'
  printf 'formal_test_previously_accessed=true\n'
} > "$OUT_ROOT/protocol_lock.txt"

cat "$OUT_ROOT/job_ids.txt"
