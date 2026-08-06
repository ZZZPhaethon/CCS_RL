#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SOURCE_ROOT="${SOURCE_ROOT:-experiments_results/E2/training_one_shot_matched_run01}"
OUT_ROOT="${OUT_ROOT:-experiments_results/E2/training_one_shot_hour_removed_budget_matched_20260730_run01}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$SOURCE_ROOT/g0/train_merged_matched.npz}"
VALIDATION_DATA_PATH="${VALIDATION_DATA_PATH:-$SOURCE_ROOT/g0/validation_merged.npz}"
SOURCE_BUDGET="${SOURCE_BUDGET:-$SOURCE_ROOT/budget.json}"

cd "$PROJECT_DIR"
if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing existing one-shot hour-removed root: $OUT_ROOT" >&2
  exit 2
fi
test -s "$TRAIN_DATA_PATH"
test -s "$VALIDATION_DATA_PATH"
test -s "$SOURCE_BUDGET"
mkdir -p "$OUT_ROOT" logs

submit_job() {
  local submitted
  submitted=$(sbatch --parsable "$@")
  printf '%s\n' "${submitted%%;*}"
}

common_export="ALL,PROJECT_DIR=$PROJECT_DIR,TRAIN_DATA_PATH=$TRAIN_DATA_PATH,VALIDATION_DATA_PATH=$VALIDATION_DATA_PATH,SOURCE_BUDGET=$SOURCE_BUDGET"
env_job=$(submit_job \
  --export="$common_export" \
  hpc/submit_iterative_q_one_shot_hour_removed_env_check.sh)

train_export="$common_export,RUN_ROOT=$OUT_ROOT,EXCLUDE_STATE_FEATURES=hour_of_week"
train_job=$(submit_job \
  --dependency=afterok:"$env_job" \
  --job-name=iq1shot_hour_train \
  --export="$train_export" \
  hpc/submit_e2_one_shot_train.sh)

eval_jobs=()
for model_seed in 0 1 2; do
  checkpoint="$PROJECT_DIR/$OUT_ROOT/model_seed_${model_seed}/p1/iterative_action_q.pt"
  eval_dir="$PROJECT_DIR/$OUT_ROOT/eval/model_seed_${model_seed}"
  eval_job=$(submit_job \
    --dependency=afterok:"$train_job" \
    --job-name="iq1shot_h_s${model_seed}" \
    --export="ALL,PROJECT_DIR=$PROJECT_DIR,CHECKPOINT=$checkpoint,OUT_DIR=$eval_dir,EVAL_NAME=one_shot_nohour_s${model_seed},STRESS_LEVEL=medium" \
    hpc/submit_locked_iterative_q_eval.sh)
  eval_jobs+=("$eval_job")
done

{
  printf 'env_check=%s\n' "$env_job"
  printf 'train_array=%s\n' "$train_job"
  for model_seed in 0 1 2; do
    printf 'seed_%s_eval=%s\n' "$model_seed" "${eval_jobs[$model_seed]}"
  done
} > "$OUT_ROOT/job_ids.txt"

{
  printf 'purpose=Greedy-only one-shot hour_of_week deletion at matched physical-simulator budget\n'
  printf 'source_root=%s\n' "$SOURCE_ROOT"
  printf 'train_data=%s\n' "$TRAIN_DATA_PATH"
  printf 'validation_data=%s\n' "$VALIDATION_DATA_PATH"
  printf 'source_budget=%s\n' "$SOURCE_BUDGET"
  printf 'excluded_state_features=hour_of_week\n'
  printf 'initial_checkpoint=none\n'
  printf 'policy_iteration=none\n'
  printf 'roll_in_policy=Greedy only\n'
  printf 'model_seeds=0 1 2\n'
  printf 'formal_eval_seeds=9000031-9000060\n'
  printf 'formal_test_previously_accessed=true\n'
} > "$OUT_ROOT/protocol_lock.txt"
cp "$SOURCE_BUDGET" "$OUT_ROOT/source_budget.json"

cat "$OUT_ROOT/job_ids.txt"
