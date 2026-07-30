#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
RUN_ROOT="experiments_results/E2/training_one_shot_matched_run01"
FORMAL_ROOT="experiments_results/E2/formal_one_shot_matched_seeds_9000031-9000060_run01"

cd "$PROJECT_DIR"
mkdir -p logs

submit_job() {
  local submitted
  submitted=$(sbatch --parsable "$@")
  printf '%s\n' "${submitted%%;*}"
}

trim_job=$(submit_job \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT" \
  hpc/submit_e2_one_shot_trim_data.sh)
merge_job=$(submit_job \
  --dependency=afterok:"$trim_job" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT" \
  hpc/submit_e2_one_shot_merge_recovery.sh)
train_job=$(submit_job \
  --dependency=afterok:"$merge_job" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",TRAIN_DATA_PATH="$RUN_ROOT/g0/train_merged_matched.npz" \
  hpc/submit_e2_one_shot_train.sh)
eval_jobs=()
for model_seed in 0 1 2; do
  out_dir="$PROJECT_DIR/$FORMAL_ROOT/model_seed_${model_seed}"
  if [[ -e "$out_dir" ]]; then
    echo "Refusing existing matched output: $out_dir" >&2
    exit 2
  fi
  job=$(submit_job \
    --dependency=afterok:"$train_job" \
    --job-name="e2m_s${model_seed}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",CHECKPOINT="$PROJECT_DIR/$RUN_ROOT/model_seed_${model_seed}/p1/iterative_action_q.pt",OUT_DIR="$out_dir",EVAL_NAME="e2_one_shot_matched_s${model_seed}",STRESS_LEVEL=medium \
    hpc/submit_locked_iterative_q_eval.sh)
  eval_jobs+=("$job")
done
IFS=:
aggregate_job=$(submit_job \
  --dependency=afterok:"${eval_jobs[*]}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",EXPERIMENT=E2 \
  hpc/submit_e2_e3_e4_aggregate.sh)
unset IFS

{
  printf 'e2_budget_trim=%s\n' "$trim_job"
  printf 'e2_merge_recovery=%s\n' "$merge_job"
  printf 'e2_train_recovery=%s\n' "$train_job"
  local_ifs=:
  printf 'e2_matched_eval_recovery_jobs=%s\n' "${eval_jobs[*]}"
  printf 'e2_aggregate_budget_recovery=%s\n' "$aggregate_job"
} | tee -a experiments_results/e2_e3_e4_job_manifest.txt
