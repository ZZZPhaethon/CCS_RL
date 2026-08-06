#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
REFRESH_ROOT="${REFRESH_ROOT:-experiments_results/refresh_no_hour_20260730}"
FORMAL_MODEL_ROOT="${FORMAL_MODEL_ROOT:-output/E1_hour_removed_formal_models}"
FORMAL_RESULT_ROOT="${FORMAL_RESULT_ROOT:-output/E1_hour_removed_formal_results}"
ONE_SHOT_ROOT="${ONE_SHOT_ROOT:-experiments_results/E2/training_one_shot_hour_removed_budget_matched_20260730_run01}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"
E3_DATA_ROOT="${E3_DATA_ROOT:-experiments_results/E3/training_future_information_run01}"
E3_ROOT="$REFRESH_ROOT/E3"
E4_ROOT="$REFRESH_ROOT/E4"

cd "$PROJECT_DIR"
mkdir -p logs

if [[ -e "$REFRESH_ROOT" ]]; then
  echo "Refusing to overwrite refresh root: $REFRESH_ROOT" >&2
  exit 2
fi
for model_seed in 0 1 2; do
  test -s "$FORMAL_MODEL_ROOT/g60_p4_model_seed_${model_seed}/iterative_action_q.pt"
  test -s "$FORMAL_RESULT_ROOT/model_seed_${model_seed}/evaluation.csv"
  test -s "$ONE_SHOT_ROOT/model_seed_${model_seed}/p1/iterative_action_q.pt"
done
for stage_index in 0 1 2 3; do
  test -s "$SOURCE_RUN_ROOT/g${stage_index}/train_merged.npz"
  test -s "$SOURCE_RUN_ROOT/g${stage_index}/validation_merged.npz"
  test -s "$E3_DATA_ROOT/augmented_data/g${stage_index}/train_forecast168.npz"
  test -s "$E3_DATA_ROOT/augmented_data/g${stage_index}/validation_forecast168.npz"
done

submit_job() {
  local submitted
  submitted=$(sbatch --parsable "$@")
  printf '%s\n' "${submitted%%;*}"
}

join_jobs() {
  local IFS=:
  printf '%s\n' "$*"
}

for model_seed in 0 1 2; do
  destination="$E3_ROOT/formal_future_information_seeds_9000031-9000060_run01/structured_summary_168/model_seed_${model_seed}"
  mkdir -p "$destination"
  cp -a "$FORMAL_RESULT_ROOT/model_seed_${model_seed}/." "$destination/"
done

e3_train_job=$(submit_job \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",SOURCE_RUN_ROOT="$SOURCE_RUN_ROOT",E3_ROOT="$E3_ROOT/training_future_information_run01",DATA_ROOT="$E3_DATA_ROOT",EXCLUDE_STATE_FEATURES=hour_of_week \
  hpc/submit_e3_future_information_train.sh)

e3_eval_jobs=()
for variant in state_only full_sequence_168; do
  for model_seed in 0 1 2; do
    checkpoint="$PROJECT_DIR/$E3_ROOT/training_future_information_run01/$variant/model_seed_${model_seed}/p4/iterative_action_q.pt"
    out_dir="$PROJECT_DIR/$E3_ROOT/formal_future_information_seeds_9000031-9000060_run01/$variant/model_seed_${model_seed}"
    job=$(submit_job \
      --dependency=afterok:"$e3_train_job" \
      --job-name="e3_nohour_${variant:0:4}_s${model_seed}" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",CHECKPOINT="$checkpoint",OUT_DIR="$out_dir",EVAL_NAME="e3_nohour_${variant}_s${model_seed}",STRESS_LEVEL=medium \
      hpc/submit_locked_iterative_q_eval.sh)
    e3_eval_jobs+=("$job")
  done
done

e4_eval_jobs=()
for family in iterative one_shot; do
  for stress in low medium high; do
    for model_seed in 0 1 2; do
      if [[ "$family" == "iterative" ]]; then
        checkpoint="$PROJECT_DIR/$FORMAL_MODEL_ROOT/g60_p4_model_seed_${model_seed}/iterative_action_q.pt"
        out_dir="$PROJECT_DIR/$E4_ROOT/formal_stress_seeds_9000031-9000060_run01/$stress/model_seed_${model_seed}"
      else
        checkpoint="$PROJECT_DIR/$ONE_SHOT_ROOT/model_seed_${model_seed}/p1/iterative_action_q.pt"
        out_dir="$PROJECT_DIR/$E4_ROOT/formal_one_shot_stress_seeds_9000031-9000060_run01/$stress/model_seed_${model_seed}"
      fi
      job=$(submit_job \
        --job-name="e4_nohour_${family:0:4}_${stress}_s${model_seed}" \
        --export=ALL,PROJECT_DIR="$PROJECT_DIR",CHECKPOINT="$checkpoint",OUT_DIR="$out_dir",EVAL_NAME="e4_nohour_${family}_${stress}_s${model_seed}",STRESS_LEVEL="$stress" \
        hpc/submit_locked_iterative_q_eval.sh)
      e4_eval_jobs+=("$job")
    done
  done
done

manifest="$REFRESH_ROOT/job_manifest.txt"
{
  printf 'formal_model_root=%s\n' "$FORMAL_MODEL_ROOT"
  printf 'one_shot_root=%s\n' "$ONE_SHOT_ROOT"
  printf 'excluded_state_features=hour_of_week\n'
  printf 'e3_train=%s\n' "$e3_train_job"
  printf 'e3_eval_jobs=%s\n' "$(join_jobs "${e3_eval_jobs[@]}")"
  printf 'e4_eval_jobs=%s\n' "$(join_jobs "${e4_eval_jobs[@]}")"
} | tee "$manifest"
