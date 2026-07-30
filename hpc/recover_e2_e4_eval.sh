#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"
E1_REPLICATION_ROOT="${E1_REPLICATION_ROOT:-experiments_results/E1/training_iterative_action_q_g60_p4_full_model_seeds_1_2_20260729}"
E2_ROOT="experiments_results/E2"
E4_ROOT="experiments_results/E4"
E2_MATCHED_EVAL_JOBS="${E2_MATCHED_EVAL_JOBS:-34540:34541:34542}"

cd "$PROJECT_DIR"
mkdir -p logs

submit_job() {
  local submitted
  submitted=$(sbatch --parsable "$@")
  printf '%s\n' "${submitted%%;*}"
}

checkpoint_for() {
  local model_seed="$1"
  local stage="$2"
  if [[ "$model_seed" == "0" ]]; then
    printf '%s\n' "$PROJECT_DIR/$SOURCE_RUN_ROOT/$stage/iterative_action_q.pt"
  else
    printf '%s\n' "$PROJECT_DIR/$E1_REPLICATION_ROOT/model_seed_${model_seed}/$stage/iterative_action_q.pt"
  fi
}

join_jobs() {
  local IFS=:
  printf '%s\n' "$*"
}

e2_jobs=()
for stage in p1 p2 p3 p4; do
  for model_seed in 0 1 2; do
    out_dir="$PROJECT_DIR/$E2_ROOT/formal_iterative_q_stages_seeds_9000031-9000060_run01/$stage/model_seed_${model_seed}"
    if [[ -e "$out_dir" ]]; then
      echo "Refusing existing E2 output: $out_dir" >&2
      exit 2
    fi
    job=$(submit_job \
      --job-name="e2r_${stage}_s${model_seed}" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",CHECKPOINT="$(checkpoint_for "$model_seed" "$stage")",OUT_DIR="$out_dir",EVAL_NAME="e2_${stage}_s${model_seed}",STRESS_LEVEL=medium \
      hpc/submit_locked_iterative_q_eval.sh)
    e2_jobs+=("$job")
  done
done

e4_jobs=()
for stress in low high; do
  for model_seed in 0 1 2; do
    out_dir="$PROJECT_DIR/$E4_ROOT/formal_stress_seeds_9000031-9000060_run01/$stress/model_seed_${model_seed}"
    if [[ -e "$out_dir" ]]; then
      echo "Refusing existing E4 output: $out_dir" >&2
      exit 2
    fi
    job=$(submit_job \
      --job-name="e4r_${stress}_s${model_seed}" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",CHECKPOINT="$(checkpoint_for "$model_seed" p4)",OUT_DIR="$out_dir",EVAL_NAME="e4_${stress}_s${model_seed}",STRESS_LEVEL="$stress" \
      hpc/submit_locked_iterative_q_eval.sh)
    e4_jobs+=("$job")
  done
done

e2_all_jobs=("${e2_jobs[@]}")
IFS=: read -r -a matched_jobs <<< "$E2_MATCHED_EVAL_JOBS"
e2_all_jobs+=("${matched_jobs[@]}")
e2_aggregate=$(submit_job \
  --dependency=afterok:"$(join_jobs "${e2_all_jobs[@]}")" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",EXPERIMENT=E2 \
  hpc/submit_e2_e3_e4_aggregate.sh)
e4_aggregate=$(submit_job \
  --dependency=afterok:"$(join_jobs "${e4_jobs[@]}")" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",EXPERIMENT=E4 \
  hpc/submit_e2_e3_e4_aggregate.sh)

{
  printf 'e2_stage_eval_recovery_jobs=%s\n' "$(join_jobs "${e2_jobs[@]}")"
  printf 'e2_aggregate_recovery=%s\n' "$e2_aggregate"
  printf 'e4_eval_recovery_jobs=%s\n' "$(join_jobs "${e4_jobs[@]}")"
  printf 'e4_aggregate_recovery=%s\n' "$e4_aggregate"
} | tee -a experiments_results/e2_e3_e4_job_manifest.txt
