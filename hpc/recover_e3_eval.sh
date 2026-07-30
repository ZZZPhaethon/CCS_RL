#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
TRAIN_ROOT="experiments_results/E3/training_future_information_run01"
FORMAL_ROOT="experiments_results/E3/formal_future_information_seeds_9000031-9000060_run01"

cd "$PROJECT_DIR"
mkdir -p logs

submit_job() {
  local submitted
  submitted=$(sbatch --parsable "$@")
  printf '%s\n' "${submitted%%;*}"
}

eval_jobs=()
for variant in state_only full_sequence_168; do
  for model_seed in 0 1 2; do
    checkpoint="$PROJECT_DIR/$TRAIN_ROOT/$variant/model_seed_${model_seed}/p4/iterative_action_q.pt"
    out_dir="$PROJECT_DIR/$FORMAL_ROOT/$variant/model_seed_${model_seed}"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing checkpoint: $checkpoint" >&2
      exit 2
    fi
    if [[ -e "$out_dir" ]]; then
      echo "Refusing existing formal output: $out_dir" >&2
      exit 2
    fi
    job=$(submit_job \
      --job-name="e3_${variant}_${model_seed}" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",CHECKPOINT="$checkpoint",OUT_DIR="$out_dir",EVAL_NAME="e3_${variant}_s${model_seed}",STRESS_LEVEL=medium \
      hpc/submit_locked_iterative_q_eval.sh)
    eval_jobs+=("$job")
  done
done

IFS=:
aggregate_job=$(submit_job \
  --dependency=afterok:"${eval_jobs[*]}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",EXPERIMENT=E3 \
  hpc/submit_e2_e3_e4_aggregate.sh)
unset IFS

{
  local_ifs=:
  printf 'e3_eval_recovery_jobs=%s\n' "${eval_jobs[*]}"
  printf 'e3_aggregate_recovery=%s\n' "$aggregate_job"
} | tee -a experiments_results/e2_e3_e4_job_manifest.txt
