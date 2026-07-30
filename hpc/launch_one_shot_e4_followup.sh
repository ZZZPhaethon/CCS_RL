#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
TRAIN_ROOT="experiments_results/E2/training_one_shot_matched_run01"
MEDIUM_ROOT="experiments_results/E2/formal_one_shot_matched_seeds_9000031-9000060_run01"
OUT_ROOT="experiments_results/E4/formal_one_shot_stress_seeds_9000031-9000060_run01"

cd "$PROJECT_DIR"
mkdir -p logs
if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing existing one-shot stress root: $OUT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT/medium"
for model_seed in 0 1 2; do
  checkpoint="$PROJECT_DIR/$TRAIN_ROOT/model_seed_${model_seed}/p1/iterative_action_q.pt"
  source_dir="$MEDIUM_ROOT/model_seed_${model_seed}"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing one-shot checkpoint: $checkpoint" >&2
    exit 2
  fi
  if [[ ! -f "$source_dir/evaluation.csv" || ! -f "$source_dir/summary.json" ]]; then
    echo "Missing one-shot medium result: $source_dir" >&2
    exit 2
  fi
  cp -r "$source_dir" "$OUT_ROOT/medium/model_seed_${model_seed}"
done

submit_job() {
  local submitted
  submitted=$(sbatch --parsable "$@")
  printf '%s\n' "${submitted%%;*}"
}

eval_jobs=()
for stress in low high; do
  for model_seed in 0 1 2; do
    checkpoint="$PROJECT_DIR/$TRAIN_ROOT/model_seed_${model_seed}/p1/iterative_action_q.pt"
    out_dir="$PROJECT_DIR/$OUT_ROOT/$stress/model_seed_${model_seed}"
    job=$(submit_job \
      --job-name="one_${stress}_s${model_seed}" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",CHECKPOINT="$checkpoint",OUT_DIR="$out_dir",EVAL_NAME="one_shot_${stress}_s${model_seed}",STRESS_LEVEL="$stress" \
      hpc/submit_locked_iterative_q_eval.sh)
    eval_jobs+=("$job")
  done
done

IFS=:
aggregate_job=$(submit_job \
  --dependency=afterok:"${eval_jobs[*]}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",FOLLOWUP_ANALYSIS=one_shot_e4 \
  hpc/submit_e2_followup_aggregate.sh)
unset IFS

{
  printf 'one_shot_e4_jobs=%s\n' "${eval_jobs[*]}"
  printf 'one_shot_e4_aggregate=%s\n' "$aggregate_job"
} | tee -a experiments_results/e2_followup_job_manifest.txt
