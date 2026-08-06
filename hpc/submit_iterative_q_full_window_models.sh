#!/usr/bin/env bash
#SBATCH --job-name=iqfull720_start
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH -o logs/iqfull720_start-%j.out
#SBATCH -e logs/iqfull720_start-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${OUT_ROOT:?OUT_ROOT must be set}"
: "${MODEL_SEEDS:?MODEL_SEEDS must be set}"
: "${TARGET_TRAIN_STEPS:?TARGET_TRAIN_STEPS must be set}"
: "${G0_ROOT_FRACTIONS:?G0_ROOT_FRACTIONS must be set}"
: "${POLICY_WINDOWS_H:?POLICY_WINDOWS_H must be set}"
: "${EVAL_SEEDS:?EVAL_SEEDS must be set}"

cd "$PROJECT_DIR"
test -s "$OUT_ROOT/common/g0/train_merged.npz"
test -s "$OUT_ROOT/common/g0/validation_merged.npz"

eval_jobs=()
audit_jobs=()
for model_seed in $MODEL_SEEDS; do
  run_root="$OUT_ROOT/model_seed_$model_seed"
  mkdir -p "$run_root/g0"
  ln -s "$PROJECT_DIR/$OUT_ROOT/common/g0/train_merged.npz" \
    "$run_root/g0/train_merged.npz"
  ln -s "$PROJECT_DIR/$OUT_ROOT/common/g0/validation_merged.npz" \
    "$run_root/g0/validation_merged.npz"

  PROJECT_DIR="$PROJECT_DIR" \
  CONFIG_NAME="iqfull720_s${model_seed}" \
  RUN_ROOT="$run_root" \
  G0_TRAIN_COUNT=180 \
  G0_VALIDATION_COUNT=40 \
  G0_TRAIN_START=1500 \
  G0_VALIDATION_START=3200 \
  G0_ROOT_FRACTIONS="$G0_ROOT_FRACTIONS" \
  G0_ROOTS_PER_SEED=12 \
  ITER_TRAIN_COUNTS=24,36,60 \
  ITER_VALIDATION_COUNTS=5,7,12 \
  ITER_TRAIN_STARTS=1500,1800,2100 \
  ITER_VALIDATION_STARTS=3200,3230,3300 \
  ITER_CHUNK_SIZE=10 \
  SCENARIO_PROTOCOL=unified_window_v1 \
  HARD_SCENARIO_PROBABILITY=0.5 \
  FORECAST_CONTEXT_HOURS=168 \
  OBSERVATION_INPUT=shared_future_summary \
  EXCLUDE_STATE_FEATURES=hour_of_week \
  POLICY_WINDOWS_H="$POLICY_WINDOWS_H" \
  MAX_OVERRIDES=12 \
  COLLECTION_REQUIRED_HEADS=4 \
  P1_RESIDUAL_MARGIN=0.40 \
  P1_ECONOMIC_MARGIN_EUR=40000 \
  ITER_RESIDUAL_MARGIN=0.40 \
  ITER_ECONOMIC_MARGIN_EUR=40000 \
  RESUME_FROM_G0=1 \
  EVAL_SEEDS="$EVAL_SEEDS" \
  VALIDATION_ONLY=0 \
  MODEL_SEED="$model_seed" \
  EVAL_EACH_STAGE=0 \
  ROOT_SELECTION=first_decision_event \
  WINDOWS_PER_SEED= \
  bash hpc/launch_iterative_action_q.sh

  eval_job=$(awk -F= '$1 == "eval" {print $2}' "$run_root/job_ids.txt")
  audit_job=$(sbatch --parsable \
    --dependency=afterok:"$eval_job" \
    --job-name="iqfull720_s${model_seed}_budget" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$run_root",TARGET_TRAIN_STEPS="$TARGET_TRAIN_STEPS" \
    hpc/submit_iterative_q_budget_audit.sh)
  audit_job="${audit_job%%;*}"
  printf 'budget_audit=%s\n' "$audit_job" | tee -a "$run_root/job_ids.txt"
  eval_jobs+=("$eval_job")
  audit_jobs+=("$audit_job")
done

{
  printf 'eval_jobs=%s\n' "${eval_jobs[*]}"
  printf 'budget_audit_jobs=%s\n' "${audit_jobs[*]}"
} | tee "$OUT_ROOT/final_job_ids.txt"
