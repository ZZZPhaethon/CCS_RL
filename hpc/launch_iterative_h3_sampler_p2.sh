#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
OUT_ROOT="${OUT_ROOT:-experiments_results/E2/iterative_h3_sampler_validation_run01}"
SOURCE_RUN="${SOURCE_RUN:-output/iterative_q_budget_search/runs/g60_p4}"
SEED_CHECKPOINT_ROOT="${SEED_CHECKPOINT_ROOT:-experiments_results/E1/training_iterative_action_q_g60_p4_full_model_seeds_1_2_20260729}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680}"
PROTOCOL="experiments/protocols/iterative_h3_sampler_validation_protocol.json"

cd "$PROJECT_DIR"
mkdir -p logs
if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing existing experiment root: $OUT_ROOT" >&2
  exit 2
fi
for path in \
  "$SOURCE_RUN/g0/train_merged.npz" \
  "$SOURCE_RUN/g0/validation_merged.npz" \
  "$SOURCE_RUN/p1/iterative_action_q.pt" \
  "$SEED_CHECKPOINT_ROOT/model_seed_1/p1/iterative_action_q.pt" \
  "$SEED_CHECKPOINT_ROOT/model_seed_2/p1/iterative_action_q.pt" \
  "$PROTOCOL"; do
  if [[ ! -s "$path" ]]; then
    echo "Missing locked input: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUT_ROOT/shared/p1"
ln -s "$PROJECT_DIR/$SOURCE_RUN/g0" "$OUT_ROOT/shared/g0"
ln -s \
  "$PROJECT_DIR/$SOURCE_RUN/p1/iterative_action_q.pt" \
  "$OUT_ROOT/shared/p1/iterative_action_q.pt"
cp "$PROTOCOL" "$OUT_ROOT/protocol_lock.json"

submit_job() {
  local submitted
  submitted=$(sbatch --parsable "$@")
  printf '%s\n' "${submitted%%;*}"
}

common_export="ALL,PROJECT_DIR=$PROJECT_DIR,OUT_ROOT=$OUT_ROOT,SOURCE_RUN=$SOURCE_RUN,SEED_CHECKPOINT_ROOT=$SEED_CHECKPOINT_ROOT,POLICY_WINDOWS_H=$POLICY_WINDOWS_H"
env_job=$(submit_job \
  --export="$common_export" \
  hpc/submit_iterative_h3_sampler_env_check.sh)
lock_job=$(submit_job \
  --dependency=afterok:"$env_job" \
  --job-name=iter_h3_p1lock \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$OUT_ROOT/shared",OUTPUT_STAGE=p1,RESIDUAL_MARGIN=0.4,ECONOMIC_MARGIN_EUR=40000,REQUIRED_HEADS=3,PROTOCOL_PREFIX=iterative_h3_collection,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12 \
  hpc/submit_iterative_q_lock.sh)
g1_job=$(submit_job \
  --dependency=afterok:"$lock_job" \
  --array=0-3%4 \
  --job-name=iter_h3_g1 \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$OUT_ROOT/shared",STAGE=g1,LOCK_CONFIG="$OUT_ROOT/shared/p1_lock.json",TRAIN_START=1500,TRAIN_COUNT=24,VALIDATION_START=3200,VALIDATION_COUNT=5,CHUNK_SIZE=10,DATASET_SEED=20260724,SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168 \
  hpc/submit_iterative_q_policy_data.sh)
merge_job=$(submit_job \
  --dependency=afterok:"$g1_job" \
  --job-name=iter_h3_g1m \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$OUT_ROOT/shared",STAGE=g1,TRAIN_START=1500,TRAIN_COUNT=24,VALIDATION_START=3200,VALIDATION_COUNT=5 \
  hpc/submit_iterative_q_merge.sh)
audit_job=$(submit_job \
  --dependency=afterok:"$merge_job" \
  --export="$common_export" \
  hpc/submit_iterative_h3_sampler_data_audit.sh)
train_job=$(submit_job \
  --dependency=afterok:"$audit_job" \
  --array=0-8%3 \
  --export="$common_export" \
  hpc/submit_iterative_h3_sampler_train.sh)
validation_job=$(submit_job \
  --dependency=afterok:"$train_job" \
  --array=0-8%3 \
  --export="$common_export" \
  hpc/submit_iterative_h3_sampler_validation.sh)
aggregate_job=$(submit_job \
  --dependency=afterok:"$validation_job" \
  --export="$common_export" \
  hpc/submit_iterative_h3_sampler_aggregate.sh)

{
  printf 'formal_test_access=false\n'
  printf 'environment_check=%s\n' "$env_job"
  printf 'p1_h3_lock=%s\n' "$lock_job"
  printf 'g1_collection=%s\n' "$g1_job"
  printf 'g1_merge=%s\n' "$merge_job"
  printf 'data_audit=%s\n' "$audit_job"
  printf 'p2_train=%s\n' "$train_job"
  printf 'p2_validation=%s\n' "$validation_job"
  printf 'p2_aggregate=%s\n' "$aggregate_job"
} | tee "$OUT_ROOT/job_manifest.txt"
