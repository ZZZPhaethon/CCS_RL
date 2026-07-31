#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
OUT_ROOT="${OUT_ROOT:-experiments_results/E2/iterative_h3_sampler_validation_run01}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680}"

cd "$PROJECT_DIR"
if [[ ! -s "$OUT_ROOT/p2_selection.json" ]]; then
  echo "Missing P2 selection: $OUT_ROOT/p2_selection.json" >&2
  exit 2
fi
if [[ -e "$OUT_ROOT/recursive_job_manifest.txt" ]]; then
  echo "Refusing existing recursive launch" >&2
  exit 2
fi
SELECTED_VARIANT=$(jq -r '.selected_reweight_variant' \
  "$OUT_ROOT/p2_selection.json")
if [[ "$SELECTED_VARIANT" != "c_dedup_balanced" \
   && "$SELECTED_VARIANT" != "d_dedup_advantage" ]]; then
  echo "Unexpected selected variant: $SELECTED_VARIANT" >&2
  exit 2
fi

for variant in b_gate_only "$SELECTED_VARIANT"; do
  for model_seed in 0 1 2; do
    branch="$OUT_ROOT/branches/$variant/model_seed_${model_seed}"
    if [[ ! -s "$branch/p2/iterative_action_q.pt" \
       || ! -s "$branch/p2_lock.json" ]]; then
      echo "Missing P2 branch input: $branch" >&2
      exit 2
    fi
    if [[ ! -e "$branch/g0" ]]; then
      ln -s "$PROJECT_DIR/$OUT_ROOT/shared/g0" "$branch/g0"
    fi
    if [[ ! -e "$branch/g1" ]]; then
      ln -s "$PROJECT_DIR/$OUT_ROOT/shared/g1" "$branch/g1"
    fi
  done
done

submit_job() {
  local submitted
  submitted=$(sbatch --parsable "$@")
  printf '%s\n' "${submitted%%;*}"
}

common_export="ALL,PROJECT_DIR=$PROJECT_DIR,OUT_ROOT=$OUT_ROOT,SELECTED_VARIANT=$SELECTED_VARIANT,POLICY_WINDOWS_H=$POLICY_WINDOWS_H"
g2_job=$(submit_job \
  --array=0-29%12 \
  --export="$common_export,STAGE=g2,PREVIOUS_STAGE=p2,TRAIN_START=1800,TRAIN_COUNT=36,VALIDATION_START=3230,VALIDATION_COUNT=7,CHUNK_SIZE=10,DATASET_SEED=20260725" \
  hpc/submit_iterative_h3_recursive_policy_data.sh)
g2_merge_job=$(submit_job \
  --dependency=afterok:"$g2_job" \
  --array=0-5%6 \
  --export="$common_export,STAGE=g2,TRAIN_START=1800,TRAIN_COUNT=36,VALIDATION_START=3230,VALIDATION_COUNT=7" \
  hpc/submit_iterative_h3_recursive_merge.sh)
p3_job=$(submit_job \
  --dependency=afterok:"$g2_merge_job" \
  --array=0-5%3 \
  --export="$common_export,OUTPUT_STAGE=p3,PREVIOUS_STAGE=p2,DATA_STAGES=g0:g1:g2,ENCODER_LR=0.00003,HEAD_LR=0.0001,CREATE_LOCK=1" \
  hpc/submit_iterative_h3_recursive_train.sh)
p3_validation_job=$(submit_job \
  --dependency=afterok:"$p3_job" \
  --array=0-5%3 \
  --export="$common_export,MODEL_STAGE=p3" \
  hpc/submit_iterative_h3_recursive_validation.sh)
g3_job=$(submit_job \
  --dependency=afterok:"$p3_job" \
  --array=0-47%12 \
  --export="$common_export,STAGE=g3,PREVIOUS_STAGE=p3,TRAIN_START=2100,TRAIN_COUNT=60,VALIDATION_START=3300,VALIDATION_COUNT=12,CHUNK_SIZE=10,DATASET_SEED=20260726" \
  hpc/submit_iterative_h3_recursive_policy_data.sh)
g3_merge_job=$(submit_job \
  --dependency=afterok:"$g3_job" \
  --array=0-5%6 \
  --export="$common_export,STAGE=g3,TRAIN_START=2100,TRAIN_COUNT=60,VALIDATION_START=3300,VALIDATION_COUNT=12" \
  hpc/submit_iterative_h3_recursive_merge.sh)
p4_job=$(submit_job \
  --dependency=afterok:"$g3_merge_job" \
  --array=0-5%3 \
  --export="$common_export,OUTPUT_STAGE=p4,PREVIOUS_STAGE=p3,DATA_STAGES=g0:g1:g2:g3,ENCODER_LR=0.00003,HEAD_LR=0.0001,CREATE_LOCK=0" \
  hpc/submit_iterative_h3_recursive_train.sh)
p4_validation_job=$(submit_job \
  --dependency=afterok:"$p4_job" \
  --array=0-5%3 \
  --export="$common_export,MODEL_STAGE=p4" \
  hpc/submit_iterative_h3_recursive_validation.sh)
aggregate_job=$(submit_job \
  --dependency=afterok:"$p3_validation_job":"$p4_validation_job" \
  --export="$common_export" \
  hpc/submit_iterative_h3_recursive_aggregate.sh)

{
  printf 'formal_test_access=false\n'
  printf 'selected_reweight_variant=%s\n' "$SELECTED_VARIANT"
  printf 'g2_collection=%s\n' "$g2_job"
  printf 'g2_merge=%s\n' "$g2_merge_job"
  printf 'p3_train=%s\n' "$p3_job"
  printf 'p3_validation=%s\n' "$p3_validation_job"
  printf 'g3_collection=%s\n' "$g3_job"
  printf 'g3_merge=%s\n' "$g3_merge_job"
  printf 'p4_train=%s\n' "$p4_job"
  printf 'p4_validation=%s\n' "$p4_validation_job"
  printf 'recursive_aggregate=%s\n' "$aggregate_job"
} | tee "$OUT_ROOT/recursive_job_manifest.txt"
