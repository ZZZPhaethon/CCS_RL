#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
OUT_ROOT="${OUT_ROOT:-output/iterative_q_full_window_0_720_20260803_run01}"
MODEL_SEEDS="${MODEL_SEEDS:-0 1 2}"
TARGET_TRAIN_STEPS="${TARGET_TRAIN_STEPS:-9526297}"
G0_ROOT_FRACTIONS="0:0.0833333333:0.1666666667:0.25:0.3333333333:0.4166666667:0.5:0.5833333333:0.6666666667:0.75:0.8333333333:0.9166666667"
POLICY_WINDOWS_H="0-59:60-119:120-179:180-239:240-299:300-359:360-419:420-479:480-539:540-599:600-659:660-719"
EVAL_SEEDS="9000031:9000032:9000033:9000034:9000035:9000036:9000037:9000038:9000039:9000040:9000041:9000042:9000043:9000044:9000045:9000046:9000047:9000048:9000049:9000050:9000051:9000052:9000053:9000054:9000055:9000056:9000057:9000058:9000059:9000060"

cd "$PROJECT_DIR"
if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing existing run root: $OUT_ROOT" >&2
  exit 2
fi
mkdir -p logs "$OUT_ROOT/common"

{
  printf 'purpose=Iterative-Q intervention-window extension from 108-680 h to 0-720 h\n'
  printf 'formal_baseline=G60-P4-no-hour\n'
  printf 'only_intended_change=G0 root times and policy intervention windows\n'
  printf 'execution_hours=[0,720)\n'
  printf 'policy_windows_h=%s\n' "$POLICY_WINDOWS_H"
  printf 'g0_root_fractions=%s\n' "$G0_ROOT_FRACTIONS"
  printf 'g0_shared_across_model_seeds=true\n'
  printf 'model_seeds=%s\n' "$MODEL_SEEDS"
  printf 'target_train_simulator_steps=%s\n' "$TARGET_TRAIN_STEPS"
  printf 'formal_test_seeds=%s\n' "$EVAL_SEEDS"
  printf 'formal_test_previously_accessed=true\n'
} > "$OUT_ROOT/protocol_lock.txt"

env_job=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR" \
  hpc/submit_iterative_q_full_window_env_check.sh)
env_job="${env_job%%;*}"

g0_job=$(sbatch --parsable \
  --dependency=afterok:"$env_job" \
  --array=0-21%22 \
  --job-name=iqfull720_g0 \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$OUT_ROOT/common",TRAIN_START=1500,TRAIN_COUNT=180,VALIDATION_START=3200,VALIDATION_COUNT=40,CHUNK_SIZE=10,G0_ROOT_FRACTIONS="$G0_ROOT_FRACTIONS",G0_ROOTS_PER_SEED=12,SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168,FUTURE_SUMMARY_WINDOWS_H= \
  hpc/submit_iterative_q_greedy_data.sh)
g0_job="${g0_job%%;*}"

g0_merge_job=$(sbatch --parsable \
  --dependency=afterok:"$g0_job" \
  --job-name=iqfull720_g0m \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$OUT_ROOT/common",STAGE=g0,TRAIN_START=1500,TRAIN_COUNT=180,VALIDATION_START=3200,VALIDATION_COUNT=40 \
  hpc/submit_iterative_q_merge.sh)
g0_merge_job="${g0_merge_job%%;*}"

start_models_job=$(sbatch --parsable \
  --dependency=afterok:"$g0_merge_job" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",OUT_ROOT="$OUT_ROOT",MODEL_SEEDS="$MODEL_SEEDS",TARGET_TRAIN_STEPS="$TARGET_TRAIN_STEPS",G0_ROOT_FRACTIONS="$G0_ROOT_FRACTIONS",POLICY_WINDOWS_H="$POLICY_WINDOWS_H",EVAL_SEEDS="$EVAL_SEEDS" \
  hpc/submit_iterative_q_full_window_models.sh)
start_models_job="${start_models_job%%;*}"

{
  printf 'environment_check=%s\n' "$env_job"
  printf 'g0=%s\n' "$g0_job"
  printf 'g0_merge=%s\n' "$g0_merge_job"
  printf 'start_models=%s\n' "$start_models_job"
} | tee "$OUT_ROOT/initial_job_ids.txt"
