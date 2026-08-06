#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SOURCE_G0="${SOURCE_G0:-output/iterative_q_budget_search/runs/g60_p4/g0}"
OUT_ROOT="${OUT_ROOT:-output/iterative_q_root_window_ablation_20260803_run01}"
MODEL_SEEDS="${MODEL_SEEDS:-0 1 2}"
CONDITIONS="${CONDITIONS:-random_root windows24}"
TARGET_TRAIN_STEPS="${TARGET_TRAIN_STEPS:-9526297}"
RANDOM_TRAIN_COUNTS="${RANDOM_TRAIN_COUNTS:-24,36,60}"
RANDOM_VALIDATION_COUNTS="${RANDOM_VALIDATION_COUNTS:-5,7,12}"
WINDOWS24_TRAIN_COUNTS="${WINDOWS24_TRAIN_COUNTS:-24,36,60}"
WINDOWS24_VALIDATION_COUNTS="${WINDOWS24_VALIDATION_COUNTS:-5,7,12}"
EVAL_SEEDS="${EVAL_SEEDS:-9000031:9000032:9000033:9000034:9000035:9000036:9000037:9000038:9000039:9000040:9000041:9000042:9000043:9000044:9000045:9000046:9000047:9000048:9000049:9000050:9000051:9000052:9000053:9000054:9000055:9000056:9000057:9000058:9000059:9000060}"
WINDOWS12="108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680"
WINDOWS24="108-131:132-155:156-179:180-203:204-227:228-251:252-275:276-299:300-323:324-347:348-371:372-395:396-419:420-443:444-467:468-491:492-515:516-539:540-563:564-587:588-611:612-635:636-659:660-680"

cd "$PROJECT_DIR"
test -s "$SOURCE_G0/train_merged.npz"
test -s "$SOURCE_G0/validation_merged.npz"
if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing existing ablation root: $OUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT"

{
  printf 'purpose=random-root and increased-window Iterative-Q ablations\n'
  printf 'formal_baseline=G60-P4-no-hour\n'
  printf 'target_train_simulator_steps=%s\n' "$TARGET_TRAIN_STEPS"
  printf 'budget_scope=training_data_generation_only\n'
  printf 'source_g0=%s\n' "$SOURCE_G0"
  printf 'g0_reused=true\n'
  printf 'conditions=%s\n' "$CONDITIONS"
  printf 'model_seeds=%s\n' "$MODEL_SEEDS"
  printf 'windows24_maximum_interventions=12\n'
  printf 'windows24_roots_per_seed=12_rotated_across_24_windows\n'
  printf 'random_train_counts=%s\n' "$RANDOM_TRAIN_COUNTS"
  printf 'random_validation_counts=%s\n' "$RANDOM_VALIDATION_COUNTS"
  printf 'windows24_train_counts=%s\n' "$WINDOWS24_TRAIN_COUNTS"
  printf 'windows24_validation_counts=%s\n' \
    "$WINDOWS24_VALIDATION_COUNTS"
  printf 'formal_test_seeds=%s\n' "$EVAL_SEEDS"
  printf 'formal_test_previously_accessed=true\n'
} > "$OUT_ROOT/protocol_lock.txt"

for condition in $CONDITIONS; do
  case "$condition" in
    random_root)
      policy_windows="$WINDOWS12"
      root_selection=random_time
      train_counts="$RANDOM_TRAIN_COUNTS"
      validation_counts="$RANDOM_VALIDATION_COUNTS"
      ;;
    windows24)
      policy_windows="$WINDOWS24"
      root_selection=first_decision_event
      train_counts="$WINDOWS24_TRAIN_COUNTS"
      validation_counts="$WINDOWS24_VALIDATION_COUNTS"
      ;;
    *)
      echo "Unknown condition: $condition" >&2
      exit 2
      ;;
  esac

  for model_seed in $MODEL_SEEDS; do
    run_root="$OUT_ROOT/$condition/model_seed_$model_seed"
    mkdir -p "$run_root/g0"
    ln -s "$PROJECT_DIR/$SOURCE_G0/train_merged.npz" \
      "$run_root/g0/train_merged.npz"
    ln -s "$PROJECT_DIR/$SOURCE_G0/validation_merged.npz" \
      "$run_root/g0/validation_merged.npz"
    config_name="iqrw_${condition}_s${model_seed}"

    PROJECT_DIR="$PROJECT_DIR" \
    CONFIG_NAME="$config_name" \
    RUN_ROOT="$run_root" \
    G0_TRAIN_COUNT=180 \
    G0_VALIDATION_COUNT=40 \
    G0_TRAIN_START=1500 \
    G0_VALIDATION_START=3200 \
    G0_ROOT_FRACTIONS="0.15:0.2166666667:0.2833333333:0.35:0.4166666667:0.4833333333:0.55:0.6166666667:0.6833333333:0.75:0.8166666667:0.8833333333" \
    G0_ROOTS_PER_SEED=12 \
    ITER_TRAIN_COUNTS="$train_counts" \
    ITER_VALIDATION_COUNTS="$validation_counts" \
    ITER_TRAIN_STARTS="1500,1800,2100" \
    ITER_VALIDATION_STARTS="3200,3230,3300" \
    ITER_CHUNK_SIZE=10 \
    SCENARIO_PROTOCOL=unified_window_v1 \
    HARD_SCENARIO_PROBABILITY=0.5 \
    FORECAST_CONTEXT_HOURS=168 \
    OBSERVATION_INPUT=shared_future_summary \
    EXCLUDE_STATE_FEATURES=hour_of_week \
    POLICY_WINDOWS_H="$policy_windows" \
    MAX_OVERRIDES=12 \
    ROOT_SELECTION="$root_selection" \
    WINDOWS_PER_SEED=12 \
    P1_RESIDUAL_MARGIN=0.40 \
    P1_ECONOMIC_MARGIN_EUR=40000 \
    ITER_RESIDUAL_MARGIN=0.40 \
    ITER_ECONOMIC_MARGIN_EUR=40000 \
    RESUME_FROM_G0=1 \
    EVAL_SEEDS="$EVAL_SEEDS" \
    VALIDATION_ONLY=0 \
    MODEL_SEED="$model_seed" \
    EVAL_EACH_STAGE=0 \
    bash hpc/launch_iterative_action_q.sh

    eval_job=$(awk -F= '$1 == "eval" {print $2}' "$run_root/job_ids.txt")
    audit_job=$(sbatch --parsable \
      --dependency=afterok:"$eval_job" \
      --job-name="${config_name}_budget" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$run_root",TARGET_TRAIN_STEPS="$TARGET_TRAIN_STEPS" \
      hpc/submit_iterative_q_budget_audit.sh)
    printf 'budget_audit=%s\n' "${audit_job%%;*}" \
      | tee -a "$run_root/job_ids.txt"
  done
done

find "$OUT_ROOT" -mindepth 3 -maxdepth 3 -name job_ids.txt -print | sort
