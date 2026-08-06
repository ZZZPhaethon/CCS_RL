#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SOURCE_RUN="${SOURCE_RUN:-output/iterative_q_budget_search/runs/g60_p4}"
OUT_ROOT="${OUT_ROOT:-output/iterative_q_state_recursive_ablation_20260730_run01}"
MODEL_SEEDS="${MODEL_SEEDS:-0 1 2}"
EVAL_SEEDS="${EVAL_SEEDS:-9000031:9000032:9000033:9000034:9000035:9000036:9000037:9000038:9000039:9000040:9000041:9000042:9000043:9000044:9000045:9000046:9000047:9000048:9000049:9000050:9000051:9000052:9000053:9000054:9000055:9000056:9000057:9000058:9000059:9000060}"
POLICY_WINDOWS_H="108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680"

cd "$PROJECT_DIR"
if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing existing recursive-ablation root: $OUT_ROOT" >&2
  exit 2
fi
test -s "$SOURCE_RUN/g0/train_merged.npz"
test -s "$SOURCE_RUN/g0/validation_merged.npz"
mkdir -p "$OUT_ROOT"

CONDITIONS=(drop_hour_of_week drop_all_three)
EXCLUSIONS=(
  "hour_of_week"
  "hour_of_week:in_transit_fill:episode_progress"
)

{
  printf 'purpose=self-consistent recursive state-feature deletion\n'
  printf 'source_g0=%s\n' "$SOURCE_RUN/g0"
  printf 'g0_reused=true\n'
  printf 'g1_g3_regenerated_per_condition_and_model_seed=true\n'
  printf 'conditions=%s\n' "${CONDITIONS[*]}"
  printf 'model_seeds=%s\n' "$MODEL_SEEDS"
  printf 'formal_eval_seeds=%s\n' "$EVAL_SEEDS"
  printf 'formal_test_previously_accessed=true\n'
} > "$OUT_ROOT/protocol_lock.txt"

for condition_index in 0 1; do
  condition="${CONDITIONS[$condition_index]}"
  exclusion="${EXCLUSIONS[$condition_index]}"
  for model_seed in $MODEL_SEEDS; do
    run_root="$OUT_ROOT/$condition/model_seed_$model_seed"
    mkdir -p "$run_root/g0"
    ln -s "$PROJECT_DIR/$SOURCE_RUN/g0/train_merged.npz" \
      "$run_root/g0/train_merged.npz"
    ln -s "$PROJECT_DIR/$SOURCE_RUN/g0/validation_merged.npz" \
      "$run_root/g0/validation_merged.npz"
    config_name="iqrec_${condition_index}_s${model_seed}"

    PROJECT_DIR="$PROJECT_DIR" \
    CONFIG_NAME="$config_name" \
    RUN_ROOT="$run_root" \
    G0_TRAIN_COUNT=180 \
    G0_VALIDATION_COUNT=40 \
    G0_TRAIN_START=1500 \
    G0_VALIDATION_START=3200 \
    G0_ROOT_FRACTIONS="0.15:0.2166666667:0.2833333333:0.35:0.4166666667:0.4833333333:0.55:0.6166666667:0.6833333333:0.75:0.8166666667:0.8833333333" \
    G0_ROOTS_PER_SEED=12 \
    ITER_TRAIN_COUNTS="24,36,60" \
    ITER_VALIDATION_COUNTS="5,7,12" \
    ITER_TRAIN_STARTS="1500,1800,2100" \
    ITER_VALIDATION_STARTS="3200,3230,3300" \
    ITER_CHUNK_SIZE=10 \
    SCENARIO_PROTOCOL=unified_window_v1 \
    HARD_SCENARIO_PROBABILITY=0.5 \
    FORECAST_CONTEXT_HOURS=168 \
    OBSERVATION_INPUT=shared_future_summary \
    EXCLUDE_STATE_FEATURES="$exclusion" \
    POLICY_WINDOWS_H="$POLICY_WINDOWS_H" \
    MAX_OVERRIDES=12 \
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
  done
done

find "$OUT_ROOT" -mindepth 3 -maxdepth 3 -name job_ids.txt -print | sort
