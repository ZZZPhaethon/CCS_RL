#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-output/iterative_q_validation_search/baseline_p1_p4}"
CONFIG_NAME="${CONFIG_NAME:-iterq_uniform_margin40_p1_p4}"
RUN_ROOT="${RUN_ROOT:-output/iterative_q_validation_search/uniform_margin40_p1_p4}"

cd "$PROJECT_DIR"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing run: $RUN_ROOT" >&2
  exit 2
fi
if [[ ! -s "$BASELINE_RUN_ROOT/g0/train_merged.npz" \
   || ! -s "$BASELINE_RUN_ROOT/g0/validation_merged.npz" \
   || ! -s "$BASELINE_RUN_ROOT/p1/iterative_action_q.pt" ]]; then
  echo "Baseline G0 data or P1 checkpoint is missing" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/g0" "$RUN_ROOT/p1"
cp "$BASELINE_RUN_ROOT/g0/train_merged.npz" "$RUN_ROOT/g0/train_merged.npz"
cp "$BASELINE_RUN_ROOT/g0/validation_merged.npz" \
  "$RUN_ROOT/g0/validation_merged.npz"
cp "$BASELINE_RUN_ROOT/p1/iterative_action_q.pt" \
  "$RUN_ROOT/p1/iterative_action_q.pt"
cp "$BASELINE_RUN_ROOT/p1/summary.json" "$RUN_ROOT/p1/summary.json"
sha256sum \
  "$BASELINE_RUN_ROOT/p1/iterative_action_q.pt" \
  "$RUN_ROOT/p1/iterative_action_q.pt" \
  > "$RUN_ROOT/reused_p1_sha256.txt"

PROJECT_DIR="$PROJECT_DIR" \
CONFIG_NAME="$CONFIG_NAME" \
RUN_ROOT="$RUN_ROOT" \
G0_TRAIN_COUNT=200 \
G0_VALIDATION_COUNT=40 \
G0_TRAIN_START=1500 \
G0_VALIDATION_START=3200 \
G0_ROOT_FRACTIONS="0.15:0.2166666667:0.2833333333:0.35:0.4166666667:0.4833333333:0.55:0.6166666667:0.6833333333:0.75:0.8166666667:0.8833333333" \
G0_ROOTS_PER_SEED=12 \
ITER_TRAIN_COUNTS="40,60,100" \
ITER_VALIDATION_COUNTS="8,12,20" \
ITER_TRAIN_STARTS="1500,1600,2000" \
ITER_VALIDATION_STARTS="3200,3230,3300" \
ITER_CHUNK_SIZE=10 \
SCENARIO_PROTOCOL=unified_window_v1 \
HARD_SCENARIO_PROBABILITY=0.5 \
FORECAST_CONTEXT_HOURS=168 \
OBSERVATION_INPUT=shared_future_summary \
POLICY_WINDOWS_H="108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680" \
MAX_OVERRIDES=12 \
P1_RESIDUAL_MARGIN=0.40 \
P1_ECONOMIC_MARGIN_EUR=40000 \
ITER_RESIDUAL_MARGIN=0.40 \
ITER_ECONOMIC_MARGIN_EUR=40000 \
RESUME_FROM_P1=1 \
EVAL_SEEDS="8100001:8100002:8100003:8100004:8100005:8100006:8100007:8100008:8100009:8100010:8100011:8100012:8100013:8100014:8100015:8100016:8100017:8100018:8100019:8100020" \
VALIDATION_ONLY=1 \
bash hpc/launch_iterative_action_q.sh
