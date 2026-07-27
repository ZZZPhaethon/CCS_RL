#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_greedy_dagger}"
SWEEP_NAME="${SWEEP_NAME:-iterative_q_block_k_sweep_20260726}"
RUN_PARENT="${RUN_PARENT:-output/rl_forecast}"
DRY_RUN="${DRY_RUN:-0}"
RESUME_FROM_P1="${RESUME_FROM_P1:-0}"

declare -A ROOT_FRACTIONS=(
  [4]="0.15:0.35:0.55:0.75"
  [8]="0.15:0.25:0.35:0.45:0.55:0.65:0.75:0.85"
  [12]="0.15:0.2166666667:0.2833333333:0.35:0.4166666667:0.4833333333:0.55:0.6166666667:0.6833333333:0.75:0.8166666667:0.8833333333"
  [16]="0.15:0.20:0.25:0.30:0.35:0.40:0.45:0.50:0.55:0.60:0.65:0.70:0.75:0.80:0.85:0.90"
)

declare -A WINDOWS_H=(
  [4]="108-251:252-395:396-539:540-680"
  [8]="108-179:180-251:252-323:324-395:396-467:468-539:540-611:612-680"
  [12]="108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680"
  [16]="108-143:144-179:180-215:216-251:252-287:288-323:324-359:360-395:396-431:432-467:468-503:504-539:540-575:576-611:612-647:648-680"
)

cd "$PROJECT_DIR"
if [[ "$DRY_RUN" != "1" ]]; then
  for k in 4 8 12 16; do
    run_root="$RUN_PARENT/${SWEEP_NAME}_k${k}"
    if [[ "$RESUME_FROM_P1" == "1" && ! -e "$run_root" ]]; then
      echo "Cannot resume missing output: $run_root" >&2
      exit 2
    fi
    if [[ "$RESUME_FROM_P1" != "1" && -e "$run_root" ]]; then
      echo "Refusing partial sweep because output already exists: $run_root" >&2
      exit 2
    fi
  done
fi

for k in 4 8 12 16; do
  CONFIG_NAME="${SWEEP_NAME}_k${k}" \
  RUN_ROOT="$RUN_PARENT/${SWEEP_NAME}_k${k}" \
  G0_ROOT_FRACTIONS="${ROOT_FRACTIONS[$k]}" \
  G0_ROOTS_PER_SEED="$k" \
  POLICY_WINDOWS_H="${WINDOWS_H[$k]}" \
  MAX_OVERRIDES="$k" \
  SCENARIO_PROTOCOL=q_original \
  OBSERVATION_INPUT=state_only \
  RESUME_FROM_P1="$RESUME_FROM_P1" \
  DRY_RUN="$DRY_RUN" \
  PROJECT_DIR="$PROJECT_DIR" \
    bash hpc/launch_iterative_action_q.sh
done
