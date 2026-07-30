#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SEARCH_ROOT="${SEARCH_ROOT:-output/iterative_q_budget_search}"
SHARED_ROOT="${SHARED_ROOT:-$SEARCH_ROOT/shared}"
TARGET_TRAIN_STEPS="${TARGET_TRAIN_STEPS:-9525119}"
EVAL_SEEDS="${EVAL_SEEDS:-8100001:8100002:8100003:8100004:8100005:8100006:8100007:8100008:8100009:8100010:8100011:8100012:8100013:8100014:8100015:8100016:8100017:8100018:8100019:8100020}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680}"

cd "$PROJECT_DIR"
if [[ ! -s "$SEARCH_ROOT/g0_pools/g0_90/g0/train_merged.npz" \
   || ! -s "$SEARCH_ROOT/g0_pools/g0_120/g0/train_merged.npz" \
   || ! -s "$SEARCH_ROOT/g0_pools/g0_150/g0/train_merged.npz" \
   || ! -s "$SEARCH_ROOT/g0_pools/g0_180/g0/train_merged.npz" ]]; then
  echo "Prepared G0 budget pools are missing" >&2
  exit 2
fi
if [[ -e "$SEARCH_ROOT/search_manifest.txt" ]]; then
  echo "Refusing to relaunch an existing budget search" >&2
  exit 2
fi
mkdir -p "$SEARCH_ROOT/runs"

CONFIGS=(
  "g30_p2|90|210|42|1500|3200"
  "g30_p3|90|84,126|17,25|1500,1800|3200,3230"
  "g30_p4|90|42,63,105|8,13,21|1500,1800,2100|3200,3230,3300"
  "g30_p5|90|21,42,63,84|4,8,13,17|1500,1800,2100,2300|3200,3230,3300,3330"
  "g40_p2|120|180|36|1500|3200"
  "g40_p3|120|72,108|14,22|1500,1800|3200,3230"
  "g40_p4|120|36,54,90|7,11,18|1500,1800,2100|3200,3230,3300"
  "g40_p5|120|18,36,54,72|4,7,11,14|1500,1800,2100,2300|3200,3230,3300,3330"
  "g50_p2|150|150|30|1500|3200"
  "g50_p3|150|60,90|12,18|1500,1800|3200,3230"
  "g50_p4|150|30,45,75|6,9,15|1500,1800,2100|3200,3230,3300"
  "g50_p5|150|15,30,45,60|4,6,9,12|1500,1800,2100,2300|3200,3230,3300,3330"
  "g60_p2|180|120|24|1500|3200"
  "g60_p3|180|48,72|10,14|1500,1800|3200,3230"
  "g60_p4|180|24,36,60|5,7,12|1500,1800,2100|3200,3230,3300"
  "g60_p5|180|12,24,36,48|4,5,7,10|1500,1800,2100,2300|3200,3230,3300,3330"
  "g40_p4_r3360|120|32,48,80|6,10,16|1500,1800,2100|3200,3230,3300"
  "g40_p4_r3840|120|40,60,100|8,12,20|1500,1800,2100|3200,3230,3300"
  "g60_p4_r3360|180|20,30,50|4,6,10|1500,1800,2100|3200,3230,3300"
  "g60_p4_r3840|180|28,42,70|6,8,14|1500,1800,2100|3200,3230,3300"
)

manifest="$SEARCH_ROOT/search_manifest.txt"
protocol="$SEARCH_ROOT/protocol_lock.txt"
{
  printf 'target_train_simulator_steps=%s\n' "$TARGET_TRAIN_STEPS"
  printf 'budget_counts_training_data_generation_only=true\n'
  printf 'validation_data_generation_excluded_from_budget=true\n'
  printf 'controller_validation_seeds=%s\n' "$EVAL_SEEDS"
  printf 'formal_test_accessed=false\n'
  printf 'model_seed=0\n'
  printf 'uniform_residual_margin=0.40\n'
  printf 'uniform_economic_margin_eur=40000\n'
  printf 'g0_budget_share_targets_pct=30,40,50,60\n'
  printf 'g0_train_seed_counts=90,120,150,180\n'
  printf 'nominal_total_root_counts=3360,3600,3840\n'
  printf 'large_regression_threshold_eur=100000\n'
  printf 'selection=mean_cost,tail_risk,stage_retention_pareto\n'
  printf 'config_count=%s\n' "${#CONFIGS[@]}"
} > "$protocol"

for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r name g0_count train_counts validation_counts train_starts validation_starts <<< "$spec"
  run_root="$SEARCH_ROOT/runs/$name"
  if [[ ! -e "$run_root" ]]; then
    g0_pool="$SEARCH_ROOT/g0_pools/g0_${g0_count}/g0"
    mkdir -p "$run_root/g0"
    cp "$g0_pool/train_merged.npz" "$run_root/g0/"
    cp "$g0_pool/validation_merged.npz" "$run_root/g0/"

    PROJECT_DIR="$PROJECT_DIR" \
    CONFIG_NAME="$name" \
    RUN_ROOT="$run_root" \
    G0_TRAIN_COUNT="$g0_count" \
    G0_VALIDATION_COUNT=40 \
    G0_TRAIN_START=1500 \
    G0_VALIDATION_START=3200 \
    G0_ROOT_FRACTIONS="0.15:0.2166666667:0.2833333333:0.35:0.4166666667:0.4833333333:0.55:0.6166666667:0.6833333333:0.75:0.8166666667:0.8833333333" \
    G0_ROOTS_PER_SEED=12 \
    ITER_TRAIN_COUNTS="$train_counts" \
    ITER_VALIDATION_COUNTS="$validation_counts" \
    ITER_TRAIN_STARTS="$train_starts" \
    ITER_VALIDATION_STARTS="$validation_starts" \
    ITER_CHUNK_SIZE=10 \
    SCENARIO_PROTOCOL=unified_window_v1 \
    HARD_SCENARIO_PROBABILITY=0.5 \
    FORECAST_CONTEXT_HOURS=168 \
    OBSERVATION_INPUT=shared_future_summary \
    POLICY_WINDOWS_H="$POLICY_WINDOWS_H" \
    MAX_OVERRIDES=12 \
    P1_RESIDUAL_MARGIN=0.40 \
    P1_ECONOMIC_MARGIN_EUR=40000 \
    ITER_RESIDUAL_MARGIN=0.40 \
    ITER_ECONOMIC_MARGIN_EUR=40000 \
    RESUME_FROM_G0=1 \
    EVAL_SEEDS="$EVAL_SEEDS" \
    VALIDATION_ONLY=1 \
    MODEL_SEED=0 \
    EVAL_EACH_STAGE=1 \
    bash hpc/launch_iterative_action_q.sh
  elif [[ ! -s "$run_root/job_ids.txt" ]]; then
    echo "Existing run lacks a complete job manifest: $run_root" >&2
    exit 2
  fi

  eval_job=$(awk -F= '$1 == "eval" {print $2}' "$run_root/job_ids.txt")
  audit_job=$(sbatch --parsable \
    --dependency=afterok:"$eval_job" \
    --job-name="${name}_budget" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$run_root",TARGET_TRAIN_STEPS="$TARGET_TRAIN_STEPS" \
    hpc/submit_iterative_q_budget_audit.sh)
  {
    printf '%s|g0_count=%s|train_counts=%s|validation_counts=%s|eval=%s|audit=%s\n' \
      "$name" "$g0_count" "$train_counts" "$validation_counts" "$eval_job" "${audit_job%%;*}"
  } | tee -a "$manifest"
done
