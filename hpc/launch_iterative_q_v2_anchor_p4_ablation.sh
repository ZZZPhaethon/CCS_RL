#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
DATA_ROOT="${DATA_ROOT:-output/iterative_q_validation_search/uniform_margin40_p1_p4}"
RUN_ROOT="${RUN_ROOT:-output/iterative_q_validation_search/iterative_q_v2_anchor_p4_ablation}"
EVAL_SEEDS="${EVAL_SEEDS:-8100001:8100002:8100003:8100004:8100005:8100006:8100007:8100008:8100009:8100010:8100011:8100012:8100013:8100014:8100015:8100016:8100017:8100018:8100019:8100020}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680}"

cd "$PROJECT_DIR"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing run: $RUN_ROOT" >&2
  exit 2
fi
for path in \
  "$DATA_ROOT/g0/train_merged.npz" \
  "$DATA_ROOT/g1/train_merged.npz" \
  "$DATA_ROOT/g2/train_merged.npz" \
  "$DATA_ROOT/g3/train_merged.npz" \
  "$DATA_ROOT/p3/iterative_action_q.pt"; do
  if [[ ! -s "$path" ]]; then
    echo "Required input is missing: $path" >&2
    exit 2
  fi
done

mkdir -p "$RUN_ROOT" logs
manifest="$RUN_ROOT/job_manifest.txt"
schedule="$RUN_ROOT/schedule.txt"
coefficients=(0.1 0.3 1.0)
variants=(anchor_010 anchor_030 anchor_100)

env_job=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",DATA_ROOT="$DATA_ROOT" \
  hpc/submit_iterative_q_v2_env_check.sh)
printf 'env_check=%s\n' "$env_job" | tee "$manifest"

for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  coefficient="${coefficients[$index]}"
  train_job=$(sbatch --parsable \
    --dependency=afterok:"$env_job" \
    --job-name="iterq_v2_${variant}_train" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",DATA_ROOT="$DATA_ROOT",OUTPUT_STAGE="$variant",DATA_STAGES="g0:g1:g2:g3",INITIAL_CHECKPOINT="$DATA_ROOT/p3/iterative_action_q.pt",ENCODER_LR=0.00003,HEAD_LR=0.0001,FOLLOW_ANCHOR=0.0,PREVIOUS_POLICY_ANCHOR="$coefficient",PREVIOUS_POLICY_RELEASE_MARGIN_EUR=40000,PREVIOUS_POLICY_ANCHOR_TEMPERATURE=0.5,OBSERVATION_INPUT=shared_future_summary,MODEL_SEED=0 \
    hpc/submit_iterative_q_train.sh)
  eval_job=$(sbatch --parsable \
    --dependency=afterok:"$train_job" \
    --job-name="iterq_v2_${variant}_eval" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",FINAL_STAGE="$variant",EVAL_NAME="iterative_q_v2_${variant}",SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,EVAL_SEEDS="$EVAL_SEEDS",VALIDATION_ONLY=1 \
    hpc/submit_iterative_q_eval.sh)
  {
    printf '%s_train=%s\n' "$variant" "$train_job"
    printf '%s_eval=%s\n' "$variant" "$eval_job"
  } | tee -a "$manifest"
done

{
  printf 'name=Iterative Q v2 P4 selective-anchor ablation\n'
  printf 'data_root=%s\n' "$DATA_ROOT"
  printf 'initial_checkpoint=%s\n' \
    "$DATA_ROOT/p3/iterative_action_q.pt"
  printf 'data_stages=g0:g1:g2:g3\n'
  printf 'anchor_coefficients=0.1,0.3,1.0\n'
  printf 'release_margin_eur=40000\n'
  printf 'anchor_temperature=0.5\n'
  printf 'model_seed=0\n'
  printf 'evaluation_seeds=%s\n' "$EVAL_SEEDS"
  printf 'validation_only=1\n'
  printf 'selection_gate=4_heads,0.40_margin,12_windows\n'
} > "$schedule"
