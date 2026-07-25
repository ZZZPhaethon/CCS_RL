#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_greedy_dagger}"
CONFIG_NAME="${CONFIG_NAME:-iterative_q_3200}"
RUN_ROOT="${RUN_ROOT:-output/rl_forecast/${CONFIG_NAME}}"
G0_TRAIN_COUNT="${G0_TRAIN_COUNT:-200}"
G0_VALIDATION_COUNT="${G0_VALIDATION_COUNT:-40}"
ITER_TRAIN_COUNTS="${ITER_TRAIN_COUNTS:-40,60,100}"
ITER_VALIDATION_COUNTS="${ITER_VALIDATION_COUNTS:-8,12,20}"
ITER_TRAIN_STARTS="${ITER_TRAIN_STARTS:-1500,1600,2000}"
ITER_VALIDATION_STARTS="${ITER_VALIDATION_STARTS:-3200,3230,3300}"
G0_TRAIN_START="${G0_TRAIN_START:-1500}"
G0_VALIDATION_START="${G0_VALIDATION_START:-3200}"
G0_CHUNK_SIZE="${G0_CHUNK_SIZE:-10}"
G0_ROOT_FRACTIONS="${G0_ROOT_FRACTIONS:-0.15:0.25:0.35:0.45:0.55:0.65:0.75:0.85}"
G0_ROOTS_PER_SEED="${G0_ROOTS_PER_SEED:-8}"
ITER_CHUNK_SIZE="${ITER_CHUNK_SIZE:-10}"
DRY_RUN="${DRY_RUN:-0}"

IFS=':' read -r -a G0_ROOT_FRACTION_VALUES <<< "$G0_ROOT_FRACTIONS"
if [[ -z "$G0_ROOTS_PER_SEED" ]]; then
  G0_ROOTS_PER_SEED=${#G0_ROOT_FRACTION_VALUES[@]}
fi
IFS=',' read -r -a TRAIN_COUNTS <<< "$ITER_TRAIN_COUNTS"
IFS=',' read -r -a VALIDATION_COUNTS <<< "$ITER_VALIDATION_COUNTS"
IFS=',' read -r -a TRAIN_STARTS <<< "$ITER_TRAIN_STARTS"
IFS=',' read -r -a VALIDATION_STARTS <<< "$ITER_VALIDATION_STARTS"
ITERATIONS=${#TRAIN_COUNTS[@]}

if (( ITERATIONS < 1 )); then
  echo "At least one policy iteration is required" >&2
  exit 2
fi
if (( ${#G0_ROOT_FRACTION_VALUES[@]} < 1 )); then
  echo "At least one G0 root fraction is required" >&2
  exit 2
fi
if ! [[ "$G0_ROOTS_PER_SEED" =~ ^[1-9][0-9]*$ ]] \
  || (( G0_ROOTS_PER_SEED > ${#G0_ROOT_FRACTION_VALUES[@]} )); then
  echo "G0 roots per seed must not exceed the root fraction count" >&2
  exit 2
fi
if (( ${#VALIDATION_COUNTS[@]} != ITERATIONS \
   || ${#TRAIN_STARTS[@]} != ITERATIONS \
   || ${#VALIDATION_STARTS[@]} != ITERATIONS )); then
  echo "All iteration schedule lists must have equal lengths" >&2
  exit 2
fi
for value in \
  "$G0_TRAIN_COUNT" "$G0_VALIDATION_COUNT" "$G0_CHUNK_SIZE" \
  "$ITER_CHUNK_SIZE" "${TRAIN_COUNTS[@]}" "${VALIDATION_COUNTS[@]}"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Counts and chunk sizes must be positive integers: $value" >&2
    exit 2
  fi
done

weighted_train_roots=$((G0_TRAIN_COUNT * G0_ROOTS_PER_SEED))
for count in "${TRAIN_COUNTS[@]}"; do
  weighted_train_roots=$((weighted_train_roots + count * 8))
done
final_stage="p$((ITERATIONS + 1))"

printf 'config=%s\n' "$CONFIG_NAME"
printf 'g0=%s/%s starts=%s/%s\n' \
  "$G0_TRAIN_COUNT" "$G0_VALIDATION_COUNT" \
  "$G0_TRAIN_START" "$G0_VALIDATION_START"
printf 'g0_root_fractions=%s roots_per_seed=%s\n' \
  "$G0_ROOT_FRACTIONS" "$G0_ROOTS_PER_SEED"
printf 'iterations=%s train_counts=%s validation_counts=%s\n' \
  "$ITERATIONS" "$ITER_TRAIN_COUNTS" "$ITER_VALIDATION_COUNTS"
printf 'train_starts=%s validation_starts=%s\n' \
  "$ITER_TRAIN_STARTS" "$ITER_VALIDATION_STARTS"
printf 'weighted_train_roots=%s final_stage=%s\n' \
  "$weighted_train_roots" "$final_stage"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

cd "$PROJECT_DIR"
mkdir -p logs
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing run: $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"

submit_job() {
  local submitted
  submitted=$(sbatch --parsable "$@")
  printf '%s\n' "${submitted%%;*}"
}

ceil_div() {
  printf '%s\n' "$((($1 + $2 - 1) / $2))"
}

g0_train_tasks=$(ceil_div "$G0_TRAIN_COUNT" "$G0_CHUNK_SIZE")
g0_validation_tasks=$(ceil_div "$G0_VALIDATION_COUNT" "$G0_CHUNK_SIZE")
g0_tasks=$((g0_train_tasks + g0_validation_tasks))
g0_throttle=$g0_tasks
if (( g0_throttle > 24 )); then
  g0_throttle=24
fi
g0_job=$(submit_job \
  --array="0-$((g0_tasks - 1))%${g0_throttle}" \
  --job-name="${CONFIG_NAME}_g0" \
  --export=ALL,RUN_ROOT="$RUN_ROOT",TRAIN_START="$G0_TRAIN_START",TRAIN_COUNT="$G0_TRAIN_COUNT",VALIDATION_START="$G0_VALIDATION_START",VALIDATION_COUNT="$G0_VALIDATION_COUNT",CHUNK_SIZE="$G0_CHUNK_SIZE",G0_ROOT_FRACTIONS="$G0_ROOT_FRACTIONS",G0_ROOTS_PER_SEED="$G0_ROOTS_PER_SEED" \
  hpc/submit_iterative_q_greedy_data.sh)
g0_merge_job=$(submit_job \
  --dependency=afterok:"$g0_job" \
  --job-name="${CONFIG_NAME}_g0m" \
  --export=ALL,RUN_ROOT="$RUN_ROOT",STAGE=g0,TRAIN_START="$G0_TRAIN_START",TRAIN_COUNT="$G0_TRAIN_COUNT",VALIDATION_START="$G0_VALIDATION_START",VALIDATION_COUNT="$G0_VALIDATION_COUNT" \
  hpc/submit_iterative_q_merge.sh)
p1_job=$(submit_job \
  --dependency=afterok:"$g0_merge_job" \
  --job-name="${CONFIG_NAME}_p1" \
  --export=ALL,RUN_ROOT="$RUN_ROOT",OUTPUT_STAGE=p1,DATA_STAGES=g0,ENCODER_LR=0.0001,HEAD_LR=0.0003,FOLLOW_ANCHOR=0.5,CREATE_LOCK=1,RESIDUAL_MARGIN=0.10,ECONOMIC_MARGIN_EUR=10000,PROTOCOL_PREFIX="iterative_q_${CONFIG_NAME}" \
  hpc/submit_iterative_q_train.sh)

job_manifest=(
  "g0=$g0_job"
  "g0_merge=$g0_merge_job"
  "p1=$p1_job"
)
previous_train_job=$p1_job
previous_stage=p1
data_stages=g0

for ((index = 0; index < ITERATIONS; index++)); do
  iteration=$((index + 1))
  stage="g${iteration}"
  output_stage="p$((iteration + 1))"
  train_count=${TRAIN_COUNTS[$index]}
  validation_count=${VALIDATION_COUNTS[$index]}
  train_start=${TRAIN_STARTS[$index]}
  validation_start=${VALIDATION_STARTS[$index]}
  train_tasks=$(ceil_div "$train_count" "$ITER_CHUNK_SIZE")
  validation_tasks=$(ceil_div "$validation_count" "$ITER_CHUNK_SIZE")
  rollout_tasks=$((train_tasks + validation_tasks))
  rollout_throttle=$rollout_tasks
  if (( rollout_throttle > 12 )); then
    rollout_throttle=12
  fi
  dataset_seed=$((20260723 + iteration))
  rollout_job=$(submit_job \
    --dependency=afterok:"$previous_train_job" \
    --array="0-$((rollout_tasks - 1))%${rollout_throttle}" \
    --job-name="${CONFIG_NAME}_${stage}" \
    --export=ALL,RUN_ROOT="$RUN_ROOT",STAGE="$stage",LOCK_CONFIG="$RUN_ROOT/${previous_stage}_lock.json",TRAIN_START="$train_start",TRAIN_COUNT="$train_count",VALIDATION_START="$validation_start",VALIDATION_COUNT="$validation_count",CHUNK_SIZE="$ITER_CHUNK_SIZE",DATASET_SEED="$dataset_seed" \
    hpc/submit_iterative_q_policy_data.sh)
  merge_job=$(submit_job \
    --dependency=afterok:"$rollout_job" \
    --job-name="${CONFIG_NAME}_${stage}m" \
    --export=ALL,RUN_ROOT="$RUN_ROOT",STAGE="$stage",TRAIN_START="$train_start",TRAIN_COUNT="$train_count",VALIDATION_START="$validation_start",VALIDATION_COUNT="$validation_count" \
    hpc/submit_iterative_q_merge.sh)
  data_stages="${data_stages}:${stage}"
  create_lock=0
  if (( index + 1 < ITERATIONS )); then
    create_lock=1
  fi
  encoder_lr=0.00003
  head_lr=0.0001
  if (( iteration == 1 )); then
    encoder_lr=0.00005
    head_lr=0.00015
  fi
  train_job=$(submit_job \
    --dependency=afterok:"$merge_job" \
    --job-name="${CONFIG_NAME}_${output_stage}" \
    --export=ALL,RUN_ROOT="$RUN_ROOT",OUTPUT_STAGE="$output_stage",DATA_STAGES="$data_stages",INITIAL_CHECKPOINT="$RUN_ROOT/$previous_stage/iterative_action_q.pt",ENCODER_LR="$encoder_lr",HEAD_LR="$head_lr",FOLLOW_ANCHOR=0.0,CREATE_LOCK="$create_lock",RESIDUAL_MARGIN=0.15,ECONOMIC_MARGIN_EUR=15000,PROTOCOL_PREFIX="iterative_q_${CONFIG_NAME}" \
    hpc/submit_iterative_q_train.sh)
  job_manifest+=(
    "$stage=$rollout_job"
    "${stage}_merge=$merge_job"
    "$output_stage=$train_job"
  )
  previous_train_job=$train_job
  previous_stage=$output_stage
done

eval_job=$(submit_job \
  --dependency=afterok:"$previous_train_job" \
  --job-name="${CONFIG_NAME}_eval" \
  --export=ALL,RUN_ROOT="$RUN_ROOT",EVAL_NAME="$CONFIG_NAME",FINAL_STAGE="$final_stage" \
  hpc/submit_iterative_q_eval.sh)
job_manifest+=("eval=$eval_job")

{
  printf '%s\n' "${job_manifest[@]}"
} | tee "$RUN_ROOT/job_ids.txt"
{
  printf 'config=%s\n' "$CONFIG_NAME"
  printf 'g0_train_count=%s\n' "$G0_TRAIN_COUNT"
  printf 'g0_validation_count=%s\n' "$G0_VALIDATION_COUNT"
  printf 'g0_root_fractions=%s\n' "$G0_ROOT_FRACTIONS"
  printf 'g0_roots_per_seed=%s\n' "$G0_ROOTS_PER_SEED"
  printf 'iteration_train_counts=%s\n' "$ITER_TRAIN_COUNTS"
  printf 'iteration_validation_counts=%s\n' "$ITER_VALIDATION_COUNTS"
  printf 'iteration_train_starts=%s\n' "$ITER_TRAIN_STARTS"
  printf 'iteration_validation_starts=%s\n' "$ITER_VALIDATION_STARTS"
  printf 'weighted_train_roots=%s\n' "$weighted_train_roots"
  printf 'final_stage=%s\n' "$final_stage"
} > "$RUN_ROOT/schedule.txt"
