#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_greedy_dagger}"
CONFIG_NAME="${CONFIG_NAME:-window_3iter_3200_eta_joint}"
RUN_ROOT="${RUN_ROOT:-output/rl_forecast/event_window_3200_eta_joint_20260724}"
FUTURE_ENCODER="${FUTURE_ENCODER:-eta_joint}"
SOURCE_ROOT="output/rl_forecast/event_window_3200_allocation_sweep_20260724/window_3iter_3200_initial"
NORMALIZATION_CHECKPOINT="output/rl_forecast/event_recurrent_q_online_100k_v1_20260723/checkpoints/recurrent_distributional_q_100000.pt"
TRAIN_COUNTS=(40 60 100)
VALIDATION_COUNTS=(8 12 20)
TRAIN_STARTS=(1500 1600 2000)
VALIDATION_STARTS=(3200 3230 3300)
ITER_CHUNK_SIZE=10

cd "$PROJECT_DIR"
mkdir -p logs
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing run: $RUN_ROOT" >&2
  exit 2
fi
for path in \
  "$SOURCE_ROOT/g0/train_merged.npz" \
  "$SOURCE_ROOT/g0/validation_merged.npz" \
  "$NORMALIZATION_CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
done

mkdir -p "$RUN_ROOT/g0"
cp "$SOURCE_ROOT/g0/train_merged.npz" "$RUN_ROOT/g0/train_merged.npz"
cp "$SOURCE_ROOT/g0/validation_merged.npz" "$RUN_ROOT/g0/validation_merged.npz"

submit_job() {
  local submitted
  submitted=$(sbatch --parsable "$@")
  printf '%s\n' "${submitted%%;*}"
}

ceil_div() {
  printf '%s\n' "$((($1 + $2 - 1) / $2))"
}

p1_job=$(submit_job \
  --job-name="${CONFIG_NAME}_p1" \
  --export=ALL,RUN_ROOT="$RUN_ROOT",OUTPUT_STAGE=p1,DATA_STAGES=g0,INITIAL_CHECKPOINT="$NORMALIZATION_CHECKPOINT",ENCODER_LR=0.0001,HEAD_LR=0.0003,FOLLOW_ANCHOR=0.5,SKIP_INITIAL_WEIGHTS=1,CREATE_LOCK=1,RESIDUAL_MARGIN=0.10,ECONOMIC_MARGIN_EUR=10000,PROTOCOL_PREFIX="event_root_${CONFIG_NAME}",OBSERVATION_INPUT=state_future,FORECAST_ENCODER="$FUTURE_ENCODER" \
  hpc/submit_event_root_schedule_train.sh)

job_manifest=("p1=$p1_job")
previous_train_job=$p1_job
previous_stage=p1
data_stages=g0

for index in "${!TRAIN_COUNTS[@]}"; do
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
    hpc/submit_event_root_schedule_rollin.sh)
  merge_job=$(submit_job \
    --dependency=afterok:"$rollout_job" \
    --job-name="${CONFIG_NAME}_${stage}m" \
    --export=ALL,RUN_ROOT="$RUN_ROOT",STAGE="$stage",TRAIN_START="$train_start",TRAIN_COUNT="$train_count",VALIDATION_START="$validation_start",VALIDATION_COUNT="$validation_count" \
    hpc/submit_event_root_schedule_merge.sh)

  data_stages="${data_stages}:${stage}"
  create_lock=0
  if (( iteration < ${#TRAIN_COUNTS[@]} )); then
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
    --export=ALL,RUN_ROOT="$RUN_ROOT",OUTPUT_STAGE="$output_stage",DATA_STAGES="$data_stages",INITIAL_CHECKPOINT="$RUN_ROOT/$previous_stage/structured_action_stateless_q.pt",ENCODER_LR="$encoder_lr",HEAD_LR="$head_lr",FOLLOW_ANCHOR=0.0,SKIP_INITIAL_WEIGHTS=0,CREATE_LOCK="$create_lock",RESIDUAL_MARGIN=0.15,ECONOMIC_MARGIN_EUR=15000,PROTOCOL_PREFIX="event_root_${CONFIG_NAME}",OBSERVATION_INPUT=state_future,FORECAST_ENCODER="$FUTURE_ENCODER" \
    hpc/submit_event_root_schedule_train.sh)

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
  --export=ALL,RUN_ROOT="$RUN_ROOT",MODEL_ARCHITECTURE=stateless_structured,EVAL_NAME="$CONFIG_NAME",FINAL_STAGE=p4 \
  hpc/submit_event_single_iter_eval.sh)
job_manifest+=("eval=$eval_job")

printf '%s\n' "${job_manifest[@]}" | tee "$RUN_ROOT/job_ids.txt"
{
  printf 'config=%s\n' "$CONFIG_NAME"
  printf 'source_g0=%s/g0\n' "$SOURCE_ROOT"
  printf 'g0_train_roots=1600\n'
  printf 'iteration_train_roots=320,480,800\n'
  printf 'allocation_percent=50,10,15,25\n'
  printf 'weighted_train_roots=3200\n'
  printf 'future_input=state_future\n'
  printf 'forecast_encoder=%s\n' "$FUTURE_ENCODER"
  printf 'final_stage=p4\n'
} > "$RUN_ROOT/schedule.txt"
