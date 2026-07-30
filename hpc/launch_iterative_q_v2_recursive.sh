#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-output/iterative_q_validation_search/uniform_margin40_p1_p4}"
RUN_ROOT="${RUN_ROOT:-output/iterative_q_validation_search/iterative_q_v2_recursive_from_p0}"
CONFIG_NAME="${CONFIG_NAME:-iterative_q_v2_recursive}"
EVAL_SEEDS="${EVAL_SEEDS:-8100001:8100002:8100003:8100004:8100005:8100006:8100007:8100008:8100009:8100010:8100011:8100012:8100013:8100014:8100015:8100016:8100017:8100018:8100019:8100020}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680}"
ANCHOR_COEFFICIENT="${ANCHOR_COEFFICIENT:-1.0}"
P1_ANCHOR_COEFFICIENT="${P1_ANCHOR_COEFFICIENT:-$ANCHOR_COEFFICIENT}"
P1_FOLLOW_ANCHOR="${P1_FOLLOW_ANCHOR:-0.0}"
ANCHOR_WEIGHTING="${ANCHOR_WEIGHTING:-hard}"
PLATEAU_MARGIN_EUR="${PLATEAU_MARGIN_EUR:-0}"
RELEASE_MARGIN_EUR="${RELEASE_MARGIN_EUR:-40000}"
ANCHOR_TEMPERATURE="${ANCHOR_TEMPERATURE:-0.5}"

cd "$PROJECT_DIR"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing recursive-v2 run: $RUN_ROOT" >&2
  exit 2
fi
for path in \
  "$SOURCE_DATA_ROOT/g0/train_merged.npz" \
  "$SOURCE_DATA_ROOT/g0/validation_merged.npz"; do
  if [[ ! -s "$path" ]]; then
    echo "Required P0/Greedy exact data is missing: $path" >&2
    exit 2
  fi
done

mkdir -p "$RUN_ROOT" logs
manifest="$RUN_ROOT/job_manifest.txt"
schedule="$RUN_ROOT/schedule.txt"
lock="$RUN_ROOT/protocol_lock.txt"

source_train_sha=$(sha256sum "$SOURCE_DATA_ROOT/g0/train_merged.npz" | awk '{print $1}')
source_validation_sha=$(sha256sum "$SOURCE_DATA_ROOT/g0/validation_merged.npz" | awk '{print $1}')
{
  printf 'algorithm=Iterative Q v2 recursive teacher\n'
  printf 'p0_teacher=Greedy/FOLLOW\n'
  printf 'p1_teacher=P0\n'
  printf 'p2_teacher=P1\n'
  printf 'p3_teacher=P2\n'
  printf 'p4_teacher=P3\n'
  printf 'student_initialization=P1_random,P2-P4_previous_checkpoint\n'
  printf 'cumulative_data=P1:G0,P2:G0-G1,P3:G0-G2,P4:G0-G3\n'
  printf 'source_g0_train=%s\n' "$SOURCE_DATA_ROOT/g0/train_merged.npz"
  printf 'source_g0_train_sha256=%s\n' "$source_train_sha"
  printf 'source_g0_validation=%s\n' "$SOURCE_DATA_ROOT/g0/validation_merged.npz"
  printf 'source_g0_validation_sha256=%s\n' "$source_validation_sha"
  printf 'anchor_weighting=%s\n' "$ANCHOR_WEIGHTING"
  printf 'plateau_margin_eur=%s\n' "$PLATEAU_MARGIN_EUR"
  printf 'p1_anchor_coefficient=%s\n' "$P1_ANCHOR_COEFFICIENT"
  printf 'p1_follow_anchor=%s\n' "$P1_FOLLOW_ANCHOR"
  printf 'anchor_coefficient=%s\n' "$ANCHOR_COEFFICIENT"
  printf 'release_margin_eur=%s\n' "$RELEASE_MARGIN_EUR"
  printf 'anchor_temperature=%s\n' "$ANCHOR_TEMPERATURE"
  printf 'model_seed=0\n'
  printf 'validation_seeds=%s\n' "$EVAL_SEEDS"
  printf 'formal_test_accessed=false\n'
} > "$lock"

env_job=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",SOURCE_DATA_ROOT="$SOURCE_DATA_ROOT" \
  hpc/submit_iterative_q_v2_recursive_env_check.sh)
prepare_job=$(sbatch --parsable \
  --dependency=afterok:"$env_job" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",SOURCE_DATA_ROOT="$SOURCE_DATA_ROOT",RUN_ROOT="$RUN_ROOT" \
  hpc/submit_iterative_q_v2_prepare_p0_data.sh)
p1_job=$(sbatch --parsable \
  --dependency=afterok:"$prepare_job" \
  --job-name="${CONFIG_NAME}_p1" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",DATA_ROOT="$RUN_ROOT",OUTPUT_STAGE=p1,DATA_STAGES=g0,ENCODER_LR=0.0001,HEAD_LR=0.0003,FOLLOW_ANCHOR="$P1_FOLLOW_ANCHOR",PREVIOUS_POLICY_ANCHOR="$P1_ANCHOR_COEFFICIENT",PREVIOUS_POLICY_RELEASE_MARGIN_EUR="$RELEASE_MARGIN_EUR",PREVIOUS_POLICY_PLATEAU_MARGIN_EUR="$PLATEAU_MARGIN_EUR",PREVIOUS_POLICY_ANCHOR_TEMPERATURE="$ANCHOR_TEMPERATURE",PREVIOUS_POLICY_ANCHOR_WEIGHTING="$ANCHOR_WEIGHTING",ALLOW_ANCHOR_WITHOUT_INITIAL_CHECKPOINT=1,CREATE_LOCK=1,RESIDUAL_MARGIN=0.40,ECONOMIC_MARGIN_EUR=40000,PROTOCOL_PREFIX="$CONFIG_NAME",OBSERVATION_INPUT=shared_future_summary,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,MODEL_SEED=0 \
  hpc/submit_iterative_q_train.sh)
p1_eval_job=$(sbatch --parsable \
  --dependency=afterok:"$p1_job" \
  --job-name="${CONFIG_NAME}_p1_eval" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",FINAL_STAGE=p1,EVAL_NAME="${CONFIG_NAME}_p1",SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,EVAL_SEEDS="$EVAL_SEEDS",VALIDATION_ONLY=1 \
  hpc/submit_iterative_q_eval.sh)

{
  printf 'env_check=%s\n' "$env_job"
  printf 'prepare_p0_data=%s\n' "$prepare_job"
  printf 'p1_train=%s\n' "$p1_job"
  printf 'p1_eval=%s\n' "$p1_eval_job"
} | tee "$manifest"

previous_train_job="$p1_job"
previous_stage=p1
data_stages=g0
train_counts=(40 60 100)
validation_counts=(8 12 20)
train_starts=(1500 1600 2000)
validation_starts=(3200 3230 3300)
rollout_tasks=(5 8 12)

for index in 0 1 2; do
  iteration=$((index + 1))
  data_stage="g$iteration"
  output_stage="p$((iteration + 1))"
  data_stages="${data_stages}:${data_stage}"
  create_lock=1
  if [[ "$output_stage" == "p4" ]]; then
    create_lock=0
  fi
  encoder_lr=0.00003
  head_lr=0.0001
  if [[ "$output_stage" == "p2" ]]; then
    encoder_lr=0.00005
    head_lr=0.00015
  fi
  dataset_seed=$((20260723 + iteration))
  rollout_job=$(sbatch --parsable \
    --dependency=afterok:"$previous_train_job" \
    --array="0-$((rollout_tasks[index] - 1))%${rollout_tasks[index]}" \
    --job-name="${CONFIG_NAME}_${data_stage}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",STAGE="$data_stage",LOCK_CONFIG="$RUN_ROOT/${previous_stage}_lock.json",TRAIN_START="${train_starts[index]}",TRAIN_COUNT="${train_counts[index]}",VALIDATION_START="${validation_starts[index]}",VALIDATION_COUNT="${validation_counts[index]}",CHUNK_SIZE=10,DATASET_SEED="$dataset_seed",SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168 \
    hpc/submit_iterative_q_policy_data.sh)
  merge_job=$(sbatch --parsable \
    --dependency=afterok:"$rollout_job" \
    --job-name="${CONFIG_NAME}_${data_stage}_merge" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",STAGE="$data_stage",TRAIN_START="${train_starts[index]}",TRAIN_COUNT="${train_counts[index]}",VALIDATION_START="${validation_starts[index]}",VALIDATION_COUNT="${validation_counts[index]}" \
    hpc/submit_iterative_q_merge.sh)
  train_job=$(sbatch --parsable \
    --dependency=afterok:"$merge_job" \
    --job-name="${CONFIG_NAME}_${output_stage}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",DATA_ROOT="$RUN_ROOT",OUTPUT_STAGE="$output_stage",DATA_STAGES="$data_stages",INITIAL_CHECKPOINT="$RUN_ROOT/$previous_stage/iterative_action_q.pt",ENCODER_LR="$encoder_lr",HEAD_LR="$head_lr",FOLLOW_ANCHOR=0.0,PREVIOUS_POLICY_ANCHOR="$ANCHOR_COEFFICIENT",PREVIOUS_POLICY_RELEASE_MARGIN_EUR="$RELEASE_MARGIN_EUR",PREVIOUS_POLICY_PLATEAU_MARGIN_EUR="$PLATEAU_MARGIN_EUR",PREVIOUS_POLICY_ANCHOR_TEMPERATURE="$ANCHOR_TEMPERATURE",PREVIOUS_POLICY_ANCHOR_WEIGHTING="$ANCHOR_WEIGHTING",CREATE_LOCK="$create_lock",RESIDUAL_MARGIN=0.40,ECONOMIC_MARGIN_EUR=40000,PROTOCOL_PREFIX="$CONFIG_NAME",OBSERVATION_INPUT=shared_future_summary,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,MODEL_SEED=0 \
    hpc/submit_iterative_q_train.sh)
  eval_job=$(sbatch --parsable \
    --dependency=afterok:"$train_job" \
    --job-name="${CONFIG_NAME}_${output_stage}_eval" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",FINAL_STAGE="$output_stage",EVAL_NAME="${CONFIG_NAME}_${output_stage}",SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,EVAL_SEEDS="$EVAL_SEEDS",VALIDATION_ONLY=1 \
    hpc/submit_iterative_q_eval.sh)
  {
    printf '%s_rollout=%s\n' "$data_stage" "$rollout_job"
    printf '%s_merge=%s\n' "$data_stage" "$merge_job"
    printf '%s_train=%s\n' "$output_stage" "$train_job"
    printf '%s_eval=%s\n' "$output_stage" "$eval_job"
  } | tee -a "$manifest"
  previous_train_job="$train_job"
  previous_stage="$output_stage"
done

{
  printf 'name=Iterative Q v2 complete recursive teacher\n'
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'teacher_chain=P0(Greedy)->P1->P2->P3->P4\n'
  printf 'data_chain=G0(P0),G1(P1),G2(P2),G3(P3)\n'
  printf 'iteration_train_counts=40,60,100\n'
  printf 'iteration_validation_counts=8,12,20\n'
  printf 'iteration_train_starts=1500,1600,2000\n'
  printf 'iteration_validation_starts=3200,3230,3300\n'
  printf 'p1_anchor_coefficient=%s\n' "$P1_ANCHOR_COEFFICIENT"
  printf 'p1_follow_anchor=%s\n' "$P1_FOLLOW_ANCHOR"
  printf 'anchor_weighting=%s\n' "$ANCHOR_WEIGHTING"
  printf 'plateau_margin_eur=%s\n' "$PLATEAU_MARGIN_EUR"
  printf 'anchor_coefficient=%s\n' "$ANCHOR_COEFFICIENT"
  printf 'release_margin_eur=%s\n' "$RELEASE_MARGIN_EUR"
  printf 'anchor_temperature=%s\n' "$ANCHOR_TEMPERATURE"
  printf 'policy_windows_h=%s\n' "$POLICY_WINDOWS_H"
  printf 'max_overrides=12\n'
  printf 'evaluation_seeds=%s\n' "$EVAL_SEEDS"
  printf 'validation_only=1\n'
} > "$schedule"
