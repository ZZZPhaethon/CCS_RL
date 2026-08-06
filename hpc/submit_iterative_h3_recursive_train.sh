#!/usr/bin/env bash
#SBATCH --job-name=iter_h3_retrain
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o logs/iterative_h3_retrain-%A_%a.out
#SBATCH -e logs/iterative_h3_retrain-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${OUT_ROOT:?OUT_ROOT must be set}"
: "${SELECTED_VARIANT:?SELECTED_VARIANT must be set}"
: "${OUTPUT_STAGE:?OUTPUT_STAGE must be set}"
: "${PREVIOUS_STAGE:?PREVIOUS_STAGE must be set}"
: "${DATA_STAGES:?DATA_STAGES must be set}"
: "${ENCODER_LR:?ENCODER_LR must be set}"
: "${HEAD_LR:?HEAD_LR must be set}"
: "${CREATE_LOCK:?CREATE_LOCK must be set}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680}"

MODEL_SEED=$((SLURM_ARRAY_TASK_ID % 3))
ROUTE_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
case "$ROUTE_INDEX" in
  0)
    VARIANT=b_gate_only
    STAGE_SAMPLING_TEMPERATURE=1.0
    NEAR_DUPLICATE_WEIGHTING=none
    ROOT_ADVANTAGE_WEIGHTING=none
    ;;
  1)
    VARIANT="$SELECTED_VARIANT"
    STAGE_SAMPLING_TEMPERATURE=0.5
    NEAR_DUPLICATE_WEIGHTING=inverse_cluster
    if [[ "$VARIANT" == "d_dedup_advantage" ]]; then
      ROOT_ADVANTAGE_WEIGHTING=stratified
    else
      ROOT_ADVANTAGE_WEIGHTING=none
    fi
    ;;
  *)
    echo "Unknown route index: $ROUTE_INDEX" >&2
    exit 2
    ;;
esac

RUN_ROOT="$OUT_ROOT/branches/$VARIANT/model_seed_${MODEL_SEED}"
if [[ -e "$RUN_ROOT/$OUTPUT_STAGE" ]]; then
  echo "Refusing existing stage output: $RUN_ROOT/$OUTPUT_STAGE" >&2
  exit 2
fi

export PROJECT_DIR RUN_ROOT
export DATA_ROOT="$RUN_ROOT"
export OUTPUT_STAGE PREVIOUS_STAGE DATA_STAGES ENCODER_LR HEAD_LR
export INITIAL_CHECKPOINT="$RUN_ROOT/$PREVIOUS_STAGE/iterative_action_q.pt"
export FOLLOW_ANCHOR=0.0
export CREATE_LOCK
export RESIDUAL_MARGIN=0.4
export ECONOMIC_MARGIN_EUR=40000
export REQUIRED_HEADS=3
export PROTOCOL_PREFIX="iterative_h3_${VARIANT}_seed${MODEL_SEED}"
export OBSERVATION_INPUT=shared_future_summary
export POLICY_WINDOWS_H
export MAX_OVERRIDES=12
export MODEL_SEED
export STAGE_SAMPLING_TEMPERATURE
export NEAR_DUPLICATE_WEIGHTING
export NEAR_DUPLICATE_COSINE_THRESHOLD=0.995
export NEAR_DUPLICATE_RMS_THRESHOLD=0.10
export ROOT_ADVANTAGE_WEIGHTING
export ROOT_ADVANTAGE_THRESHOLD_EUR=40000
export ROOT_NO_IMPROVEMENT_WEIGHT=0.5
export ROOT_MODERATE_IMPROVEMENT_WEIGHT=1.0
export ROOT_STRONG_IMPROVEMENT_WEIGHT=2.0
export PREVIOUS_POLICY_ANCHOR=0.0

exec bash "$PROJECT_DIR/hpc/submit_iterative_q_train.sh"
