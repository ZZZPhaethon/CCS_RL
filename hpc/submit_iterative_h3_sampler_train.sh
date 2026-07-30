#!/usr/bin/env bash
#SBATCH --job-name=iter_h3_train
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH -o logs/iterative_h3_train-%A_%a.out
#SBATCH -e logs/iterative_h3_train-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${OUT_ROOT:?OUT_ROOT must be set}"
: "${SOURCE_RUN:?SOURCE_RUN must be set}"
: "${SEED_CHECKPOINT_ROOT:?SEED_CHECKPOINT_ROOT must be set}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680}"

MODEL_SEED=$((SLURM_ARRAY_TASK_ID % 3))
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
case "$VARIANT_INDEX" in
  0)
    VARIANT=b_gate_only
    STAGE_SAMPLING_TEMPERATURE=1.0
    NEAR_DUPLICATE_WEIGHTING=none
    ROOT_ADVANTAGE_WEIGHTING=none
    ;;
  1)
    VARIANT=c_dedup_balanced
    STAGE_SAMPLING_TEMPERATURE=0.5
    NEAR_DUPLICATE_WEIGHTING=inverse_cluster
    ROOT_ADVANTAGE_WEIGHTING=none
    ;;
  2)
    VARIANT=d_dedup_advantage
    STAGE_SAMPLING_TEMPERATURE=0.5
    NEAR_DUPLICATE_WEIGHTING=inverse_cluster
    ROOT_ADVANTAGE_WEIGHTING=stratified
    ;;
  *)
    echo "Unknown variant index: $VARIANT_INDEX" >&2
    exit 2
    ;;
esac

if [[ "$MODEL_SEED" == "0" ]]; then
  INITIAL_CHECKPOINT="$SOURCE_RUN/p1/iterative_action_q.pt"
else
  INITIAL_CHECKPOINT="$SEED_CHECKPOINT_ROOT/model_seed_${MODEL_SEED}/p1/iterative_action_q.pt"
fi
RUN_ROOT="$OUT_ROOT/branches/$VARIANT/model_seed_${MODEL_SEED}"
if [[ -e "$RUN_ROOT/p2" || -e "$RUN_ROOT/p2_lock.json" ]]; then
  echo "Refusing existing P2 output: $RUN_ROOT" >&2
  exit 2
fi

export PROJECT_DIR OUT_ROOT SOURCE_RUN SEED_CHECKPOINT_ROOT
export RUN_ROOT
export DATA_ROOT="$OUT_ROOT/shared"
export OUTPUT_STAGE=p2
export DATA_STAGES=g0:g1
export INITIAL_CHECKPOINT
export ENCODER_LR=0.00005
export HEAD_LR=0.00015
export FOLLOW_ANCHOR=0.0
export CREATE_LOCK=1
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
