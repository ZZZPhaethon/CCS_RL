#!/usr/bin/env bash
#SBATCH --job-name=iter_h3_roll
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/iterative_h3_roll-%A_%a.out
#SBATCH -e logs/iterative_h3_roll-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${OUT_ROOT:?OUT_ROOT must be set}"
: "${SELECTED_VARIANT:?SELECTED_VARIANT must be set}"
: "${STAGE:?STAGE must be set}"
: "${PREVIOUS_STAGE:?PREVIOUS_STAGE must be set}"
: "${TRAIN_START:?TRAIN_START must be set}"
: "${TRAIN_COUNT:?TRAIN_COUNT must be set}"
: "${VALIDATION_START:?VALIDATION_START must be set}"
: "${VALIDATION_COUNT:?VALIDATION_COUNT must be set}"
: "${CHUNK_SIZE:?CHUNK_SIZE must be set}"
: "${DATASET_SEED:?DATASET_SEED must be set}"

TRAIN_TASKS=$(((TRAIN_COUNT + CHUNK_SIZE - 1) / CHUNK_SIZE))
VALIDATION_TASKS=$(((VALIDATION_COUNT + CHUNK_SIZE - 1) / CHUNK_SIZE))
TASKS_PER_BRANCH=$((TRAIN_TASKS + VALIDATION_TASKS))
BRANCH_INDEX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_BRANCH))
LOCAL_TASK_ID=$((SLURM_ARRAY_TASK_ID % TASKS_PER_BRANCH))
MODEL_SEED=$((BRANCH_INDEX % 3))
ROUTE_INDEX=$((BRANCH_INDEX / 3))
case "$ROUTE_INDEX" in
  0) VARIANT=b_gate_only ;;
  1) VARIANT="$SELECTED_VARIANT" ;;
  *)
    echo "Unknown route index: $ROUTE_INDEX" >&2
    exit 2
    ;;
esac

export PROJECT_DIR
export RUN_ROOT="$OUT_ROOT/branches/$VARIANT/model_seed_${MODEL_SEED}"
export LOCK_CONFIG="$RUN_ROOT/${PREVIOUS_STAGE}_lock.json"
export SLURM_ARRAY_TASK_ID="$LOCAL_TASK_ID"
export STAGE TRAIN_START TRAIN_COUNT VALIDATION_START VALIDATION_COUNT
export CHUNK_SIZE DATASET_SEED
export SCENARIO_PROTOCOL=unified_window_v1
export HARD_SCENARIO_PROBABILITY=0.5
export FORECAST_CONTEXT_HOURS=168

exec bash "$PROJECT_DIR/hpc/submit_iterative_q_policy_data.sh"
