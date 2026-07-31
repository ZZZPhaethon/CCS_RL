#!/usr/bin/env bash
#SBATCH --job-name=iter_h3_merge
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH -o logs/iterative_h3_merge-%A_%a.out
#SBATCH -e logs/iterative_h3_merge-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${OUT_ROOT:?OUT_ROOT must be set}"
: "${SELECTED_VARIANT:?SELECTED_VARIANT must be set}"
: "${STAGE:?STAGE must be set}"
: "${TRAIN_START:?TRAIN_START must be set}"
: "${TRAIN_COUNT:?TRAIN_COUNT must be set}"
: "${VALIDATION_START:?VALIDATION_START must be set}"
: "${VALIDATION_COUNT:?VALIDATION_COUNT must be set}"

MODEL_SEED=$((SLURM_ARRAY_TASK_ID % 3))
ROUTE_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
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
export STAGE TRAIN_START TRAIN_COUNT VALIDATION_START VALIDATION_COUNT

exec bash "$PROJECT_DIR/hpc/submit_iterative_q_merge.sh"
