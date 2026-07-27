#!/usr/bin/env bash
#SBATCH --job-name=ccs_iter_q_merge
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH -o logs/iterative_q_merge-%j.out
#SBATCH -e logs/iterative_q_merge-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_greedy_dagger}"
: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${STAGE:?STAGE must be set}"
: "${TRAIN_START:?TRAIN_START must be set}"
: "${TRAIN_COUNT:?TRAIN_COUNT must be set}"
: "${VALIDATION_START:?VALIDATION_START must be set}"
: "${VALIDATION_COUNT:?VALIDATION_COUNT must be set}"

TRAIN_END=$((TRAIN_START + TRAIN_COUNT - 1))
VALIDATION_END=$((VALIDATION_START + VALIDATION_COUNT - 1))
cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
mapfile -t TRAIN_SEEDS < <(seq "$TRAIN_START" "$TRAIN_END")
mapfile -t VALIDATION_SEEDS < <(seq "$VALIDATION_START" "$VALIDATION_END")
TRAIN_SHARDS=("$RUN_ROOT/$STAGE"/train/shard_*.npz)
VALIDATION_SHARDS=("$RUN_ROOT/$STAGE"/validation/shard_*.npz)

python -u experiments/merge_iterative_q_data.py \
  --shards "${TRAIN_SHARDS[@]}" \
  --out-path "$RUN_ROOT/$STAGE/train_merged.npz" \
  --expected-split train \
  --expected-seeds "${TRAIN_SEEDS[@]}"

python -u experiments/merge_iterative_q_data.py \
  --shards "${VALIDATION_SHARDS[@]}" \
  --out-path "$RUN_ROOT/$STAGE/validation_merged.npz" \
  --expected-split validation \
  --expected-seeds "${VALIDATION_SEEDS[@]}"
