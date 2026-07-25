#!/usr/bin/env bash
#SBATCH --job-name=ccs_iter_q_g0
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -o logs/iterative_q_g0-%A_%a.out
#SBATCH -e logs/iterative_q_g0-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_greedy_dagger}"
: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${TRAIN_START:?TRAIN_START must be set}"
: "${TRAIN_COUNT:?TRAIN_COUNT must be set}"
: "${VALIDATION_START:?VALIDATION_START must be set}"
: "${VALIDATION_COUNT:?VALIDATION_COUNT must be set}"
: "${CHUNK_SIZE:?CHUNK_SIZE must be set}"
G0_ROOT_FRACTIONS="${G0_ROOT_FRACTIONS:-0.15:0.25:0.35:0.45:0.55:0.65:0.75:0.85}"
IFS=':' read -r -a ROOT_FRACTIONS <<< "$G0_ROOT_FRACTIONS"
G0_ROOTS_PER_SEED="${G0_ROOTS_PER_SEED:-${#ROOT_FRACTIONS[@]}}"

TRAIN_TASKS=$(((TRAIN_COUNT + CHUNK_SIZE - 1) / CHUNK_SIZE))
if (( SLURM_ARRAY_TASK_ID < TRAIN_TASKS )); then
  SPLIT=train
  RANGE_START=$((TRAIN_START + SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
  LIMIT=$((TRAIN_START + TRAIN_COUNT - 1))
else
  SPLIT=validation
  VALIDATION_TASK_ID=$((SLURM_ARRAY_TASK_ID - TRAIN_TASKS))
  RANGE_START=$((VALIDATION_START + VALIDATION_TASK_ID * CHUNK_SIZE))
  LIMIT=$((VALIDATION_START + VALIDATION_COUNT - 1))
fi
RANGE_END=$((RANGE_START + CHUNK_SIZE - 1))
if (( RANGE_END > LIMIT )); then
  RANGE_END=$LIMIT
fi
mapfile -t SEEDS < <(seq "$RANGE_START" "$RANGE_END")

cd "$PROJECT_DIR"
mkdir -p logs "$RUN_ROOT/g0/$SPLIT"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u experiments/generate_iterative_q_greedy_data.py \
  --out-path "$RUN_ROOT/g0/$SPLIT/shard_${RANGE_START}_${RANGE_END}.npz" \
  --split "$SPLIT" \
  --seeds "${SEEDS[@]}" \
  --root-fractions "${ROOT_FRACTIONS[@]}" \
  --roots-per-seed "$G0_ROOTS_PER_SEED" \
  --max-two-vessel-actions 8 \
  --max-three-vessel-actions 4 \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --dataset-seed 20260723 \
  --variant future_mlp_mode_destination \
  --device cpu
