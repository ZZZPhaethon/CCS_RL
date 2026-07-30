#!/usr/bin/env bash
#SBATCH --job-name=e2_g0_extra
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-12%13
#SBATCH -o logs/e2_g0_extra-%A_%a.out
#SBATCH -e logs/e2_g0_extra-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
RUN_ROOT="${RUN_ROOT:-experiments_results/E2/training_one_shot_matched_run01}"
TRAIN_START=1680
TRAIN_COUNT=121
CHUNK_SIZE=10
RANGE_START=$((TRAIN_START + SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
LIMIT=$((TRAIN_START + TRAIN_COUNT - 1))
RANGE_END=$((RANGE_START + CHUNK_SIZE - 1))
if (( RANGE_END > LIMIT )); then
  RANGE_END=$LIMIT
fi
mapfile -t SEEDS < <(seq "$RANGE_START" "$RANGE_END")

cd "$PROJECT_DIR"
mkdir -p logs "$RUN_ROOT/g0/extra"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u experiments/generate_iterative_q_greedy_data.py \
  --out-path "$RUN_ROOT/g0/extra/shard_${RANGE_START}_${RANGE_END}.npz" \
  --split train \
  --seeds "${SEEDS[@]}" \
  --root-fractions \
    0.15 0.2166666667 0.2833333333 0.35 0.4166666667 0.4833333333 \
    0.55 0.6166666667 0.6833333333 0.75 0.8166666667 0.8833333333 \
  --roots-per-seed 12 \
  --max-two-vessel-actions 8 \
  --max-three-vessel-actions 4 \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --dataset-seed 20260723 \
  --variant future_mlp_mode_destination \
  --scenario-protocol unified_window_v1 \
  --stress-level medium \
  --hard-scenario-probability 0.5 \
  --forecast-context-hours 168 \
  --device cpu
