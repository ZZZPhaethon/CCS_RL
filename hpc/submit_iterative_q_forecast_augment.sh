#!/usr/bin/env bash
#SBATCH --job-name=ccs_iq_fc_aug
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH -o logs/iterative_q_forecast_augment-%A_%a.out
#SBATCH -e logs/iterative_q_forecast_augment-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
: "${SOURCE_RUN_ROOT:?SOURCE_RUN_ROOT must be set}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

STAGE_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
SPLIT_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
STAGE="g${STAGE_INDEX}"
if [[ "$SPLIT_INDEX" == "0" ]]; then
  SPLIT="train"
else
  SPLIT="validation"
fi

python -u experiments/augment_iterative_q_forecasts.py \
  --input-path "$SOURCE_RUN_ROOT/$STAGE/${SPLIT}_merged.npz" \
  --out-path "$SOURCE_RUN_ROOT/$STAGE/${SPLIT}_forecast168.npz" \
  --horizon-h 168
