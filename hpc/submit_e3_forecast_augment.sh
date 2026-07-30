#!/usr/bin/env bash
#SBATCH --job-name=e3_fc_aug
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH -o logs/e3_fc_aug-%A_%a.out
#SBATCH -e logs/e3_fc_aug-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"
OUT_ROOT="${OUT_ROOT:-experiments_results/E3/training_future_information_run01/augmented_data}"
STAGE_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
SPLIT_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
STAGE="g${STAGE_INDEX}"
if [[ "$SPLIT_INDEX" == "0" ]]; then
  SPLIT=train
else
  SPLIT=validation
fi

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u experiments/augment_iterative_q_forecasts.py \
  --input-path "$SOURCE_RUN_ROOT/$STAGE/${SPLIT}_merged.npz" \
  --out-path "$OUT_ROOT/$STAGE/${SPLIT}_forecast168.npz" \
  --horizon-h 168
