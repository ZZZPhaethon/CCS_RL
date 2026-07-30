#!/usr/bin/env bash
#SBATCH --job-name=iterq_v2_p0_data
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH -o logs/iterative_q_v2_prepare_p0-%j.out
#SBATCH -e logs/iterative_q_v2_prepare_p0-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${SOURCE_DATA_ROOT:?SOURCE_DATA_ROOT must be set}"
: "${RUN_ROOT:?RUN_ROOT must be set}"

cd "$PROJECT_DIR"
mkdir -p logs "$RUN_ROOT/g0"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python experiments/prepare_iterative_q_p0_anchor_data.py \
  --source "$SOURCE_DATA_ROOT/g0/train_merged.npz" \
  --output "$RUN_ROOT/g0/train_merged.npz"
python experiments/prepare_iterative_q_p0_anchor_data.py \
  --source "$SOURCE_DATA_ROOT/g0/validation_merged.npz" \
  --output "$RUN_ROOT/g0/validation_merged.npz"

sha256sum \
  "$SOURCE_DATA_ROOT/g0/train_merged.npz" \
  "$SOURCE_DATA_ROOT/g0/validation_merged.npz" \
  "$RUN_ROOT/g0/train_merged.npz" \
  "$RUN_ROOT/g0/validation_merged.npz" \
  > "$RUN_ROOT/g0/p0_anchor_data_sha256.txt"
