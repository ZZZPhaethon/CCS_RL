#!/usr/bin/env bash
#SBATCH --job-name=iter_h3_audit
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH -o logs/iterative_h3_audit-%j.out
#SBATCH -e logs/iterative_h3_audit-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${OUT_ROOT:?OUT_ROOT must be set}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u experiments/audit_iterative_sampling_data.py \
  --train-data \
    "$OUT_ROOT/shared/g0/train_merged.npz" \
    "$OUT_ROOT/shared/g1/train_merged.npz" \
  --stage-names g0 g1 \
  --observation-input shared_future_summary \
  --out-path "$OUT_ROOT/shared/data_quality.json"
