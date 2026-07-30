#!/usr/bin/env bash
#SBATCH --job-name=iter_h3_agg
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o logs/iterative_h3_agg-%j.out
#SBATCH -e logs/iterative_h3_agg-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${OUT_ROOT:?OUT_ROOT must be set}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python -u experiments/aggregate_iterative_h3_sampler_validation.py \
  --run-root "$OUT_ROOT"
