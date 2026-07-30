#!/usr/bin/env bash
#SBATCH --job-name=e234_aggregate
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o logs/e234_aggregate-%j.out
#SBATCH -e logs/e234_aggregate-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${EXPERIMENT:?EXPERIMENT must be E2, E3, or E4}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u experiments/aggregate_e2_e3_e4.py "$EXPERIMENT"
