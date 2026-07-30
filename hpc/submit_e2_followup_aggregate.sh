#!/usr/bin/env bash
#SBATCH --job-name=e2_follow_agg
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH -o logs/e2_follow_agg-%j.out
#SBATCH -e logs/e2_follow_agg-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${FOLLOWUP_ANALYSIS:?FOLLOWUP_ANALYSIS must be set}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u experiments/aggregate_e2_followup.py "$FOLLOWUP_ANALYSIS"
