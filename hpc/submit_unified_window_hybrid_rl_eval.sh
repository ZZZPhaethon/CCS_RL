#!/usr/bin/env bash
#SBATCH --job-name=ccs_u_hybrid_eval
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/unified_window_hybrid_eval-%j.out
#SBATCH -e logs/unified_window_hybrid_eval-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
RUN_DIR="${RUN_DIR:-output/unified_window12/hybrid_rl_50k_20260726}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mapfile -t TEST_SEEDS < <(seq 8000001 8000030)
EXECUTOR="${EXECUTOR:-rule}"

python -u -m sim.control.event_based.rl.evaluate_high_level_ppo \
  --run-dir "$RUN_DIR" \
  --executor "$EXECUTOR" \
  --seeds "${TEST_SEEDS[@]}"
