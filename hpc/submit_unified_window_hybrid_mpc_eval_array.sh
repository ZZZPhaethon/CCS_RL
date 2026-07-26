#!/usr/bin/env bash
#SBATCH --job-name=ccs_u_hybrid_mpc
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --array=0-29%30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o logs/unified_window_hybrid_mpc-%A_%a.out
#SBATCH -e logs/unified_window_hybrid_mpc-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
RUN_DIR="${RUN_DIR:-output/unified_window12/hybrid_rl_50k_20260726}"
SEED=$((8000001 + SLURM_ARRAY_TASK_ID))
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -u -m sim.control.event_based.rl.evaluate_high_level_ppo \
  --run-dir "$RUN_DIR" \
  --executor mpc \
  --seeds "$SEED"
