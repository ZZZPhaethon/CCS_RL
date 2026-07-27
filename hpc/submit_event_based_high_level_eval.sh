#!/usr/bin/env bash
#SBATCH --job-name=ccs_high_level_eval
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o logs/high_level_eval-%j.out
#SBATCH -e logs/high_level_eval-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_control_20260725}"
RUN_DIR="${RUN_DIR:-output/unified_physics/high_level_ppo_50k_seed0_20260725_aligned}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mapfile -t TEST_SEEDS < <(seq 7000 7029)

python -u -m sim.control.event_based.rl.evaluate_high_level_ppo \
  --run-dir "$RUN_DIR" \
  --seeds "${TEST_SEEDS[@]}"
