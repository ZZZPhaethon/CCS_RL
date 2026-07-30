#!/usr/bin/env bash
#SBATCH --job-name=native_mpc_cleanup
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH -o logs/native_mpc_cleanup-%A_%a.out
#SBATCH -e logs/native_mpc_cleanup-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_native_mpc_cleanup_20260728}"
RUN_ROOT="${RUN_ROOT:-output/native_mpc_cleanup_seeds_8100001_8100003}"
SEEDS=(8100001 8100002 8100003)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u experiments/run_native_mpc_cleanup.py \
  --out-dir "$RUN_ROOT/seed_$SEED" \
  --seed "$SEED" \
  --episode-hours 720 \
  --forecast-context-hours 168 \
  --replan-hours 24 \
  --planning-horizon-hours 168
