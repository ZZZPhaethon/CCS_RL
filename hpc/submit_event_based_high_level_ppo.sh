#!/usr/bin/env bash
#SBATCH --job-name=ccs_high_level_ppo
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/high_level_ppo-%j.out
#SBATCH -e logs/high_level_ppo-%j.err

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

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing output collision: $RUN_DIR" >&2
  exit 2
fi

python -u -m sim.control.event_based.rl.train_high_level_ppo \
  --timesteps 50000 \
  --seed 0 \
  --scenario northern_lights_phase1_3vessels \
  --episode-hours 720 \
  --forecast-context-hours 169 \
  --decision-interval-h 24 \
  --event-triggered \
  --weather-mode block \
  --warm-start \
  --device cpu \
  --ent-coef 0.01 \
  --log-dir "$RUN_DIR" \
  --status-every-steps 1000 \
  --progress-mode lines
