#!/usr/bin/env bash
#SBATCH --job-name=ccs_u_hybrid
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/unified_window_hybrid-%j.out
#SBATCH -e logs/unified_window_hybrid-%j.err

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

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing output collision: $RUN_DIR" >&2
  exit 2
fi

python -u -m sim.control.event_based.rl.train_high_level_ppo \
  --timesteps 50000 \
  --seed 0 \
  --scenario northern_lights_phase1_3vessels \
  --episode-hours 720 \
  --forecast-context-hours 168 \
  --decision-interval-h 24 \
  --event-triggered \
  --weather-mode window \
  --warm-start \
  --scenario-protocol unified_window_v1 \
  --device cpu \
  --ent-coef 0.01 \
  --log-dir "$RUN_DIR" \
  --status-every-steps 1000 \
  --progress-mode lines
