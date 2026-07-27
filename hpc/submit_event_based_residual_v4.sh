#!/usr/bin/env bash
#SBATCH --job-name=ccs_residual_v4
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/residual_v4-%j.out
#SBATCH -e logs/residual_v4-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_control_20260725}"
RUN_DIR="${RUN_DIR:-output/unified_physics/residual_v4_seed0_100k_20260725_noref}"
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

python -u -m sim.control.event_based.residual_rl_v4.train_tail_robust_ppo \
  --scenario northern_lights_phase1_3vessels \
  --episode-hours 720 \
  --forecast-context-hours 168 \
  --decision-interval-h 24 \
  --event-triggered \
  --weather-mode window \
  --timesteps 100000 \
  --num-envs 4 \
  --vec-env subproc \
  --validation-every-steps 10000 \
  --replay-probability 0.30 \
  --replay-capacity 20 \
  --minimum-replay-pool 4 \
  --seed 0 \
  --device cpu \
  --no-reference-constraints \
  --log-dir "$RUN_DIR"
