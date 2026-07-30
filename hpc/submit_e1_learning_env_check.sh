#!/usr/bin/env bash
#SBATCH --job-name=ccs_e1_env
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o logs/e1_learning_env-%j.out
#SBATCH -e logs/e1_learning_env-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_e1_20260728}"
SMOKE_ROOT="${SMOKE_ROOT:-output/e1_learning_smoke_${SLURM_JOB_ID}}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

which python
python --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'PY'
import torch
import stable_baselines3
import sb3_contrib

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name", torch.cuda.get_device_name(0))
print("stable_baselines3", stable_baselines3.__version__)
print("sb3_contrib", sb3_contrib.__version__)
PY

python -m unittest tests.test_event_based_algorithms -q
python -m pytest tests/test_hourly_ppo.py -q

python -u -m sim.control.hourly_ppo.train_hourly_ppo \
  --timesteps 512 \
  --seed 0 \
  --episode-hours 24 \
  --forecast-context-hours 168 \
  --future-summary-windows-h 168 \
  --scenario-protocol unified_window_v1 \
  --gamma 1 \
  --max-simulator-hour-steps 48 \
  --training-seed-min 100000 \
  --training-seed-max 100010 \
  --validation-seeds 8100001 \
  --validation-every-simulator-hour-steps 24 \
  --n-steps 16 \
  --batch-size 16 \
  --device cuda \
  --log-dir "$SMOKE_ROOT/centralized"

python -u -m \
  sim.control.event_based.residual_rl_v4.train_objective_aligned_ppo \
  --timesteps 512 \
  --seed 0 \
  --episode-hours 24 \
  --forecast-context-hours 168 \
  --future-summary-windows-h 168 \
  --decision-interval-h 24 \
  --max-simulator-hour-steps 48 \
  --validation-every-simulator-hour-steps 24 \
  --validation-seeds 8100001 \
  --training-seed-min 100000 \
  --training-seed-max 100010 \
  --n-steps 16 \
  --batch-size 16 \
  --log-dir "$SMOKE_ROOT/event_residual"

test -f "$SMOKE_ROOT/centralized/ppo_hourly_best_validation.zip"
test -f "$SMOKE_ROOT/centralized/training_complete.json"
test -f "$SMOKE_ROOT/event_residual/event_residual_e1_best_validation.zip"
test -f "$SMOKE_ROOT/event_residual/training_complete.json"
