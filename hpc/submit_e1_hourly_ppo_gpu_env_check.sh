#!/usr/bin/env bash
#SBATCH --job-name=ccs_hourly_ppo_gpu_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -o logs/hourly_ppo_gpu_check-%j.out
#SBATCH -e logs/hourly_ppo_gpu_check-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_e1_20260728}"
SMOKE_ROOT="${SMOKE_ROOT:-output/hourly_ppo_gpu_smoke_${SLURM_JOB_ID}}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

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
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the SLURM GPU job")
print("cuda_device_name", torch.cuda.get_device_name(0))
print("stable_baselines3", stable_baselines3.__version__)
print("sb3_contrib", sb3_contrib.__version__)
PY

python -u -m sim.control.hourly_ppo.train_hourly_ppo \
  --timesteps 2048 \
  --seed 0 \
  --episode-hours 24 \
  --forecast-context-hours 168 \
  --future-summary-windows-h 168 \
  --scenario-protocol unified_window_v1 \
  --gamma 1 \
  --num-envs 4 \
  --max-simulator-hour-steps 1024 \
  --training-seed-min 100000 \
  --training-seed-max 100010 \
  --validation-seeds 8100001 \
  --validation-every-simulator-hour-steps 512 \
  --n-steps 32 \
  --batch-size 64 \
  --device cuda \
  --log-dir "$SMOKE_ROOT"

test -f "$SMOKE_ROOT/ppo_hourly_best_validation.zip"
test -f "$SMOKE_ROOT/ppo_hourly_final.zip"
test -f "$SMOKE_ROOT/training_complete.json"
grep -q '"simulator_hour_steps": 1024.0' \
  "$SMOKE_ROOT/training_complete.json"
