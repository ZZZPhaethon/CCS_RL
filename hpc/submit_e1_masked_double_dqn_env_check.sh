#!/usr/bin/env bash
#SBATCH --job-name=ccs_dqn_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH -o logs/e1_masked_double_dqn_check-%j.out
#SBATCH -e logs/e1_masked_double_dqn_check-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_dqn_20260804}"
SMOKE_ROOT="${SMOKE_ROOT:-output/e1_masked_double_dqn_smoke_${SLURM_JOB_ID}}"
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

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the SLURM GPU job")
print("cuda_device_name", torch.cuda.get_device_name(0))
print("stable_baselines3", stable_baselines3.__version__)
PY

python -m compileall -q src/sim/control/hourly_dqn
python - <<'PY'
import numpy as np

from sim.control.hourly_dqn.model import joint_action_mask, joint_action_table

actions = joint_action_table((5, 5, 5))
mask = joint_action_mask(
    (
        np.asarray([1, 0, 1, 0, 1], dtype=np.int8),
        np.asarray([0, 1, 0, 1, 0], dtype=np.int8),
        np.asarray([1, 1, 0, 0, 0], dtype=np.int8),
    ),
    actions,
)
assert actions.shape == (125, 3)
assert int(mask.sum()) == 3 * 2 * 2
print("remote_import_and_mask_check_ok")
PY

python -u -m sim.control.hourly_dqn.train_hourly_dqn \
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
  --hidden-sizes 64 64 \
  --batch-size 64 \
  --replay-capacity 2048 \
  --learning-starts 128 \
  --gradient-steps-per-vector-step 1 \
  --target-update-interval 128 \
  --log-every-simulator-hour-steps 128 \
  --device cuda \
  --log-dir "$SMOKE_ROOT"

test -f "$SMOKE_ROOT/masked_double_dqn_best_validation.pt"
test -f "$SMOKE_ROOT/masked_double_dqn_final.pt"
test -f "$SMOKE_ROOT/training_complete.json"
grep -q '"simulator_hour_steps": 1024.0' \
  "$SMOKE_ROOT/training_complete.json"
