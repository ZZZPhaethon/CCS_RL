#!/usr/bin/env bash
#SBATCH --job-name=iterq_rootwin_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o logs/iterq_rootwin_check-%j.out
#SBATCH -e logs/iterq_rootwin_check-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

which python
python --version
python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
PY
nvidia-smi
python -m compileall -q experiments/generate_iterative_q_policy_data.py
python - <<'PY'
from argparse import Namespace

from experiments.generate_iterative_q_policy_data import (
    select_target_root_h,
    select_window_indices,
)

random_args = Namespace(root_selection="random_time", dataset_seed=20260803)
first = select_target_root_h(random_args, 1500, 3, 252, 299)
second = select_target_root_h(random_args, 1500, 3, 252, 299)
assert first == second
assert 252 <= first <= 299

window_args = Namespace(window_indices=None, windows_per_seed=12)
selected = select_window_indices(window_args, seed=24, window_count=24)
assert selected == list(range(12))
assert len(select_window_indices(window_args, seed=25, window_count=24)) == 12
print("root_window_assertions=passed")
PY
python experiments/generate_iterative_q_policy_data.py --help \
  | grep -q -- '--root-selection'
python experiments/generate_iterative_q_policy_data.py --help \
  | grep -q -- '--windows-per-seed'
