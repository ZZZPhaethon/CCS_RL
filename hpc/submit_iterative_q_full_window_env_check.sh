#!/usr/bin/env bash
#SBATCH --job-name=iqfull720_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o logs/iqfull720_check-%j.out
#SBATCH -e logs/iqfull720_check-%j.err

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
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
PY
nvidia-smi
python -m compileall -q \
  experiments/generate_iterative_q_greedy_data.py \
  experiments/generate_iterative_q_policy_data.py \
  experiments/analyze_iterative_q_window_extension.py
python - <<'PY'
from experiments.generate_iterative_q_greedy_data import parse_args
from experiments.evaluate_iterative_action_q import parse_gate

args = parse_args([
    "--out-path", "/tmp/unused.npz",
    "--split", "train",
    "--seeds", "1500",
    "--root-fractions", "0", "0.9166666667",
])
assert args.root_fractions == [0.0, 0.9166666667]
gate = parse_gate(
    "full720:4:0.40:12:"
    "0-59,60-119,120-179,180-239,240-299,300-359,"
    "360-419,420-479,480-539,540-599,600-659,660-719"
)
assert len(gate["windows"]) == 12
assert gate["windows"][0] == [0.0, 59.0]
assert gate["windows"][-1] == [660.0, 719.0]
print("full_window_assertions=passed")
PY
