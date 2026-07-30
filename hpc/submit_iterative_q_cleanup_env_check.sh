#!/usr/bin/env bash
#SBATCH --job-name=iterq_cleanup_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o logs/iterative_q_cleanup_check-%j.out
#SBATCH -e logs/iterative_q_cleanup_check-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_cleanup_matched_20260728}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

which python
python --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'PY'
import pulp
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("cbc_available", pulp.PULP_CBC_CMD(msg=False).available())
if not torch.cuda.is_available():
    raise RuntimeError("Iterative Q training requires CUDA.")
if not pulp.PULP_CBC_CMD(msg=False).available():
    raise RuntimeError("Terminal cleanup evaluation requires CBC.")
PY

SMOKE_PATH="output/iterative_q_cleanup_check_${SLURM_JOB_ID}.npz"
python -u experiments/generate_iterative_q_greedy_data.py \
  --out-path "$SMOKE_PATH" \
  --split validation \
  --seeds 3200 \
  --root-fractions 0.15 \
  --roots-per-seed 1 \
  --max-two-vessel-actions 0 \
  --max-three-vessel-actions 0 \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --dataset-seed 20260723 \
  --variant future_mlp_mode_destination \
  --scenario-protocol unified_window_v1 \
  --hard-scenario-probability 0.5 \
  --forecast-context-hours 168 \
  --future-summary-windows-h 24 72 \
  --device cpu

python - "$SMOKE_PATH" <<'PY'
import json
import sys

import numpy as np

path = sys.argv[1]
with np.load(path, allow_pickle=False) as payload:
    metadata = json.loads(str(payload["metadata_json"].item()))
    assert metadata["future_summary_windows_h"] == [24, 72]
    assert "terminal cleanup" in metadata["objective"]
    baseline = payload["baseline_terminal_cleanup_operating_cost_eur"]
    candidate = payload["candidate_terminal_cleanup_operating_cost_eur"]
    expected = 1e-5 * (
        payload["baseline_total_cost_eur"]
        - payload["candidate_total_cost_eur"]
    )
    assert np.all(baseline >= 0.0)
    assert np.all(candidate >= 0.0)
    assert np.allclose(payload["return_to_go"][:, 0], expected, atol=2e-5)
    print("smoke_candidates", len(expected))
    print("cleanup_mean_eur", float(candidate.mean()))
PY

DRY_RUN=1 bash hpc/launch_unified_window12_iterative_q_cleanup.sh
