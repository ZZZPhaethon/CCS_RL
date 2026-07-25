#!/usr/bin/env bash
#SBATCH --job-name=ccs_window_summary_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o logs/event_window_summary_check-%j.out
#SBATCH -e logs/event_window_summary_check-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_greedy_dagger}"
cd "$PROJECT_DIR"
export PYTHONPATH=src
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

which python
python --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"
python - <<'PY'
import numpy as np
import torch

from sim.control.recurrent_distributional_q import (
    StatelessStructuredActionQuantileQ,
)

emitters = ("a", "b", "c")
vessels = ("ship_a", "ship_b", "ship_c")
state_names = [f"{name}.fill" for name in (*emitters, "oygarden_terminal")]
state_names.append("weather.speed_now")
for vessel in vessels:
    for destination in ("oygarden_terminal", *emitters):
        state_names.append(f"{vessel}.to_{destination}.travel_hours_now")
for vessel in vessels:
    state_names.extend(
        f"greedy_proposal.{vessel}.native_action_{action}"
        for action in range(5)
    )
forecast_names = [
    *(f"capture.{name}" for name in emitters),
    *(f"emitter_available.{name}" for name in emitters),
    "well_available.well",
    "injectivity.well",
    "weather.global_speed_factor",
]
joint_actions = np.asarray([(action, action, action) for action in range(6)])
states = torch.zeros(1, 1, len(state_names), device="cuda")
forecasts = torch.ones(1, 1, 168, len(forecast_names), device="cuda")
for encoder in (
    "window_summary_24_72",
    "window_summary_168",
    "window_summary_24_72_168",
    "window_summary_joint_168",
):
    model = StatelessStructuredActionQuantileQ(
        state_names,
        (168, len(forecast_names)),
        joint_actions,
        state_mean=np.zeros(len(state_names)),
        state_std=np.ones(len(state_names)),
        forecast_mean=np.zeros(len(forecast_names)),
        forecast_std=np.ones(len(forecast_names)),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder=encoder,
        forecast_channel_names=forecast_names,
    ).cuda()
    with torch.no_grad():
        q = model(states, forecasts)
    assert q.shape == (1, 1, 2, len(joint_actions), 3)
    module = (
        model.window_summary_joint_q
        if model.window_summary_joint_q is not None
        else model.window_summary_residual
    )
    print(encoder, module.horizons, tuple(q.shape))
PY
