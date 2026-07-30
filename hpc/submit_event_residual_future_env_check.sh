#!/usr/bin/env bash
#SBATCH --job-name=ccs_er_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -o logs/event_residual_future_check-%j.out
#SBATCH -e logs/event_residual_future_check-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
cd "$PROJECT_DIR"
export PYTHONPATH=src:.:scripts

which python
python --version
nvidia-smi || true
python - <<'PY'
import torch

from sim.control.event_based.residual_rl_v4.factory import (
    make_tail_robust_native_env,
)

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
expected = {
    (): 89,
    (24, 72): 103,
    (168,): 96,
    (24, 72, 168): 110,
}
for windows_h, expected_size in expected.items():
    env = make_tail_robust_native_env(
        episode_hours=24,
        forecast_context_hours=168,
        future_summary_windows_h=windows_h,
        scenario_protocol="unified_window_v1",
        hard_scenario_probability=0.0,
    )
    observation = env.reset(seed=123)
    assert observation.shape == (expected_size,)
    print(f"windows={windows_h or 'none'} observation={observation.shape}")
print("event_residual_future_env_check=ok")
PY
