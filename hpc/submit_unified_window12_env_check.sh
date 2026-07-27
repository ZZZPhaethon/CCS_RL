#!/usr/bin/env bash
#SBATCH --job-name=ccs_u12_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o logs/unified_window12_check-%j.out
#SBATCH -e logs/unified_window12_check-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

which python
python --version
nvidia-smi
python - <<'PY'
from types import SimpleNamespace

import torch

from experiments import iterative_q_data_common as common
from sim.control.event_based.residual_rl_v4.factory import (
    make_tail_robust_native_env,
)
from sim.control.event_based.rl.train_high_level_ppo import (
    make_high_level_native_env,
)

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("Iterative Q training requires CUDA.")

args = SimpleNamespace(
    episode_hours=24,
    reward_scale=1e-5,
    variant=common.DEFAULT_VARIANT,
    scenario_protocol="unified_window_v1",
    hard_scenario_probability=0.5,
    forecast_context_hours=168,
    seeds=[123],
)
q_env = common.make_native_env(args)
q_env.reset(seed=123)
config = q_env.scenario_generator.normal.config
assert config.capture_noise_std == 0.30
assert config.weather_window_rate_per_week == 0.5
assert config.weather_window_speed_factor_range == (0.5, 0.8)
assert config.well_maintenance_mean_hours == 12.0
assert config.warm_start

v4 = make_tail_robust_native_env(
    episode_hours=24,
    scenario_protocol="unified_window_v1",
    gate_mode="off",
    override_windows_h=((0.0, 23.0),),
)
v4.reset(seed=123)
hybrid = make_high_level_native_env(
    episode_hours=24,
    forecast_context_hours=168,
    weather_mode="window",
    scenario_protocol="unified_window_v1",
)
hybrid.reset(seed=123)
hybrid_config = hybrid.env.scenario_generator.normal.config
assert hybrid_config == config
print("q_scenario_steps", q_env.scenario.n_steps)
print("v4_action_count", v4.action_count)
print("hybrid_action_count", hybrid.action_count)
print("hybrid_observation_size", hybrid.observation_size)
PY
