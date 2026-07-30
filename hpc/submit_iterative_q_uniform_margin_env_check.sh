#!/usr/bin/env bash
#SBATCH --job-name=iterq_m40_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o logs/iterative_q_uniform_margin_check-%j.out
#SBATCH -e logs/iterative_q_uniform_margin_check-%j.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
cd "$PROJECT_DIR"
mkdir -p logs

which python
python --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("Iterative Q training requires CUDA.")
PY

bash -n hpc/launch_iterative_action_q.sh
bash -n hpc/launch_iterative_q_uniform_margin40.sh

PROJECT_DIR="$PROJECT_DIR" \
CONFIG_NAME=iterq_uniform_margin40_dry_run \
RUN_ROOT=output/iterative_q_validation_search/uniform_margin40_dry_run \
G0_TRAIN_COUNT=200 \
G0_VALIDATION_COUNT=40 \
G0_ROOT_FRACTIONS="0.15:0.2166666667:0.2833333333:0.35:0.4166666667:0.4833333333:0.55:0.6166666667:0.6833333333:0.75:0.8166666667:0.8833333333" \
G0_ROOTS_PER_SEED=12 \
ITER_TRAIN_COUNTS="40,60,100" \
ITER_VALIDATION_COUNTS="8,12,20" \
ITER_TRAIN_STARTS="1500,1600,2000" \
ITER_VALIDATION_STARTS="3200,3230,3300" \
SCENARIO_PROTOCOL=unified_window_v1 \
OBSERVATION_INPUT=shared_future_summary \
POLICY_WINDOWS_H="108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680" \
MAX_OVERRIDES=12 \
P1_RESIDUAL_MARGIN=0.40 \
P1_ECONOMIC_MARGIN_EUR=40000 \
ITER_RESIDUAL_MARGIN=0.40 \
ITER_ECONOMIC_MARGIN_EUR=40000 \
DRY_RUN=1 \
bash hpc/launch_iterative_action_q.sh
