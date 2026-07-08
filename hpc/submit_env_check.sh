#!/usr/bin/env bash
#SBATCH --job-name=ccs_env_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o logs/env_check-%j.out
#SBATCH -e logs/env_check-%j.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

cd /scratch_root/hx721/CCS_RLLLM
export MPLCONFIGDIR=/scratch_root/hx721/CCS_RLLLM/.cache/matplotlib
mkdir -p "$MPLCONFIGDIR"

echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

which python
python --version
nvidia-smi
PYTHONPATH=src python -c "import torch, gymnasium, stable_baselines3, sb3_contrib; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count()); print('gymnasium', gymnasium.__version__); print('stable_baselines3', stable_baselines3.__version__); print('sb3_contrib', sb3_contrib.__version__)"

echo "Job finished at $(date)"
