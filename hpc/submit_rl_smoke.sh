#!/usr/bin/env bash
#SBATCH --job-name=ccs_rl_smoke
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o logs/rl_smoke-%j.out
#SBATCH -e logs/rl_smoke-%j.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

cd /scratch_root/hx721/CCS_RLLLM
export PYTHONPATH=src
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR=/scratch_root/hx721/CCS_RLLLM/.cache/matplotlib
mkdir -p "$MPLCONFIGDIR"

echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

python hpc/rl_smoke.py

echo "Job finished at $(date)"
