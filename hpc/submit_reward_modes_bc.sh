#!/usr/bin/env bash
#SBATCH --job-name=ccs_reward_bc
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o logs/reward_bc-%j.out
#SBATCH -e logs/reward_bc-%j.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM}"
YARA_BUFFER_CAPACITY="${YARA_BUFFER_CAPACITY:-7500}"
cd "$PROJECT_DIR"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs output/rl_ppo

LIVE_LOG="logs/reward_modes_${SLURM_JOB_ID:-manual}.live.log"

echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Yara buffer capacity: $YARA_BUFFER_CAPACITY"
which python
python --version
nvidia-smi

python -u scripts/compare_reward_modes_bc.py \
  --scenario northern_lights_phase1_3vessels \
  --episode-hours 720 \
  --timesteps 100000 \
  --bc-episodes 30 \
  --bc-epochs 20 \
  --kickstart-coef 1.0 \
  --yara-buffer-capacity "$YARA_BUFFER_CAPACITY" \
  --eval-seeds 101 102 103 104 105 \
  --device cuda \
  --verbose 1 \
  2>&1 | tee "$LIVE_LOG"

echo "Job finished at $(date)"
