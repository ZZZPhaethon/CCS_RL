#!/bin/bash
#SBATCH --job-name=ccs_bc_weight
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=20,30
#SBATCH -o logs/bc_weight_sweep-%A_%a.out
#SBATCH -e logs/bc_weight_sweep-%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_git}
CONDA_ENV=${CONDA_ENV:-mas-ccus}
YARA_BUFFER_CAPACITY=${YARA_BUFFER_CAPACITY:-7500}
NONWAIT_WEIGHT=${NONWAIT_WEIGHT:-${SLURM_ARRAY_TASK_ID}}
BC_EPISODES=${BC_EPISODES:-30}

cd "$PROJECT_DIR"
mkdir -p logs

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

echo "Job ${SLURM_JOB_ID:-manual}, task ${SLURM_ARRAY_TASK_ID:-manual}"
echo "Project: $PROJECT_DIR"
echo "Non-WAIT weight: $NONWAIT_WEIGHT"
echo "BC episodes: $BC_EPISODES"
echo "Yara buffer capacity: $YARA_BUFFER_CAPACITY"
which python
python --version
nvidia-smi

python -u experiments/compare_reward_modes_bc.py \
  --scenario northern_lights_phase1_3vessels \
  --episode-hours 720 \
  --timesteps 0 \
  --bc-episodes "$BC_EPISODES" \
  --bc-epochs 20 \
  --nonwait-weight "$NONWAIT_WEIGHT" \
  --yara-buffer-capacity "$YARA_BUFFER_CAPACITY" \
  --eval-seeds 101 102 103 104 105 \
  --device cuda \
  --verbose 0

echo "Job finished at $(date)"
