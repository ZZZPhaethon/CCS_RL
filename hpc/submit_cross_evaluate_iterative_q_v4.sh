#!/usr/bin/env bash
#SBATCH --job-name=ccs_q_v4_cross_eval
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o logs/q_v4_cross_eval-%j.out
#SBATCH -e logs/q_v4_cross_eval-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_control_20260725}"
Q_CHECKPOINT="${Q_CHECKPOINT:-output/cross_scenario_inputs/iterative_action_q_p4.pt}"
V4_RUN_DIR="${V4_RUN_DIR:-output/unified_physics/residual_v4_seed0_100k_20260725_noref}"
OUT_DIR="${OUT_DIR:-output/unified_physics/q_v4_zero_shot_cross_20260725}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u experiments/cross_evaluate_iterative_q_v4.py \
  --iterative-q-checkpoint "$Q_CHECKPOINT" \
  --v4-run-dir "$V4_RUN_DIR" \
  --out-dir "$OUT_DIR" \
  --device cpu
