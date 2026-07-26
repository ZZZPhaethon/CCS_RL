#!/usr/bin/env bash
#SBATCH --job-name=ccs_residual_v4_eval
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o logs/residual_v4_eval-%j.out
#SBATCH -e logs/residual_v4_eval-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_control_20260725}"
RUN_DIR="${RUN_DIR:-output/unified_physics/residual_v4_seed0_100k_20260725_noref}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mapfile -t NORMAL_SEEDS < <(seq 6000001 6000020)
mapfile -t HARD_SEEDS < <(seq 7000001 7000020)

python -u -m sim.control.event_based.residual_rl_v4.evaluate_ppo \
  --run-dir "$RUN_DIR" \
  --model best \
  --hard-scenario-probability 0 \
  --seeds "${NORMAL_SEEDS[@]}"

python -u -m sim.control.event_based.residual_rl_v4.evaluate_ppo \
  --run-dir "$RUN_DIR" \
  --model best \
  --hard-scenario-probability 1 \
  --seeds "${HARD_SEEDS[@]}"

python -u -m sim.control.event_based.residual_rl_v4.evaluate_greedy \
  --reference-run-dir "$RUN_DIR" \
  --hard-scenario-probability 0 \
  --seeds "${NORMAL_SEEDS[@]}" \
  --output-dir "$RUN_DIR/evaluation/greedy_normal__seeds_6000001-6000020__n20"

python -u -m sim.control.event_based.residual_rl_v4.evaluate_greedy \
  --reference-run-dir "$RUN_DIR" \
  --hard-scenario-probability 1 \
  --seeds "${HARD_SEEDS[@]}" \
  --output-dir "$RUN_DIR/evaluation/greedy_hard__seeds_7000001-7000020__n20"
