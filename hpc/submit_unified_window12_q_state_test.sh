#!/usr/bin/env bash
#SBATCH --job-name=ccs_u12_qstate_test
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o logs/unified_window12_q_state_test-%j.out
#SBATCH -e logs/unified_window12_q_state_test-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
RUN_ROOT="${RUN_ROOT:-output/unified_window12/iterative_q_state_4800_20260726}"
OUT_DIR="${OUT_DIR:-output/unified_window12/q_state_test_30seeds_20260726}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mapfile -t TEST_SEEDS < <(seq 8000001 8000030)

python -u experiments/evaluate_iterative_action_q.py \
  --checkpoint "$RUN_ROOT/p4/iterative_action_q.pt" \
  --out-dir "$OUT_DIR" \
  --eval-seeds "${TEST_SEEDS[@]}" \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --scenario-protocol unified_window_v1 \
  --hard-scenario-probability 0.5 \
  --forecast-context-hours 168 \
  --gates state_only_strict4_margin40k_12windows:4:0.40:12:108-155,156-203,204-251,252-299,300-347,348-395,396-443,444-491,492-539,540-587,588-635,636-680 \
  --device cuda
