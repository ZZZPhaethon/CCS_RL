#!/usr/bin/env bash
#SBATCH --job-name=ccs_u12_compare
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o logs/unified_window12_compare-%j.out
#SBATCH -e logs/unified_window12_compare-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
Q_RUN="${Q_RUN:-output/unified_window12/iterative_q_future_4800_20260726}"
V4_RUN="${V4_RUN:-output/unified_window12/residual_v4_100k_20260726}"
OUT_DIR="${OUT_DIR:-output/unified_window12/comparison_30seeds_20260726}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mapfile -t TEST_SEEDS < <(seq 8000001 8000030)

python -u experiments/compare_unified_window_controls.py \
  --iterative-q-checkpoint "$Q_RUN/p4/iterative_action_q.pt" \
  --v4-run-dir "$V4_RUN" \
  --out-dir "$OUT_DIR" \
  --seeds "${TEST_SEEDS[@]}" \
  --device cpu

