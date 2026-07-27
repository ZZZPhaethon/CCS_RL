#!/usr/bin/env bash
#SBATCH --job-name=ccs_er_report
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/event_residual_future_report-%j.out
#SBATCH -e logs/event_residual_future_report-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
RUN_ROOT="${RUN_ROOT:-output/unified_window12/event_residual_future_ablation_20260727}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mapfile -t TEST_SEEDS < <(seq 8000001 8000030)
python -u -m sim.control.event_based.residual_rl_v4.evaluate_greedy \
  --reference-run-dir "$RUN_ROOT/state_only" \
  --seeds "${TEST_SEEDS[@]}" \
  --hard-scenario-probability 0 \
  --output-dir "$RUN_ROOT/greedy"

python -u experiments/analyze_event_residual_future_ablation.py \
  --run-root "$RUN_ROOT"
