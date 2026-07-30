#!/usr/bin/env bash
#SBATCH --job-name=iterq_budget_audit
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -o logs/iterq_budget_audit-%j.out
#SBATCH -e logs/iterq_budget_audit-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${RUN_ROOT:?RUN_ROOT must be set}"
TARGET_TRAIN_STEPS="${TARGET_TRAIN_STEPS:-9525119}"

cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
python -u experiments/summarize_iterative_q_budget.py \
  --run-root "$RUN_ROOT" \
  --target-train-steps "$TARGET_TRAIN_STEPS" \
  --output "$RUN_ROOT/budget.json"
