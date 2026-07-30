#!/usr/bin/env bash
#SBATCH --job-name=iterq_budget_analysis
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -o logs/iterq_budget_analysis-%j.out
#SBATCH -e logs/iterq_budget_analysis-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SEARCH_ROOT="${SEARCH_ROOT:-$PROJECT_DIR/output/iterative_q_budget_search}"

cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
python -u experiments/analyze_iterative_q_budget_search.py \
  --search-root "$SEARCH_ROOT" \
  --output-json "$SEARCH_ROOT/validation_analysis.json" \
  --output-csv "$SEARCH_ROOT/validation_analysis.csv"
