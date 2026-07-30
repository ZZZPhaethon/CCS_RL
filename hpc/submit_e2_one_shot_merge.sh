#!/usr/bin/env bash
#SBATCH --job-name=e2_g0_merge
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH -o logs/e2_g0_merge-%j.out
#SBATCH -e logs/e2_g0_merge-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"
RUN_ROOT="${RUN_ROOT:-experiments_results/E2/training_one_shot_matched_run01}"

cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
mapfile -t TRAIN_SEEDS < <(seq 1500 1803)
EXTRA_SHARDS=("$RUN_ROOT/g0/extra"/shard_*.npz)

python -u experiments/merge_iterative_q_data.py \
  --shards "$SOURCE_RUN_ROOT/g0/train_merged.npz" "${EXTRA_SHARDS[@]}" \
  --out-path "$RUN_ROOT/g0/train_merged.npz" \
  --expected-split train \
  --expected-seeds "${TRAIN_SEEDS[@]}"
cp "$SOURCE_RUN_ROOT/g0/validation_merged.npz" \
  "$RUN_ROOT/g0/validation_merged.npz"
python -u experiments/summarize_e2_matched_budget.py \
  --train-data "$RUN_ROOT/g0/train_merged.npz" \
  --out-path "$RUN_ROOT/budget.json" \
  --target-simulator-calls 9526297 \
  --max-relative-error-pct 0.5
