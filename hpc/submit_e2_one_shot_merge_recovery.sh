#!/usr/bin/env bash
#SBATCH --job-name=e2_g0_merge2
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH -o logs/e2_g0_merge2-%j.out
#SBATCH -e logs/e2_g0_merge2-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"
RUN_ROOT="${RUN_ROOT:-experiments_results/E2/training_one_shot_matched_run01}"
OUT_PATH="$RUN_ROOT/g0/train_merged_matched.npz"

cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
mapfile -t TRAIN_SEEDS < <(seq 1500 1800)
mapfile -t EXTRA_SHARDS < <(
  find "$RUN_ROOT/g0/extra" -maxdepth 1 -type f -name 'shard_*.npz' \
    ! -name 'shard_1800_1803.npz' | sort
)

python -u experiments/merge_iterative_q_data.py \
  --shards \
    "$SOURCE_RUN_ROOT/g0/train_merged.npz" \
    "${EXTRA_SHARDS[@]}" \
    "$RUN_ROOT/g0/trim/shard_1800_1800.npz" \
  --out-path "$OUT_PATH" \
  --expected-split train \
  --expected-seeds "${TRAIN_SEEDS[@]}"
if [[ -s "$RUN_ROOT/budget.json" && ! -e "$RUN_ROOT/budget_attempt1.json" ]]; then
  cp "$RUN_ROOT/budget.json" "$RUN_ROOT/budget_attempt1.json"
fi
python -u experiments/summarize_e2_matched_budget.py \
  --train-data "$OUT_PATH" \
  --out-path "$RUN_ROOT/budget.json" \
  --target-simulator-calls 9526297 \
  --max-relative-error-pct 0.5
