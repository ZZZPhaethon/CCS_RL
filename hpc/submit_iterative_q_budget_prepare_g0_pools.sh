#!/usr/bin/env bash
#SBATCH --job-name=iterq_g0_pools
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o logs/iterq_g0_pools-%j.out
#SBATCH -e logs/iterq_g0_pools-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SOURCE_G0_ROOT="${SOURCE_G0_ROOT:-output/iterative_q_validation_search/baseline_p1_p4/g0}"
SEARCH_ROOT="${SEARCH_ROOT:-output/iterative_q_budget_search}"

cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
pool_root="$SEARCH_ROOT/g0_pools"
if [[ -e "$pool_root" ]]; then
  echo "Refusing to overwrite G0 pools: $pool_root" >&2
  exit 2
fi
mkdir -p "$pool_root"

mapfile -t ALL_TRAIN_SHARDS < <(
  find "$SOURCE_G0_ROOT/train" -maxdepth 1 -name 'shard_*.npz' -type f | sort
)
mapfile -t VALIDATION_SHARDS < <(
  find "$SOURCE_G0_ROOT/validation" -maxdepth 1 -name 'shard_*.npz' -type f | sort
)
mapfile -t VALIDATION_SEEDS < <(seq 3200 3239)
if (( ${#ALL_TRAIN_SHARDS[@]} != 20 )); then
  echo "Expected 20 ten-seed G0 training shards" >&2
  exit 2
fi

for train_count in 90 120 150 180; do
  pool="$pool_root/g0_${train_count}"
  mkdir -p "$pool/g0"
  shard_count=$((train_count / 10))
  train_shards=("${ALL_TRAIN_SHARDS[@]:0:shard_count}")
  mapfile -t train_seeds < <(seq 1500 "$((1500 + train_count - 1))")
  python -u experiments/merge_iterative_q_data.py \
    --shards "${train_shards[@]}" \
    --out-path "$pool/g0/train_merged.npz" \
    --expected-split train \
    --expected-seeds "${train_seeds[@]}"
  python -u experiments/merge_iterative_q_data.py \
    --shards "${VALIDATION_SHARDS[@]}" \
    --out-path "$pool/g0/validation_merged.npz" \
    --expected-split validation \
    --expected-seeds "${VALIDATION_SEEDS[@]}"
  python -u experiments/summarize_iterative_q_budget.py \
    --run-root "$pool" \
    --output "$pool/g0_budget.json"
done
