#!/usr/bin/env bash
#SBATCH --job-name=iterq_budget_shared
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o logs/iterq_budget_shared-%j.out
#SBATCH -e logs/iterq_budget_shared-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SOURCE_G0_ROOT="${SOURCE_G0_ROOT:-output/iterative_q_validation_search/baseline_p1_p4/g0}"
SOURCE_P1_ROOT="${SOURCE_P1_ROOT:-output/iterative_q_validation_search/uniform_margin40_p1_p4/p1}"
SHARED_ROOT="${SHARED_ROOT:-output/iterative_q_budget_search/shared}"

cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
if [[ -e "$SHARED_ROOT" ]]; then
  echo "Refusing to overwrite shared budget-search data: $SHARED_ROOT" >&2
  exit 2
fi
mkdir -p "$SHARED_ROOT/g0" "$SHARED_ROOT/p1"

mapfile -t TRAIN_SHARDS < <(
  find "$SOURCE_G0_ROOT/train" -maxdepth 1 -name 'shard_*.npz' -type f | sort
)
mapfile -t VALIDATION_SHARDS < <(
  find "$SOURCE_G0_ROOT/validation" -maxdepth 1 -name 'shard_*.npz' -type f | sort
)
mapfile -t TRAIN_SEEDS < <(seq 1500 1699)
mapfile -t VALIDATION_SEEDS < <(seq 3200 3239)

python -u experiments/merge_iterative_q_data.py \
  --shards "${TRAIN_SHARDS[@]}" \
  --out-path "$SHARED_ROOT/g0/train_merged.npz" \
  --expected-split train \
  --expected-seeds "${TRAIN_SEEDS[@]}"
python -u experiments/merge_iterative_q_data.py \
  --shards "${VALIDATION_SHARDS[@]}" \
  --out-path "$SHARED_ROOT/g0/validation_merged.npz" \
  --expected-split validation \
  --expected-seeds "${VALIDATION_SEEDS[@]}"

cp "$SOURCE_P1_ROOT/iterative_action_q.pt" "$SHARED_ROOT/p1/"
cp "$SOURCE_P1_ROOT/summary.json" "$SHARED_ROOT/p1/"
sha256sum \
  "$SOURCE_P1_ROOT/iterative_action_q.pt" \
  "$SHARED_ROOT/p1/iterative_action_q.pt" \
  > "$SHARED_ROOT/p1_sha256.txt"
python -u experiments/summarize_iterative_q_budget.py \
  --run-root "$SHARED_ROOT" \
  --output "$SHARED_ROOT/g0_budget.json"
