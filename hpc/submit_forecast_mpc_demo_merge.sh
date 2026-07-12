#!/usr/bin/env bash
#SBATCH --job-name=ccs_mpc_demo_merge
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/mpc_demo_merge-%j.out
#SBATCH -e logs/mpc_demo_merge-%j.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM}"
cd "$PROJECT_DIR"
export PYTHONPATH="src"
export PYTHONUNBUFFERED=1

SHARD_DIR="${SHARD_DIR:-output/rl_forecast/demos/mpc_720h_100seeds_shards}"
DEMO_CACHE="${DEMO_CACHE:-output/rl_forecast/demos/mpc_720h_100seeds.npz}"
EPISODE_HOURS="${EPISODE_HOURS:-720}"
SEED_START="${SEED_START:-0}"
TASK_COUNT="${TASK_COUNT:-10}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-10}"
SHARDS=()
for TASK_ID in $(seq 0 $((TASK_COUNT - 1))); do
  START_SEED=$((SEED_START + TASK_ID * SEEDS_PER_TASK))
  END_SEED=$((START_SEED + SEEDS_PER_TASK - 1))
  SHARD="$SHARD_DIR/mpc_720h_seeds_${START_SEED}_${END_SEED}.npz"
  if [[ ! -f "$SHARD" ]]; then
    echo "ERROR: missing demonstration shard: $SHARD" >&2
    exit 1
  fi
  SHARDS+=("$SHARD")
done
FINAL_SEED=$((SEED_START + TASK_COUNT * SEEDS_PER_TASK - 1))
mapfile -t EXPECTED_SEEDS < <(seq "$SEED_START" "$FINAL_SEED")

mkdir -p logs "$(dirname "$DEMO_CACHE")"
if [[ -z "${GIT_COMMIT:-}" ]]; then
  GIT_COMMIT="$(git rev-parse HEAD)"
fi
export GIT_COMMIT

python -u scripts/compare_forecast_encoders_rl.py merge-demos \
  --shards "${SHARDS[@]}" \
  --demo-cache "$DEMO_CACHE" \
  --expected-seeds "${EXPECTED_SEEDS[@]}" \
  --episode-hours "$EPISODE_HOURS"
