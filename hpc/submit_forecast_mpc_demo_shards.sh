#!/usr/bin/env bash
#SBATCH --job-name=ccs_mpc_demo_shard
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o logs/mpc_demo_shard-%A_%a.out
#SBATCH -e logs/mpc_demo_shard-%A_%a.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM}"
cd "$PROJECT_DIR"
export PYTHONPATH="src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ ! "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID > 9 )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be in 0-9; got '${TASK_ID:-unset}'." >&2
  exit 1
fi

SHARD_DIR="${SHARD_DIR:-output/rl_forecast/demos/mpc_720h_100seeds_shards}"
EPISODE_HOURS="${EPISODE_HOURS:-720}"
SEED_START="${SEED_START:-0}"
SEEDS_PER_TASK="${SEEDS_PER_TASK:-10}"
START_SEED=$((SEED_START + TASK_ID * SEEDS_PER_TASK))
END_SEED=$((START_SEED + SEEDS_PER_TASK - 1))
mapfile -t DEMO_SEEDS < <(seq "$START_SEED" "$END_SEED")
DEMO_CACHE="$SHARD_DIR/mpc_720h_seeds_${START_SEED}_${END_SEED}.npz"

mkdir -p logs "$SHARD_DIR"
if [[ -z "${GIT_COMMIT:-}" ]]; then
  GIT_COMMIT="$(git rev-parse HEAD)"
fi
export GIT_COMMIT

echo "Collecting MPC seeds $START_SEED..$END_SEED into $DEMO_CACHE"
python -u scripts/compare_forecast_encoders_rl.py generate-demos \
  --demo-cache "$DEMO_CACHE" \
  --demo-seeds "${DEMO_SEEDS[@]}" \
  --episode-hours "$EPISODE_HOURS"
