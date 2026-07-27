#!/usr/bin/env bash
#SBATCH --job-name=ccs_iter_q_rollin
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/iterative_q_rollin-%A_%a.out
#SBATCH -e logs/iterative_q_rollin-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_greedy_dagger}"
: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${STAGE:?STAGE must be set}"
: "${LOCK_CONFIG:?LOCK_CONFIG must be set}"
: "${TRAIN_START:?TRAIN_START must be set}"
: "${TRAIN_COUNT:?TRAIN_COUNT must be set}"
: "${VALIDATION_START:?VALIDATION_START must be set}"
: "${VALIDATION_COUNT:?VALIDATION_COUNT must be set}"
: "${CHUNK_SIZE:?CHUNK_SIZE must be set}"
: "${DATASET_SEED:?DATASET_SEED must be set}"
WORKERS=4
SCENARIO_PROTOCOL="${SCENARIO_PROTOCOL:-q_original}"
HARD_SCENARIO_PROBABILITY="${HARD_SCENARIO_PROBABILITY:-0.5}"
FORECAST_CONTEXT_HOURS="${FORECAST_CONTEXT_HOURS:-168}"

TRAIN_TASKS=$(((TRAIN_COUNT + CHUNK_SIZE - 1) / CHUNK_SIZE))
if (( SLURM_ARRAY_TASK_ID < TRAIN_TASKS )); then
  SPLIT=train
  RANGE_START=$((TRAIN_START + SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
  LIMIT=$((TRAIN_START + TRAIN_COUNT - 1))
else
  SPLIT=validation
  VALIDATION_TASK_ID=$((SLURM_ARRAY_TASK_ID - TRAIN_TASKS))
  RANGE_START=$((VALIDATION_START + VALIDATION_TASK_ID * CHUNK_SIZE))
  LIMIT=$((VALIDATION_START + VALIDATION_COUNT - 1))
fi
RANGE_END=$((RANGE_START + CHUNK_SIZE - 1))
if (( RANGE_END > LIMIT )); then
  RANGE_END=$LIMIT
fi
mapfile -t SEEDS < <(seq "$RANGE_START" "$RANGE_END")
SEED_COUNT=${#SEEDS[@]}
BASE_COUNT=$((SEED_COUNT / WORKERS))
EXTRA_COUNT=$((SEED_COUNT % WORKERS))

cd "$PROJECT_DIR"
mkdir -p logs "$RUN_ROOT/$STAGE/$SPLIT"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / WORKERS))
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

pids=()
offset=0
for ((worker = 0; worker < WORKERS; worker++)); do
  worker_count=$BASE_COUNT
  if (( worker < EXTRA_COUNT )); then
    worker_count=$((worker_count + 1))
  fi
  if (( worker_count == 0 )); then
    continue
  fi
  worker_seeds=("${SEEDS[@]:offset:worker_count}")
  first=${worker_seeds[0]}
  last=${worker_seeds[worker_count - 1]}
  python -u experiments/generate_iterative_q_policy_data.py \
    --lock-config "$LOCK_CONFIG" \
    --out-path "$RUN_ROOT/$STAGE/$SPLIT/shard_${first}_${last}_w${worker}.npz" \
    --split "$SPLIT" \
    --seeds "${worker_seeds[@]}" \
    --max-two-vessel-actions 8 \
    --max-three-vessel-actions 4 \
    --episode-hours 720 \
    --reward-scale 0.00001 \
    --dataset-seed "$DATASET_SEED" \
    --variant future_mlp_mode_destination \
    --scenario-protocol "$SCENARIO_PROTOCOL" \
    --hard-scenario-probability "$HARD_SCENARIO_PROBABILITY" \
    --forecast-context-hours "$FORECAST_CONTEXT_HOURS" \
    --device cpu &
  pids+=("$!")
  offset=$((offset + worker_count))
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
