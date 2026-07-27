#!/usr/bin/env bash
#SBATCH --job-name=ccs_iq_repro_train
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-14%4
#SBATCH -o logs/iterative_q_repro_train-%A_%a.out
#SBATCH -e logs/iterative_q_repro_train-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
: "${SOURCE_RUN_ROOT:?SOURCE_RUN_ROOT must be set}"
: "${OUT_ROOT:?OUT_ROOT must be set}"
: "${EXPERIMENT_AXIS:?EXPERIMENT_AXIS must be model_seed or root_sample}"

METHODS=(state_only original_14d summary_168)
INPUTS=(state_only v4_future_24_72 forecast_summary_168)
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
REPLICATE=$((SLURM_ARRAY_TASK_ID % 5))
METHOD="${METHODS[$METHOD_INDEX]}"
OBSERVATION_INPUT="${INPUTS[$METHOD_INDEX]}"

case "$EXPERIMENT_AXIS" in
  model_seed)
    MODEL_SEED="$REPLICATE"
    ROOT_SAMPLE_FRACTION=1.0
    ROOT_SAMPLE_SEED=0
    RUN_NAME="model_seed_${MODEL_SEED}"
    ;;
  root_sample)
    MODEL_SEED=0
    ROOT_SAMPLE_FRACTION=0.8
    ROOT_SAMPLE_SEED=$((101 + REPLICATE))
    RUN_NAME="root_seed_${ROOT_SAMPLE_SEED}"
    ;;
  *)
    echo "Unknown EXPERIMENT_AXIS: $EXPERIMENT_AXIS" >&2
    exit 2
    ;;
esac

RUN_ROOT="$OUT_ROOT/$EXPERIMENT_AXIS/$METHOD/$RUN_NAME"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

train_stage() {
  local output_stage="$1"
  local stage_count="$2"
  local encoder_lr="$3"
  local head_lr="$4"
  local follow_anchor="$5"
  local initial_checkpoint="${6:-}"
  local train_data=()
  local validation_data=()
  local stage_index
  for ((stage_index = 0; stage_index < stage_count; stage_index++)); do
    train_data+=("$SOURCE_RUN_ROOT/g${stage_index}/train_forecast168.npz")
    validation_data+=("$SOURCE_RUN_ROOT/g${stage_index}/validation_forecast168.npz")
  done
  local initial_args=()
  if [[ -n "$initial_checkpoint" ]]; then
    initial_args+=(--initial-checkpoint "$initial_checkpoint")
  fi
  python -u scripts/train_iterative_action_q.py \
    --train-data "${train_data[@]}" \
    --validation-data "${validation_data[@]}" \
    "${initial_args[@]}" \
    --out-dir "$RUN_ROOT/$output_stage" \
    --observation-input "$OBSERVATION_INPUT" \
    --epochs 40 \
    --patience 8 \
    --batch-size 16 \
    --encoder-learning-rate "$encoder_lr" \
    --head-learning-rate "$head_lr" \
    --heads 5 \
    --quantiles 51 \
    --prior-scale 0.25 \
    --action-embedding-size 16 \
    --action-feature-size 64 \
    --bootstrap-probability 0.8 \
    --improving-sample-weight 2.5 \
    --ranking-coefficient 1.0 \
    --listwise-coefficient 0.25 \
    --classification-coefficient 1.0 \
    --follow-anchor-coefficient "$follow_anchor" \
    --pairwise-min-cost-eur 10000 \
    --ranking-temperature 0.5 \
    --model-seed "$MODEL_SEED" \
    --root-sample-fraction "$ROOT_SAMPLE_FRACTION" \
    --root-sample-seed "$ROOT_SAMPLE_SEED" \
    --device cuda
}

train_stage p1 1 0.0001 0.0003 0.5
train_stage p2 2 0.00005 0.00015 0.0 \
  "$RUN_ROOT/p1/iterative_action_q.pt"
train_stage p3 3 0.00003 0.0001 0.0 \
  "$RUN_ROOT/p2/iterative_action_q.pt"
train_stage p4 4 0.00003 0.0001 0.0 \
  "$RUN_ROOT/p3/iterative_action_q.pt"
