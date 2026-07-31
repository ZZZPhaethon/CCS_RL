#!/usr/bin/env bash
#SBATCH --job-name=e3_future_train
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH -o logs/e3_future_train-%A_%a.out
#SBATCH -e logs/e3_future_train-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"
E3_ROOT="${E3_ROOT:-experiments_results/E3/training_future_information_run01}"
DATA_ROOT="${DATA_ROOT:-$E3_ROOT}"
EXCLUDE_STATE_FEATURES="${EXCLUDE_STATE_FEATURES:-}"
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
MODEL_SEED=$((SLURM_ARRAY_TASK_ID % 3))
if [[ "$VARIANT_INDEX" == "0" ]]; then
  VARIANT=state_only
  OBSERVATION_INPUT=state_only
else
  VARIANT=full_sequence_168
  OBSERVATION_INPUT=forecast_168
fi
OUT_ROOT="$E3_ROOT/$VARIANT/model_seed_${MODEL_SEED}"

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
    if [[ "$VARIANT" == "state_only" ]]; then
      train_data+=("$SOURCE_RUN_ROOT/g${stage_index}/train_merged.npz")
      validation_data+=("$SOURCE_RUN_ROOT/g${stage_index}/validation_merged.npz")
    else
      train_data+=("$DATA_ROOT/augmented_data/g${stage_index}/train_forecast168.npz")
      validation_data+=("$DATA_ROOT/augmented_data/g${stage_index}/validation_forecast168.npz")
    fi
  done
  local initial_args=()
  local exclusion_args=()
  if [[ -n "$initial_checkpoint" ]]; then
    initial_args+=(--initial-checkpoint "$initial_checkpoint")
  fi
  if [[ -n "$EXCLUDE_STATE_FEATURES" ]]; then
    read -r -a excluded_features <<< "$EXCLUDE_STATE_FEATURES"
    exclusion_args+=(--exclude-state-features "${excluded_features[@]}")
  fi
  python -u scripts/train_iterative_action_q.py \
    --train-data "${train_data[@]}" \
    --validation-data "${validation_data[@]}" \
    "${initial_args[@]}" \
    "${exclusion_args[@]}" \
    --out-dir "$OUT_ROOT/$output_stage" \
    --observation-input "$OBSERVATION_INPUT" \
    --forecast-encoder small_mlp \
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
    --device cuda
}

train_stage p1 1 0.0001 0.0003 0.5
train_stage p2 2 0.00005 0.00015 0.0 \
  "$OUT_ROOT/p1/iterative_action_q.pt"
train_stage p3 3 0.00003 0.0001 0.0 \
  "$OUT_ROOT/p2/iterative_action_q.pt"
train_stage p4 4 0.00003 0.0001 0.0 \
  "$OUT_ROOT/p3/iterative_action_q.pt"
