#!/usr/bin/env bash
#SBATCH --job-name=iq_state_ind_train
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH -o logs/iterative_q_state_individual_retrain-%A_%a.out
#SBATCH -e logs/iterative_q_state_individual_retrain-%A_%a.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
DATA_ROOT="${DATA_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"
OUT_ROOT="${OUT_ROOT:-experiments_results/E1/iterative_q_state_individual_retrain_20260731_run01}"
MODEL_SEED="${MODEL_SEED:-0}"

CONDITIONS=(drop_in_transit_fill drop_episode_progress)
EXCLUSIONS=(
  "hour_of_week in_transit_fill"
  "hour_of_week episode_progress"
)
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
read -r -a EXCLUDE_FEATURES <<< "${EXCLUSIONS[$SLURM_ARRAY_TASK_ID]}"
RUN_ROOT="$OUT_ROOT/$CONDITION/model_seed_$MODEL_SEED"

cd "$PROJECT_DIR"
mkdir -p logs "$OUT_ROOT"
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing existing output: $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"

export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

{
  printf 'condition=%s\n' "$CONDITION"
  printf 'excluded_state_features=%s\n' "${EXCLUDE_FEATURES[*]}"
  printf 'model_seed=%s\n' "$MODEL_SEED"
  printf 'data_root=%s\n' "$DATA_ROOT"
  printf 'stages=P1:G0,P2:G0-G1,P3:G0-G2,P4:G0-G3\n'
  printf 'observation_input=shared_future_summary\n'
  printf 'epochs=40\n'
  printf 'patience=8\n'
  printf 'batch_size=16\n'
  printf 'checkpoint_selection_metric=composite\n'
} > "$RUN_ROOT/protocol_lock.txt"

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
    train_data+=("$DATA_ROOT/g${stage_index}/train_merged.npz")
    validation_data+=("$DATA_ROOT/g${stage_index}/validation_merged.npz")
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
    --exclude-state-features "${EXCLUDE_FEATURES[@]}" \
    --observation-input shared_future_summary \
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
    --checkpoint-selection-metric composite \
    --device cuda
}

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "job_id=$SLURM_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "condition=$CONDITION"
echo "excluded_state_features=${EXCLUDE_FEATURES[*]}"

train_stage p1 1 0.0001 0.0003 0.5
train_stage p2 2 0.00005 0.00015 0.0 \
  "$RUN_ROOT/p1/iterative_action_q.pt"
train_stage p3 3 0.00003 0.0001 0.0 \
  "$RUN_ROOT/p2/iterative_action_q.pt"
train_stage p4 4 0.00003 0.0001 0.0 \
  "$RUN_ROOT/p3/iterative_action_q.pt"

echo "finished_at=$(date --iso-8601=seconds)"
