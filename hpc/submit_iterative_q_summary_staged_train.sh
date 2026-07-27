#!/usr/bin/env bash
#SBATCH --job-name=ccs_iq_sum_stage
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH -o logs/iterative_q_summary_staged-%A_%a.out
#SBATCH -e logs/iterative_q_summary_staged-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
: "${SOURCE_RUN_ROOT:?SOURCE_RUN_ROOT must be set}"
: "${OUT_ROOT:?OUT_ROOT must be set}"
NAMES=(summary_24_72 summary_168 summary_24_72_168 summary_bands_24_72_168)
INPUTS=(forecast_summary_24_72 forecast_summary_168 forecast_summary_24_72_168 forecast_summary_bands_24_72_168)
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
OBSERVATION_INPUT="${INPUTS[$SLURM_ARRAY_TASK_ID]}"

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
    --out-dir "$OUT_ROOT/$NAME/$output_stage" \
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
    --model-seed 0 \
    --device cuda
}

train_stage p1 1 0.0001 0.0003 0.5
train_stage p2 2 0.00005 0.00015 0.0 \
  "$OUT_ROOT/$NAME/p1/iterative_action_q.pt"
train_stage p3 3 0.00003 0.0001 0.0 \
  "$OUT_ROOT/$NAME/p2/iterative_action_q.pt"
train_stage p4 4 0.00003 0.0001 0.0 \
  "$OUT_ROOT/$NAME/p3/iterative_action_q.pt"
