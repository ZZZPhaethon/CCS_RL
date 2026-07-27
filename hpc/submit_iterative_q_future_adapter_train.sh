#!/usr/bin/env bash
#SBATCH --job-name=ccs_iq_adapter_train
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-14%4
#SBATCH -o logs/iterative_q_future_adapter_train-%A_%a.out
#SBATCH -e logs/iterative_q_future_adapter_train-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
: "${SOURCE_RUN_ROOT:?SOURCE_RUN_ROOT must be set}"
: "${BASE_ROOT:?BASE_ROOT must be set}"
: "${OUT_ROOT:?OUT_ROOT must be set}"

NAMES=(frozen_scale025 frozen_scale100 tune_scale025_dropout25)
FUSIONS=(residual_frozen residual_frozen residual_tune)
SCALE_LIMITS=(0.25 1.0 0.25)
DROPOUTS=(0.0 0.0 0.25)
CANDIDATE_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
MODEL_SEED=$((SLURM_ARRAY_TASK_ID % 5))
NAME="${NAMES[$CANDIDATE_INDEX]}"
FUSION="${FUSIONS[$CANDIDATE_INDEX]}"
SCALE_LIMIT="${SCALE_LIMITS[$CANDIDATE_INDEX]}"
DROPOUT="${DROPOUTS[$CANDIDATE_INDEX]}"
RUN_ROOT="$OUT_ROOT/$NAME/model_seed_${MODEL_SEED}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TRAIN_DATA=()
VALIDATION_DATA=()
for stage_index in 0 1 2 3; do
  TRAIN_DATA+=("$SOURCE_RUN_ROOT/g${stage_index}/train_forecast168.npz")
  VALIDATION_DATA+=("$SOURCE_RUN_ROOT/g${stage_index}/validation_forecast168.npz")
done

python -u scripts/train_iterative_action_q.py \
  --train-data "${TRAIN_DATA[@]}" \
  --validation-data "${VALIDATION_DATA[@]}" \
  --initial-checkpoint "$BASE_ROOT/model_seed_${MODEL_SEED}/p4/iterative_action_q.pt" \
  --out-dir "$RUN_ROOT" \
  --observation-input forecast_summary_168 \
  --future-fusion "$FUSION" \
  --future-residual-scale-limit "$SCALE_LIMIT" \
  --future-dropout "$DROPOUT" \
  --epochs 40 \
  --patience 8 \
  --batch-size 16 \
  --encoder-learning-rate 0.00003 \
  --head-learning-rate 0.0001 \
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
  --follow-anchor-coefficient 0.0 \
  --pairwise-min-cost-eur 10000 \
  --ranking-temperature 0.5 \
  --model-seed "$MODEL_SEED" \
  --device cuda
