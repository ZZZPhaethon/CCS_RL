#!/usr/bin/env bash
#SBATCH --job-name=ccs_iq_encoder
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH -o logs/iterative_q_encoder-%A_%a.out
#SBATCH -e logs/iterative_q_encoder-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
: "${SOURCE_RUN_ROOT:?SOURCE_RUN_ROOT must be set}"
: "${OUT_ROOT:?OUT_ROOT must be set}"

ENCODERS=(small_mlp tcn gru)
ENCODER="${ENCODERS[$SLURM_ARRAY_TASK_ID]}"
TRAIN_DATA=()
VALIDATION_DATA=()
for stage in g0 g1 g2 g3; do
  TRAIN_DATA+=("$SOURCE_RUN_ROOT/$stage/train_forecast168.npz")
  VALIDATION_DATA+=("$SOURCE_RUN_ROOT/$stage/validation_forecast168.npz")
done

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u scripts/train_iterative_action_q.py \
  --train-data "${TRAIN_DATA[@]}" \
  --validation-data "${VALIDATION_DATA[@]}" \
  --out-dir "$OUT_ROOT/$ENCODER" \
  --observation-input forecast_168 \
  --forecast-encoder "$ENCODER" \
  --epochs 40 \
  --patience 8 \
  --batch-size 16 \
  --encoder-learning-rate 0.0001 \
  --head-learning-rate 0.0003 \
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
  --follow-anchor-coefficient 0.5 \
  --pairwise-min-cost-eur 10000 \
  --ranking-temperature 0.5 \
  --model-seed 0 \
  --device cuda
