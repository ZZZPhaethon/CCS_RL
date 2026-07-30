#!/usr/bin/env bash
#SBATCH --job-name=e2_one_train
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH -o logs/e2_one_train-%A_%a.out
#SBATCH -e logs/e2_one_train-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
RUN_ROOT="${RUN_ROOT:-experiments_results/E2/training_one_shot_matched_run01}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$RUN_ROOT/g0/train_merged.npz}"
MODEL_SEED="$SLURM_ARRAY_TASK_ID"
OUT_DIR="$RUN_ROOT/model_seed_${MODEL_SEED}/p1"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u scripts/train_iterative_action_q.py \
  --train-data "$TRAIN_DATA_PATH" \
  --validation-data "$RUN_ROOT/g0/validation_merged.npz" \
  --out-dir "$OUT_DIR" \
  --observation-input shared_future_summary \
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
  --model-seed "$MODEL_SEED" \
  --device cuda
