#!/usr/bin/env bash
#SBATCH --job-name=ccs_iq_adapter_eval
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-14%4
#SBATCH -o logs/iterative_q_future_adapter_eval-%A_%a.out
#SBATCH -e logs/iterative_q_future_adapter_eval-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
: "${OUT_ROOT:?OUT_ROOT must be set}"

NAMES=(frozen_scale025 frozen_scale100 tune_scale025_dropout25)
CANDIDATE_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
MODEL_SEED=$((SLURM_ARRAY_TASK_ID % 5))
NAME="${NAMES[$CANDIDATE_INDEX]}"
RUN_ROOT="$OUT_ROOT/$NAME/model_seed_${MODEL_SEED}"
FUTURE_ABLATION="${FUTURE_ABLATION:-none}"
EVAL_SUBDIR="${EVAL_SUBDIR:-eval_formal}"
WINDOWS="108-155,156-203,204-251,252-299,300-347,348-395,396-443,444-491,492-539,540-587,588-635,636-680"
mapfile -t TEST_SEEDS < <(seq 8000001 8000030)

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u experiments/evaluate_iterative_action_q.py \
  --checkpoint "$RUN_ROOT/iterative_action_q.pt" \
  --out-dir "$RUN_ROOT/$EVAL_SUBDIR" \
  --eval-seeds "${TEST_SEEDS[@]}" \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --scenario-protocol unified_window_v1 \
  --hard-scenario-probability 0.5 \
  --forecast-context-hours 168 \
  --future-ablation "$FUTURE_ABLATION" \
  --gates "${NAME}_strict4_margin40k":4:0.40:12:"$WINDOWS" \
  --device cuda
