#!/usr/bin/env bash
#SBATCH --job-name=ccs_iq_enc_eval
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH -o logs/iterative_q_encoder_eval-%A_%a.out
#SBATCH -e logs/iterative_q_encoder_eval-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
: "${OUT_ROOT:?OUT_ROOT must be set}"
ENCODERS=(small_mlp tcn gru)
ENCODER="${ENCODERS[$SLURM_ARRAY_TASK_ID]}"
WINDOWS="108-155,156-203,204-251,252-299,300-347,348-395,396-443,444-491,492-539,540-587,588-635,636-680"
EVAL_SEED_START="${EVAL_SEED_START:-7000}"
EVAL_SEED_END="${EVAL_SEED_END:-7029}"
EVAL_SUBDIR="${EVAL_SUBDIR:-eval}"
CHECKPOINT_SUBDIR="${CHECKPOINT_SUBDIR:-}"
mapfile -t TEST_SEEDS < <(seq "$EVAL_SEED_START" "$EVAL_SEED_END")
CHECKPOINT_DIR="$OUT_ROOT/$ENCODER"
if [[ -n "$CHECKPOINT_SUBDIR" ]]; then
  CHECKPOINT_DIR="$CHECKPOINT_DIR/$CHECKPOINT_SUBDIR"
fi

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u experiments/evaluate_iterative_action_q.py \
  --checkpoint "$CHECKPOINT_DIR/iterative_action_q.pt" \
  --out-dir "$OUT_ROOT/$ENCODER/$EVAL_SUBDIR" \
  --eval-seeds "${TEST_SEEDS[@]}" \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --scenario-protocol unified_window_v1 \
  --hard-scenario-probability 0.5 \
  --forecast-context-hours 168 \
  --gates "${ENCODER}_strict4_margin40k":4:0.40:12:"$WINDOWS" \
  --device cuda
