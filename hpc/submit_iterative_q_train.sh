#!/usr/bin/env bash
#SBATCH --job-name=ccs_iter_q_train
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o logs/iterative_q_train-%j.out
#SBATCH -e logs/iterative_q_train-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_greedy_dagger}"
: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${OUTPUT_STAGE:?OUTPUT_STAGE must be set}"
: "${DATA_STAGES:?DATA_STAGES must be set}"
: "${ENCODER_LR:?ENCODER_LR must be set}"
: "${HEAD_LR:?HEAD_LR must be set}"
: "${FOLLOW_ANCHOR:?FOLLOW_ANCHOR must be set}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-}"
CREATE_LOCK="${CREATE_LOCK:-0}"
PROTOCOL_PREFIX="${PROTOCOL_PREFIX:-iterative_q}"

IFS=':' read -r -a STAGES <<< "$DATA_STAGES"
TRAIN_DATA=()
VALIDATION_DATA=()
for stage in "${STAGES[@]}"; do
  TRAIN_DATA+=("$RUN_ROOT/$stage/train_merged.npz")
  VALIDATION_DATA+=("$RUN_ROOT/$stage/validation_merged.npz")
done

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

INITIAL_ARGS=()
if [[ -n "$INITIAL_CHECKPOINT" ]]; then
  INITIAL_ARGS+=(--initial-checkpoint "$INITIAL_CHECKPOINT")
fi

python -u scripts/train_iterative_action_q.py \
  --train-data "${TRAIN_DATA[@]}" \
  --validation-data "${VALIDATION_DATA[@]}" \
  "${INITIAL_ARGS[@]}" \
  --out-dir "$RUN_ROOT/$OUTPUT_STAGE" \
  --epochs 40 \
  --patience 8 \
  --batch-size 16 \
  --encoder-learning-rate "$ENCODER_LR" \
  --head-learning-rate "$HEAD_LR" \
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
  --follow-anchor-coefficient "$FOLLOW_ANCHOR" \
  --pairwise-min-cost-eur 10000 \
  --ranking-temperature 0.5 \
  --model-seed 0 \
  --device cuda

if [[ "$CREATE_LOCK" == "1" ]]; then
  : "${RESIDUAL_MARGIN:?RESIDUAL_MARGIN must be set when CREATE_LOCK=1}"
  : "${ECONOMIC_MARGIN_EUR:?ECONOMIC_MARGIN_EUR must be set when CREATE_LOCK=1}"
  python -u experiments/create_iterative_q_lock.py \
    --checkpoint "$RUN_ROOT/$OUTPUT_STAGE/iterative_action_q.pt" \
    --out-path "$RUN_ROOT/${OUTPUT_STAGE}_lock.json" \
    --protocol-id "${PROTOCOL_PREFIX}_${OUTPUT_STAGE}" \
    --residual-margin "$RESIDUAL_MARGIN" \
    --economic-margin-eur "$ECONOMIC_MARGIN_EUR"
fi
