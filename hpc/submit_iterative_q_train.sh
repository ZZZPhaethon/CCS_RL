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
: "${DATA_ROOT:=$RUN_ROOT}"
: "${OUTPUT_STAGE:?OUTPUT_STAGE must be set}"
: "${DATA_STAGES:?DATA_STAGES must be set}"
: "${ENCODER_LR:?ENCODER_LR must be set}"
: "${HEAD_LR:?HEAD_LR must be set}"
: "${FOLLOW_ANCHOR:?FOLLOW_ANCHOR must be set}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-}"
CREATE_LOCK="${CREATE_LOCK:-0}"
PROTOCOL_PREFIX="${PROTOCOL_PREFIX:-iterative_q}"
OBSERVATION_INPUT="${OBSERVATION_INPUT:-state_only}"
FORECAST_ENCODER="${FORECAST_ENCODER:-small_mlp}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-179:180-251:252-323:324-395:396-467:468-539:540-611:612-680}"
MAX_OVERRIDES="${MAX_OVERRIDES:-8}"
REQUIRED_HEADS="${REQUIRED_HEADS:-4}"
MODEL_SEED="${MODEL_SEED:-0}"
STAGE_SAMPLING_TEMPERATURE="${STAGE_SAMPLING_TEMPERATURE:-1.0}"
NEAR_DUPLICATE_WEIGHTING="${NEAR_DUPLICATE_WEIGHTING:-none}"
NEAR_DUPLICATE_COSINE_THRESHOLD="${NEAR_DUPLICATE_COSINE_THRESHOLD:-0.995}"
NEAR_DUPLICATE_RMS_THRESHOLD="${NEAR_DUPLICATE_RMS_THRESHOLD:-0.10}"
ROOT_ADVANTAGE_WEIGHTING="${ROOT_ADVANTAGE_WEIGHTING:-none}"
ROOT_ADVANTAGE_THRESHOLD_EUR="${ROOT_ADVANTAGE_THRESHOLD_EUR:-40000}"
ROOT_NO_IMPROVEMENT_WEIGHT="${ROOT_NO_IMPROVEMENT_WEIGHT:-0.5}"
ROOT_MODERATE_IMPROVEMENT_WEIGHT="${ROOT_MODERATE_IMPROVEMENT_WEIGHT:-1.0}"
ROOT_STRONG_IMPROVEMENT_WEIGHT="${ROOT_STRONG_IMPROVEMENT_WEIGHT:-2.0}"
PREVIOUS_POLICY_ANCHOR="${PREVIOUS_POLICY_ANCHOR:-0.0}"
PREVIOUS_POLICY_RELEASE_MARGIN_EUR="${PREVIOUS_POLICY_RELEASE_MARGIN_EUR:-40000}"
PREVIOUS_POLICY_PLATEAU_MARGIN_EUR="${PREVIOUS_POLICY_PLATEAU_MARGIN_EUR:-0}"
PREVIOUS_POLICY_ANCHOR_TEMPERATURE="${PREVIOUS_POLICY_ANCHOR_TEMPERATURE:-0.5}"
PREVIOUS_POLICY_ANCHOR_WEIGHTING="${PREVIOUS_POLICY_ANCHOR_WEIGHTING:-hard}"
ALLOW_ANCHOR_WITHOUT_INITIAL_CHECKPOINT="${ALLOW_ANCHOR_WITHOUT_INITIAL_CHECKPOINT:-0}"
POLICY_WINDOWS_CSV="${POLICY_WINDOWS_H//:/,}"

IFS=':' read -r -a STAGES <<< "$DATA_STAGES"
TRAIN_DATA=()
VALIDATION_DATA=()
for stage in "${STAGES[@]}"; do
  TRAIN_DATA+=("$DATA_ROOT/$stage/train_merged.npz")
  VALIDATION_DATA+=("$DATA_ROOT/$stage/validation_merged.npz")
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
ANCHOR_WITHOUT_INITIAL_ARGS=()
if [[ "$ALLOW_ANCHOR_WITHOUT_INITIAL_CHECKPOINT" == "1" ]]; then
  ANCHOR_WITHOUT_INITIAL_ARGS+=(--allow-anchor-without-initial-checkpoint)
fi

python -u scripts/train_iterative_action_q.py \
  --train-data "${TRAIN_DATA[@]}" \
  --validation-data "${VALIDATION_DATA[@]}" \
  "${INITIAL_ARGS[@]}" \
  --out-dir "$RUN_ROOT/$OUTPUT_STAGE" \
  --observation-input "$OBSERVATION_INPUT" \
  --forecast-encoder "$FORECAST_ENCODER" \
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
  --stage-sampling-temperature "$STAGE_SAMPLING_TEMPERATURE" \
  --near-duplicate-weighting "$NEAR_DUPLICATE_WEIGHTING" \
  --near-duplicate-cosine-threshold \
    "$NEAR_DUPLICATE_COSINE_THRESHOLD" \
  --near-duplicate-rms-threshold "$NEAR_DUPLICATE_RMS_THRESHOLD" \
  --root-advantage-weighting "$ROOT_ADVANTAGE_WEIGHTING" \
  --root-advantage-threshold-eur "$ROOT_ADVANTAGE_THRESHOLD_EUR" \
  --root-no-improvement-weight "$ROOT_NO_IMPROVEMENT_WEIGHT" \
  --root-moderate-improvement-weight \
    "$ROOT_MODERATE_IMPROVEMENT_WEIGHT" \
  --root-strong-improvement-weight "$ROOT_STRONG_IMPROVEMENT_WEIGHT" \
  --previous-policy-anchor-coefficient "$PREVIOUS_POLICY_ANCHOR" \
  --previous-policy-release-margin-eur \
    "$PREVIOUS_POLICY_RELEASE_MARGIN_EUR" \
  --previous-policy-anchor-plateau-margin-eur \
    "$PREVIOUS_POLICY_PLATEAU_MARGIN_EUR" \
  --previous-policy-anchor-temperature \
    "$PREVIOUS_POLICY_ANCHOR_TEMPERATURE" \
  --previous-policy-anchor-weighting \
    "$PREVIOUS_POLICY_ANCHOR_WEIGHTING" \
  "${ANCHOR_WITHOUT_INITIAL_ARGS[@]}" \
  --pairwise-min-cost-eur 10000 \
  --ranking-temperature 0.5 \
  --model-seed "$MODEL_SEED" \
  --device cuda

if [[ "$CREATE_LOCK" == "1" ]]; then
  : "${RESIDUAL_MARGIN:?RESIDUAL_MARGIN must be set when CREATE_LOCK=1}"
  : "${ECONOMIC_MARGIN_EUR:?ECONOMIC_MARGIN_EUR must be set when CREATE_LOCK=1}"
  python -u experiments/create_iterative_q_lock.py \
    --checkpoint "$RUN_ROOT/$OUTPUT_STAGE/iterative_action_q.pt" \
    --out-path "$RUN_ROOT/${OUTPUT_STAGE}_lock.json" \
    --protocol-id "${PROTOCOL_PREFIX}_${OUTPUT_STAGE}" \
    --residual-margin "$RESIDUAL_MARGIN" \
    --economic-margin-eur "$ECONOMIC_MARGIN_EUR" \
    --required-heads "$REQUIRED_HEADS" \
    --max-overrides "$MAX_OVERRIDES" \
    --windows-h "$POLICY_WINDOWS_CSV"
fi
