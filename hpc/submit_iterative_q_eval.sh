#!/usr/bin/env bash
#SBATCH --job-name=ccs_iter_q_eval
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o logs/iterative_q_eval-%j.out
#SBATCH -e logs/iterative_q_eval-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_greedy_dagger}"
: "${RUN_ROOT:?RUN_ROOT must be set}"
FINAL_STAGE="${FINAL_STAGE:-p4}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-179:180-251:252-323:324-395:396-467:468-539:540-611:612-680}"
MAX_OVERRIDES="${MAX_OVERRIDES:-8}"
POLICY_WINDOWS_CSV="${POLICY_WINDOWS_H//:/,}"
EVAL_NAME="${EVAL_NAME:-iterative_action_q}"
SCENARIO_PROTOCOL="${SCENARIO_PROTOCOL:-q_original}"
HARD_SCENARIO_PROBABILITY="${HARD_SCENARIO_PROBABILITY:-0.5}"
FORECAST_CONTEXT_HOURS="${FORECAST_CONTEXT_HOURS:-168}"
FUTURE_SUMMARY_WINDOWS_H="${FUTURE_SUMMARY_WINDOWS_H:-}"
V4_RUN_DIR="${V4_RUN_DIR:-output/unified_physics/residual_v4_seed0_100k_20260725_noref}"
EVAL_SEEDS="${EVAL_SEEDS:-}"
VALIDATION_ONLY="${VALIDATION_ONLY:-0}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
if [[ -n "$EVAL_SEEDS" ]]; then
  if [[ "$EVAL_SEEDS" == *:* ]]; then
    IFS=':' read -r -a TEST_SEEDS <<< "$EVAL_SEEDS"
  else
    read -r -a TEST_SEEDS <<< "$EVAL_SEEDS"
  fi
else
  mapfile -t TEST_SEEDS < <(seq 7000 7029)
fi
VALIDATION_ARGS=()
if [[ "$VALIDATION_ONLY" == "1" ]]; then
  VALIDATION_ARGS+=(--validation-only)
fi
FUTURE_SUMMARY_ARGS=()
if [[ -n "$FUTURE_SUMMARY_WINDOWS_H" ]]; then
  IFS=':' read -r -a FUTURE_SUMMARY_WINDOWS <<< "$FUTURE_SUMMARY_WINDOWS_H"
  FUTURE_SUMMARY_ARGS+=(
    --future-summary-windows-h "${FUTURE_SUMMARY_WINDOWS[@]}"
  )
fi

if [[ "$SCENARIO_PROTOCOL" == "v4_mixed_window" ]]; then
  python -u experiments/evaluate_iterative_action_q.py \
    --checkpoint "$RUN_ROOT/$FINAL_STAGE/iterative_action_q.pt" \
    --out-dir "$RUN_ROOT/eval/q_original" \
    --eval-seeds "${TEST_SEEDS[@]}" \
    --episode-hours 720 \
    --reward-scale 0.00001 \
    --gates "$EVAL_NAME":4:0.40:"$MAX_OVERRIDES":"$POLICY_WINDOWS_CSV" \
    "${FUTURE_SUMMARY_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    --device cuda

  python -u experiments/cross_evaluate_iterative_q_v4.py \
    --iterative-q-checkpoint "$RUN_ROOT/$FINAL_STAGE/iterative_action_q.pt" \
    --v4-run-dir "$V4_RUN_DIR" \
    --out-dir "$RUN_ROOT/eval/window_normal_hard" \
    --tasks q_on_v4_normal q_on_v4_hard \
    --device cuda
else
  python -u experiments/evaluate_iterative_action_q.py \
    --checkpoint "$RUN_ROOT/$FINAL_STAGE/iterative_action_q.pt" \
    --out-dir "$RUN_ROOT/eval/$EVAL_NAME" \
    --eval-seeds "${TEST_SEEDS[@]}" \
    --episode-hours 720 \
    --reward-scale 0.00001 \
    --scenario-protocol "$SCENARIO_PROTOCOL" \
    --hard-scenario-probability "$HARD_SCENARIO_PROBABILITY" \
    --forecast-context-hours "$FORECAST_CONTEXT_HOURS" \
    --gates "$EVAL_NAME":4:0.40:"$MAX_OVERRIDES":"$POLICY_WINDOWS_CSV" \
    "${FUTURE_SUMMARY_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    --device cuda
fi
