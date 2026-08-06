#!/usr/bin/env bash
#SBATCH --job-name=iq_state_eval
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH -o logs/iterative_q_state_retrain_eval-%A_%a.out
#SBATCH -e logs/iterative_q_state_retrain_eval-%A_%a.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
OUT_ROOT="${OUT_ROOT:-experiments_results/E1/iterative_q_state_retrain_ablation_20260730_run01}"
MODEL_SEED="${MODEL_SEED:-0}"
CONDITIONS=(drop_hour_of_week drop_all_three)
CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
RUN_ROOT="$OUT_ROOT/$CONDITION/model_seed_$MODEL_SEED"
CHECKPOINT="$RUN_ROOT/p4/iterative_action_q.pt"
EVAL_DIR="$RUN_ROOT/eval/formal_9000031_9000060"
POLICY_WINDOWS="108-155,156-203,204-251,252-299,300-347,348-395,396-443,444-491,492-539,540-587,588-635,636-680"
mapfile -t EVAL_SEEDS < <(seq 9000031 9000060)

cd "$PROJECT_DIR"
mkdir -p logs
test -s "$CHECKPOINT"
if [[ -e "$EVAL_DIR" ]]; then
  echo "Refusing existing evaluation output: $EVAL_DIR" >&2
  exit 2
fi

export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "job_id=$SLURM_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "condition=$CONDITION"

python -u -m experiments.evaluate_iterative_action_q \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$EVAL_DIR" \
  --eval-seeds "${EVAL_SEEDS[@]}" \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --scenario-protocol unified_window_v1 \
  --hard-scenario-probability 0.5 \
  --forecast-context-hours 168 \
  --gates \
    "formal_state_retrain:4:0.40:12:$POLICY_WINDOWS" \
  --device cpu

echo "finished_at=$(date --iso-8601=seconds)"
