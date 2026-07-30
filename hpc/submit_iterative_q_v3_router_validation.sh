#!/usr/bin/env bash
#SBATCH --job-name=iterq_v3_router
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o logs/iterative_q_v3_router-%j.out
#SBATCH -e logs/iterative_q_v3_router-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
RUN_ROOT="${RUN_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"
: "${ROUTER_SPEC:?ROUTER_SPEC must be set}"
ROUTER_NAME="${ROUTER_SPEC%%:*}"
OUT_ROOT="${OUT_ROOT:-$RUN_ROOT/eval/v3_router_validation}"
EVAL_SEEDS="${EVAL_SEEDS:-8100001:8100002:8100003:8100004:8100005:8100006:8100007:8100008:8100009:8100010:8100011:8100012:8100013:8100014:8100015:8100016:8100017:8100018:8100019:8100020}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
IFS=':' read -r -a VALIDATION_SEEDS <<< "$EVAL_SEEDS"

python -u experiments/evaluate_iterative_q_checkpoint_router.py \
  --checkpoints \
  "p1=$RUN_ROOT/p1/iterative_action_q.pt" \
  "p2=$RUN_ROOT/p2/iterative_action_q.pt" \
  "p3=$RUN_ROOT/p3/iterative_action_q.pt" \
  "p4=$RUN_ROOT/p4/iterative_action_q.pt" \
  --routers "$ROUTER_SPEC" \
  --out-dir "$OUT_ROOT/$ROUTER_NAME" \
  --eval-seeds "${VALIDATION_SEEDS[@]}" \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --scenario-protocol unified_window_v1 \
  --hard-scenario-probability 0.5 \
  --forecast-context-hours 168 \
  --max-overrides 12 \
  --validation-only \
  --device cuda
