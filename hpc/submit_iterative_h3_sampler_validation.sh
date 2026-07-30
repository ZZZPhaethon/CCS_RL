#!/usr/bin/env bash
#SBATCH --job-name=iter_h3_val
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH -o logs/iterative_h3_val-%A_%a.out
#SBATCH -e logs/iterative_h3_val-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${OUT_ROOT:?OUT_ROOT must be set}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680}"
POLICY_WINDOWS_CSV="${POLICY_WINDOWS_H//:/,}"

MODEL_SEED=$((SLURM_ARRAY_TASK_ID % 3))
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
case "$VARIANT_INDEX" in
  0) VARIANT=b_gate_only ;;
  1) VARIANT=c_dedup_balanced ;;
  2) VARIANT=d_dedup_advantage ;;
  *)
    echo "Unknown variant index: $VARIANT_INDEX" >&2
    exit 2
    ;;
esac
CHECKPOINT="$OUT_ROOT/branches/$VARIANT/model_seed_${MODEL_SEED}/p2/iterative_action_q.pt"
OUT_DIR="$OUT_ROOT/validation/$VARIANT/model_seed_${MODEL_SEED}"
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "Missing P2 checkpoint: $CHECKPOINT" >&2
  exit 2
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing existing validation output: $OUT_DIR" >&2
  exit 2
fi

VALIDATION_SEEDS=(
  8100001 8100002 8100003 8100004 8100005
  8100006 8100007 8100008 8100009 8100010
  8100011 8100012 8100013 8100014 8100015
  8100016 8100017 8100018 8100019 8100020
)

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u experiments/evaluate_iterative_action_q.py \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --eval-seeds "${VALIDATION_SEEDS[@]}" \
  --validation-only \
  --seed-manifest experiments/protocols/unified_window_v1_seed_manifest.json \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --scenario-protocol unified_window_v1 \
  --stress-level medium \
  --hard-scenario-probability 0.5 \
  --forecast-context-hours 168 \
  --gates \
    "h4_m040_w12_c12:4:0.4:12:$POLICY_WINDOWS_CSV" \
    "h3_m040_w12_c12:3:0.4:12:$POLICY_WINDOWS_CSV" \
  --device cuda
