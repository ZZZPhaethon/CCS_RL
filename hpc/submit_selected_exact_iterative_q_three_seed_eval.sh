#!/usr/bin/env bash
#SBATCH --job-name=iterq_exact3
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o logs/iterq_exact3-%j.out
#SBATCH -e logs/iterq_exact3-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
CHECKPOINT="${CHECKPOINT:-output/iterative_q_validation_search/baseline_p1_p4/p3/iterative_action_q.pt}"
OUT_DIR="${OUT_DIR:-output/selected_exact_single168_seed0_three_seed_eval_20260728}"
EXPECTED_SHA256="c70bc00967579594c02d233af66cf287631e6a65866a36f3209bac98678ffa75"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

which python
python --version
python - <<'PY'
import pulp
import torch

print("torch", torch.__version__)
print("cbc_available", pulp.PULP_CBC_CMD(msg=False).available())
PY

ACTUAL_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "checkpoint SHA-256 mismatch: $ACTUAL_SHA256" >&2
  exit 1
fi
echo "checkpoint_sha256=$ACTUAL_SHA256"

python -u experiments/evaluate_iterative_action_q.py \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --eval-seeds 8100001 8100002 8100003 \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --scenario-protocol unified_window_v1 \
  --hard-scenario-probability 0.5 \
  --forecast-context-hours 168 \
  --future-summary-windows-h 168 \
  --gates selected_exact_single168_seed0:4:0.40:12:"108-155,156-203,204-251,252-299,300-347,348-395,396-443,444-491,492-539,540-587,588-635,636-680" \
  --validation-only \
  --device cpu

test -f "$OUT_DIR/evaluation.csv"
test -f "$OUT_DIR/summary.json"
