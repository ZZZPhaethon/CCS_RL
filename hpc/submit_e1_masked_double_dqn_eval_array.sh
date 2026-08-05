#!/usr/bin/env bash
#SBATCH --job-name=ccs_e1_dqn_eval
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH -o logs/e1_masked_double_dqn_eval-%A_%a.out
#SBATCH -e logs/e1_masked_double_dqn_eval-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_dqn_20260804}"
RESULT_ROOT="${RESULT_ROOT:-experiments_results/E1_addendum_masked_double_dqn_20260804}"
MODEL_SEED="${SLURM_ARRAY_TASK_ID}"
RUN_DIR="$RESULT_ROOT/training/model_seed_${MODEL_SEED}"
OUT_DIR="$RESULT_ROOT/formal_test/model_seed_${MODEL_SEED}"
TEST_SEEDS=($(seq 9000031 9000060))

cd "$PROJECT_DIR"
mkdir -p logs "$RESULT_ROOT/formal_test"
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

test -f "$RUN_DIR/masked_double_dqn_best_validation.pt"
test -f "$RUN_DIR/training_complete.json"
python - "$RUN_DIR/training_complete.json" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert record["simulator_budget_stop_reached"] is True
assert int(record["simulator_step_calls"]) == 9_505_312
assert int(record["effective_vectorized_budget_stop"]) == 9_505_312
print("verified_training_budget", record["simulator_step_calls"])
PY
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing output collision: $OUT_DIR" >&2
  exit 2
fi

python -u -m sim.control.hourly_dqn.evaluate_hourly_dqn \
  --run-dir "$RUN_DIR" \
  --model best \
  --seeds "${TEST_SEEDS[@]}" \
  --out-dir "$OUT_DIR"
