#!/usr/bin/env bash
#SBATCH --job-name=e1_ppo_test_smoke
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -o logs/e1_ppo_test_smoke-%j.out
#SBATCH -e logs/e1_ppo_test_smoke-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_e1_20260728}"
SOURCE_ROOT="${SOURCE_ROOT:-experiments_results/E1/matched_learning_algorithms_20260728}"
RESULT_ROOT="${RESULT_ROOT:-experiments_results/E1/learning_algorithms_formal_9000031_9000060_20260729}"
SMOKE_SEED=8100001
SMOKE_ROOT="$RESULT_ROOT/smoke_validation_seed_${SMOKE_SEED}_job_${SLURM_JOB_ID}"

cd "$PROJECT_DIR"
mkdir -p logs "$SMOKE_ROOT"
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

for algorithm in centralized_maskable_ppo event_residual_ppo; do
  source_dir="$SOURCE_ROOT/$algorithm/model_seed_0"
  run_dir="$SMOKE_ROOT/$algorithm"
  mkdir -p "$run_dir"
  cp "$source_dir/config.json" "$run_dir/config.json"

  if [[ "$algorithm" == "centralized_maskable_ppo" ]]; then
    cp \
      "$source_dir/ppo_high_level_best_validation.zip" \
      "$run_dir/ppo_high_level_best_validation.zip"
    python -u -m sim.control.event_based.rl.evaluate_high_level_ppo \
      --run-dir "$run_dir" \
      --seeds "$SMOKE_SEED" \
      --model best
    result_json="$(find "$run_dir/evaluation" -maxdepth 1 -name '*.json' -print -quit)"
  else
    cp \
      "$source_dir/event_residual_e1_best_validation.zip" \
      "$run_dir/event_residual_e1_best_validation.zip"
    python -u -m sim.control.event_based.residual_rl_v4.evaluate_ppo \
      --run-dir "$run_dir" \
      --seeds "$SMOKE_SEED" \
      --model best \
      --hard-scenario-probability 0
    result_json="$(find "$run_dir/evaluation" -name 'results.json' -print -quit)"
  fi

  python - "$result_json" "$SMOKE_SEED" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_seed = int(sys.argv[2])
rows = payload["per_seed"]
if [int(row["seed"]) for row in rows] != [expected_seed]:
    raise SystemExit("Unexpected smoke seed.")
row = rows[0]
expected_total = (
    float(row["episode_total_cost_eur"])
    + float(row["terminal_cleanup_operating_cost_eur"])
)
if not math.isclose(
    float(row["total_cost_eur"]),
    expected_total,
    rel_tol=0.0,
    abs_tol=1e-6,
):
    raise SystemExit("Smoke total-cost cleanup identity failed.")
print(
    f"SMOKE_OK seed={expected_seed} "
    f"total_cost_eur={float(row['total_cost_eur']):.6f}"
)
PY
done

printf 'SMOKE_COMPLETE result_root=%s\n' "$SMOKE_ROOT"
