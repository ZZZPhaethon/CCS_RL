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
HIGH_LEVEL_RESULT_ROOT="${HIGH_LEVEL_RESULT_ROOT:-experiments_results/E1/formal_ppo_high_level_seeds_9000031-9000060_run01}"
EVENT_RESIDUAL_RESULT_ROOT="${EVENT_RESIDUAL_RESULT_ROOT:-experiments_results/E1/formal_ppo_event_residual_seeds_9000031-9000060_run01}"
SMOKE_SEED=8100001

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

for source_algorithm in centralized_maskable_ppo event_residual_ppo; do
  if [[ "$source_algorithm" == "centralized_maskable_ppo" ]]; then
    algorithm="ppo_high_level"
    result_root="$HIGH_LEVEL_RESULT_ROOT"
  else
    algorithm="ppo_event_residual"
    result_root="$EVENT_RESIDUAL_RESULT_ROOT"
  fi
  smoke_root="$result_root/smoke_validation_seed_${SMOKE_SEED}_job_${SLURM_JOB_ID}"
  source_dir="$SOURCE_ROOT/$source_algorithm/model_seed_0"
  run_dir="$smoke_root/$algorithm"
  mkdir -p "$run_dir"
  cp "$source_dir/config.json" "$run_dir/config.json"

  if [[ "$algorithm" == "ppo_high_level" ]]; then
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

printf 'SMOKE_COMPLETE high_level_root=%s event_residual_root=%s\n' \
  "$HIGH_LEVEL_RESULT_ROOT" "$EVENT_RESIDUAL_RESULT_ROOT"
