#!/usr/bin/env bash
#SBATCH --job-name=rolling_milp_val
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-39%4
#SBATCH -o logs/rolling_milp_val-%A_%a.out
#SBATCH -e logs/rolling_milp_val-%A_%a.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mas-ccus}"

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_milp_validation}"
CPLEX_BIN="${CPLEX_BIN:-$(command -v cplex || true)}"
RUN_LABEL="${RUN_LABEL:-hpc_cplex_validation_v1}"
if [[ -z "$CPLEX_BIN" || ! -x "$CPLEX_BIN" ]]; then
  echo "CPLEX_BIN must point to an executable Linux CPLEX binary." >&2
  exit 2
fi
if [[ ! "$RUN_LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_LABEL must contain only letters, digits, dots, underscores, or hyphens." >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 39 )); then
  echo "SLURM_ARRAY_TASK_ID must be between 0 and 39." >&2
  exit 2
fi
if (( ${SLURM_CPUS_PER_TASK:-0} < 4 )); then
  echo "Each array task requires at least four allocated CPUs." >&2
  exit 2
fi

# Adjacent task IDs form a paired 30 s/300 s comparison for one seed.
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
LIMIT_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
SEED=$((8100001 + SEED_INDEX))
TIME_LIMITS=(30 300)
TIME_LIMIT="${TIME_LIMITS[$LIMIT_INDEX]}"

cd "$PROJECT_DIR"
mkdir -p logs
export PATH="$(dirname "$CPLEX_BIN"):$PATH"
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

OUT_DIR="$PROJECT_DIR/experiments_results/E1/rolling_milp_budget_validation_${RUN_LABEL}/${TIME_LIMIT}s/seed_${SEED}"
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing result directory: $OUT_DIR" >&2
  exit 3
fi

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "job_id=$SLURM_JOB_ID"
echo "array_job_id=$SLURM_ARRAY_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "seed=$SEED"
echo "time_limit_seconds=$TIME_LIMIT"
echo "solver_threads=4"
echo "cplex=$CPLEX_BIN"
echo "out_dir=$OUT_DIR"

python -u experiments/smoke_test_paper_controllers.py \
  --out-dir "$OUT_DIR" \
  --controllers rolling_milp \
  --seed "$SEED" \
  --online-episode-hours 720 \
  --forecast-context-hours 168 \
  --rolling-replan-hours 24 \
  --rolling-planning-horizon-hours 168 \
  --rolling-time-limit-seconds "$TIME_LIMIT" \
  --solver-threads 4 \
  --purpose rolling_milp_hpc_time_budget_validation_not_formal_results

python - "$OUT_DIR/smoke_summary.json" "$SEED" "$TIME_LIMIT" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_seed = int(sys.argv[2])
expected_limit = float(sys.argv[3])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
row = summary["rows"][0]
replans = summary["diagnostics"]["rolling_milp_replans"]
assert row["seed"] == expected_seed, row
assert row["run_status"] == "completed", row
assert row["solver_failure_count"] == 0, row
assert row["fallback_used"] is False, row
assert row["solver_replan_count"] == 30, row
assert row["solver_threads"] == 4, row
assert row["solver_time_limit_seconds_per_replan"] == expected_limit, row
assert len(replans) == 30, len(replans)
assert all(item["solver_is_valid"] for item in replans), replans
assert all(item["execution_replay_is_valid"] for item in replans), replans
print("rolling_milp_validation_task=passed")
print("summary_path", summary_path)
PY

echo "finished_at=$(date --iso-8601=seconds)"
