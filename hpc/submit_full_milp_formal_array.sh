#!/usr/bin/env bash
#SBATCH --job-name=full_milp_formal
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-29%8
#SBATCH -o logs/full_milp_formal-%A_%a.out
#SBATCH -e logs/full_milp_formal-%A_%a.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mas-ccus}"

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_milp_validation_20260728}"
CPLEX_BIN="${CPLEX_BIN:-/scratch_root/hx721/software/CPLEX_Studio2220/cplex/bin/x86-64_linux/cplex}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/experiments_results/E5/full_milp_formal_9000031_9000060_cplex222_7200s}"
PURPOSE="formal_test_E5_full_milp_cplex222_7200s"

if [[ ! -x "$CPLEX_BIN" ]]; then
  echo "CPLEX_BIN is not executable: $CPLEX_BIN" >&2
  exit 2
fi
if (( ${SLURM_ARRAY_TASK_ID:--1} < 0 || ${SLURM_ARRAY_TASK_ID:--1} > 29 )); then
  echo "SLURM_ARRAY_TASK_ID must be between 0 and 29." >&2
  exit 2
fi
if (( ${SLURM_CPUS_PER_TASK:-0} != 4 )); then
  echo "Formal Full MILP requires exactly four allocated CPUs." >&2
  exit 2
fi

SEED=$((9000031 + SLURM_ARRAY_TASK_ID))
OUT_DIR="$RESULT_ROOT/seed_$SEED"

cd "$PROJECT_DIR"
mkdir -p logs
export PATH="$(dirname "$CPLEX_BIN"):$PATH"
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite formal result directory: $OUT_DIR" >&2
  exit 3
fi

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "job_id=$SLURM_JOB_ID"
echo "array_job_id=$SLURM_ARRAY_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "seed=$SEED"
echo "solver_threads=4"
echo "time_limit_seconds=7200"
echo "cplex=$CPLEX_BIN"
echo "out_dir=$OUT_DIR"

python -u experiments/smoke_test_paper_controllers.py \
  --out-dir "$OUT_DIR" \
  --controllers full_milp \
  --seed "$SEED" \
  --online-episode-hours 720 \
  --forecast-context-hours 168 \
  --full-milp-horizon-hours 720 \
  --full-milp-time-limit-seconds 7200 \
  --solver-threads 4 \
  --purpose "$PURPOSE"

python - "$OUT_DIR" "$SEED" "$PURPOSE" "$CPLEX_BIN" <<'PY'
import json
import os
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
expected_seed = int(sys.argv[2])
expected_purpose = sys.argv[3]
cplex_bin = sys.argv[4]
summary = json.loads((out_dir / "smoke_summary.json").read_text(encoding="utf-8"))
actions = json.loads((out_dir / "executed_actions.json").read_text(encoding="utf-8"))
row = summary["rows"][0]
config = summary["configuration"]
trace = actions["actions_by_controller"]["full_milp"]

assert summary["protocol"] == "unified_window_v1"
assert summary["purpose"] == expected_purpose
assert row["controller"] == "full_milp"
assert row["seed"] == expected_seed
assert row["evaluation_role"] == "offline_reference"
assert row["online_comparable"] is False
assert row["solver_threads"] == 4
assert row["solver_time_limit_seconds"] == 7200.0
assert row["planning_horizon_hours"] == 720
assert row["evaluation_horizon_hours"] == 720
assert row["warm_start_mode"] == "greedy"
assert row["fallback_used"] is False
assert config["forecast_context_hours"] == 168
assert config["full_milp_horizon_hours"] == 720
assert config["mip_gap_relative"] is None
assert actions["seed"] == expected_seed
assert len(trace) == row["executed_action_count"]

if row["run_status"] in {"completed", "replay_failed"}:
    assert len(trace) == 720

metadata = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
    "array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
    "host": os.uname().nodename,
    "seed": expected_seed,
    "purpose": expected_purpose,
    "cplex_bin": cplex_bin,
    "run_status": row["run_status"],
}
(out_dir / "formal_task_metadata.json").write_text(
    json.dumps(metadata, indent=2) + "\n",
    encoding="utf-8",
)
print("formal_output_check=passed")
print("scientific_run_status", row["run_status"])
PY

echo "finished_at=$(date --iso-8601=seconds)"
