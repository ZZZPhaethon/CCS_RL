#!/usr/bin/env bash
#SBATCH --job-name=rolling_h72_t600
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:30:00
#SBATCH --array=0-29%1
#SBATCH -o logs/rolling_h72_t600-%A_%a.out
#SBATCH -e logs/rolling_h72_t600-%A_%a.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mas-ccus}"

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_milp_validation_20260728}"
CPLEX_BIN="${CPLEX_BIN:-/scratch_root/hx721/software/CPLEX_Studio2220/cplex/bin/x86-64_linux/cplex}"
RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT must be set}"
PURPOSE="${PURPOSE:?PURPOSE must be set}"

if [[ ! -x "$CPLEX_BIN" ]]; then
  echo "CPLEX_BIN is not executable: $CPLEX_BIN" >&2
  exit 2
fi
if (( ${SLURM_ARRAY_TASK_ID:--1} < 0 || ${SLURM_ARRAY_TASK_ID:--1} > 29 )); then
  echo "SLURM_ARRAY_TASK_ID must be between 0 and 29." >&2
  exit 2
fi
if (( ${SLURM_CPUS_PER_TASK:-0} != 4 )); then
  echo "H72 Rolling MILP requires exactly four allocated CPUs." >&2
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
  echo "Refusing to overwrite H72 result directory: $OUT_DIR" >&2
  exit 3
fi

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "job_id=$SLURM_JOB_ID"
echo "array_job_id=$SLURM_ARRAY_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "seed=$SEED"
echo "solver_threads=4"
echo "planning_horizon_hours=72"
echo "replan_interval_hours=24"
echo "time_limit_seconds_per_replan=600"
echo "cplex=$CPLEX_BIN"
echo "out_dir=$OUT_DIR"

python -u experiments/smoke_test_paper_controllers.py \
  --out-dir "$OUT_DIR" \
  --controllers rolling_milp \
  --seed "$SEED" \
  --online-episode-hours 720 \
  --forecast-context-hours 168 \
  --rolling-replan-hours 24 \
  --rolling-planning-horizon-hours 72 \
  --rolling-time-limit-seconds 600 \
  --solver-threads 4 \
  --purpose "$PURPOSE"

python - "$OUT_DIR" "$SEED" "$PURPOSE" "$CPLEX_BIN" <<'PY'
import collections
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
replans = summary["diagnostics"]["rolling_milp_replans"]
trace = actions["actions_by_controller"]["rolling_milp"]

assert summary["protocol"] == "unified_window_v1"
assert summary["purpose"] == expected_purpose
assert row["controller"] == "rolling_milp"
assert row["seed"] == expected_seed
assert row["evaluation_role"] == "online_controller"
assert row["solver_threads"] == 4
assert row["solver_time_limit_seconds_per_replan"] == 600.0
assert row["warm_start_mode"] == "greedy"
assert row["shifted_warm_start"] is False
assert row["fallback_used"] is False
assert config["online_episode_hours"] == 720
assert config["forecast_context_hours"] == 168
assert config["rolling_replan_hours"] == 24
assert config["rolling_planning_horizon_hours"] == 72
assert config["mip_gap_relative"] is None
assert actions["seed"] == expected_seed
assert len(trace) == row["executed_action_count"]

if row["run_status"] == "completed":
    assert len(trace) == 720
    assert row["solver_replan_count"] == 30
    assert row["solver_failure_count"] == 0
    assert len(replans) == 30
    assert all(item["solver_is_valid"] for item in replans)
    assert all(item["execution_replay_is_valid"] for item in replans)

status_counts = collections.Counter(str(item["status"]) for item in replans)
optimal_replans = status_counts.get("Optimal", 0)
metadata = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
    "array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
    "host": os.uname().nodename,
    "seed": expected_seed,
    "purpose": expected_purpose,
    "cplex_bin": cplex_bin,
    "planning_horizon_hours": 72,
    "replan_interval_hours": 24,
    "solver_threads": 4,
    "solver_time_limit_seconds_per_replan": 600,
    "run_status": row["run_status"],
    "solver_status_counts": dict(status_counts),
    "optimal_replan_count": optimal_replans,
    "all_replans_optimal": len(replans) == 30 and optimal_replans == 30,
}
(out_dir / "h72_r24_t600s_task_metadata.json").write_text(
    json.dumps(metadata, indent=2) + "\n",
    encoding="utf-8",
)
print("h72_r24_t600s_output_check=passed")
print("scientific_run_status", row["run_status"])
print("solver_status_counts", dict(status_counts))
print("all_replans_optimal", metadata["all_replans_optimal"])
PY

echo "finished_at=$(date --iso-8601=seconds)"
