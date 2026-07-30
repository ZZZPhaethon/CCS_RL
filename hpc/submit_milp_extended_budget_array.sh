#!/usr/bin/env bash
#SBATCH --job-name=milp_extended
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:30:00
#SBATCH --array=0-29%1
#SBATCH -o logs/milp_extended-%A_%a.out
#SBATCH -e logs/milp_extended-%A_%a.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mas-ccus}"

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_milp_validation_20260728}"
CPLEX_BIN="${CPLEX_BIN:-/scratch_root/hx721/software/CPLEX_Studio2220/cplex/bin/x86-64_linux/cplex}"
MODE="${MODE:?MODE must be rolling or full}"
RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT must be set}"
PURPOSE="${PURPOSE:?PURPOSE must be set}"

if [[ "$MODE" != "rolling" && "$MODE" != "full" ]]; then
  echo "MODE must be rolling or full, got: $MODE" >&2
  exit 2
fi
if [[ ! -x "$CPLEX_BIN" ]]; then
  echo "CPLEX_BIN is not executable: $CPLEX_BIN" >&2
  exit 2
fi
if (( ${SLURM_ARRAY_TASK_ID:--1} < 0 || ${SLURM_ARRAY_TASK_ID:--1} > 29 )); then
  echo "SLURM_ARRAY_TASK_ID must be between 0 and 29." >&2
  exit 2
fi
if (( ${SLURM_CPUS_PER_TASK:-0} != 4 )); then
  echo "Extended-budget MILP requires exactly four allocated CPUs." >&2
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
  echo "Refusing to overwrite extended-budget result directory: $OUT_DIR" >&2
  exit 3
fi

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "job_id=$SLURM_JOB_ID"
echo "array_job_id=$SLURM_ARRAY_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
echo "mode=$MODE"
echo "seed=$SEED"
echo "solver_threads=4"
echo "cplex=$CPLEX_BIN"
echo "out_dir=$OUT_DIR"

if [[ "$MODE" == "rolling" ]]; then
  echo "time_limit_seconds_per_replan=600"
  python -u experiments/smoke_test_paper_controllers.py \
    --out-dir "$OUT_DIR" \
    --controllers rolling_milp \
    --seed "$SEED" \
    --online-episode-hours 720 \
    --forecast-context-hours 168 \
    --rolling-replan-hours 24 \
    --rolling-planning-horizon-hours 168 \
    --rolling-time-limit-seconds 600 \
    --solver-threads 4 \
    --purpose "$PURPOSE"
else
  echo "time_limit_seconds_per_seed=18000"
  python -u experiments/smoke_test_paper_controllers.py \
    --out-dir "$OUT_DIR" \
    --controllers full_milp \
    --seed "$SEED" \
    --online-episode-hours 720 \
    --forecast-context-hours 168 \
    --full-milp-horizon-hours 720 \
    --full-milp-time-limit-seconds 18000 \
    --solver-threads 4 \
    --purpose "$PURPOSE"
fi

python - "$OUT_DIR" "$SEED" "$MODE" "$PURPOSE" "$CPLEX_BIN" <<'PY'
import json
import os
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
expected_seed = int(sys.argv[2])
mode = sys.argv[3]
expected_purpose = sys.argv[4]
cplex_bin = sys.argv[5]

summary = json.loads((out_dir / "smoke_summary.json").read_text(encoding="utf-8"))
actions = json.loads((out_dir / "executed_actions.json").read_text(encoding="utf-8"))
row = summary["rows"][0]
config = summary["configuration"]
controller = "rolling_milp" if mode == "rolling" else "full_milp"
trace = actions["actions_by_controller"][controller]

assert summary["protocol"] == "unified_window_v1"
assert summary["purpose"] == expected_purpose
assert row["controller"] == controller
assert row["seed"] == expected_seed
assert row["solver_threads"] == 4
assert row["warm_start_mode"] == "greedy"
assert row["fallback_used"] is False
assert config["forecast_context_hours"] == 168
assert config["mip_gap_relative"] is None
assert actions["seed"] == expected_seed
assert len(trace) == row["executed_action_count"]

if mode == "rolling":
    replans = summary["diagnostics"]["rolling_milp_replans"]
    assert row["evaluation_role"] == "online_controller"
    assert row["solver_time_limit_seconds_per_replan"] == 600.0
    assert row["shifted_warm_start"] is False
    assert config["online_episode_hours"] == 720
    assert config["rolling_replan_hours"] == 24
    assert config["rolling_planning_horizon_hours"] == 168
    if row["run_status"] == "completed":
        assert len(trace) == 720
        assert row["solver_replan_count"] == 30
        assert row["solver_failure_count"] == 0
        assert len(replans) == 30
        assert all(item["solver_is_valid"] for item in replans)
        assert all(item["execution_replay_is_valid"] for item in replans)
else:
    assert row["evaluation_role"] == "offline_reference"
    assert row["online_comparable"] is False
    assert row["solver_time_limit_seconds"] == 18000.0
    assert row["planning_horizon_hours"] == 720
    assert row["evaluation_horizon_hours"] == 720
    assert config["full_milp_horizon_hours"] == 720
    if row["run_status"] in {"completed", "replay_failed"}:
        assert len(trace) == 720

metadata = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
    "array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
    "host": os.uname().nodename,
    "mode": mode,
    "seed": expected_seed,
    "purpose": expected_purpose,
    "cplex_bin": cplex_bin,
    "solver_threads": 4,
    "solver_time_limit_seconds": 600 if mode == "rolling" else 18000,
    "run_status": row["run_status"],
}
(out_dir / "extended_budget_task_metadata.json").write_text(
    json.dumps(metadata, indent=2) + "\n",
    encoding="utf-8",
)
print("extended_budget_output_check=passed")
print("scientific_run_status", row["run_status"])
PY

echo "finished_at=$(date --iso-8601=seconds)"
