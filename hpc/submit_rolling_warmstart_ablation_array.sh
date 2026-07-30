#!/usr/bin/env bash
#SBATCH --job-name=rolling_ws_ablation
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH -o logs/rolling_ws_ablation-%A_%a.out
#SBATCH -e logs/rolling_ws_ablation-%A_%a.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mas-ccus}"

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_milp_validation_20260728}"
CPLEX_BIN="${CPLEX_BIN:-/scratch_root/hx721/software/CPLEX_Studio2220/cplex/bin/x86-64_linux/cplex}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/experiments_results/E1/ablation_rolling_milp_warmstart_cplex222_t300s_n3_run01}"
PURPOSE="ablation_E1_rolling_milp_greedy_vs_none_3seeds_300s"
SEEDS=(9000042 9000034 9000047)
MODES=(greedy none)

if [[ ! -x "$CPLEX_BIN" ]]; then
  echo "CPLEX_BIN is not executable: $CPLEX_BIN" >&2
  exit 2
fi
if (( ${SLURM_ARRAY_TASK_ID:--1} < 0 || ${SLURM_ARRAY_TASK_ID:--1} > 5 )); then
  echo "SLURM_ARRAY_TASK_ID must be between 0 and 5." >&2
  exit 2
fi
if (( ${SLURM_CPUS_PER_TASK:-0} != 4 )); then
  echo "Rolling MILP ablation requires exactly four allocated CPUs." >&2
  exit 2
fi

SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
MODE_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
SEED="${SEEDS[$SEED_INDEX]}"
MODE="${MODES[$MODE_INDEX]}"
OUT_DIR="$RESULT_ROOT/$MODE/seed_$SEED"

cd "$PROJECT_DIR"
mkdir -p logs
export PATH="$(dirname "$CPLEX_BIN"):$PATH"
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite ablation result directory: $OUT_DIR" >&2
  exit 3
fi

python -u experiments/smoke_test_paper_controllers.py \
  --out-dir "$OUT_DIR" \
  --controllers rolling_milp \
  --seed "$SEED" \
  --online-episode-hours 720 \
  --forecast-context-hours 168 \
  --rolling-replan-hours 24 \
  --rolling-planning-horizon-hours 168 \
  --rolling-time-limit-seconds 300 \
  --rolling-warm-start-mode "$MODE" \
  --solver-threads 4 \
  --purpose "$PURPOSE"

python - "$OUT_DIR" "$SEED" "$MODE" "$PURPOSE" "$CPLEX_BIN" <<'PY'
import json
import os
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
expected_seed = int(sys.argv[2])
expected_mode = sys.argv[3]
expected_purpose = sys.argv[4]
cplex_bin = sys.argv[5]
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
assert row["solver_time_limit_seconds_per_replan"] == 300.0
assert row["warm_start_mode"] == expected_mode
assert row["shifted_warm_start"] is False
assert row["fallback_used"] is False
assert config["online_episode_hours"] == 720
assert config["forecast_context_hours"] == 168
assert config["rolling_replan_hours"] == 24
assert config["rolling_planning_horizon_hours"] == 168
assert config["rolling_warm_start_mode"] == expected_mode
assert config["mip_gap_relative"] is None
assert len(trace) == row["executed_action_count"]
assert all("first_incumbent_time_s" in item for item in replans)
assert all("first_incumbent_objective" in item for item in replans)

if expected_mode == "none":
    assert all(item["warm_start_accepted"] is None for item in replans)
    assert all(item["warm_start_score"] is None for item in replans)
    assert all(item["warm_start_source"] == "none" for item in replans)
else:
    assert all(item["warm_start_source"] == "greedy" for item in replans)

if row["run_status"] == "completed":
    assert len(trace) == 720
    assert row["solver_replan_count"] == 30
    assert row["solver_failure_count"] == 0
    assert len(replans) == 30
    assert all(item["solver_is_valid"] for item in replans)
    assert all(item["execution_replay_is_valid"] for item in replans)
    assert all(item["first_incumbent_time_s"] is not None for item in replans)
    assert all(item["first_incumbent_objective"] is not None for item in replans)

metadata = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
    "array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
    "host": os.uname().nodename,
    "seed": expected_seed,
    "warm_start_mode": expected_mode,
    "purpose": expected_purpose,
    "cplex_bin": cplex_bin,
    "run_status": row["run_status"],
    "replan_count": len(replans),
}
(out_dir / "ablation_task_metadata.json").write_text(
    json.dumps(metadata, indent=2) + "\n",
    encoding="utf-8",
)
print("ablation_output_check=passed")
print("scientific_run_status", row["run_status"])
PY
