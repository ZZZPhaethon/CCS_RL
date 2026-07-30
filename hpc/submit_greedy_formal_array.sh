#!/usr/bin/env bash
#SBATCH --job-name=greedy_formal
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-29%30
#SBATCH -o logs/greedy_formal-%A_%a.out
#SBATCH -e logs/greedy_formal-%A_%a.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mas-ccus}"

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_milp_validation_20260728}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/experiments_results/E1/formal_greedy_seeds_9000031-9000060_run01}"
PURPOSE="formal_test_E1_greedy_9000031_9000060"

if (( ${SLURM_ARRAY_TASK_ID:--1} < 0 || ${SLURM_ARRAY_TASK_ID:--1} > 29 )); then
  echo "SLURM_ARRAY_TASK_ID must be between 0 and 29." >&2
  exit 2
fi

SEED=$((9000031 + SLURM_ARRAY_TASK_ID))
OUT_DIR="$RESULT_ROOT/seed_$SEED"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite formal result directory: $OUT_DIR" >&2
  exit 3
fi

python -u experiments/smoke_test_paper_controllers.py \
  --out-dir "$OUT_DIR" \
  --controllers greedy \
  --seed "$SEED" \
  --online-episode-hours 720 \
  --forecast-context-hours 168 \
  --purpose "$PURPOSE"

python - "$OUT_DIR" "$SEED" "$PURPOSE" <<'PY'
import json
import os
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
expected_seed = int(sys.argv[2])
expected_purpose = sys.argv[3]
summary = json.loads((out_dir / "smoke_summary.json").read_text(encoding="utf-8"))
actions = json.loads((out_dir / "executed_actions.json").read_text(encoding="utf-8"))
row = summary["rows"][0]
config = summary["configuration"]
trace = actions["actions_by_controller"]["greedy"]

assert summary["protocol"] == "unified_window_v1"
assert summary["purpose"] == expected_purpose
assert row["controller"] == "greedy"
assert row["seed"] == expected_seed
assert row["evaluation_role"] == "online_controller"
assert row["online_comparable"] is True
assert row["run_status"] == "completed"
assert row["terminal_cleanup_included"] is True
assert config["online_episode_hours"] == 720
assert config["forecast_context_hours"] == 168
assert actions["seed"] == expected_seed
assert len(trace) == row["executed_action_count"] == 720

metadata = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
    "array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
    "host": os.uname().nodename,
    "seed": expected_seed,
    "purpose": expected_purpose,
    "run_status": row["run_status"],
}
(out_dir / "formal_task_metadata.json").write_text(
    json.dumps(metadata, indent=2) + "\n",
    encoding="utf-8",
)
print("formal_output_check=passed")
PY
