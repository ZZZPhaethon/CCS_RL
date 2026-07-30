#!/usr/bin/env bash
#SBATCH --job-name=rolling_milp_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH -o logs/rolling_milp_check-%j.out
#SBATCH -e logs/rolling_milp_check-%j.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-mas-ccus}"

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_milp_validation}"
CPLEX_BIN="${CPLEX_BIN:-$(command -v cplex || true)}"
if [[ -z "$CPLEX_BIN" || ! -x "$CPLEX_BIN" ]]; then
  echo "CPLEX_BIN must point to an executable Linux CPLEX binary." >&2
  exit 2
fi

cd "$PROJECT_DIR"
mkdir -p logs
export PATH="$(dirname "$CPLEX_BIN"):$PATH"
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "job_id=$SLURM_JOB_ID"
echo "project_dir=$PROJECT_DIR"
echo "python=$(command -v python)"
echo "cplex=$CPLEX_BIN"
python --version
"$CPLEX_BIN" -c "quit"

python - <<'PY'
import shutil

import pulp

cplex = shutil.which("cplex")
print("pulp", pulp.__version__)
print("cplex_on_path", cplex)
if not cplex:
    raise RuntimeError("CPLEX is not available on PATH")
if not pulp.CPLEX_CMD(path=cplex, msg=False).available():
    raise RuntimeError("PuLP cannot execute CPLEX")
PY

python -m py_compile \
  experiments/smoke_test_paper_controllers.py \
  src/sim/control/cplex_milp.py \
  src/sim/control/rolling_milp.py

SMOKE_ROOT="$PROJECT_DIR/experiments_results/E0/rolling_milp_hpc_check_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
python -u experiments/smoke_test_paper_controllers.py \
  --out-dir "$SMOKE_ROOT" \
  --controllers rolling_milp \
  --seed 8100001 \
  --online-episode-hours 24 \
  --forecast-context-hours 168 \
  --rolling-replan-hours 24 \
  --rolling-planning-horizon-hours 168 \
  --rolling-time-limit-seconds 5 \
  --solver-threads 4 \
  --purpose rolling_milp_hpc_environment_check_not_formal_results

python - "$SMOKE_ROOT/smoke_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
row = summary["rows"][0]
replans = summary["diagnostics"]["rolling_milp_replans"]
assert row["run_status"] == "completed", row
assert row["solver_failure_count"] == 0, row
assert row["fallback_used"] is False, row
assert len(replans) == 1, len(replans)
assert replans[0]["solver_is_valid"] is True, replans[0]
assert replans[0]["execution_replay_is_valid"] is True, replans[0]
print("rolling_milp_hpc_check=passed")
print("summary_path", summary_path)
PY

echo "finished_at=$(date --iso-8601=seconds)"
