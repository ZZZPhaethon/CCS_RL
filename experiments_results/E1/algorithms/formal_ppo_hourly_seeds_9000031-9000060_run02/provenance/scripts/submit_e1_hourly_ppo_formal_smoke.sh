#!/usr/bin/env bash
#SBATCH --job-name=e1_hourly_smoke
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o logs/e1_hourly_smoke-%j.out
#SBATCH -e logs/e1_hourly_smoke-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_e1_20260728}"
SOURCE_ROOT="${SOURCE_ROOT:-experiments_results/E1/hourly_ppo_gpu_20260728/centralized_maskable_ppo}"
RESULT_ROOT="${RESULT_ROOT:-experiments_results/E1/formal_hourly_centralized_maskable_ppo_seeds_9000031-9000060_run02}"
SMOKE_SEED=8100001
SMOKE_ROOT="$RESULT_ROOT/smoke_validation_seed_${SMOKE_SEED}_job_${SLURM_JOB_ID}"
RUN_DIR="$SMOKE_ROOT/hourly_centralized_maskable_ppo/model_seed_0"

cd "$PROJECT_DIR"
mkdir -p logs
if [[ -e "$SMOKE_ROOT" ]]; then
  printf 'Refusing output collision: %s\n' "$SMOKE_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

which python
python --version
python - <<'PY'
import stable_baselines3
import sb3_contrib
import torch

print("torch", torch.__version__)
print("stable_baselines3", stable_baselines3.__version__)
print("sb3_contrib", sb3_contrib.__version__)
PY
SOURCE_DIR="$SOURCE_ROOT/model_seed_0"
cp "$SOURCE_DIR/config.json" "$RUN_DIR/config.json"
cp \
  "$SOURCE_DIR/ppo_hourly_best_validation.zip" \
  "$RUN_DIR/checkpoint_best_validation.zip"

python -u -m sim.control.hourly_ppo.evaluate_hourly_ppo \
  --run-dir "$RUN_DIR" \
  --model checkpoint_best_validation.zip \
  --seeds "$SMOKE_SEED" \
  --out-dir "$RUN_DIR/evaluation"

python - "$RUN_DIR/evaluation/summary.json" "$SMOKE_SEED" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_seed = int(sys.argv[2])
rows = payload["per_seed"]
if payload["paper_name"] != "Hourly Centralized Maskable PPO":
    raise SystemExit("Unexpected paper_name.")
if [int(row["seed"]) for row in rows] != [expected_seed]:
    raise SystemExit("Unexpected smoke seed.")
row = rows[0]
if int(row["decisions"]) != 720:
    raise SystemExit("Hourly PPO did not make 720 direct decisions.")
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

printf 'SMOKE_COMPLETE result_root=%s\n' "$SMOKE_ROOT"
