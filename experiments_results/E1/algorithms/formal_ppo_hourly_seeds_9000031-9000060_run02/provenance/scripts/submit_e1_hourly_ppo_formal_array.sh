#!/usr/bin/env bash
#SBATCH --job-name=e1_hourly_formal
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH -o logs/e1_hourly_formal-%A_%a.out
#SBATCH -e logs/e1_hourly_formal-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_e1_20260728}"
SOURCE_ROOT="${SOURCE_ROOT:-experiments_results/E1/hourly_ppo_gpu_20260728/centralized_maskable_ppo}"
RESULT_ROOT="${RESULT_ROOT:-experiments_results/E1/formal_hourly_centralized_maskable_ppo_seeds_9000031-9000060_run02}"
FORMAL_SEEDS=(
  9000031 9000032 9000033 9000034 9000035
  9000036 9000037 9000038 9000039 9000040
  9000041 9000042 9000043 9000044 9000045
  9000046 9000047 9000048 9000049 9000050
  9000051 9000052 9000053 9000054 9000055
  9000056 9000057 9000058 9000059 9000060
)

MODEL_SEED="${SLURM_ARRAY_TASK_ID}"
ALGORITHM="hourly_centralized_maskable_ppo"

cd "$PROJECT_DIR"
mkdir -p logs "$RESULT_ROOT/$ALGORITHM"
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

SOURCE_DIR="$SOURCE_ROOT/model_seed_$MODEL_SEED"
FINAL_DIR="$RESULT_ROOT/$ALGORITHM/model_seed_$MODEL_SEED"
STAGE_DIR="${FINAL_DIR}.inprogress.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

if [[ -e "$FINAL_DIR" || -e "$STAGE_DIR" ]]; then
  printf 'Refusing output collision: %s or %s\n' \
    "$FINAL_DIR" "$STAGE_DIR" >&2
  exit 2
fi

mkdir -p "$STAGE_DIR"
cp "$SOURCE_DIR/config.json" "$STAGE_DIR/config.json"
cp "$SOURCE_DIR/training_complete.json" "$STAGE_DIR/training_complete.json"
cp "$SOURCE_DIR/validation/best.json" "$STAGE_DIR/best_validation.json"
cp \
  "$SOURCE_DIR/ppo_hourly_best_validation.zip" \
  "$STAGE_DIR/checkpoint_best_validation.zip"

python -u -m sim.control.hourly_ppo.evaluate_hourly_ppo \
  --run-dir "$STAGE_DIR" \
  --model checkpoint_best_validation.zip \
  --seeds "${FORMAL_SEEDS[@]}" \
  --out-dir "$STAGE_DIR/evaluation"

mv "$STAGE_DIR/evaluation/evaluation.csv" "$STAGE_DIR/results.csv"
mv "$STAGE_DIR/evaluation/summary.json" "$STAGE_DIR/results.json"
rmdir "$STAGE_DIR/evaluation"
(cd "$STAGE_DIR" && \
  sha256sum checkpoint_best_validation.zip > checkpoint.sha256)

python - \
  "$STAGE_DIR/config.json" \
  "$STAGE_DIR/results.json" \
  "$STAGE_DIR/audit.json" \
  "$MODEL_SEED" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
results_path = Path(sys.argv[2])
audit_path = Path(sys.argv[3])
model_seed = int(sys.argv[4])
expected_seeds = list(range(9_000_031, 9_000_061))

config = json.loads(config_path.read_text(encoding="utf-8"))
expected_config = {
    "paper_name": "Hourly Centralized Maskable PPO",
    "episode_hours": 720,
    "forecast_context_hours": 168,
    "future_summary_windows_h": [168],
    "decision_interval_h": 1.0,
    "direct_native_action": True,
    "uses_event_trigger": False,
    "uses_goal_executor": False,
    "uses_greedy_default": False,
    "uses_residual_actions": False,
    "valid_fraction_feature": False,
}
for key, expected in expected_config.items():
    if config.get(key) != expected:
        raise SystemExit(
            f"Unexpected config {key}: "
            f"{config.get(key)!r} != {expected!r}"
        )

payload = json.loads(results_path.read_text(encoding="utf-8"))
if payload["paper_name"] != "Hourly Centralized Maskable PPO":
    raise SystemExit("Unexpected paper_name.")
payload["model_path"] = "checkpoint_best_validation.zip"
results_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
rows = payload["per_seed"]
actual_seeds = [int(row["seed"]) for row in rows]
if actual_seeds != expected_seeds:
    raise SystemExit(f"Unexpected formal seeds: {actual_seeds!r}")
for row in rows:
    if int(row["decisions"]) != 720:
        raise SystemExit(
            f"Seed {row['seed']} did not make 720 decisions."
        )
    if not math.isclose(
        float(row["simulated_hours"]),
        720.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise SystemExit(
            f"Seed {row['seed']} did not simulate 720 hours."
        )
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
        raise SystemExit(
            f"Cleanup identity failed for seed {row['seed']}."
        )
    for key in (
        "total_cost_eur",
        "vented_t",
        "stored_t",
        "captured_t",
    ):
        if not math.isfinite(float(row[key])):
            raise SystemExit(
                f"Non-finite {key} for seed {row['seed']}."
            )

audit = {
    "algorithm": "Hourly Centralized Maskable PPO",
    "model_seed": model_seed,
    "model_choice": "best_validation",
    "formal_seeds": expected_seeds,
    "episodes": len(rows),
    "cleanup_identity_failures": 0,
    "direct_hourly_decision_failures": 0,
    "configuration": expected_config,
    "slurm_array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
    "slurm_array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
}
audit_path.write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(
    "FORMAL_AUDIT_OK "
    f"algorithm=hourly_centralized_maskable_ppo "
    f"model_seed={model_seed} episodes={len(rows)}"
)
PY

mv "$STAGE_DIR" "$FINAL_DIR"
printf 'FORMAL_COMPLETE model_seed=%s result_dir=%s\n' \
  "$MODEL_SEED" "$FINAL_DIR"
