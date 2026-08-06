#!/usr/bin/env bash
#SBATCH --job-name=e1_ppo_formal
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-5%6
#SBATCH -o logs/e1_ppo_formal-%A_%a.out
#SBATCH -e logs/e1_ppo_formal-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_e1_20260728}"
SOURCE_ROOT="${SOURCE_ROOT:-experiments_results/E1/matched_learning_algorithms_20260728}"
HIGH_LEVEL_RESULT_ROOT="${HIGH_LEVEL_RESULT_ROOT:-experiments_results/E1/algorithms/formal_ppo_high_level_seeds_9000031-9000060_run01}"
EVENT_RESIDUAL_RESULT_ROOT="${EVENT_RESIDUAL_RESULT_ROOT:-experiments_results/E1/algorithms/formal_ppo_event_residual_seeds_9000031-9000060_run01}"
FORMAL_SEEDS=(
  9000031 9000032 9000033 9000034 9000035
  9000036 9000037 9000038 9000039 9000040
  9000041 9000042 9000043 9000044 9000045
  9000046 9000047 9000048 9000049 9000050
  9000051 9000052 9000053 9000054 9000055
  9000056 9000057 9000058 9000059 9000060
)

task_id="${SLURM_ARRAY_TASK_ID}"
if (( task_id < 3 )); then
  source_algorithm="centralized_maskable_ppo"
  algorithm="ppo_high_level"
  result_root="$HIGH_LEVEL_RESULT_ROOT"
  model_seed="$task_id"
else
  source_algorithm="event_residual_ppo"
  algorithm="ppo_event_residual"
  result_root="$EVENT_RESIDUAL_RESULT_ROOT"
  model_seed="$((task_id - 3))"
fi

cd "$PROJECT_DIR"
mkdir -p logs "$result_root/$algorithm"
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

source_dir="$SOURCE_ROOT/$source_algorithm/model_seed_$model_seed"
final_dir="$result_root/$algorithm/model_seed_$model_seed"
stage_dir="${final_dir}.inprogress.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

if [[ -e "$final_dir" || -e "$stage_dir" ]]; then
  printf 'Refusing output collision: %s or %s\n' "$final_dir" "$stage_dir" >&2
  exit 2
fi

mkdir -p "$stage_dir"
cp "$source_dir/config.json" "$stage_dir/config.json"
cp "$source_dir/training_complete.json" "$stage_dir/training_complete.json"
cp "$source_dir/validation/best.json" "$stage_dir/best_validation.json"

if [[ "$algorithm" == "ppo_high_level" ]]; then
  checkpoint="$source_dir/ppo_high_level_best_validation.zip"
  cp "$checkpoint" "$stage_dir/ppo_high_level_best_validation.zip"
  python -u -m sim.control.event_based.rl.evaluate_high_level_ppo \
    --run-dir "$stage_dir" \
    --seeds "${FORMAL_SEEDS[@]}" \
    --model best
  result_json="$(find "$stage_dir/evaluation" -maxdepth 1 -name '*.json' -print -quit)"
  result_csv="$(find "$stage_dir/evaluation" -maxdepth 1 -name '*.csv' -print -quit)"
else
  checkpoint="$source_dir/event_residual_e1_best_validation.zip"
  cp "$checkpoint" "$stage_dir/event_residual_e1_best_validation.zip"
  python -u -m sim.control.event_based.residual_rl_v4.evaluate_ppo \
    --run-dir "$stage_dir" \
    --seeds "${FORMAL_SEEDS[@]}" \
    --model best \
    --hard-scenario-probability 0
  result_json="$(find "$stage_dir/evaluation" -name 'results.json' -print -quit)"
  result_csv="$(find "$stage_dir/evaluation" -name 'results.csv' -print -quit)"
fi

cp "$result_json" "$stage_dir/results.json"
cp "$result_csv" "$stage_dir/results.csv"
sha256sum "$checkpoint" > "$stage_dir/source_checkpoint.sha256"

python - "$stage_dir/results.json" "$stage_dir/audit.json" "$algorithm" "$model_seed" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
audit_path = Path(sys.argv[2])
algorithm = sys.argv[3]
model_seed = int(sys.argv[4])
expected_seeds = list(range(9_000_031, 9_000_061))
payload = json.loads(results_path.read_text(encoding="utf-8"))
rows = payload["per_seed"]
actual_seeds = [int(row["seed"]) for row in rows]
if actual_seeds != expected_seeds:
    raise SystemExit(
        f"Unexpected formal seeds: {actual_seeds!r}"
    )
for row in rows:
    expected_total = (
        float(row["episode_total_cost_eur"])
        + float(row["terminal_cleanup_operating_cost_eur"])
    )
    expected_episode_operating = sum(
        float(row[key])
        for key in (
            "episode_vessel_fuel_eur",
            "episode_conditioning_eur",
            "episode_reconditioning_eur",
            "episode_loading_eur",
            "episode_unloading_eur",
        )
    )
    expected_episode_total = (
        expected_episode_operating
        + float(row["episode_vent_penalty_eur"])
        + float(row["episode_storage_shortfall_penalty_eur"])
    )
    if not math.isclose(
        float(row["episode_operating_cost_eur"]),
        expected_episode_operating,
        rel_tol=0.0,
        abs_tol=1e-6,
    ) or not math.isclose(
        float(row["episode_total_cost_eur"]),
        expected_episode_total,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise SystemExit(
            f"Episode cost breakdown failed for seed {row['seed']}."
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
audit = {
    "algorithm": algorithm,
    "model_seed": model_seed,
    "model_choice": "best_validation",
    "formal_seeds": expected_seeds,
    "episodes": len(rows),
    "cleanup_identity_failures": 0,
    "episode_cost_breakdown_identity_failures": 0,
    "detailed_episode_cost_fields": True,
    "terminal_cleanup_component_breakdown": False,
    "slurm_array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
    "slurm_array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
}
audit_path.write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(
    f"FORMAL_AUDIT_OK algorithm={algorithm} "
    f"model_seed={model_seed} episodes={len(rows)}"
)
PY

mv "$stage_dir" "$final_dir"
printf 'FORMAL_COMPLETE algorithm=%s model_seed=%s result_dir=%s\n' \
  "$algorithm" "$model_seed" "$final_dir"
