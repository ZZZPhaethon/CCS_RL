#!/usr/bin/env bash
#SBATCH --job-name=iq1shot_hour_env
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -o logs/iterative_q_one_shot_hour_env-%j.out
#SBATCH -e logs/iterative_q_one_shot_hour_env-%j.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${TRAIN_DATA_PATH:?TRAIN_DATA_PATH must be set}"
: "${VALIDATION_DATA_PATH:?VALIDATION_DATA_PATH must be set}"
: "${SOURCE_BUDGET:?SOURCE_BUDGET must be set}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "job_id=$SLURM_JOB_ID"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
which python
python --version
nvidia-smi

python -m py_compile \
  scripts/train_iterative_action_q.py \
  experiments/evaluate_iterative_action_q.py
bash -n \
  hpc/submit_e2_one_shot_train.sh \
  hpc/submit_locked_iterative_q_eval.sh \
  hpc/launch_iterative_q_one_shot_hour_removed.sh

test -s "$TRAIN_DATA_PATH"
test -s "$VALIDATION_DATA_PATH"
test -s "$SOURCE_BUDGET"

python - "$TRAIN_DATA_PATH" "$VALIDATION_DATA_PATH" "$SOURCE_BUDGET" <<'PY'
import json
import sys

import torch

from scripts.train_iterative_action_q import (
    _load_collection,
    dataset_normalization,
    exclude_state_features,
)
from sim.control.iterative_action_q import IterativeFutureActionQuantileQ

train_data, validation_data, budget_path = sys.argv[1:]
budget = json.load(open(budget_path, encoding="utf-8"))
actual_calls = int(budget["train_simulator_step_calls"])
target_calls = int(budget["target_simulator_step_calls"])
relative_error_pct = 100.0 * (actual_calls - target_calls) / target_calls
if abs(relative_error_pct) > 0.5:
    raise RuntimeError(
        f"one-shot budget mismatch {relative_error_pct:.6f}% exceeds 0.5%"
    )

rows = _load_collection([train_data, validation_data])
exclude_state_features(rows, ["hour_of_week"])
data, metadata = rows[0]
if len(metadata["source_state_feature_names"]) != 94:
    raise RuntimeError("expected 94 source state features")
if data["states"].shape[-1] != 93:
    raise RuntimeError(
        f"expected projected state width 93, got {data['states'].shape[-1]}"
    )
if metadata["excluded_state_feature_names"] != ["hour_of_week"]:
    raise RuntimeError("hour_of_week exclusion metadata mismatch")

normalization = dataset_normalization(
    [rows[0]],
    "shared_future_summary",
)
model = IterativeFutureActionQuantileQ(
    metadata["state_feature_names"],
    metadata["future_feature_names"],
    metadata["joint_actions"],
    state_mean=normalization["state_mean"],
    state_std=normalization["state_std"],
    future_mean=normalization["future_mean"],
    future_std=normalization["future_std"],
    return_scale=normalization["return_scale"],
).cuda()
states = torch.as_tensor(data["states"][:2], dtype=torch.float32).cuda()
futures = torch.as_tensor(
    data["future_summaries"][:2],
    dtype=torch.float32,
).cuda()
with torch.no_grad():
    q_values = model(states, futures)
if not torch.isfinite(q_values).all():
    raise RuntimeError("one-shot hour-removed smoke forward is non-finite")

print("train_simulator_step_calls", actual_calls)
print("target_simulator_step_calls", target_calls)
print("relative_error_pct", relative_error_pct)
print("source_state_width", len(metadata["source_state_feature_names"]))
print("projected_state_width", data["states"].shape[-1])
print("q_shape", tuple(q_values.shape))
PY

echo "one_shot_hour_removed_env_check=ok"
echo "finished_at=$(date --iso-8601=seconds)"
