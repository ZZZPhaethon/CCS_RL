#!/usr/bin/env bash
#SBATCH --job-name=iq_state_env
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -o logs/iterative_q_state_retrain_env-%j.out
#SBATCH -e logs/iterative_q_state_retrain_env-%j.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
DATA_ROOT="${DATA_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"

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
python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("state-retrain ablation requires CUDA")
PY

python -m py_compile \
  scripts/train_iterative_action_q.py \
  experiments/evaluate_iterative_action_q.py

for stage in g0 g1 g2 g3; do
  test -s "$DATA_ROOT/$stage/train_merged.npz"
  test -s "$DATA_ROOT/$stage/validation_merged.npz"
done

python - "$DATA_ROOT" <<'PY'
import sys

import torch

from scripts.train_iterative_action_q import (
    _load_collection,
    dataset_normalization,
    exclude_state_features,
)
from sim.control.iterative_action_q import IterativeFutureActionQuantileQ

data_root = sys.argv[1]
for excluded, expected_width in (
    (["hour_of_week"], 93),
    (["hour_of_week", "in_transit_fill"], 92),
    (["hour_of_week", "episode_progress"], 92),
    (["hour_of_week", "in_transit_fill", "episode_progress"], 91),
):
    rows = _load_collection(
        [
            f"{data_root}/g0/train_merged.npz",
            f"{data_root}/g0/validation_merged.npz",
        ]
    )
    exclude_state_features(rows, excluded)
    data, metadata = rows[0]
    if data["states"].shape[-1] != expected_width:
        raise RuntimeError(
            f"expected projected state width {expected_width}, "
            f"got {data['states'].shape[-1]}"
        )
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
        q = model(states, futures)
    if not torch.isfinite(q).all():
        raise RuntimeError("state-retrain smoke forward produced non-finite Q values")
    print(
        "smoke",
        ",".join(excluded),
        "state_width",
        expected_width,
        "q_shape",
        tuple(q.shape),
    )
PY

echo "state_retrain_env_check=ok"
echo "finished_at=$(date --iso-8601=seconds)"
