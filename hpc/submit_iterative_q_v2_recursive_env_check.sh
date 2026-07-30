#!/usr/bin/env bash
#SBATCH --job-name=iterq_v2_rec_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o logs/iterative_q_v2_recursive_check-%j.out
#SBATCH -e logs/iterative_q_v2_recursive_check-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${SOURCE_DATA_ROOT:?SOURCE_DATA_ROOT must be set}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

which python
python --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'PY'
import pulp
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("cbc_available", pulp.PULP_CBC_CMD(msg=False).available())
if not torch.cuda.is_available():
    raise RuntimeError("Recursive Iterative Q v2 training requires CUDA.")
if not pulp.PULP_CBC_CMD(msg=False).available():
    raise RuntimeError("Exact terminal-cleanup evaluation requires CBC.")
PY

python -m py_compile \
  experiments/prepare_iterative_q_p0_anchor_data.py \
  scripts/train_iterative_action_q.py \
  experiments/evaluate_iterative_action_q.py

smoke_path="${SLURM_TMPDIR:-/tmp}/iterative_q_p0_anchor_${SLURM_JOB_ID}.npz"
python experiments/prepare_iterative_q_p0_anchor_data.py \
  --source "$SOURCE_DATA_ROOT/g0/validation_merged.npz" \
  --output "$smoke_path"
python - "$smoke_path" <<'PY'
import json
import sys

import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as payload:
    metadata = json.loads(str(payload["metadata_json"]))
    keys = np.stack(
        (payload["scenario_seed"], payload["root_time_h"]), axis=1
    )
    roots = len(np.unique(keys, axis=0))
    follow = int(metadata["follow_action_index"])
    follow_rows = int((payload["actions"][:, 0] == follow).sum())
    assert metadata["anchors_in_data"] is True
    assert metadata["anchor_policy"] == "greedy_follow_p0"
    assert follow_rows == roots
    assert np.array_equal(np.unique(payload["anchor_action"]), [follow])
    print("p0_anchor_roots", roots)
    print("p0_follow_rows", follow_rows)
PY

python - <<'PY'
from scripts.train_iterative_action_q import parse_args

args = parse_args(
    [
        "--train-data",
        "train.npz",
        "--validation-data",
        "validation.npz",
        "--out-dir",
        "out",
        "--previous-policy-anchor-coefficient",
        "1.0",
        "--allow-anchor-without-initial-checkpoint",
    ]
)
assert args.initial_checkpoint is None
assert args.allow_anchor_without_initial_checkpoint is True
print("p0_non_neural_teacher_cli", True)
PY
