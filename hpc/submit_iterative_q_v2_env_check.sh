#!/usr/bin/env bash
#SBATCH --job-name=iterq_v2_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o logs/iterative_q_v2_check-%j.out
#SBATCH -e logs/iterative_q_v2_check-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
DATA_ROOT="${DATA_ROOT:-output/iterative_q_validation_search/uniform_margin40_p1_p4}"

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
    raise RuntimeError("Iterative Q v2 training requires CUDA.")
if not pulp.PULP_CBC_CMD(msg=False).available():
    raise RuntimeError("Exact terminal-cleanup evaluation requires CBC.")
PY

python -m py_compile \
  scripts/train_iterative_action_q.py \
  experiments/evaluate_iterative_action_q.py

python - <<'PY'
import torch

from scripts.train_iterative_action_q import (
    selective_previous_policy_anchor_loss,
)

expected = torch.tensor(
    [
        [[0.0, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [100.0, 100.0]],
    ],
    requires_grad=True,
)
loss, metrics = selective_previous_policy_anchor_loss(
    expected=expected,
    actions=torch.tensor([[0, 1], [0, 1]]),
    targets=torch.tensor([[0.0, 0.3], [0.0, 0.5]]),
    valid=torch.ones((2, 2), dtype=torch.bool),
    anchor_actions=torch.tensor([0, 0]),
    release_margin=0.4,
    temperature=0.5,
)
loss.backward()
assert metrics["protected_roots"] == 1
assert metrics["released_roots"] == 1
assert expected.grad[0].abs().sum() > 0
assert expected.grad[1].abs().sum() == 0
print("selective_anchor_smoke", metrics)

linear_expected = torch.zeros((6, 2, 2), requires_grad=True)
linear_improvements = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
linear_loss, linear_metrics = selective_previous_policy_anchor_loss(
    expected=linear_expected,
    actions=torch.tensor([[0, 1]]).expand(6, -1),
    targets=torch.stack(
        (torch.zeros_like(linear_improvements), linear_improvements), dim=1
    ),
    valid=torch.ones((6, 2), dtype=torch.bool),
    anchor_actions=torch.zeros(6, dtype=torch.int64),
    release_margin=0.5,
    temperature=0.5,
    weighting="linear",
)
linear_loss.backward()
assert linear_metrics["protected_roots"] == 5
assert linear_metrics["released_roots"] == 1
assert abs(linear_metrics["effective_weight"] - 3.0) < 1e-6
assert linear_expected.grad[-1].abs().sum() == 0
print("linear_anchor_smoke", linear_metrics)

plateau_expected = torch.zeros((6, 2, 2), requires_grad=True)
plateau_loss, plateau_metrics = selective_previous_policy_anchor_loss(
    expected=plateau_expected,
    actions=torch.tensor([[0, 1]]).expand(6, -1),
    targets=torch.stack(
        (torch.zeros_like(linear_improvements), linear_improvements), dim=1
    ),
    valid=torch.ones((6, 2), dtype=torch.bool),
    anchor_actions=torch.zeros(6, dtype=torch.int64),
    release_margin=0.5,
    temperature=0.5,
    weighting="plateau_linear",
    plateau_margin=0.2,
)
plateau_loss.backward()
assert plateau_metrics["protected_roots"] == 5
assert plateau_metrics["released_roots"] == 1
assert abs(plateau_metrics["effective_weight"] - 4.0) < 1e-6
assert plateau_expected.grad[-1].abs().sum() == 0
print("plateau_anchor_smoke", plateau_metrics)
PY

python - "$DATA_ROOT/g3/train_merged.npz" <<'PY'
import sys

import numpy as np

path = sys.argv[1]
with np.load(path, allow_pickle=False) as payload:
    keys = np.stack(
        (payload["scenario_seed"], payload["root_time_h"]), axis=1
    )
    protected = 0
    released = 0
    for key in np.unique(keys, axis=0):
        indices = np.flatnonzero(np.all(keys == key, axis=1))
        actions = payload["actions"][indices, 0]
        targets = payload["return_to_go"][indices, 0]
        anchors = np.asarray(payload["anchor_action"][indices]).reshape(-1)
        assert len(np.unique(anchors)) == 1
        matches = actions == anchors[0]
        assert matches.any()
        assert np.allclose(targets[matches], 0.0, atol=2e-5)
        if targets.max() > 0.40:
            released += 1
        else:
            protected += 1
    print("g3_protected_roots", protected)
    print("g3_released_roots", released)
    if protected == 0 or released == 0:
        raise RuntimeError("Selective anchor requires both protected and released roots.")
PY
