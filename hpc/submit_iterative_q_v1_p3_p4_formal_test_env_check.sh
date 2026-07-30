#!/usr/bin/env bash
#SBATCH --job-name=iterq_v1_test_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o logs/iterative_q_v1_formal_test_check-%j.out
#SBATCH -e logs/iterative_q_v1_formal_test_check-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${P3_CHECKPOINT:?P3_CHECKPOINT must be set}"
: "${P4_CHECKPOINT:?P4_CHECKPOINT must be set}"
: "${P3_EXPECTED_SHA256:?P3_EXPECTED_SHA256 must be set}"
: "${P4_EXPECTED_SHA256:?P4_EXPECTED_SHA256 must be set}"
: "${P3_EVAL_OUT_DIR:?P3_EVAL_OUT_DIR must be set}"
: "${P4_EVAL_OUT_DIR:?P4_EVAL_OUT_DIR must be set}"

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
    raise RuntimeError("Formal Iterative Q evaluation requires CUDA.")
if not pulp.PULP_CBC_CMD(msg=False).available():
    raise RuntimeError("Exact terminal-cleanup evaluation requires CBC.")
PY

p3_actual_sha256=$(sha256sum "$P3_CHECKPOINT" | awk '{print $1}')
p4_actual_sha256=$(sha256sum "$P4_CHECKPOINT" | awk '{print $1}')
if [[ "$p3_actual_sha256" != "$P3_EXPECTED_SHA256" ]]; then
  echo "P3 checkpoint SHA mismatch: $p3_actual_sha256" >&2
  exit 2
fi
if [[ "$p4_actual_sha256" != "$P4_EXPECTED_SHA256" ]]; then
  echo "P4 checkpoint SHA mismatch: $p4_actual_sha256" >&2
  exit 2
fi

python - \
  "$P3_CHECKPOINT" "$P4_CHECKPOINT" \
  "$P3_EVAL_OUT_DIR" "$P4_EVAL_OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

import torch

p3_path = Path(sys.argv[1])
p4_path = Path(sys.argv[2])
output_paths = [Path(sys.argv[3]), Path(sys.argv[4])]
manifest = json.loads(
    Path(
        "experiments/protocols/unified_window_v1_seed_manifest.json"
    ).read_text(encoding="utf-8")
)
assert manifest["formal_test"]["range_inclusive"] == [9000001, 9000030]
assert manifest["formal_test"]["count"] == 30
for output_path in output_paths:
    if output_path.exists():
        raise FileExistsError(
            f"formal-test output already exists: {output_path}"
        )

for stage, checkpoint_path in (("P3", p3_path), ("P4", p4_path)):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    configuration = checkpoint["configuration"]
    assert configuration["model_seed"] == 0
    assert configuration["q_head"] == "iterative_action_q_future_summary"
    assert configuration["observation_input"] == "shared_future_summary"
    print(stage, checkpoint_path)

print("formal_test_range", manifest["formal_test"]["range_inclusive"])
print("formal_test_count", manifest["formal_test"]["count"])
PY

python -m py_compile experiments/evaluate_iterative_action_q.py
