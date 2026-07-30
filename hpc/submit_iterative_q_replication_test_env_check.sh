#!/usr/bin/env bash
#SBATCH --job-name=iterq_repl_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o logs/iterative_q_replication_test_check-%j.out
#SBATCH -e logs/iterative_q_replication_test_check-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${P3_CHECKPOINT:?P3_CHECKPOINT must be set}"
: "${P4_CHECKPOINT:?P4_CHECKPOINT must be set}"
: "${V2_CHECKPOINT:?V2_CHECKPOINT must be set}"
: "${P3_EXPECTED_SHA256:?P3_EXPECTED_SHA256 must be set}"
: "${P4_EXPECTED_SHA256:?P4_EXPECTED_SHA256 must be set}"
: "${V2_EXPECTED_SHA256:?V2_EXPECTED_SHA256 must be set}"
: "${P3_EVAL_OUT_DIR:?P3_EVAL_OUT_DIR must be set}"
: "${P4_EVAL_OUT_DIR:?P4_EVAL_OUT_DIR must be set}"
: "${V2_EVAL_OUT_DIR:?V2_EVAL_OUT_DIR must be set}"
: "${EVAL_SEEDS:?EVAL_SEEDS must be set}"

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
    raise RuntimeError("Iterative Q replication evaluation requires CUDA.")
if not pulp.PULP_CBC_CMD(msg=False).available():
    raise RuntimeError("Exact terminal-cleanup evaluation requires CBC.")
PY

for spec in \
  "$P3_CHECKPOINT:$P3_EXPECTED_SHA256" \
  "$P4_CHECKPOINT:$P4_EXPECTED_SHA256" \
  "$V2_CHECKPOINT:$V2_EXPECTED_SHA256"; do
  checkpoint="${spec%:*}"
  expected_sha256="${spec##*:}"
  actual_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Checkpoint SHA mismatch for $checkpoint: $actual_sha256" >&2
    exit 2
  fi
done

python - \
  "$P3_CHECKPOINT" "$P4_CHECKPOINT" "$V2_CHECKPOINT" \
  "$P3_EVAL_OUT_DIR" "$P4_EVAL_OUT_DIR" "$V2_EVAL_OUT_DIR" \
  "$EVAL_SEEDS" <<'PY'
import sys
from pathlib import Path

import torch

p3_path, p4_path, v2_path = map(Path, sys.argv[1:4])
output_paths = list(map(Path, sys.argv[4:7]))
seeds = [int(value) for value in sys.argv[7].split(":")]
assert seeds == list(range(9000031, 9000061))
assert not set(seeds).intersection(range(9000001, 9000031))
for output_path in output_paths:
    if output_path.exists():
        raise FileExistsError(f"replication output already exists: {output_path}")

for stage, checkpoint_path in (
    ("P3", p3_path),
    ("P4", p4_path),
    ("v2", v2_path),
):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    configuration = checkpoint["configuration"]
    assert configuration["model_seed"] == 0
    assert configuration["q_head"] == "iterative_action_q_future_summary"
    if stage == "v2":
        assert configuration["previous_policy_anchor_coefficient"] == 1.0
        assert configuration["previous_policy_release_margin_eur"] == 40000.0
        assert configuration.get(
            "previous_policy_anchor_weighting", "hard"
        ) == "hard"
    print(stage, checkpoint_path)

print("replication_seed_range", [seeds[0], seeds[-1]])
print("replication_seed_count", len(seeds))
PY

python -m py_compile experiments/evaluate_iterative_action_q.py
