#!/usr/bin/env bash
#SBATCH --job-name=iterq_seed_blocks
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH -o logs/iterative_q_seed_blocks-%A_%a.out
#SBATCH -e logs/iterative_q_seed_blocks-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
RUN_ROOT="${RUN_ROOT:-output/iterative_q_validation_search/root_reallocation_exact_b_20260728}"
CHECKPOINT="${CHECKPOINT:-$RUN_ROOT/p3/iterative_action_q.pt}"
EXPECTED_SHA256="${EXPECTED_SHA256:-c97f018e63b336fe15c34b4546ea7908a2c271dccb7cc23b32c3aad8f44d1ed4}"
RESULT_ROOT="${RESULT_ROOT:-$RUN_ROOT/eval/seed_block_sensitivity_20260728}"
WINDOWS="108-155,156-203,204-251,252-299,300-347,348-395,396-443,444-491,492-539,540-587,588-635,636-680"

BLOCK_STARTS=(9000031 9000061 9000091 9000121)
BLOCK_ENDS=(9000060 9000090 9000120 9000150)
EVAL_SEED_START="${EVAL_SEED_START:-}"
EVAL_SEED_END="${EVAL_SEED_END:-}"
if [[ -n "$EVAL_SEED_START" && -n "$EVAL_SEED_END" ]]; then
  start="$EVAL_SEED_START"
  end="$EVAL_SEED_END"
elif [[ -z "$EVAL_SEED_START" && -z "$EVAL_SEED_END" ]]; then
  start="${BLOCK_STARTS[$SLURM_ARRAY_TASK_ID]}"
  end="${BLOCK_ENDS[$SLURM_ARRAY_TASK_ID]}"
else
  echo "EVAL_SEED_START and EVAL_SEED_END must be set together" >&2
  exit 2
fi
name="reallocated_b_margin50_${start}_${end}"
out_dir="$RESULT_ROOT/$name"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

actual_sha256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "checkpoint SHA-256 mismatch: $actual_sha256" >&2
  exit 2
fi
if [[ -e "$out_dir" ]]; then
  echo "refusing existing output directory: $out_dir" >&2
  exit 2
fi

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
    raise RuntimeError("Iterative Q evaluation requires CUDA.")
if not pulp.PULP_CBC_CMD(msg=False).available():
    raise RuntimeError("Terminal-cleanup evaluation requires CBC.")
PY

mapfile -t eval_seeds < <(seq "$start" "$end")
echo "checkpoint_sha256=$actual_sha256"
echo "seed_block=$start-$end"

python -u experiments/evaluate_iterative_action_q.py \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$out_dir" \
  --eval-seeds "${eval_seeds[@]}" \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --scenario-protocol unified_window_v1 \
  --hard-scenario-probability 0.5 \
  --forecast-context-hours 168 \
  --future-summary-windows-h 168 \
  --gates "$name":4:0.50:12:"$WINDOWS" \
  --device cuda

test -s "$out_dir/evaluation.csv"
test -s "$out_dir/summary.json"
