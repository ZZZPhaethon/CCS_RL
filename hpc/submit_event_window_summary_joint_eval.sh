#!/usr/bin/env bash
#SBATCH --job-name=ccs_window_joint_eval
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH -o logs/event_window_joint_eval-%A_%a.out
#SBATCH -e logs/event_window_joint_eval-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_greedy_dagger}"
OUT_ROOT="${OUT_ROOT:-output/rl_forecast/event_window_3200_summary_joint_finetune_20260724_v2}"
WINDOWS="108-179,180-251,252-323,324-395,396-467,468-539,540-611,612-680"
NAMES=(
  summary_168_and_base_head
  base_head_only
)

name="${NAMES[$SLURM_ARRAY_TASK_ID]}"
run_dir="$OUT_ROOT/$name"
checkpoint="$run_dir/model/structured_action_stateless_q.pt"

cd "$PROJECT_DIR"
mkdir -p logs
if [[ ! -f "$checkpoint" ]]; then
  echo "Missing trained checkpoint: $checkpoint" >&2
  exit 2
fi
if [[ -e "$run_dir/eval" ]]; then
  echo "Refusing to overwrite existing evaluation: $run_dir/eval" >&2
  exit 2
fi
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mapfile -t TEST_SEEDS < <(seq 7000 7029)
python -u experiments/evaluate_event_recurrent_q_policy.py \
  --checkpoint "$checkpoint" \
  --out-dir "$run_dir/eval" \
  --eval-seeds "${TEST_SEEDS[@]}" \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --gates "$name":4:0.40:8:"$WINDOWS" \
  --device cuda
