#!/usr/bin/env bash
#SBATCH --job-name=ccs_window_joint
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH -o logs/event_window_joint-%A_%a.out
#SBATCH -e logs/event_window_joint-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_greedy_dagger}"
SOURCE_ROOT="output/rl_forecast/event_window_3200_allocation_sweep_20260724/window_3iter_3200_initial"
OUT_ROOT="${OUT_ROOT:-output/rl_forecast/event_window_3200_summary_joint_finetune_20260724}"
WINDOWS="108-179,180-251,252-323,324-395,396-467,468-539,540-611,612-680"
MODES=(
  residual_and_base_head
  base_head_only
)
OBSERVATIONS=(
  state_future
  state_only
)
NAMES=(
  summary_168_and_base_head
  base_head_only
)

mode="${MODES[$SLURM_ARRAY_TASK_ID]}"
observation="${OBSERVATIONS[$SLURM_ARRAY_TASK_ID]}"
name="${NAMES[$SLURM_ARRAY_TASK_ID]}"
run_dir="$OUT_ROOT/$name"

cd "$PROJECT_DIR"
mkdir -p logs "$OUT_ROOT"
if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing output: $run_dir" >&2
  exit 2
fi
mkdir -p "$run_dir"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

{
  printf 'source_checkpoint=%s\n' "$SOURCE_ROOT/p4/structured_action_stateless_q.pt"
  printf 'trainable_components=%s\n' "$mode"
  printf 'observation_input=%s\n' "$observation"
  printf 'base_head_learning_rate=0.00001\n'
  printf 'future_learning_rate=0.0001\n'
  printf 'model_seed=0\n'
} > "$run_dir/config.txt"

python -u scripts/train_event_structured_action_q.py \
  --train-data \
    "$SOURCE_ROOT/g0/train_merged.npz" \
    "$SOURCE_ROOT/g1/train_merged.npz" \
    "$SOURCE_ROOT/g2/train_merged.npz" \
    "$SOURCE_ROOT/g3/train_merged.npz" \
  --validation-data \
    "$SOURCE_ROOT/g0/validation_merged.npz" \
    "$SOURCE_ROOT/g1/validation_merged.npz" \
    "$SOURCE_ROOT/g2/validation_merged.npz" \
    "$SOURCE_ROOT/g3/validation_merged.npz" \
  --initial-checkpoint "$SOURCE_ROOT/p4/structured_action_stateless_q.pt" \
  --out-dir "$run_dir/model" \
  --epochs 40 \
  --patience 8 \
  --batch-size 16 \
  --encoder-learning-rate 0.00001 \
  --head-learning-rate 0.0001 \
  --heads 5 \
  --quantiles 51 \
  --prior-scale 0.25 \
  --action-embedding-size 16 \
  --action-feature-size 64 \
  --bootstrap-probability 0.8 \
  --improving-sample-weight 2.5 \
  --ranking-coefficient 1.0 \
  --listwise-coefficient 0.25 \
  --classification-coefficient 1.0 \
  --follow-anchor-coefficient 0.0 \
  --pairwise-min-cost-eur 10000 \
  --ranking-temperature 0.5 \
  --model-seed 0 \
  --model-architecture stateless_structured \
  --observation-input "$observation" \
  --forecast-encoder window_summary_168 \
  --trainable-components "$mode" \
  --require-trained-checkpoint \
  --device cuda

mapfile -t TEST_SEEDS < <(seq 7000 7029)
python -u experiments/evaluate_event_recurrent_q_policy.py \
  --checkpoint "$run_dir/model/structured_action_stateless_q.pt" \
  --out-dir "$run_dir/eval" \
  --eval-seeds "${TEST_SEEDS[@]}" \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --gates "$name":4:0.40:8:"$WINDOWS" \
  --device cuda
