#!/usr/bin/env bash
#SBATCH --job-name=ccs_iter_q_eval
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o logs/iterative_q_eval-%j.out
#SBATCH -e logs/iterative_q_eval-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_greedy_dagger}"
: "${RUN_ROOT:?RUN_ROOT must be set}"
FINAL_STAGE="${FINAL_STAGE:-p4}"
WINDOWS="108-179,180-251,252-323,324-395,396-467,468-539,540-611,612-680"
EVAL_NAME="${EVAL_NAME:-iterative_action_q}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mapfile -t TEST_SEEDS < <(seq 7000 7029)

python -u experiments/evaluate_iterative_action_q.py \
  --checkpoint "$RUN_ROOT/$FINAL_STAGE/iterative_action_q.pt" \
  --out-dir "$RUN_ROOT/eval/$EVAL_NAME" \
  --eval-seeds "${TEST_SEEDS[@]}" \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --gates "$EVAL_NAME":4:0.40:8:"$WINDOWS" \
  --device cuda
