#!/usr/bin/env bash
#SBATCH --job-name=iq_locked_eval
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o logs/locked_iterative_q_eval-%j.out
#SBATCH -e logs/locked_iterative_q_eval-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${CHECKPOINT:?CHECKPOINT must be set}"
: "${OUT_DIR:?OUT_DIR must be set}"
: "${EVAL_NAME:?EVAL_NAME must be set}"
STRESS_LEVEL="${STRESS_LEVEL:-medium}"
POLICY_WINDOWS_H="108-155,156-203,204-251,252-299,300-347,348-395,396-443,444-491,492-539,540-587,588-635,636-680"
TEST_SEEDS=(
  9000031 9000032 9000033 9000034 9000035
  9000036 9000037 9000038 9000039 9000040
  9000041 9000042 9000043 9000044 9000045
  9000046 9000047 9000048 9000049 9000050
  9000051 9000052 9000053 9000054 9000055
  9000056 9000057 9000058 9000059 9000060
)

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -c "import json; p=json.load(open('experiments/protocols/e2_e3_e4_iterative_q_protocol.json')); assert p['formal_test']['range_inclusive'] == [9000031, 9000060] and p['formal_test']['count'] == 30"
python -u experiments/evaluate_iterative_action_q.py \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --eval-seeds "${TEST_SEEDS[@]}" \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --scenario-protocol unified_window_v1 \
  --stress-level "$STRESS_LEVEL" \
  --hard-scenario-probability 0.5 \
  --forecast-context-hours 168 \
  --gates "$EVAL_NAME":4:0.40:12:"$POLICY_WINDOWS_H" \
  --device cuda
