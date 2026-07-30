#!/usr/bin/env bash
#SBATCH --job-name=iter_h3_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o logs/iterative_h3_check-%j.out
#SBATCH -e logs/iterative_h3_check-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
: "${OUT_ROOT:?OUT_ROOT must be set}"
: "${SOURCE_RUN:?SOURCE_RUN must be set}"
: "${SEED_CHECKPOINT_ROOT:?SEED_CHECKPOINT_ROOT must be set}"
PROTOCOL="experiments/protocols/iterative_h3_sampler_validation_protocol.json"

cd "$PROJECT_DIR"
mkdir -p logs "$OUT_ROOT"
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1

python -m py_compile \
  scripts/train_iterative_action_q.py \
  experiments/create_iterative_q_lock.py \
  experiments/audit_iterative_sampling_data.py \
  experiments/check_iterative_h3_sampler_experiment.py \
  experiments/aggregate_iterative_h3_sampler_validation.py
bash -n hpc/submit_iterative_q_lock.sh
bash -n hpc/submit_iterative_q_train.sh
bash -n hpc/submit_iterative_h3_sampler_train.sh
bash -n hpc/submit_iterative_h3_sampler_validation.sh
bash -n hpc/launch_iterative_h3_sampler_p2.sh
python -u experiments/check_iterative_h3_sampler_experiment.py \
  --protocol "$PROTOCOL" \
  --source-run "$SOURCE_RUN" \
  --seed-checkpoint-root "$SEED_CHECKPOINT_ROOT" \
  --out-path "$OUT_ROOT/environment_check.json"
