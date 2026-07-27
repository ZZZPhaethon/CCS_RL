#!/usr/bin/env bash
#SBATCH --job-name=ccs_iter_q_lock
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o logs/iterative_q_lock-%j.out
#SBATCH -e logs/iterative_q_lock-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_greedy_dagger}"
: "${RUN_ROOT:?RUN_ROOT must be set}"
: "${OUTPUT_STAGE:?OUTPUT_STAGE must be set}"
: "${RESIDUAL_MARGIN:?RESIDUAL_MARGIN must be set}"
: "${ECONOMIC_MARGIN_EUR:?ECONOMIC_MARGIN_EUR must be set}"
: "${MAX_OVERRIDES:?MAX_OVERRIDES must be set}"
: "${POLICY_WINDOWS_H:?POLICY_WINDOWS_H must be set}"
PROTOCOL_PREFIX="${PROTOCOL_PREFIX:-iterative_q}"
POLICY_WINDOWS_CSV="${POLICY_WINDOWS_H//:/,}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

python -u experiments/create_iterative_q_lock.py \
  --checkpoint "$RUN_ROOT/$OUTPUT_STAGE/iterative_action_q.pt" \
  --out-path "$RUN_ROOT/${OUTPUT_STAGE}_lock.json" \
  --protocol-id "${PROTOCOL_PREFIX}_${OUTPUT_STAGE}" \
  --residual-margin "$RESIDUAL_MARGIN" \
  --economic-margin-eur "$ECONOMIC_MARGIN_EUR" \
  --max-overrides "$MAX_OVERRIDES" \
  --windows-h "$POLICY_WINDOWS_CSV"
