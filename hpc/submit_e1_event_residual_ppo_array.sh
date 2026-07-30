#!/usr/bin/env bash
#SBATCH --job-name=ccs_e1_erppo
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH -o logs/e1_event_residual_ppo-%A_%a.out
#SBATCH -e logs/e1_event_residual_ppo-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_e1_20260728}"
RESULT_ROOT="${RESULT_ROOT:-experiments_results/E1/matched_learning_algorithms_20260728}"
MODEL_SEED="${SLURM_ARRAY_TASK_ID}"
RUN_DIR="$RESULT_ROOT/event_residual_ppo/model_seed_${MODEL_SEED}"
VALIDATION_SEEDS=(
  8100001 8100002 8100003 8100004 8100005
  8100006 8100007 8100008 8100009 8100010
  8100011 8100012 8100013 8100014 8100015
  8100016 8100017 8100018 8100019 8100020
)

cd "$PROJECT_DIR"
mkdir -p logs "$RESULT_ROOT/event_residual_ppo"
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing output collision: $RUN_DIR" >&2
  exit 2
fi

python -u -m \
  sim.control.event_based.residual_rl_v4.train_objective_aligned_ppo \
  --timesteps 10000000 \
  --seed "$MODEL_SEED" \
  --scenario northern_lights_phase1_3vessels \
  --episode-hours 720 \
  --forecast-context-hours 168 \
  --future-summary-windows-h 168 \
  --decision-interval-h 24 \
  --event-triggered \
  --gate-mode hard \
  --risk-hours-threshold-h 48 \
  --risk-fill-threshold 0.80 \
  --max-simulator-hour-steps 9505319 \
  --validation-every-simulator-hour-steps 950531 \
  --validation-seeds "${VALIDATION_SEEDS[@]}" \
  --training-seed-min 100000 \
  --training-seed-max 999999 \
  --n-steps 256 \
  --batch-size 64 \
  --learning-rate 0.0003 \
  --ent-coef 0.01 \
  --reward-scale 0.000001 \
  --device cpu \
  --log-dir "$RUN_DIR"
