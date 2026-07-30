#!/usr/bin/env bash
#SBATCH --job-name=ccs_e1_cppo
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH -o logs/e1_centralized_ppo-%A_%a.out
#SBATCH -e logs/e1_centralized_ppo-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_e1_20260728}"
RESULT_ROOT="${RESULT_ROOT:-experiments_results/E1/matched_learning_algorithms_20260728}"
MODEL_SEED="${SLURM_ARRAY_TASK_ID}"
RUN_DIR="$RESULT_ROOT/centralized_maskable_ppo/model_seed_${MODEL_SEED}"
VALIDATION_SEEDS=(
  8100001 8100002 8100003 8100004 8100005
  8100006 8100007 8100008 8100009 8100010
  8100011 8100012 8100013 8100014 8100015
  8100016 8100017 8100018 8100019 8100020
)

cd "$PROJECT_DIR"
mkdir -p logs "$RESULT_ROOT/centralized_maskable_ppo"
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing output collision: $RUN_DIR" >&2
  exit 2
fi

python -u -m sim.control.hourly_ppo.train_hourly_ppo \
  --timesteps 10000000 \
  --seed "$MODEL_SEED" \
  --scenario northern_lights_phase1_3vessels \
  --episode-hours 720 \
  --forecast-context-hours 168 \
  --future-summary-windows-h 168 \
  --weather-mode window \
  --warm-start \
  --scenario-protocol unified_window_v1 \
  --gamma 1 \
  --num-envs 8 \
  --n-steps 256 \
  --max-simulator-hour-steps 9505319 \
  --training-seed-min 100000 \
  --training-seed-max 999999 \
  --validation-seeds "${VALIDATION_SEEDS[@]}" \
  --validation-every-simulator-hour-steps 950531 \
  --device cuda \
  --ent-coef 0.01 \
  --log-dir "$RUN_DIR"
