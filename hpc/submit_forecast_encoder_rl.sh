#!/usr/bin/env bash
#SBATCH --job-name=ccs_forecast_rl
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o logs/forecast_rl-%A_%a.out
#SBATCH -e logs/forecast_rl-%A_%a.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM}"
cd "$PROJECT_DIR"

export PYTHONPATH="src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

if [[ -z "${GIT_COMMIT:-}" ]]; then
  if ! GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null)"; then
    echo "ERROR: cannot determine GIT_COMMIT from git rev-parse HEAD." >&2
    exit 1
  fi
fi
if [[ -z "$GIT_COMMIT" ]]; then
  echo "ERROR: cannot determine GIT_COMMIT: resolved value is empty." >&2
  exit 1
fi
export GIT_COMMIT

VARIANTS=(state flat tcn)
MODEL_SEEDS=(0 1 2)
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ ! "$TASK_ID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be an integer in the range 0-8; got '${TASK_ID:-unset}'." >&2
  exit 1
fi
VARIANT_INDEX=$((TASK_ID % 3))
SEED_INDEX=$((TASK_ID / 3))
if (( TASK_ID > 8 || SEED_INDEX >= ${#MODEL_SEEDS[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID $TASK_ID is out of range; expected 0-8." >&2
  exit 1
fi
VARIANT="${VARIANTS[$VARIANT_INDEX]}"
MODEL_SEED="${MODEL_SEEDS[$SEED_INDEX]}"

DEMO_CACHE="${DEMO_CACHE:-output/rl_forecast/demos/mpc_720h_30eps.npz}"
OUT_DIR="${OUT_DIR:-output/rl_forecast/pilot}"
TIMESTEPS="${TIMESTEPS:-100000}"
BC_EPOCHS="${BC_EPOCHS:-20}"
N_STEPS="${N_STEPS:-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-256}"
EVAL_SEEDS="${EVAL_SEEDS:-101 102 103 104 105}"
DEVICE="${DEVICE:-cuda}"
EPISODE_HOURS="${EPISODE_HOURS:-720}"
read -r -a EVAL_SEED_ARGS <<< "$EVAL_SEEDS"

if [[ ! -f "$DEMO_CACHE" ]]; then
  echo "ERROR: demonstration cache not found: $DEMO_CACHE" >&2
  exit 1
fi

mkdir -p logs "$OUT_DIR"

echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Array task ID: $TASK_ID"
echo "Project directory: $PROJECT_DIR"
echo "Git commit: $GIT_COMMIT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Variant: $VARIANT"
echo "Model seed: $MODEL_SEED"
echo "Demo cache: $DEMO_CACHE"
echo "Output directory: $OUT_DIR"
which python
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

python -u scripts/compare_forecast_encoders_rl.py train \
  --variant "$VARIANT" \
  --demo-cache "$DEMO_CACHE" \
  --timesteps "$TIMESTEPS" \
  --bc-epochs "$BC_EPOCHS" \
  --n-steps "$N_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --bc-batch-size "$BC_BATCH_SIZE" \
  --model-seed "$MODEL_SEED" \
  --eval-seeds "${EVAL_SEED_ARGS[@]}" \
  --device "$DEVICE" \
  --out-dir "$OUT_DIR" \
  --episode-hours "$EPISODE_HOURS"

echo "Job finished at $(date)"

# Smoke (after the one-seed cache exists):
# sbatch --qos=short --time=01:00:00 --array=0-2 --export=ALL,TIMESTEPS=2048,BC_EPOCHS=1,DEMO_CACHE=output/rl_forecast/demos/mpc_smoke.npz,OUT_DIR=output/rl_forecast/smoke hpc/submit_forecast_encoder_rl.sh
# Formal comparison (three variants by three model seeds):
# sbatch --array=0-8 --export=ALL,OUT_DIR=output/rl_forecast/formal,EVAL_SEEDS='101 102 103 104 105 106 107 108 109 110' hpc/submit_forecast_encoder_rl.sh
