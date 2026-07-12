#!/usr/bin/env bash
#SBATCH --job-name=ccs_bc_objective
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --array=0-9%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o logs/bc_objective-%A_%a.out
#SBATCH -e logs/bc_objective-%A_%a.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_bc_objective}"
cd "$PROJECT_DIR"
export PYTHONPATH="src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

BC_OBJECTIVE="${BC_OBJECTIVE:-}"
if [[ "$BC_OBJECTIVE" != "decision_only" && "$BC_OBJECTIVE" != "decision_balanced" ]]; then
  echo "ERROR: BC_OBJECTIVE must be decision_only or decision_balanced." >&2
  exit 1
fi

VARIANTS=(state_mode tcn_mode)
MODEL_SEEDS=(0 1 2 3 4)
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ ! "$TASK_ID" =~ ^[0-9]+$ ]] || (( TASK_ID > 9 )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be in 0-9; got '${TASK_ID:-unset}'." >&2
  exit 1
fi
VARIANT="${VARIANTS[$((TASK_ID % 2))]}"
MODEL_SEED="${MODEL_SEEDS[$((TASK_ID / 2))]}"

DEMO_CACHE="${DEMO_CACHE:-/scratch_root/hx721/CCS_RLLLM_operation_mode_41ad6d4/output/rl_forecast/demos/mpc_720h_100seeds.npz}"
OUT_DIR="${OUT_DIR:-output/rl_forecast/bc_objective_${BC_OBJECTIVE}}"
BC_EPOCHS="${BC_EPOCHS:-50}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-256}"
EVAL_SEEDS="${EVAL_SEEDS:-101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120}"
EPISODE_HOURS="${EPISODE_HOURS:-720}"
DEVICE="${DEVICE:-cuda}"
read -r -a EVAL_SEED_ARGS <<< "$EVAL_SEEDS"

if [[ ! -f "$DEMO_CACHE" ]]; then
  echo "ERROR: demonstration cache not found: $DEMO_CACHE" >&2
  exit 1
fi
mkdir -p logs "$OUT_DIR"

echo "Objective: $BC_OBJECTIVE"
echo "Variant: $VARIANT"
echo "Model seed: $MODEL_SEED"
echo "Demo cache: $DEMO_CACHE"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPU count:', torch.cuda.device_count())"

python -u scripts/compare_forecast_encoders_rl.py train \
  --variant "$VARIANT" \
  --demo-cache "$DEMO_CACHE" \
  --bc-objective "$BC_OBJECTIVE" \
  --bc-only \
  --timesteps 0 \
  --bc-epochs "$BC_EPOCHS" \
  --bc-batch-size "$BC_BATCH_SIZE" \
  --model-seed "$MODEL_SEED" \
  --eval-seeds "${EVAL_SEED_ARGS[@]}" \
  --device "$DEVICE" \
  --out-dir "$OUT_DIR" \
  --episode-hours "$EPISODE_HOURS"

