#!/usr/bin/env bash
#SBATCH --job-name=ccs_future_mlp_bc
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --array=0-4%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o logs/future_mlp_bc-%A_%a.out
#SBATCH -e logs/future_mlp_bc-%A_%a.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_gnn_bc}"
cd "$PROJECT_DIR"
export PYTHONPATH="src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

MODEL_SEED="${SLURM_ARRAY_TASK_ID:-}"
if [[ ! "$MODEL_SEED" =~ ^[0-4]$ ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be in 0-4; got '${MODEL_SEED:-unset}'." >&2
  exit 1
fi

DEMO_CACHE="${DEMO_CACHE:-output/rl_forecast/corrected_forecast_cache/destination_mask_train_0_99_v4.npz}"
HELDOUT_DEMO_CACHE="${HELDOUT_DEMO_CACHE:-output/rl_forecast/corrected_forecast_cache/destination_mask_heldout_121_140_v4.npz}"
OUT_DIR="${OUT_DIR:-output/rl_forecast/future_mlp_bc}"
BC_EPOCHS="${BC_EPOCHS:-50}"
BC_BATCH_SIZE="${BC_BATCH_SIZE:-256}"
EVAL_SEEDS="${EVAL_SEEDS:-101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120}"
EPISODE_HOURS="${EPISODE_HOURS:-720}"
DEVICE="${DEVICE:-cuda}"
read -r -a EVAL_SEED_ARGS <<< "$EVAL_SEEDS"

for CACHE in "$DEMO_CACHE" "$HELDOUT_DEMO_CACHE"; do
  if [[ ! -f "$CACHE" ]]; then
    echo "ERROR: demonstration cache not found: $CACHE" >&2
    exit 1
  fi
done
mkdir -p logs "$OUT_DIR"

echo "Variant: future_mlp_mode_destination"
echo "Model seed: $MODEL_SEED"
echo "Training cache: $DEMO_CACHE"
echo "Held-out cache: $HELDOUT_DEMO_CACHE"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPU count:', torch.cuda.device_count())"

python -u scripts/compare_forecast_encoders_rl.py train \
  --variant future_mlp_mode_destination \
  --demo-cache "$DEMO_CACHE" \
  --heldout-demo-cache "$HELDOUT_DEMO_CACHE" \
  --bc-objective decision_only \
  --bc-only \
  --imitation-only \
  --timesteps 0 \
  --bc-epochs "$BC_EPOCHS" \
  --bc-batch-size "$BC_BATCH_SIZE" \
  --model-seed "$MODEL_SEED" \
  --eval-seeds "${EVAL_SEED_ARGS[@]}" \
  --device "$DEVICE" \
  --out-dir "$OUT_DIR" \
  --episode-hours "$EPISODE_HOURS"
