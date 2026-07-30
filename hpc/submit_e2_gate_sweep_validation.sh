#!/usr/bin/env bash
#SBATCH --job-name=e2_gate_val
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH -o logs/e2_gate_val-%A_%a.out
#SBATCH -e logs/e2_gate_val-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
OUT_ROOT="experiments_results/E2/validation_gate_sweep_seeds_8100001-8100020_run01"
MODEL_SEED=$((SLURM_ARRAY_TASK_ID % 3))
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
if [[ "$METHOD_INDEX" == "0" ]]; then
  METHOD=iterative_p4
  if [[ "$MODEL_SEED" == "0" ]]; then
    CHECKPOINT="output/iterative_q_budget_search/runs/g60_p4/p4/iterative_action_q.pt"
  else
    CHECKPOINT="experiments_results/E1/training_iterative_action_q_g60_p4_full_model_seeds_1_2_20260729/model_seed_${MODEL_SEED}/p4/iterative_action_q.pt"
  fi
else
  METHOD=one_shot_matched
  CHECKPOINT="experiments_results/E2/training_one_shot_matched_run01/model_seed_${MODEL_SEED}/p1/iterative_action_q.pt"
fi
OUT_DIR="$PROJECT_DIR/$OUT_ROOT/$METHOD/model_seed_${MODEL_SEED}"
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing existing validation gate output: $OUT_DIR" >&2
  exit 2
fi

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mapfile -t GATE_VALUES < <(
  python -c 'from experiments.gate_sweep_configs import gate_cli_values; print(*gate_cli_values(), sep="\n")'
)
VALIDATION_SEEDS=(
  8100001 8100002 8100003 8100004 8100005
  8100006 8100007 8100008 8100009 8100010
  8100011 8100012 8100013 8100014 8100015
  8100016 8100017 8100018 8100019 8100020
)

python -u experiments/evaluate_iterative_action_q.py \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --eval-seeds "${VALIDATION_SEEDS[@]}" \
  --validation-only \
  --seed-manifest experiments/protocols/unified_window_v1_seed_manifest.json \
  --episode-hours 720 \
  --reward-scale 0.00001 \
  --scenario-protocol unified_window_v1 \
  --stress-level medium \
  --hard-scenario-probability 0.5 \
  --forecast-context-hours 168 \
  --gates "${GATE_VALUES[@]}" \
  --device cuda
