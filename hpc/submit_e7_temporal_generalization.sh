#!/usr/bin/env bash
#SBATCH --job-name=e7_horizon
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodelist=rootrunner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-143%8
#SBATCH -o logs/e7_horizon-%A_%a.out
#SBATCH -e logs/e7_horizon-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/experiments_results/E7/formal_run01}"
ITERATIVE_Q_MODEL_ROOT="${ITERATIVE_Q_MODEL_ROOT:-$PROJECT_DIR/experiments_results/E1/models/iterative_q}"
controllers=(
  fixed_assignment fixed_assignment fixed_assignment
  greedy greedy greedy
  iterative_q_direct iterative_q_direct iterative_q_direct
  iterative_q_direct iterative_q_direct iterative_q_direct
  iterative_q_direct iterative_q_direct iterative_q_direct
  iterative_q_receding iterative_q_receding iterative_q_receding
  iterative_q_receding iterative_q_receding iterative_q_receding
  iterative_q_receding iterative_q_receding iterative_q_receding
)
horizons=(
  720 2160 8760
  720 2160 8760
  720 2160 8760
  720 2160 8760
  720 2160 8760
  720 2160 8760
  720 2160 8760
  720 2160 8760
)
model_seeds=(
  -1 -1 -1
  -1 -1 -1
  0 0 0
  1 1 1
  2 2 2
  0 0 0
  1 1 1
  2 2 2
)

task_id="${SLURM_ARRAY_TASK_ID}"
combo_id="$((task_id / 6))"
seed_block="$((task_id % 6))"
controller="${controllers[$combo_id]}"
horizon="${horizons[$combo_id]}"
model_seed="${model_seeds[$combo_id]}"
first_seed="$((9000031 + 5 * seed_block))"
seeds=(
  "$first_seed"
  "$((first_seed + 1))"
  "$((first_seed + 2))"
  "$((first_seed + 3))"
  "$((first_seed + 4))"
)
if (( model_seed < 0 )); then
  final_dir="$RESULT_ROOT/$controller/h${horizon}/block_$seed_block"
else
  final_dir="$RESULT_ROOT/$controller/h${horizon}/model_seed_$model_seed/block_$seed_block"
fi
stage_dir="${final_dir}.inprogress.${SLURM_ARRAY_JOB_ID}_${task_id}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if (( ${SLURM_CPUS_PER_TASK:-0} != 4 )); then
  echo "E7 requires exactly four allocated CPUs." >&2
  exit 2
fi
if [[ -e "$final_dir" || -e "$stage_dir" ]]; then
  echo "Refusing output collision: $final_dir or $stage_dir" >&2
  exit 3
fi

mkdir -p "$(dirname "$final_dir")" "$stage_dir"
command=(
  python -u experiments/run_e7_temporal_generalization.py
  --controller "$controller"
  --horizon-hours "$horizon"
  --seeds "${seeds[@]}"
  --output-dir "$stage_dir"
)
if (( model_seed >= 0 )); then
  command+=(
    --model-seed "$model_seed"
    --iterative-q-model-root "$ITERATIVE_Q_MODEL_ROOT"
  )
fi

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "controller=$controller"
echo "horizon_hours=$horizon"
echo "model_seed=$model_seed"
echo "seeds=${seeds[*]}"
"${command[@]}"
mv "$stage_dir" "$final_dir"
echo "finished_at=$(date --iso-8601=seconds)"
echo "E7_TASK_COMPLETE output=$final_dir"
