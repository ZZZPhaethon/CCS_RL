#!/usr/bin/env bash
#SBATCH --job-name=e7_direct_global
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --nodelist=rootrunner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-53%24
#SBATCH -o logs/e7_direct_global-%A_%a.out
#SBATCH -e logs/e7_direct_global-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="/scratch_root/hx721/CCS_RLLLM_e1_20260728"
RESULT_ROOT="$PROJECT_DIR/experiments_results/E7/formal_run05_direct_global"
MODEL_ROOT="/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728/output/E1_hour_removed_formal_models"
expected_hashes=(
  962373fc027107000979ba26209e1b2be781d992dc4abc45038a5f2f3285c3dc
  f4cfbeab74b072d22dcb3801890f95c3660d4b77293f52542d31879cc894ccbc
  6e8f7dba48ce08b1a299327ca8c95f7b72c67bdb0164b05806464cfe6e28d50f
)
horizons=(720 2160 8760)

task_id="${SLURM_ARRAY_TASK_ID}"
combo_id="$((task_id / 6))"
seed_block="$((task_id % 6))"
model_seed="$((combo_id / 3))"
horizon="${horizons[$((combo_id % 3))]}"
first_seed="$((9000031 + 5 * seed_block))"
seeds=(
  "$first_seed"
  "$((first_seed + 1))"
  "$((first_seed + 2))"
  "$((first_seed + 3))"
  "$((first_seed + 4))"
)
checkpoint="$MODEL_ROOT/g60_p4_model_seed_$model_seed/iterative_action_q.pt"
actual_hash="$(sha256sum "$checkpoint" | awk '{print $1}')"
if [[ "$actual_hash" != "${expected_hashes[$model_seed]}" ]]; then
  echo "Checkpoint hash mismatch for model seed $model_seed: $actual_hash" >&2
  exit 2
fi

final_dir="$RESULT_ROOT/iterative_q_direct/h${horizon}/model_seed_$model_seed/block_$seed_block"
stage_dir="${final_dir}.inprogress.${SLURM_ARRAY_JOB_ID}_${task_id}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if (( ${SLURM_CPUS_PER_TASK:-0} != 4 )); then
  echo "E7 requires exactly four allocated CPUs." >&2
  exit 3
fi
if [[ -e "$final_dir" || -e "$stage_dir" ]]; then
  echo "Refusing output collision: $final_dir or $stage_dir" >&2
  exit 4
fi

mkdir -p "$(dirname "$final_dir")" "$stage_dir"
echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "controller=iterative_q_direct"
echo "horizon_hours=$horizon"
echo "model_seed=$model_seed"
echo "checkpoint=$checkpoint"
echo "checkpoint_sha256=$actual_hash"
echo "seeds=${seeds[*]}"
python -u experiments/run_e7_temporal_generalization.py \
  --controller iterative_q_direct \
  --horizon-hours "$horizon" \
  --model-seed "$model_seed" \
  --seeds "${seeds[@]}" \
  --output-dir "$stage_dir" \
  --iterative-q-model-root "$MODEL_ROOT"
mv "$stage_dir" "$final_dir"
echo "finished_at=$(date --iso-8601=seconds)"
echo "E7_DIRECT_GLOBAL_TASK_COMPLETE output=$final_dir"
