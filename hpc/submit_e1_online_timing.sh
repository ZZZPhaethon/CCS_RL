#!/usr/bin/env bash
#SBATCH --job-name=e1_online_time
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodelist=rootrunner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-13%1
#SBATCH -o logs/e1_online_time-%A_%a.out
#SBATCH -e logs/e1_online_time-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/experiments_results/E1/timing/online_timing_hpc_run01}"
ITERATIVE_Q_MODEL_ROOT="${ITERATIVE_Q_MODEL_ROOT:-$PROJECT_DIR/experiments_results/E1/models/iterative_q}"
methods=(
  fixed_assignment
  greedy
  ppo_hourly ppo_hourly ppo_hourly
  ppo_high_level ppo_high_level ppo_high_level
  ppo_event_residual ppo_event_residual ppo_event_residual
  iterative_action_q_g60_p4 iterative_action_q_g60_p4 iterative_action_q_g60_p4
)
model_seeds=(-1 -1 0 1 2 0 1 2 0 1 2 0 1 2)

task_id="${SLURM_ARRAY_TASK_ID}"
method="${methods[$task_id]}"
model_seed="${model_seeds[$task_id]}"
if (( model_seed < 0 )); then
  final_dir="$RESULT_ROOT/$method"
else
  final_dir="$RESULT_ROOT/$method/model_seed_$model_seed"
fi
stage_dir="${final_dir}.inprogress.${SLURM_ARRAY_JOB_ID}_${task_id}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if [[ "$(hostname)" != "rootrunner" ]]; then
  echo "E1 timing must run on rootrunner." >&2
  exit 2
fi
if (( ${SLURM_CPUS_PER_TASK:-0} != 4 )); then
  echo "E1 timing requires exactly four allocated CPUs." >&2
  exit 2
fi
if [[ -e "$final_dir" || -e "$stage_dir" ]]; then
  echo "Refusing output collision: $final_dir or $stage_dir" >&2
  exit 3
fi

mkdir -p "$(dirname "$final_dir")" "$stage_dir"
command=(
  python -u experiments/run_e1_online_timing.py
  --method "$method"
  --output-dir "$stage_dir"
)
if (( model_seed >= 0 )); then
  command+=(--model-seed "$model_seed")
fi
if [[ "$method" == "iterative_action_q_g60_p4" ]]; then
  command+=(--iterative-q-model-root "$ITERATIVE_Q_MODEL_ROOT")
fi

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "method=$method"
echo "model_seed=$model_seed"
"${command[@]}"
mv "$stage_dir" "$final_dir"
echo "finished_at=$(date --iso-8601=seconds)"
echo "E1_ONLINE_TIMING_TASK_COMPLETE output=$final_dir"
