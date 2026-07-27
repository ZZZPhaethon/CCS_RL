#!/usr/bin/env bash
#SBATCH --job-name=ccs_er_future
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH -o logs/event_residual_future-%A_%a.out
#SBATCH -e logs/event_residual_future-%A_%a.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_unified_window12_20260726}"
RUN_ROOT="${RUN_ROOT:-output/unified_window12/event_residual_future_ablation_20260727}"
NAMES=(
  state_only
  summary_24_72
  summary_168
  summary_24_72_168
)

name="${NAMES[$SLURM_ARRAY_TASK_ID]}"
case "$name" in
  state_only)
    summary_windows=()
    ;;
  summary_24_72)
    summary_windows=(24 72)
    ;;
  summary_168)
    summary_windows=(168)
    ;;
  summary_24_72_168)
    summary_windows=(24 72 168)
    ;;
  *)
    echo "Unknown ablation variant: $name" >&2
    exit 2
    ;;
esac

run_dir="$RUN_ROOT/$name"
cd "$PROJECT_DIR"
mkdir -p logs "$RUN_ROOT"
if [[ -e "$run_dir" ]]; then
  echo "Refusing output collision: $run_dir" >&2
  exit 2
fi

export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u -m sim.control.event_based.residual_rl_v4.train_tail_robust_ppo \
  --scenario northern_lights_phase1_3vessels \
  --episode-hours 720 \
  --forecast-context-hours 168 \
  --future-summary-windows-h "${summary_windows[@]}" \
  --decision-interval-h 24 \
  --event-triggered \
  --weather-mode window \
  --scenario-protocol unified_window_v1 \
  --override-windows-h \
    108-155 156-203 204-251 252-299 300-347 348-395 \
    396-443 444-491 492-539 540-587 588-635 636-680 \
  --curriculum-stages 0:0 \
  --timesteps 100000 \
  --num-envs 4 \
  --vec-env subproc \
  --validation-every-steps 10000 \
  --replay-probability 0.30 \
  --replay-capacity 20 \
  --minimum-replay-pool 4 \
  --seed 0 \
  --device cpu \
  --no-reference-constraints \
  --log-dir "$run_dir"

mapfile -t TEST_SEEDS < <(seq 8000001 8000030)
python -u -m sim.control.event_based.residual_rl_v4.evaluate_ppo \
  --run-dir "$run_dir" \
  --seeds "${TEST_SEEDS[@]}" \
  --model best \
  --hard-scenario-probability 0
