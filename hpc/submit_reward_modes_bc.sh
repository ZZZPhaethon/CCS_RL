#!/usr/bin/env bash
#SBATCH --job-name=ccs_reward_bc
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o logs/reward_bc-%j.out
#SBATCH -e logs/reward_bc-%j.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM}"
YARA_BUFFER_CAPACITY="${YARA_BUFFER_CAPACITY:-7500}"
TIMESTEPS="${TIMESTEPS:-100000}"
BC_EPISODES="${BC_EPISODES:-30}"
BC_EPOCHS="${BC_EPOCHS:-20}"
NONWAIT_WEIGHT="${NONWAIT_WEIGHT:-10}"
KICKSTART_COEF="${KICKSTART_COEF:-1.0}"
REWARD_MODES="${REWARD_MODES:-economic vent_first}"
CAPTURE_NOISE_STD="${CAPTURE_NOISE_STD:-0.30}"
INITIAL_INVENTORY_FILL_MAX="${INITIAL_INVENTORY_FILL_MAX:-0.5}"
LEG_WAVE_SLOWDOWN_MULTIPLIER="${LEG_WAVE_SLOWDOWN_MULTIPLIER:-1.0}"
LEG_WAVE_SPEED_FACTOR_FLOOR="${LEG_WAVE_SPEED_FACTOR_FLOOR:-0.0}"
WEATHER_MODE="${WEATHER_MODE:-window}"
WEATHER_WINDOW_RATE_PER_WEEK="${WEATHER_WINDOW_RATE_PER_WEEK:-1.0}"
WEATHER_OBS="${WEATHER_OBS:-1}"
WEATHER_OBS_ARGS=()
if [[ "$WEATHER_OBS" == "1" || "$WEATHER_OBS" == "true" || "$WEATHER_OBS" == "TRUE" || "$WEATHER_OBS" == "yes" || "$WEATHER_OBS" == "YES" ]]; then
  WEATHER_OBS_ARGS=(--weather-obs)
fi
cd "$PROJECT_DIR"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs output/rl_ppo

LIVE_LOG="logs/reward_modes_${SLURM_JOB_ID:-manual}.live.log"

echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Yara buffer capacity: $YARA_BUFFER_CAPACITY"
echo "Reward modes: $REWARD_MODES"
echo "BC episodes: $BC_EPISODES"
echo "BC epochs: $BC_EPOCHS"
echo "Non-WAIT weight: $NONWAIT_WEIGHT"
echo "Kickstart coef: $KICKSTART_COEF"
echo "Timesteps: $TIMESTEPS"
echo "Capture noise std: $CAPTURE_NOISE_STD"
echo "Initial inventory fill max: $INITIAL_INVENTORY_FILL_MAX"
echo "Leg-wave slowdown multiplier: $LEG_WAVE_SLOWDOWN_MULTIPLIER"
echo "Leg-wave speed factor floor: $LEG_WAVE_SPEED_FACTOR_FLOOR"
echo "Weather mode: $WEATHER_MODE"
echo "Weather window rate per week: $WEATHER_WINDOW_RATE_PER_WEEK"
echo "Weather obs: $WEATHER_OBS"
which python
python --version
nvidia-smi

# shellcheck disable=SC2086
python -u scripts/compare_reward_modes_bc.py \
  --scenario northern_lights_phase1_3vessels \
  --episode-hours 720 \
  --timesteps "$TIMESTEPS" \
  --bc-episodes "$BC_EPISODES" \
  --bc-epochs "$BC_EPOCHS" \
  --nonwait-weight "$NONWAIT_WEIGHT" \
  --kickstart-coef "$KICKSTART_COEF" \
  --yara-buffer-capacity "$YARA_BUFFER_CAPACITY" \
  --capture-noise-std "$CAPTURE_NOISE_STD" \
  --initial-inventory-fill-max "$INITIAL_INVENTORY_FILL_MAX" \
  --leg-wave-slowdown-multiplier "$LEG_WAVE_SLOWDOWN_MULTIPLIER" \
  --leg-wave-speed-factor-floor "$LEG_WAVE_SPEED_FACTOR_FLOOR" \
  --weather-mode "$WEATHER_MODE" \
  --weather-window-rate-per-week "$WEATHER_WINDOW_RATE_PER_WEEK" \
  "${WEATHER_OBS_ARGS[@]}" \
  --eval-seeds 101 102 103 104 105 \
  --reward-modes $REWARD_MODES \
  --device cuda \
  --verbose 1 \
  2>&1 | tee "$LIVE_LOG"

echo "Job finished at $(date)"
