#!/usr/bin/env bash
#SBATCH --job-name=ccs_mpc_demos
#SBATCH --partition=root
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o logs/mpc_demos-%j.out
#SBATCH -e logs/mpc_demos-%j.err

set -euo pipefail

source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM}"
cd "$PROJECT_DIR"

export PYTHONPATH="src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

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

DEMO_CACHE="${DEMO_CACHE:-output/rl_forecast/demos/mpc_720h_30eps.npz}"
EPISODE_HOURS="${EPISODE_HOURS:-720}"
DEMO_SEEDS="${DEMO_SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29}"
read -r -a DEMO_SEED_ARGS <<< "$DEMO_SEEDS"

mkdir -p logs output/rl_forecast/demos

echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Project directory: $PROJECT_DIR"
echo "Git commit: $GIT_COMMIT"
echo "Demo cache: $DEMO_CACHE"
echo "Episode hours: $EPISODE_HOURS"
echo "Demo seeds: $DEMO_SEEDS"
which python
python --version

# The runner locks forecast horizon 168h, block weather, vent-first reward,
# partial dispatch, and the 889h demonstration environment (720 + 168 + 1).
python -u scripts/compare_forecast_encoders_rl.py generate-demos \
  --demo-cache "$DEMO_CACHE" \
  --demo-seeds "${DEMO_SEED_ARGS[@]}" \
  --episode-hours "$EPISODE_HOURS"

echo "Job finished at $(date)"

# Smoke example (creates the one-seed cache required by the training smoke run):
# sbatch --qos=short --time=01:00:00 --export=ALL,DEMO_SEEDS=0,DEMO_CACHE=output/rl_forecast/demos/mpc_smoke.npz hpc/submit_forecast_mpc_demos.sh
