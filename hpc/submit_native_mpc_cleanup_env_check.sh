#!/usr/bin/env bash
#SBATCH --job-name=native_mpc_cleanup_check
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH -o logs/native_mpc_cleanup_check-%j.out
#SBATCH -e logs/native_mpc_cleanup_check-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_native_mpc_cleanup_20260728}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

which python
python --version
python - <<'PY'
import pulp

print("cbc_available", pulp.PULP_CBC_CMD(msg=False).available())
if not pulp.PULP_CBC_CMD(msg=False).available():
    raise RuntimeError("Terminal cleanup evaluation requires CBC.")
PY

python -m unittest tests.test_rolling_milp.NativeMpcTests -q
python -u experiments/run_native_mpc_cleanup.py \
  --out-dir "output/native_mpc_cleanup_smoke_${SLURM_JOB_ID}" \
  --seed 8100001 \
  --episode-hours 24 \
  --forecast-context-hours 24 \
  --replan-hours 24 \
  --planning-horizon-hours 24
