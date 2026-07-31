#!/usr/bin/env bash
#SBATCH --job-name=e1_e7_env
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH -o logs/e1_e7_env-%j.out
#SBATCH -e logs/e1_e7_env-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts

which python
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import numpy, matplotlib, stable_baselines3, sb3_contrib; print('python_dependencies=ok')"
python -c "from types import SimpleNamespace; from experiments.iterative_q_data_common import make_native_env; a=SimpleNamespace(scenario_protocol='unified_window_v1', hard_scenario_probability=0.5, forecast_context_hours=168, scenario_episode_hours=8928, episode_hours=720, stress_level='medium', reward_scale=0.00001); e=make_native_env(a); e.reset(seed=9000031); assert e.n_steps == 720; print('e7_scenario_reset=ok')"
python -m py_compile \
  experiments/run_e1_online_timing.py \
  experiments/aggregate_e1_cost_timing.py \
  experiments/run_e7_temporal_generalization.py \
  experiments/aggregate_e7_temporal_generalization.py \
  experiments/evaluate_iterative_action_q.py \
  experiments/iterative_q_data_common.py \
  src/sim/control/hourly_ppo/train_hourly_ppo.py
bash -n \
  hpc/submit_e1_online_timing.sh \
  hpc/submit_e7_temporal_generalization.sh
test -s experiments_results/E1/models/iterative_q/g60_p4_model_seed_0/iterative_action_q.pt
test -s experiments_results/E1/models/iterative_q/g60_p4_model_seed_1/iterative_action_q.pt
test -s experiments_results/E1/models/iterative_q/g60_p4_model_seed_2/iterative_action_q.pt
test -s experiments_results/E1/models/ppo_hourly/model_seed_0/ppo_hourly_best_validation.zip
test -s experiments_results/E1/models/ppo_high_level/model_seed_0/ppo_high_level_best_validation.zip
test -s experiments_results/E1/models/ppo_event_residual/model_seed_0/event_residual_e1_best_validation.zip
echo "e1_e7_env_check=ok"
