#!/usr/bin/env bash
#SBATCH --job-name=e234_env
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o logs/e234_env-%j.out
#SBATCH -e logs/e234_env-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.

which python
python --version
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import json; p=json.load(open('experiments/protocols/e2_e3_e4_iterative_q_protocol.json')); assert p['formal_test']['range_inclusive'] == [9000031, 9000060] and p['formal_test']['count'] == 30"
python -c "from sim.control.event_based.residual_rl_v4.scenario import unified_window_scenario_config; assert unified_window_scenario_config(888, 'low').weather_window_mean_hours == 24.0; assert unified_window_scenario_config(888, 'medium').weather_window_mean_hours == 48.0; assert unified_window_scenario_config(888, 'high').weather_window_mean_hours == 96.0"
python -m py_compile \
  experiments/aggregate_e2_e3_e4.py \
  experiments/evaluate_iterative_action_q.py \
  experiments/summarize_e2_matched_budget.py
bash -n \
  hpc/launch_e2_e3_e4.sh \
  hpc/submit_locked_iterative_q_eval.sh \
  hpc/submit_e2_one_shot_extra_data.sh \
  hpc/submit_e2_one_shot_merge.sh \
  hpc/submit_e2_one_shot_train.sh \
  hpc/submit_e3_forecast_augment.sh \
  hpc/submit_e3_future_information_train.sh \
  hpc/submit_e2_e3_e4_aggregate.sh
test -s output/iterative_q_budget_search/runs/g60_p4/p4/iterative_action_q.pt
echo "e2_e3_e4_env_check=ok"
