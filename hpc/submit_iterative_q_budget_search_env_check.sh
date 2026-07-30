#!/usr/bin/env bash
#SBATCH --job-name=iterq_budget_env
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o logs/iterq_budget_env-%j.out
#SBATCH -e logs/iterq_budget_env-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
cd "$PROJECT_DIR"
export PYTHONPATH=src:.

which python
python --version
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "from experiments.merge_iterative_q_data import merge_shards; from experiments.summarize_iterative_q_budget import summarize; print('budget_modules=ok')"
bash -n hpc/launch_iterative_action_q.sh
bash -n hpc/launch_iterative_q_budget_search.sh
bash -n hpc/submit_iterative_q_budget_prepare_shared.sh
bash -n hpc/submit_iterative_q_budget_prepare_g0_pools.sh
bash -n hpc/submit_iterative_q_budget_audit.sh
test -s output/iterative_q_validation_search/baseline_p1_p4/g0/train/shard_1500_1509.npz
test -s output/iterative_q_validation_search/uniform_margin40_p1_p4/p1/iterative_action_q.pt
echo "iterative_q_budget_search_env_check=ok"
