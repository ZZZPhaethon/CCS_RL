#!/usr/bin/env bash
#SBATCH --job-name=e34_nohour_env
#SBATCH --partition=root
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -o logs/e34_nohour_env-%j.out
#SBATCH -e logs/e34_nohour_env-%j.err

set -euo pipefail
source /scratch_root/hx721/miniconda3/etc/profile.d/conda.sh
conda activate mas-ccus

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
FORMAL_MODEL_ROOT="${FORMAL_MODEL_ROOT:-output/E1_hour_removed_formal_models}"
ONE_SHOT_ROOT="${ONE_SHOT_ROOT:-experiments_results/E2/training_one_shot_hour_removed_budget_matched_20260730_run01}"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONPATH=src:.:scripts
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

which python
python --version
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())"
python -m py_compile \
  scripts/train_iterative_action_q.py \
  experiments/evaluate_iterative_action_q.py \
  experiments/run_e7_temporal_generalization.py
bash -n \
  hpc/submit_e3_future_information_train.sh \
  hpc/launch_e3_e4_no_hour_refresh.sh \
  hpc/submit_e7_temporal_generalization.sh
python - "$FORMAL_MODEL_ROOT" "$ONE_SHOT_ROOT" <<'PY'
import sys
from pathlib import Path

import torch

for root, stage in ((Path(sys.argv[1]), "iterative"), (Path(sys.argv[2]), "one_shot")):
    for seed in (0, 1, 2):
        path = (
            root / f"g60_p4_model_seed_{seed}" / "iterative_action_q.pt"
            if stage == "iterative"
            else root / f"model_seed_{seed}" / "p1" / "iterative_action_q.pt"
        )
        metadata = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )["metadata"]
        if len(metadata["state_feature_names"]) != 93:
            raise RuntimeError(f"unexpected state width: {path}")
        if metadata.get("excluded_state_feature_names") != ["hour_of_week"]:
            raise RuntimeError(f"unexpected state exclusion: {path}")
        print("checkpoint_ok", stage, seed, path)
PY
echo "e3_e4_no_hour_env_check=ok"
