#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
G60_RUN_ROOT="${G60_RUN_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"
PLATEAU_RUN_ROOT="${PLATEAU_RUN_ROOT:-output/iterative_q_validation_search/iterative_q_v2_teacher_from_p2_plateau20_50}"
RESULT_ROOT="${RESULT_ROOT:-output/iterative_q_validation_search/g60_plateau_test_9000031_9000060}"
G60_EVAL_NAME="${G60_EVAL_NAME:-g60_p4_test_9000031_9000060}"
PLATEAU_EVAL_NAME="${PLATEAU_EVAL_NAME:-iterative_q_v2_teacher_from_p2_plateau20_50_p4_test_9000031_9000060}"
G60_EXPECTED_SHA256="${G60_EXPECTED_SHA256:-e529eb06038a4842f58eb97a912ad0f72de9000f03d39d9bc8839e8690febedc}"
PLATEAU_EXPECTED_SHA256="${PLATEAU_EXPECTED_SHA256:-8439aab87231419e3d57a67d59cd862dd55e0541044c989e6c45c34268138160}"
EVAL_SEEDS="${EVAL_SEEDS:-9000031:9000032:9000033:9000034:9000035:9000036:9000037:9000038:9000039:9000040:9000041:9000042:9000043:9000044:9000045:9000046:9000047:9000048:9000049:9000050:9000051:9000052:9000053:9000054:9000055:9000056:9000057:9000058:9000059:9000060}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680}"

cd "$PROJECT_DIR"

g60_checkpoint="$G60_RUN_ROOT/p4/iterative_action_q.pt"
plateau_checkpoint="$PLATEAU_RUN_ROOT/p4/iterative_action_q.pt"
g60_eval_dir="$G60_RUN_ROOT/eval/$G60_EVAL_NAME"
plateau_eval_dir="$PLATEAU_RUN_ROOT/eval/$PLATEAU_EVAL_NAME"

for checkpoint in "$g60_checkpoint" "$plateau_checkpoint"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "Checkpoint is missing: $checkpoint" >&2
    exit 2
  fi
done
if [[ "$(sha256sum "$g60_checkpoint" | awk '{print $1}')" != "$G60_EXPECTED_SHA256" ]]; then
  echo "G60-P4 checkpoint SHA256 mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "$plateau_checkpoint" | awk '{print $1}')" != "$PLATEAU_EXPECTED_SHA256" ]]; then
  echo "Plateau P4 checkpoint SHA256 mismatch" >&2
  exit 2
fi
grep -qx 'uniform_residual_margin=0.40' output/iterative_q_budget_search/protocol_lock.txt
grep -qx 'uniform_economic_margin_eur=40000' output/iterative_q_budget_search/protocol_lock.txt

for output_path in "$g60_eval_dir" "$plateau_eval_dir" "$RESULT_ROOT"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite existing output: $output_path" >&2
    exit 2
  fi
done

mkdir -p "$RESULT_ROOT"
{
  printf 'purpose=paired_test_evaluation\n'
  printf 'used_for_training=false\n'
  printf 'configuration_changed=false\n'
  printf 'g60_uniform_residual_margin=0.40\n'
  printf 'g60_uniform_economic_margin_eur=40000\n'
  printf 'g60_checkpoint=%s\n' "$g60_checkpoint"
  printf 'g60_checkpoint_sha256=%s\n' "$G60_EXPECTED_SHA256"
  printf 'plateau_checkpoint=%s\n' "$plateau_checkpoint"
  printf 'plateau_checkpoint_sha256=%s\n' "$PLATEAU_EXPECTED_SHA256"
  printf 'plateau_weighting=plateau_linear\n'
  printf 'plateau_margin_eur=20000\n'
  printf 'plateau_release_margin_eur=50000\n'
  printf 'model_seed=0\n'
  printf 'inference_required_heads=4\n'
  printf 'inference_margin_reward_units=0.40\n'
  printf 'inference_margin_eur=40000\n'
  printf 'maximum_interventions=12\n'
  printf 'policy_windows_h=%s\n' "$POLICY_WINDOWS_H"
  printf 'scenario_protocol=unified_window_v1\n'
  printf 'hard_scenario_probability=0.5\n'
  printf 'forecast_context_hours=168\n'
  printf 'test_seeds=%s\n' "$EVAL_SEEDS"
  printf 'test_count=30\n'
} > "$RESULT_ROOT/protocol_lock.txt"

env_job=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",DATA_ROOT="$PLATEAU_RUN_ROOT" \
  hpc/submit_iterative_q_v2_env_check.sh)
g60_job=$(sbatch --parsable \
  --dependency=afterok:"$env_job" \
  --job-name="g60_p4_test" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$G60_RUN_ROOT",FINAL_STAGE=p4,EVAL_NAME="$G60_EVAL_NAME",SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,EVAL_SEEDS="$EVAL_SEEDS",VALIDATION_ONLY=0 \
  hpc/submit_iterative_q_eval.sh)
plateau_job=$(sbatch --parsable \
  --dependency=afterok:"$env_job" \
  --job-name="plateau_p4_test" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$PLATEAU_RUN_ROOT",FINAL_STAGE=p4,EVAL_NAME="$PLATEAU_EVAL_NAME",SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,EVAL_SEEDS="$EVAL_SEEDS",VALIDATION_ONLY=0 \
  hpc/submit_iterative_q_eval.sh)

{
  printf 'env_check=%s\n' "$env_job"
  printf 'g60_p4_test=%s\n' "$g60_job"
  printf 'plateau_p4_test=%s\n' "$plateau_job"
} | tee "$RESULT_ROOT/job_manifest.txt"
