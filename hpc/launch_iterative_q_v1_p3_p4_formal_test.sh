#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
RUN_ROOT="${RUN_ROOT:-output/iterative_q_validation_search/uniform_margin40_p1_p4}"
P3_EXPECTED_SHA256="${P3_EXPECTED_SHA256:-dd125d5fa883c8984c49b7189431eea001c449cf200074361e70f07a9a7e82d3}"
P4_EXPECTED_SHA256="${P4_EXPECTED_SHA256:-f4b66a55beca6fa5f915a6dd7b0bfe6f76d40c1a8c01f07ad3cbaaa1f4e39795}"
P3_EVAL_NAME="${P3_EVAL_NAME:-iterative_q_v1_p3_formal_test}"
P4_EVAL_NAME="${P4_EVAL_NAME:-iterative_q_v1_p4_formal_test}"
EVAL_SEEDS="${EVAL_SEEDS:-9000001:9000002:9000003:9000004:9000005:9000006:9000007:9000008:9000009:9000010:9000011:9000012:9000013:9000014:9000015:9000016:9000017:9000018:9000019:9000020:9000021:9000022:9000023:9000024:9000025:9000026:9000027:9000028:9000029:9000030}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680}"

cd "$PROJECT_DIR"

p3_checkpoint="$RUN_ROOT/p3/iterative_action_q.pt"
p4_checkpoint="$RUN_ROOT/p4/iterative_action_q.pt"
p3_eval_out_dir="$RUN_ROOT/eval/$P3_EVAL_NAME"
p4_eval_out_dir="$RUN_ROOT/eval/$P4_EVAL_NAME"
lock_path="$RUN_ROOT/formal_test_p3_p4_lock.txt"
manifest_path="$RUN_ROOT/formal_test_p3_p4_job_manifest.txt"

for checkpoint in "$p3_checkpoint" "$p4_checkpoint"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "Baseline checkpoint is missing: $checkpoint" >&2
    exit 2
  fi
done
if [[ -e "$p3_eval_out_dir" || -e "$p4_eval_out_dir" \
   || -e "$lock_path" || -e "$manifest_path" ]]; then
  echo "Refusing repeated P3/P4 formal-test access" >&2
  exit 2
fi

{
  printf 'purpose=retrospective_paired_baseline_comparison\n'
  printf 'used_for_v2_selection=false\n'
  printf 'v2_configuration_changed=false\n'
  printf 'p3_checkpoint=%s\n' "$p3_checkpoint"
  printf 'p3_checkpoint_sha256=%s\n' "$P3_EXPECTED_SHA256"
  printf 'p4_checkpoint=%s\n' "$p4_checkpoint"
  printf 'p4_checkpoint_sha256=%s\n' "$P4_EXPECTED_SHA256"
  printf 'model_seed=0\n'
  printf 'inference_required_heads=4\n'
  printf 'inference_margin_reward_units=0.40\n'
  printf 'inference_margin_eur=40000\n'
  printf 'maximum_interventions=12\n'
  printf 'policy_windows_h=%s\n' "$POLICY_WINDOWS_H"
  printf 'scenario_protocol=unified_window_v1\n'
  printf 'hard_scenario_probability=0.5\n'
  printf 'forecast_context_hours=168\n'
  printf 'formal_test_seeds=%s\n' "$EVAL_SEEDS"
  printf 'formal_test_count=30\n'
} > "$lock_path"

env_job=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",P3_CHECKPOINT="$p3_checkpoint",P4_CHECKPOINT="$p4_checkpoint",P3_EXPECTED_SHA256="$P3_EXPECTED_SHA256",P4_EXPECTED_SHA256="$P4_EXPECTED_SHA256",P3_EVAL_OUT_DIR="$p3_eval_out_dir",P4_EVAL_OUT_DIR="$p4_eval_out_dir" \
  hpc/submit_iterative_q_v1_p3_p4_formal_test_env_check.sh)
p3_eval_job=$(sbatch --parsable \
  --dependency=afterok:"$env_job" \
  --job-name="iterq_v1_p3_formal_test" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",FINAL_STAGE=p3,EVAL_NAME="$P3_EVAL_NAME",SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,EVAL_SEEDS="$EVAL_SEEDS",VALIDATION_ONLY=0 \
  hpc/submit_iterative_q_eval.sh)
p4_eval_job=$(sbatch --parsable \
  --dependency=afterok:"$env_job" \
  --job-name="iterq_v1_p4_formal_test" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",FINAL_STAGE=p4,EVAL_NAME="$P4_EVAL_NAME",SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,EVAL_SEEDS="$EVAL_SEEDS",VALIDATION_ONLY=0 \
  hpc/submit_iterative_q_eval.sh)

{
  printf 'env_check=%s\n' "$env_job"
  printf 'p3_formal_test_eval=%s\n' "$p3_eval_job"
  printf 'p4_formal_test_eval=%s\n' "$p4_eval_job"
} | tee "$manifest_path"
