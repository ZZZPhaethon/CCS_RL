#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
P3_P4_RUN_ROOT="${P3_P4_RUN_ROOT:-output/iterative_q_validation_search/uniform_margin40_p1_p4}"
V2_RUN_ROOT="${V2_RUN_ROOT:-output/iterative_q_validation_search/iterative_q_v2_anchor_p4_ablation}"
REPLICATION_ROOT="${REPLICATION_ROOT:-output/iterative_q_validation_search/replication_test_9000031_9000060}"
P3_EXPECTED_SHA256="${P3_EXPECTED_SHA256:-dd125d5fa883c8984c49b7189431eea001c449cf200074361e70f07a9a7e82d3}"
P4_EXPECTED_SHA256="${P4_EXPECTED_SHA256:-f4b66a55beca6fa5f915a6dd7b0bfe6f76d40c1a8c01f07ad3cbaaa1f4e39795}"
V2_EXPECTED_SHA256="${V2_EXPECTED_SHA256:-ea82fa0aec2d312cefac9e4c52a5bdc03e1a21c11a5b21aae70e37013ad4053b}"
P3_EVAL_NAME="${P3_EVAL_NAME:-iterative_q_v1_p3_replication_9000031_9000060}"
P4_EVAL_NAME="${P4_EVAL_NAME:-iterative_q_v1_p4_replication_9000031_9000060}"
V2_EVAL_NAME="${V2_EVAL_NAME:-iterative_q_v2_replication_9000031_9000060}"
EVAL_SEEDS="${EVAL_SEEDS:-9000031:9000032:9000033:9000034:9000035:9000036:9000037:9000038:9000039:9000040:9000041:9000042:9000043:9000044:9000045:9000046:9000047:9000048:9000049:9000050:9000051:9000052:9000053:9000054:9000055:9000056:9000057:9000058:9000059:9000060}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680}"

cd "$PROJECT_DIR"

p3_checkpoint="$P3_P4_RUN_ROOT/p3/iterative_action_q.pt"
p4_checkpoint="$P3_P4_RUN_ROOT/p4/iterative_action_q.pt"
v2_checkpoint="$V2_RUN_ROOT/anchor_100/iterative_action_q.pt"
p3_eval_out_dir="$P3_P4_RUN_ROOT/eval/$P3_EVAL_NAME"
p4_eval_out_dir="$P3_P4_RUN_ROOT/eval/$P4_EVAL_NAME"
v2_eval_out_dir="$V2_RUN_ROOT/eval/$V2_EVAL_NAME"
lock_path="$REPLICATION_ROOT/replication_lock.txt"
manifest_path="$REPLICATION_ROOT/job_manifest.txt"

for checkpoint in "$p3_checkpoint" "$p4_checkpoint" "$v2_checkpoint"; do
  if [[ ! -s "$checkpoint" ]]; then
    echo "Checkpoint is missing: $checkpoint" >&2
    exit 2
  fi
done
for output_path in \
  "$p3_eval_out_dir" "$p4_eval_out_dir" "$v2_eval_out_dir" \
  "$REPLICATION_ROOT"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing repeated replication access: $output_path" >&2
    exit 2
  fi
done

mkdir -p "$REPLICATION_ROOT"
{
  printf 'purpose=locked_out_of_sample_replication\n'
  printf 'replaces_original_formal_test=false\n'
  printf 'used_for_model_selection=false\n'
  printf 'configuration_changed=false\n'
  printf 'p3_checkpoint=%s\n' "$p3_checkpoint"
  printf 'p3_checkpoint_sha256=%s\n' "$P3_EXPECTED_SHA256"
  printf 'p4_checkpoint=%s\n' "$p4_checkpoint"
  printf 'p4_checkpoint_sha256=%s\n' "$P4_EXPECTED_SHA256"
  printf 'v2_checkpoint=%s\n' "$v2_checkpoint"
  printf 'v2_checkpoint_sha256=%s\n' "$V2_EXPECTED_SHA256"
  printf 'model_seed=0\n'
  printf 'inference_required_heads=4\n'
  printf 'inference_margin_reward_units=0.40\n'
  printf 'inference_margin_eur=40000\n'
  printf 'maximum_interventions=12\n'
  printf 'policy_windows_h=%s\n' "$POLICY_WINDOWS_H"
  printf 'scenario_protocol=unified_window_v1\n'
  printf 'hard_scenario_probability=0.5\n'
  printf 'forecast_context_hours=168\n'
  printf 'replication_seeds=%s\n' "$EVAL_SEEDS"
  printf 'replication_count=30\n'
} > "$lock_path"

env_job=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",P3_CHECKPOINT="$p3_checkpoint",P4_CHECKPOINT="$p4_checkpoint",V2_CHECKPOINT="$v2_checkpoint",P3_EXPECTED_SHA256="$P3_EXPECTED_SHA256",P4_EXPECTED_SHA256="$P4_EXPECTED_SHA256",V2_EXPECTED_SHA256="$V2_EXPECTED_SHA256",P3_EVAL_OUT_DIR="$p3_eval_out_dir",P4_EVAL_OUT_DIR="$p4_eval_out_dir",V2_EVAL_OUT_DIR="$v2_eval_out_dir",EVAL_SEEDS="$EVAL_SEEDS" \
  hpc/submit_iterative_q_replication_test_env_check.sh)
p3_eval_job=$(sbatch --parsable \
  --dependency=afterok:"$env_job" \
  --job-name="iterq_v1_p3_repl" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$P3_P4_RUN_ROOT",FINAL_STAGE=p3,EVAL_NAME="$P3_EVAL_NAME",SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,EVAL_SEEDS="$EVAL_SEEDS",VALIDATION_ONLY=0 \
  hpc/submit_iterative_q_eval.sh)
p4_eval_job=$(sbatch --parsable \
  --dependency=afterok:"$env_job" \
  --job-name="iterq_v1_p4_repl" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$P3_P4_RUN_ROOT",FINAL_STAGE=p4,EVAL_NAME="$P4_EVAL_NAME",SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,EVAL_SEEDS="$EVAL_SEEDS",VALIDATION_ONLY=0 \
  hpc/submit_iterative_q_eval.sh)
v2_eval_job=$(sbatch --parsable \
  --dependency=afterok:"$env_job" \
  --job-name="iterq_v2_repl" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$V2_RUN_ROOT",FINAL_STAGE=anchor_100,EVAL_NAME="$V2_EVAL_NAME",SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,EVAL_SEEDS="$EVAL_SEEDS",VALIDATION_ONLY=0 \
  hpc/submit_iterative_q_eval.sh)

{
  printf 'env_check=%s\n' "$env_job"
  printf 'p3_replication_eval=%s\n' "$p3_eval_job"
  printf 'p4_replication_eval=%s\n' "$p4_eval_job"
  printf 'v2_replication_eval=%s\n' "$v2_eval_job"
} | tee "$manifest_path"
