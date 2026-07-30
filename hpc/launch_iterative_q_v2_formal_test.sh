#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
RUN_ROOT="${RUN_ROOT:-output/iterative_q_validation_search/iterative_q_v2_anchor_p4_ablation}"
FINAL_STAGE="${FINAL_STAGE:-anchor_100}"
EVAL_NAME="${EVAL_NAME:-iterative_q_v2_formal_test}"
EXPECTED_SHA256="${EXPECTED_SHA256:-ea82fa0aec2d312cefac9e4c52a5bdc03e1a21c11a5b21aae70e37013ad4053b}"
EVAL_SEEDS="${EVAL_SEEDS:-9000001:9000002:9000003:9000004:9000005:9000006:9000007:9000008:9000009:9000010:9000011:9000012:9000013:9000014:9000015:9000016:9000017:9000018:9000019:9000020:9000021:9000022:9000023:9000024:9000025:9000026:9000027:9000028:9000029:9000030}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680}"

cd "$PROJECT_DIR"

checkpoint="$RUN_ROOT/$FINAL_STAGE/iterative_action_q.pt"
eval_out_dir="$RUN_ROOT/eval/$EVAL_NAME"
lock_path="$RUN_ROOT/formal_test_lock.txt"
manifest_path="$RUN_ROOT/formal_test_job_manifest.txt"

if [[ ! -s "$checkpoint" ]]; then
  echo "Selected checkpoint is missing: $checkpoint" >&2
  exit 2
fi
if [[ -e "$eval_out_dir" || -e "$lock_path" || -e "$manifest_path" ]]; then
  echo "Refusing repeated formal-test access for $RUN_ROOT" >&2
  exit 2
fi

{
  printf 'access=authorized_one_shot_formal_test\n'
  printf 'model=Iterative Q v2 hard-anchor\n'
  printf 'checkpoint=%s\n' "$checkpoint"
  printf 'checkpoint_sha256=%s\n' "$EXPECTED_SHA256"
  printf 'model_seed=0\n'
  printf 'anchor_weighting=hard\n'
  printf 'anchor_coefficient=1.0\n'
  printf 'anchor_release_margin_eur=40000\n'
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
  printf 'test_set_used_for_selection=false\n'
  printf 'continuous_teacher_evaluated=false\n'
} > "$lock_path"

env_job=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CHECKPOINT="$checkpoint",EXPECTED_SHA256="$EXPECTED_SHA256",EVAL_OUT_DIR="$eval_out_dir" \
  hpc/submit_iterative_q_v2_formal_test_env_check.sh)
eval_job=$(sbatch --parsable \
  --dependency=afterok:"$env_job" \
  --job-name="iterq_v2_formal_test" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",FINAL_STAGE="$FINAL_STAGE",EVAL_NAME="$EVAL_NAME",SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168,POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,EVAL_SEEDS="$EVAL_SEEDS",VALIDATION_ONLY=0 \
  hpc/submit_iterative_q_eval.sh)

{
  printf 'env_check=%s\n' "$env_job"
  printf 'formal_test_eval=%s\n' "$eval_job"
} | tee "$manifest_path"
