#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SEARCH_ROOT="${SEARCH_ROOT:-$PROJECT_DIR/output/iterative_q_budget_search}"
EVAL_SEEDS="${EVAL_SEEDS:-8100001:8100002:8100003:8100004:8100005:8100006:8100007:8100008:8100009:8100010:8100011:8100012:8100013:8100014:8100015:8100016:8100017:8100018:8100019:8100020}"
POLICY_WINDOWS_H="${POLICY_WINDOWS_H:-108-155:156-203:204-251:252-299:300-347:348-395:396-443:444-491:492-539:540-587:588-635:636-680}"

cd "$PROJECT_DIR"
manifest="$SEARCH_ROOT/p1_eval_jobs.txt"
: > "$manifest"
for run_root in "$SEARCH_ROOT"/runs/*; do
  [[ -d "$run_root" ]] || continue
  config=$(basename "$run_root")
  p1_job=$(awk -F= '$1 == "p1" {print $2}' "$run_root/job_ids.txt")
  p1_state=$(
    sacct -j "$p1_job" -X -n -o State \
      | awk 'NF {print $1; exit}'
  )
  dependency_args=()
  case "$p1_state" in
    COMPLETED)
      ;;
    PENDING | RUNNING | CONFIGURING | COMPLETING)
      dependency_args=(--dependency=afterok:"$p1_job")
      ;;
    *)
      printf 'Refusing %s: P1 job %s state is %s\n' \
        "$config" "$p1_job" "$p1_state" >&2
      exit 1
      ;;
  esac
  eval_name="${config}_p1"
  eval_job=$(
    sbatch --parsable \
      "${dependency_args[@]}" \
      --job-name="${config}_p1e" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$run_root",EVAL_NAME="$eval_name",FINAL_STAGE=p1,SCENARIO_PROTOCOL=unified_window_v1,HARD_SCENARIO_PROBABILITY=0.5,FORECAST_CONTEXT_HOURS=168,FUTURE_SUMMARY_WINDOWS_H="",POLICY_WINDOWS_H="$POLICY_WINDOWS_H",MAX_OVERRIDES=12,EVAL_SEEDS="$EVAL_SEEDS",VALIDATION_ONLY=1 \
      hpc/submit_iterative_q_eval.sh
  )
  printf '%s=%s\n' "$config" "$eval_job" | tee -a "$manifest"
done
