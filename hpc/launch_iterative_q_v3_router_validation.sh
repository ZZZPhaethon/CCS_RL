#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
RUN_ROOT="${RUN_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"
RESULT_ROOT="$RUN_ROOT/eval/v3_router_validation"
MANIFEST="$RESULT_ROOT/job_manifest.txt"

cd "$PROJECT_DIR"
if [[ -e "$RESULT_ROOT" ]]; then
  echo "Refusing repeated router-validation access: $RESULT_ROOT" >&2
  exit 2
fi
mkdir -p "$RESULT_ROOT" logs

ROUTERS=(
  "p4_reference:confidence:p4:4:0.40:0.0"
  "p3_p4_confidence:confidence:p3,p4:4:0.40:0.0"
  "all_confidence:confidence:p1,p2,p3,p4:4:0.40:0.0"
  "p3_p4_lcb:confidence:p3,p4:4:0.40:1.0"
  "all_lcb:confidence:p1,p2,p3,p4:4:0.40:1.0"
  "p3_p4_pooled:pooled:p3,p4:8:0.40:0.0"
  "all_pooled:pooled:p1,p2,p3,p4:16:0.40:0.0"
)

{
  printf 'kind=iterative_q_v3_router_validation\n'
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'validation_seeds=8100001-8100020\n'
  printf 'formal_test_accessed=false\n'
  printf 'router_count=%s\n' "${#ROUTERS[@]}"
} > "$MANIFEST"

for router in "${ROUTERS[@]}"; do
  name="${router%%:*}"
  job_id=$(sbatch --parsable \
    --job-name="iterq_v3_${name}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$RUN_ROOT",ROUTER_SPEC="$router" \
    hpc/submit_iterative_q_v3_router_validation.sh)
  printf '%s=%s\n' "$name" "${job_id%%;*}" | tee -a "$MANIFEST"
done
