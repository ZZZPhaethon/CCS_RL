#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
OUT_ROOT="experiments_results/E2/validation_gate_sweep_seeds_8100001-8100020_run01"

cd "$PROJECT_DIR"
mkdir -p logs
if [[ -e "$OUT_ROOT" ]]; then
  echo "Refusing existing validation gate-sweep root: $OUT_ROOT" >&2
  exit 2
fi

submit_job() {
  local submitted
  submitted=$(sbatch --parsable "$@")
  printf '%s\n' "${submitted%%;*}"
}

sweep_job=$(submit_job \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR" \
  hpc/submit_e2_gate_sweep_validation.sh)
aggregate_job=$(submit_job \
  --dependency=afterok:"$sweep_job" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",FOLLOWUP_ANALYSIS=gate_sweep \
  hpc/submit_e2_followup_aggregate.sh)

{
  printf 'validation_gate_sweep=%s\n' "$sweep_job"
  printf 'validation_gate_sweep_aggregate=%s\n' "$aggregate_job"
} | tee -a experiments_results/e2_followup_job_manifest.txt
