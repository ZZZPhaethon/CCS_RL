#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch_root/hx721/CCS_RLLLM_iterq_validation_search_20260728}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-output/iterative_q_budget_search/runs/g60_p4}"
E1_REPLICATION_ROOT="${E1_REPLICATION_ROOT:-experiments_results/E1/training_iterative_action_q_g60_p4_full_model_seeds_1_2_20260729}"
E2_ROOT="experiments_results/E2"
E3_ROOT="experiments_results/E3"
E4_ROOT="experiments_results/E4"

cd "$PROJECT_DIR"
mkdir -p logs

for required in \
  "$SOURCE_RUN_ROOT/p1/iterative_action_q.pt" \
  "$SOURCE_RUN_ROOT/p2/iterative_action_q.pt" \
  "$SOURCE_RUN_ROOT/p3/iterative_action_q.pt" \
  "$SOURCE_RUN_ROOT/p4/iterative_action_q.pt" \
  "$E1_REPLICATION_ROOT/model_seed_1/p1/iterative_action_q.pt" \
  "$E1_REPLICATION_ROOT/model_seed_1/p2/iterative_action_q.pt" \
  "$E1_REPLICATION_ROOT/model_seed_1/p3/iterative_action_q.pt" \
  "$E1_REPLICATION_ROOT/model_seed_1/p4/iterative_action_q.pt" \
  "$E1_REPLICATION_ROOT/model_seed_2/p1/iterative_action_q.pt" \
  "$E1_REPLICATION_ROOT/model_seed_2/p2/iterative_action_q.pt" \
  "$E1_REPLICATION_ROOT/model_seed_2/p3/iterative_action_q.pt" \
  "$E1_REPLICATION_ROOT/model_seed_2/p4/iterative_action_q.pt" \
  "$SOURCE_RUN_ROOT/eval/formal_9000031_9000060_cost_fields_run02/evaluation.csv" \
  "$E1_REPLICATION_ROOT/model_seed_1/eval/formal_9000031_9000060_cost_fields_run01/evaluation.csv" \
  "$E1_REPLICATION_ROOT/model_seed_2/eval/formal_9000031_9000060_cost_fields_run01/evaluation.csv"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing required E1 artifact: $required" >&2
    exit 2
  fi
done

for result_root in "$E2_ROOT" "$E3_ROOT" "$E4_ROOT"; do
  if [[ -e "$result_root" ]]; then
    echo "Refusing to overwrite existing result root: $result_root" >&2
    exit 2
  fi
done

/scratch_root/hx721/miniconda3/envs/mas-ccus/bin/python -c "import json; p=json.load(open('experiments/protocols/e2_e3_e4_iterative_q_protocol.json')); assert p['formal_test'] == {'range_inclusive': [9000031, 9000060], 'count': 30, 'paired_across_model_seeds': True, 'paired_across_stress_levels': True}"
mkdir -p "$E2_ROOT" "$E3_ROOT" "$E4_ROOT"

submit_job() {
  local submitted
  submitted=$(sbatch --parsable "$@")
  printf '%s\n' "${submitted%%;*}"
}

checkpoint_for() {
  local model_seed="$1"
  local stage="$2"
  if [[ "$model_seed" == "0" ]]; then
    printf '%s\n' "$PROJECT_DIR/$SOURCE_RUN_ROOT/$stage/iterative_action_q.pt"
  else
    printf '%s\n' "$PROJECT_DIR/$E1_REPLICATION_ROOT/model_seed_${model_seed}/$stage/iterative_action_q.pt"
  fi
}

e1_formal_dir_for() {
  local model_seed="$1"
  if [[ "$model_seed" == "0" ]]; then
    printf '%s\n' "$SOURCE_RUN_ROOT/eval/formal_9000031_9000060_cost_fields_run02"
  else
    printf '%s\n' "$E1_REPLICATION_ROOT/model_seed_${model_seed}/eval/formal_9000031_9000060_cost_fields_run01"
  fi
}

join_jobs() {
  local IFS=:
  printf '%s\n' "$*"
}

# Reuse the completed E1 medium-stress P4 results inside E3 and E4.
for model_seed in 0 1 2; do
  e1_source=$(e1_formal_dir_for "$model_seed")
  e3_dest="$E3_ROOT/formal_future_information_seeds_9000031-9000060_run01/structured_summary_168/model_seed_${model_seed}"
  e4_dest="$E4_ROOT/formal_stress_seeds_9000031-9000060_run01/medium/model_seed_${model_seed}"
  mkdir -p "$e3_dest" "$e4_dest"
  cp -a "$e1_source/." "$e3_dest/"
  cp -a "$e1_source/." "$e4_dest/"
done

# E2: evaluate the P1-P4 lineage of the exact E1 G60-P4 models.
e2_stage_jobs=()
for stage in p1 p2 p3 p4; do
  for model_seed in 0 1 2; do
    checkpoint=$(checkpoint_for "$model_seed" "$stage")
    out_dir="$PROJECT_DIR/$E2_ROOT/formal_iterative_q_stages_seeds_9000031-9000060_run01/$stage/model_seed_${model_seed}"
    job=$(submit_job \
      --job-name="e2_${stage}_s${model_seed}" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",CHECKPOINT="$checkpoint",OUT_DIR="$out_dir",EVAL_NAME="e2_${stage}_s${model_seed}",STRESS_LEVEL=medium \
      hpc/submit_locked_iterative_q_eval.sh)
    e2_stage_jobs+=("$job")
  done
done

# E2: build and train the Greedy-only matched-budget one-shot control.
e2_data_job=$(submit_job \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$E2_ROOT/training_one_shot_matched_run01" \
  hpc/submit_e2_one_shot_extra_data.sh)
e2_merge_job=$(submit_job \
  --dependency=afterok:"$e2_data_job" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",SOURCE_RUN_ROOT="$SOURCE_RUN_ROOT",RUN_ROOT="$E2_ROOT/training_one_shot_matched_run01" \
  hpc/submit_e2_one_shot_merge.sh)
e2_train_job=$(submit_job \
  --dependency=afterok:"$e2_merge_job" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",RUN_ROOT="$E2_ROOT/training_one_shot_matched_run01" \
  hpc/submit_e2_one_shot_train.sh)
e2_matched_eval_jobs=()
for model_seed in 0 1 2; do
  checkpoint="$PROJECT_DIR/$E2_ROOT/training_one_shot_matched_run01/model_seed_${model_seed}/p1/iterative_action_q.pt"
  out_dir="$PROJECT_DIR/$E2_ROOT/formal_one_shot_matched_seeds_9000031-9000060_run01/model_seed_${model_seed}"
  job=$(submit_job \
    --dependency=afterok:"$e2_train_job" \
    --job-name="e2_one_s${model_seed}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",CHECKPOINT="$checkpoint",OUT_DIR="$out_dir",EVAL_NAME="e2_one_shot_matched_s${model_seed}",STRESS_LEVEL=medium \
    hpc/submit_locked_iterative_q_eval.sh)
  e2_matched_eval_jobs+=("$job")
done
e2_all_eval_jobs=("${e2_stage_jobs[@]}" "${e2_matched_eval_jobs[@]}")
e2_aggregate_job=$(submit_job \
  --dependency=afterok:"$(join_jobs "${e2_all_eval_jobs[@]}")" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",EXPERIMENT=E2 \
  hpc/submit_e2_e3_e4_aggregate.sh)

# E3: augment the frozen roots, then train State-only and full-sequence variants.
e3_augment_job=$(submit_job \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",SOURCE_RUN_ROOT="$SOURCE_RUN_ROOT",OUT_ROOT="$E3_ROOT/training_future_information_run01/augmented_data" \
  hpc/submit_e3_forecast_augment.sh)
e3_train_job=$(submit_job \
  --dependency=afterok:"$e3_augment_job" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",SOURCE_RUN_ROOT="$SOURCE_RUN_ROOT",E3_ROOT="$E3_ROOT/training_future_information_run01" \
  hpc/submit_e3_future_information_train.sh)
e3_eval_jobs=()
for variant in state_only full_sequence_168; do
  for model_seed in 0 1 2; do
    checkpoint="$PROJECT_DIR/$E3_ROOT/training_future_information_run01/$variant/model_seed_${model_seed}/p4/iterative_action_q.pt"
    out_dir="$PROJECT_DIR/$E3_ROOT/formal_future_information_seeds_9000031-9000060_run01/$variant/model_seed_${model_seed}"
    job=$(submit_job \
      --dependency=afterok:"$e3_train_job" \
      --job-name="e3_${variant:0:4}_s${model_seed}" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",CHECKPOINT="$checkpoint",OUT_DIR="$out_dir",EVAL_NAME="e3_${variant}_s${model_seed}",STRESS_LEVEL=medium \
      hpc/submit_locked_iterative_q_eval.sh)
    e3_eval_jobs+=("$job")
  done
done
e3_aggregate_job=$(submit_job \
  --dependency=afterok:"$(join_jobs "${e3_eval_jobs[@]}")" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",EXPERIMENT=E3 \
  hpc/submit_e2_e3_e4_aggregate.sh)

# E4: keep the E1 weights and gate frozen; only Low and High are new runs.
e4_eval_jobs=()
for stress in low high; do
  for model_seed in 0 1 2; do
    checkpoint=$(checkpoint_for "$model_seed" p4)
    out_dir="$PROJECT_DIR/$E4_ROOT/formal_stress_seeds_9000031-9000060_run01/$stress/model_seed_${model_seed}"
    job=$(submit_job \
      --job-name="e4_${stress}_s${model_seed}" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",CHECKPOINT="$checkpoint",OUT_DIR="$out_dir",EVAL_NAME="e4_${stress}_s${model_seed}",STRESS_LEVEL="$stress" \
      hpc/submit_locked_iterative_q_eval.sh)
    e4_eval_jobs+=("$job")
  done
done
e4_aggregate_job=$(submit_job \
  --dependency=afterok:"$(join_jobs "${e4_eval_jobs[@]}")" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",EXPERIMENT=E4 \
  hpc/submit_e2_e3_e4_aggregate.sh)

manifest="$PROJECT_DIR/experiments_results/e2_e3_e4_job_manifest.txt"
{
  printf 'formal_test_seeds=9000031-9000060\n'
  printf 'e2_stage_eval_jobs=%s\n' "$(join_jobs "${e2_stage_jobs[@]}")"
  printf 'e2_data=%s\n' "$e2_data_job"
  printf 'e2_merge=%s\n' "$e2_merge_job"
  printf 'e2_train=%s\n' "$e2_train_job"
  printf 'e2_matched_eval_jobs=%s\n' "$(join_jobs "${e2_matched_eval_jobs[@]}")"
  printf 'e2_aggregate=%s\n' "$e2_aggregate_job"
  printf 'e3_augment=%s\n' "$e3_augment_job"
  printf 'e3_train=%s\n' "$e3_train_job"
  printf 'e3_eval_jobs=%s\n' "$(join_jobs "${e3_eval_jobs[@]}")"
  printf 'e3_aggregate=%s\n' "$e3_aggregate_job"
  printf 'e4_eval_jobs=%s\n' "$(join_jobs "${e4_eval_jobs[@]}")"
  printf 'e4_aggregate=%s\n' "$e4_aggregate_job"
} | tee "$manifest"
