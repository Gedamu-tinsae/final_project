#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
project_root="$(cd "$repo_root/.." && pwd)"
log_root="$project_root/logs/runs"
mkdir -p "$log_root"

ts="$(date +%Y%m%d_%H%M%S)"
run_tag="group2_tpu"
log_file="$log_root/${ts}_${run_tag}.log"
meta_file="$log_root/${ts}_${run_tag}.meta.txt"

python_bin="$project_root/group1_baseline/.venv/bin/python"
entry_script="$repo_root/scripts/run_group2_workflow.py"
subset_script="$repo_root/scripts/create_stage2_subset_profile.py"
config_rel="configs/workflow_paths_subset_10000.json"
run_stage4_experiments="${RUN_STAGE4_EXPERIMENTS:-0}"
run_stage5_experiments="${RUN_STAGE5_EXPERIMENTS:-0}"
with_experiments=0

if [[ "${1:-}" == "--with-experiments" ]]; then
  with_experiments=1
  shift
fi
if [[ "$with_experiments" == "1" ]]; then
  run_stage4_experiments=1
  run_stage5_experiments=1
fi

experiment_args=(
  --experiment-epochs "${EXPERIMENT_EPOCHS:-1}"
  --experiment-batch-size "${EXPERIMENT_BATCH_SIZE:-8}"
  --experiment-log-every-steps "${EXPERIMENT_LOG_EVERY_STEPS:-20}"
  --experiment-learning-rate "${EXPERIMENT_LEARNING_RATE:-2e-5}"
  --experiment-weight-decay "${EXPERIMENT_WEIGHT_DECAY:-0.0}"
  --experiment-dtype "${EXPERIMENT_DTYPE:-bfloat16}"
)

if [[ ! -x "$python_bin" ]]; then
  echo "Missing python executable: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$entry_script" ]]; then
  echo "Missing entry script: $entry_script" >&2
  exit 1
fi
if [[ ! -f "$subset_script" ]]; then
  echo "Missing subset script: $subset_script" >&2
  exit 1
fi

{
  echo "timestamp=$ts"
  echo "run_tag=$run_tag"
  echo "repo_root=$repo_root"
  echo "project_root=$project_root"
  echo "python_bin=$python_bin"
  echo "entry_script=$entry_script"
  echo "subset_script=$subset_script"
  echo "config_rel=$config_rel"
  echo "run_stage4_experiments=$run_stage4_experiments"
  echo "run_stage5_experiments=$run_stage5_experiments"
  echo "with_experiments=$with_experiments"
  echo "experiment_args=${experiment_args[*]}"
  echo "log_file=$log_file"
} > "$meta_file"

run_logged_workflow() {
  local label="$1"
  shift
  echo "==== ${label} ====" | tee -a "$log_file"
  set +e
  /usr/bin/time -v "$python_bin" "$entry_script" "$@" 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  echo "${label}_exit_code=${status}" >> "$meta_file"
  if [[ $status -ne 0 ]]; then
    echo "exit_code=$status" >> "$meta_file"
    echo "log_file=$log_file"
    echo "meta_file=$meta_file"
    exit "$status"
  fi
}

run_logged_subset() {
  local label="$1"
  shift
  echo "==== ${label} ====" | tee -a "$log_file"
  set +e
  /usr/bin/time -v "$python_bin" "$subset_script" "$@" 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  echo "${label}_exit_code=${status}" >> "$meta_file"
  if [[ $status -ne 0 ]]; then
    echo "exit_code=$status" >> "$meta_file"
    echo "log_file=$log_file"
    echo "meta_file=$meta_file"
    exit "$status"
  fi
}

run_logged_subset "group2_subset10k" --config "$config_rel" --rows 10000 --seed 42 --overwrite

if [[ "$#" -gt 0 ]]; then
  echo "custom_args=$*" >> "$meta_file"
  run_logged_workflow "group2_custom" "$@"
else
  run_logged_workflow "group2_stage1236" --config "$config_rel" --max-rows-guard 10000 --stages 1,2,3,6 --stage2-variants baseline --stage2-splits val --overwrite
  stage4_args=(--config "$config_rel" --max-rows-guard 10000 --stages 4 --overwrite)
  stage5_args=(--config "$config_rel" --max-rows-guard 10000 --stages 5 --stage5-prepare-inputs --overwrite)

  if [[ "$run_stage4_experiments" == "1" ]]; then
    stage4_args+=(--stage4-run-experiments --stage4-run-all-missing "${experiment_args[@]}")
  fi
  if [[ "$run_stage5_experiments" == "1" ]]; then
    stage5_args+=(--stage5-run-experiments "${experiment_args[@]}")
  fi

  run_logged_workflow "group2_stage4" "${stage4_args[@]}"
  run_logged_workflow "group2_stage5" "${stage5_args[@]}"
fi

echo "exit_code=0" >> "$meta_file"
echo "log_file=$log_file"
echo "meta_file=$meta_file"
exit 0
