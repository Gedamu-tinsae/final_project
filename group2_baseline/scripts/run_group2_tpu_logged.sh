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
group1_root="$project_root/group1_baseline"
config_rel="configs/workflow_paths_subset_10000.json"
run_stage4_experiments="${RUN_STAGE4_EXPERIMENTS:-0}"
run_stage5_experiments="${RUN_STAGE5_EXPERIMENTS:-0}"
with_experiments=0
loaded_env_file=""

if [[ "${1:-}" == "--with-experiments" ]]; then
  with_experiments=1
  shift
fi
if [[ "$with_experiments" == "1" ]]; then
  run_stage4_experiments=1
  run_stage5_experiments=1
fi

load_env_file_if_present() {
  local env_file="$1"
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
    loaded_env_file="$env_file"
  fi
}

if [[ -z "${HF_TOKEN:-}" ]]; then
  load_env_file_if_present "$repo_root/.env"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  load_env_file_if_present "$group1_root/.env"
fi

experiment_args=(
  --experiment-epochs "${EXPERIMENT_EPOCHS:-1}"
  --experiment-batch-size "${EXPERIMENT_BATCH_SIZE:-1}"
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
  echo "hf_token_present=$([[ -n \"${HF_TOKEN:-}\" ]] && echo 1 || echo 0)"
  echo "loaded_env_file=${loaded_env_file:-none}"
  echo "experiment_args=${experiment_args[*]}"
  echo "log_file=$log_file"
} > "$meta_file"

if [[ "$with_experiments" == "1" ]]; then
  llama_dir="$group1_root/data/models/Llama-3.2-1B-Instruct"
  projector_subset="$group1_root/artifacts/subsets/subset_10000_seed42/projector_stage1.pkl"
  projector_legacy="$group1_root/artifacts/projector_stage1.pkl"

  if [[ ! -d "$llama_dir" ]]; then
    echo "Missing Group1 LLaMA local dir: $llama_dir" >&2
    echo "Run Group1 bootstrap/stage4.5 first (or sync model artifacts) before Group2 experiments." >&2
    exit 1
  fi
  if [[ ! -f "$projector_subset" && ! -f "$projector_legacy" ]]; then
    echo "Missing Group1 Stage1 projector artifact." >&2
    echo "Expected one of:" >&2
    echo "  - $projector_subset" >&2
    echo "  - $projector_legacy" >&2
    echo "Run Group1 Stage5 first (or sync artifacts) before Group2 experiments." >&2
    exit 1
  fi
fi

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
  stage1236_variants="baseline"
  stage1236_splits="val"
  if [[ "$run_stage4_experiments" == "1" || "$run_stage5_experiments" == "1" ]]; then
    stage1236_variants="all"
    stage1236_splits="train,val"
  fi
  run_logged_workflow "group2_stage1236" --config "$config_rel" --max-rows-guard 10000 --stages 1,2,3,6 --stage2-variants "$stage1236_variants" --stage2-splits "$stage1236_splits" --overwrite
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
