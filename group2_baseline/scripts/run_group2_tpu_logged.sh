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
config_rel="configs/workflow_paths_subset_10000.json"

if [[ ! -x "$python_bin" ]]; then
  echo "Missing python executable: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$entry_script" ]]; then
  echo "Missing entry script: $entry_script" >&2
  exit 1
fi

{
  echo "timestamp=$ts"
  echo "run_tag=$run_tag"
  echo "repo_root=$repo_root"
  echo "project_root=$project_root"
  echo "python_bin=$python_bin"
  echo "entry_script=$entry_script"
  echo "config_rel=$config_rel"
  echo "log_file=$log_file"
} > "$meta_file"

run_logged() {
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

if [[ "$#" -gt 0 ]]; then
  echo "custom_args=$*" >> "$meta_file"
  run_logged "group2_custom" "$@"
else
  run_logged "group2_stage1236" --config "$config_rel" --max-rows-guard 10000 --stages 1,2,3,6 --stage2-variants baseline --stage2-splits val --overwrite
  run_logged "group2_stage4" --config "$config_rel" --max-rows-guard 10000 --stages 4 --overwrite
  run_logged "group2_stage5" --config "$config_rel" --max-rows-guard 10000 --stages 5 --stage5-prepare-inputs --overwrite
fi

echo "exit_code=0" >> "$meta_file"
echo "log_file=$log_file"
echo "meta_file=$meta_file"
exit 0
