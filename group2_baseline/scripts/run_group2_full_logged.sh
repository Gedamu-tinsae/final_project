#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
project_root="$(cd "$repo_root/.." && pwd)"
log_root="$project_root/logs/runs"
mkdir -p "$log_root"

ts="$(date +%Y%m%d_%H%M%S)"
run_tag="group2_full"
log_file="$log_root/${ts}_${run_tag}.log"
meta_file="$log_root/${ts}_${run_tag}.meta.txt"

gpuspec="${GPUSPEC:-EUNH100x1}"
python_bin="$project_root/group1_baseline/.venv/bin/python"
entry_script="$repo_root/scripts/run_group2_workflow.py"

if [[ ! -x "$python_bin" ]]; then
  echo "Missing python executable: $python_bin" >&2
  exit 1
fi

if [[ ! -f "$entry_script" ]]; then
  echo "Missing entry script: $entry_script" >&2
  exit 1
fi

declare -a workflow_args
if [[ "$#" -eq 0 ]]; then
  workflow_args=(--stages 1,2,3,6 --stage2-variants baseline --stage2-splits val --overwrite)
else
  workflow_args=("$@")
fi

{
  echo "timestamp=$ts"
  echo "run_tag=$run_tag"
  echo "gpuspec=$gpuspec"
  echo "repo_root=$repo_root"
  echo "project_root=$project_root"
  echo "python_bin=$python_bin"
  echo "entry_script=$entry_script"
  echo "workflow_args=${workflow_args[*]}"
  echo "log_file=$log_file"
} > "$meta_file"

set +e
cloudexe --gpuspec "$gpuspec" -- "$python_bin" "$entry_script" "${workflow_args[@]}" 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
set -e

echo "exit_code=$status" >> "$meta_file"
echo "log_file=$log_file"
echo "meta_file=$meta_file"
exit "$status"
