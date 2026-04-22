#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
project_root="$(cd "$repo_root/.." && pwd)"
log_root="$project_root/logs/runs"
mkdir -p "$log_root"

ts="$(date +%Y%m%d_%H%M%S)"
run_tag="group4_full"
log_file="$log_root/${ts}_${run_tag}.log"
meta_file="$log_root/${ts}_${run_tag}.meta.txt"

gpuspec="${GPUSPEC:-EUNH100x1}"
python_bin="$project_root/group1_baseline/.venv/bin/python"
workflow_script="$repo_root/scripts/run_group4_workflow.py"
peft_script="$repo_root/scripts/run_group4_peft_smoke.py"

for p in "$python_bin" "$workflow_script" "$peft_script"; do
  if [[ ! -e "$p" ]]; then
    echo "Missing required path: $p" >&2
    exit 1
  fi
done

{
  echo "timestamp=$ts"
  echo "run_tag=$run_tag"
  echo "gpuspec=$gpuspec"
  echo "repo_root=$repo_root"
  echo "project_root=$project_root"
  echo "python_bin=$python_bin"
  echo "workflow_script=$workflow_script"
  echo "peft_script=$peft_script"
  echo "log_file=$log_file"
} > "$meta_file"

run_logged() {
  local label="$1"
  shift
  echo "==== ${label} ====" | tee -a "$log_file"
  set +e
  cloudexe --gpuspec "$gpuspec" -- "$python_bin" "$@" 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  echo "${label}_exit_code=${status}" >> "$meta_file"
  if [[ $status -ne 0 ]]; then
    echo "FAILED: ${label} (exit ${status})"
    echo "exit_code=${status}" >> "$meta_file"
    exit "$status"
  fi
}

run_logged "workflow_stage123" "$workflow_script" --stages 1,2,3 --overwrite
run_logged "peft_lora_qv" "$peft_script" --method lora --lora-variant qv --target-modules qv --max-rows 64 --batch-size 1 --epochs 1 --append-manual-results --overwrite
run_logged "peft_selective_ft_qv" "$peft_script" --method selective_ft --target-modules qv --selection-strategy magnitude --budget-pct 1.0 --max-rows 64 --batch-size 1 --epochs 1 --append-manual-results --overwrite

echo "exit_code=0" >> "$meta_file"
echo "log_file=$log_file"
echo "meta_file=$meta_file"
