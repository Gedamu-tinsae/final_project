#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
project_root="$(cd "$repo_root/.." && pwd)"
log_root="$project_root/logs/runs"
mkdir -p "$log_root"

ts="$(date +%Y%m%d_%H%M%S)"
run_tag="group4_subset10k_tpu"
log_file="$log_root/${ts}_${run_tag}.log"
meta_file="$log_root/${ts}_${run_tag}.meta.txt"

python_bin="$project_root/group1_baseline/.venv/bin/python"
workflow_script="$repo_root/scripts/run_group4_workflow.py"
peft_script="$repo_root/scripts/run_group4_peft_smoke.py"
comparison_script="$project_root/common/generate_comparison_report.py"
config_rel="configs/workflow_paths_subset_10000.json"
plan_overwrite="${PLAN_OVERWRITE:-0}"
allow_overwrite_outputs="${ALLOW_OVERWRITE_OUTPUTS:-0}"

for p in "$python_bin" "$workflow_script" "$peft_script" "$comparison_script"; do
  if [[ ! -e "$p" ]]; then
    echo "Missing required path: $p" >&2
    exit 1
  fi
done

{
  echo "timestamp=$ts"
  echo "run_tag=$run_tag"
  echo "repo_root=$repo_root"
  echo "project_root=$project_root"
  echo "python_bin=$python_bin"
  echo "workflow_script=$workflow_script"
  echo "peft_script=$peft_script"
  echo "comparison_script=$comparison_script"
  echo "config_rel=$config_rel"
  echo "max_rows=10000"
  echo "batch_size=1"
  echo "epochs=1"
  echo "plan_overwrite=$plan_overwrite"
  echo "allow_overwrite_outputs=$allow_overwrite_outputs"
  echo "log_file=$log_file"
} > "$meta_file"

run_logged() {
  local label="$1"
  shift
  echo "==== ${label} ====" | tee -a "$log_file"
  set +e
  /usr/bin/time -v "$python_bin" "$@" 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  echo "${label}_exit_code=${status}" >> "$meta_file"
  if [[ $status -ne 0 ]]; then
    echo "exit_code=${status}" >> "$meta_file"
    echo "log_file=$log_file"
    echo "meta_file=$meta_file"
    exit "$status"
  fi
}

workflow_args=( "$workflow_script" --config "$config_rel" --stages 1,2,3 )
if [[ "$plan_overwrite" == "1" ]]; then
  workflow_args+=( --overwrite )
fi
run_logged "workflow_stage123" "${workflow_args[@]}"

peft_lora_args=( "$peft_script" \
  --config "$config_rel" \
  --method lora \
  --lora-variant qv \
  --target-modules qv \
  --max-rows 10000 \
  --max-rows-guard 10000 \
  --batch-size 1 \
  --epochs 1 \
  --append-manual-results )
if [[ "$allow_overwrite_outputs" == "1" ]]; then
  peft_lora_args+=( --overwrite )
fi
run_logged "peft_lora_qv_10k" "${peft_lora_args[@]}"

peft_selective_args=( "$peft_script" \
  --config "$config_rel" \
  --method selective_ft \
  --target-modules qv \
  --selection-strategy magnitude \
  --budget-pct 1.0 \
  --max-rows 10000 \
  --max-rows-guard 10000 \
  --batch-size 1 \
  --epochs 1 \
  --append-manual-results )
if [[ "$allow_overwrite_outputs" == "1" ]]; then
  peft_selective_args+=( --overwrite )
fi
run_logged "peft_selective_qv_10k" "${peft_selective_args[@]}"

run_logged "workflow_stage4" "$workflow_script" --config "$config_rel" --stages 4 --overwrite
run_logged "comparison_report" "$comparison_script" --outputs-root "$project_root/outputs" --run-name "tpu_presentation"

echo "exit_code=0" >> "$meta_file"
echo "log_file=$log_file"
echo "meta_file=$meta_file"
