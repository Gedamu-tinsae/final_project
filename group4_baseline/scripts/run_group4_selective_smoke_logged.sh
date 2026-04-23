#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
project_root="$(cd "$repo_root/.." && pwd)"
log_root="$project_root/logs/runs"
mkdir -p "$log_root"

ts="$(date +%Y%m%d_%H%M%S)"
run_tag="group4_selective_smoke"
log_file="$log_root/${ts}_${run_tag}.log"
meta_file="$log_root/${ts}_${run_tag}.meta.txt"

gpuspec="${GPUSPEC:-H100x1}"
python_bin="$project_root/group1_baseline/.venv/bin/python"
peft_script="$repo_root/scripts/run_group4_peft_smoke.py"

for p in "$python_bin" "$peft_script"; do
  if [[ ! -e "$p" ]]; then
    echo "Missing required path: $p" >&2
    exit 1
  fi
done

declare -a peft_args
if [[ "$#" -eq 0 ]]; then
  peft_args=(
    --method selective_ft
    --target-modules qv
    --selection-strategy magnitude
    --budget-pct 1.0
    --max-rows 64
    --batch-size 1
    --epochs 1
    --append-manual-results
    --overwrite
  )
else
  peft_args=("$@")
fi

{
  echo "timestamp=$ts"
  echo "run_tag=$run_tag"
  echo "gpuspec=$gpuspec"
  echo "repo_root=$repo_root"
  echo "project_root=$project_root"
  echo "python_bin=$python_bin"
  echo "peft_script=$peft_script"
  echo "peft_args=${peft_args[*]}"
  echo "log_file=$log_file"
} > "$meta_file"

set +e
cloudexe --gpuspec "$gpuspec" -- "$python_bin" "$peft_script" "${peft_args[@]}" 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
set -e

echo "exit_code=$status" >> "$meta_file"
echo "log_file=$log_file"
echo "meta_file=$meta_file"
exit "$status"
