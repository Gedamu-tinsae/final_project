#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
project_root="$(cd "$repo_root/.." && pwd)"
log_root="$project_root/logs/runs"
mkdir -p "$log_root"

ts="$(date +%Y%m%d_%H%M%S)"
run_tag="group1_subset10k_tpu"
log_file="$log_root/${ts}_${run_tag}.log"
meta_file="$log_root/${ts}_${run_tag}.meta.txt"

python_bin="$repo_root/.venv/bin/python"
subset_builder="$repo_root/scripts/create_subset_profile.py"
workflow_script="$repo_root/scripts/run_baseline_workflow.py"
config_rel="configs/workflow_paths_subset_10000.json"

if [[ ! -x "$python_bin" ]]; then
  echo "Missing python executable: $python_bin" >&2
  exit 1
fi

for p in "$subset_builder" "$workflow_script"; do
  if [[ ! -f "$p" ]]; then
    echo "Missing required script: $p" >&2
    exit 1
  fi
done

{
  echo "timestamp=$ts"
  echo "run_tag=$run_tag"
  echo "repo_root=$repo_root"
  echo "project_root=$project_root"
  echo "python_bin=$python_bin"
  echo "subset_builder=$subset_builder"
  echo "workflow_script=$workflow_script"
  echo "config_rel=$config_rel"
  echo "log_file=$log_file"
} > "$meta_file"

echo "==== create_subset_profile ====" | tee "$log_file"
set +e
/usr/bin/time -v "$python_bin" "$subset_builder" \
  --rows 10000 \
  --seed 42 \
  --profile-name subset_10000_seed42 \
  --output-config "$config_rel" \
  --download \
  --extract 2>&1 | tee -a "$log_file"
status_subset=${PIPESTATUS[0]}
set -e
echo "create_subset_profile_exit_code=$status_subset" >> "$meta_file"
if [[ $status_subset -ne 0 ]]; then
  echo "exit_code=$status_subset" >> "$meta_file"
  echo "log_file=$log_file"
  echo "meta_file=$meta_file"
  exit "$status_subset"
fi

echo "==== run_baseline_workflow_subset10k ====" | tee -a "$log_file"
set +e
/usr/bin/time -v "$python_bin" "$workflow_script" \
  --config "$config_rel" \
  --overwrite \
  --train both \
  --batch-size 1 \
  --epochs 1 \
  --no-mesh 2>&1 | tee -a "$log_file"
status_workflow=${PIPESTATUS[0]}
set -e
echo "run_baseline_workflow_exit_code=$status_workflow" >> "$meta_file"

echo "exit_code=$status_workflow" >> "$meta_file"
echo "log_file=$log_file"
echo "meta_file=$meta_file"
exit "$status_workflow"
