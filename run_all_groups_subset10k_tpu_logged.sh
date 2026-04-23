#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")" && pwd)"
log_root="$project_root/logs/runs"
mkdir -p "$log_root"

ts="$(date +%Y%m%d_%H%M%S)"
run_tag="all_groups_subset10k_tpu"
log_file="$log_root/${ts}_${run_tag}.log"
meta_file="$log_root/${ts}_${run_tag}.meta.txt"

g1_script="$project_root/group1_baseline/scripts/run_group1_subset10k_tpu_logged.sh"
g2_script="$project_root/group2_baseline/scripts/run_group2_tpu_logged.sh"
g4_script="$project_root/group4_baseline/scripts/run_group4_subset10k_tpu_logged.sh"

for p in "$g1_script" "$g2_script" "$g4_script"; do
  if [[ ! -f "$p" ]]; then
    echo "Missing required script: $p" >&2
    exit 1
  fi
done

{
  echo "timestamp=$ts"
  echo "run_tag=$run_tag"
  echo "project_root=$project_root"
  echo "group1_script=$g1_script"
  echo "group2_script=$g2_script"
  echo "group4_script=$g4_script"
  echo "log_file=$log_file"
} > "$meta_file"

run_logged() {
  local label="$1"
  shift
  echo "==== ${label} ====" | tee -a "$log_file"
  set +e
  /usr/bin/time -v "$@" 2>&1 | tee -a "$log_file"
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

run_logged "group1_subset10k" "$g1_script"
run_logged "group2_subset10k" "$g2_script"
run_logged "group4_subset10k" "$g4_script"

echo "exit_code=0" >> "$meta_file"
echo "log_file=$log_file"
echo "meta_file=$meta_file"
exit 0
