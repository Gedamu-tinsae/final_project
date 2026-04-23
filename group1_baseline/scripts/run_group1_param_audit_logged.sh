#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
project_root="$(cd "$repo_root/.." && pwd)"
log_root="$project_root/logs/runs"
mkdir -p "$log_root"

ts="$(date +%Y%m%d_%H%M%S)"
run_tag="group1_param_audit"
log_file="$log_root/${ts}_${run_tag}.log"
meta_file="$log_root/${ts}_${run_tag}.meta.txt"

python_bin="$repo_root/.venv/bin/python"
entry_script="$repo_root/scripts/export_group1_trainable_params.py"

if [[ ! -x "$python_bin" ]]; then
  echo "Missing python executable: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$entry_script" ]]; then
  echo "Missing entry script: $entry_script" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "$repo_root/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$repo_root/.env"
    set +a
  fi
fi

export JAX_PLATFORMS=cpu

{
  echo "timestamp=$ts"
  echo "run_tag=$run_tag"
  echo "repo_root=$repo_root"
  echo "project_root=$project_root"
  echo "python_bin=$python_bin"
  echo "entry_script=$entry_script"
  echo "hf_token_present=$([[ -n \"${HF_TOKEN:-}\" ]] && echo 1 || echo 0)"
  echo "log_file=$log_file"
} > "$meta_file"

set +e
PYTHONUNBUFFERED=1 /usr/bin/time -v "$python_bin" -u "$entry_script" "$@" 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
set -e

echo "exit_code=$status" >> "$meta_file"
echo "log_file=$log_file"
echo "meta_file=$meta_file"
exit "$status"
