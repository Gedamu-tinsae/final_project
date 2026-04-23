#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
project_root="$(cd "$repo_root/.." && pwd)"
log_root="$project_root/logs/runs"
mkdir -p "$log_root"

ts="$(date +%Y%m%d_%H%M%S)"
run_tag="group4_predictions"
log_file="$log_root/${ts}_${run_tag}.log"
meta_file="$log_root/${ts}_${run_tag}.meta.txt"

gpuspec="${GPUSPEC:-H100x1}"
python_bin="$project_root/group1_baseline/.venv/bin/python"
pred_script="$repo_root/scripts/run_group4_smoke_predictions.py"
config_rel="configs/workflow_paths_subset_10000.json"
config_path="$repo_root/$config_rel"

for p in "$python_bin" "$pred_script" "$config_path"; do
  if [[ ! -e "$p" ]]; then
    echo "Missing required path: $p" >&2
    exit 1
  fi
done

# Make JAX CUDA libs discoverable at runtime.
export LD_LIBRARY_PATH="$("$python_bin" - <<'PY'
import site, glob, os
libs = []
for r in site.getsitepackages():
    libs += glob.glob(os.path.join(r, "nvidia", "*", "lib"))
print(":".join(libs))
PY
):${LD_LIBRARY_PATH:-}"

mapfile -t default_metrics < <(ls -1t "$repo_root"/artifacts/peft_smoke/*_rows-10000_*_metrics.json 2>/dev/null | head -n 3)

declare -a pred_args
if [[ "$#" -eq 0 ]]; then
  if [[ ${#default_metrics[@]} -lt 1 ]]; then
    echo "No 10k metrics files found under $repo_root/artifacts/peft_smoke" >&2
    echo "Run the 10k subset Group4 jobs first, then rerun this script." >&2
    exit 1
  fi
  pred_args=(
    --metrics-json
    "${default_metrics[@]}"
    --max-samples 5
  )
else
  pred_args=("$@")
fi

{
  echo "timestamp=$ts"
  echo "run_tag=$run_tag"
  echo "gpuspec=$gpuspec"
  echo "repo_root=$repo_root"
  echo "project_root=$project_root"
  echo "python_bin=$python_bin"
  echo "pred_script=$pred_script"
  echo "config_path=$config_path"
  echo "pred_args=${pred_args[*]}"
  echo "log_file=$log_file"
} > "$meta_file"

set +e
cloudexe --gpuspec "$gpuspec" -- "$python_bin" "$pred_script" "${pred_args[@]}" 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
set -e

echo "exit_code=$status" >> "$meta_file"
echo "log_file=$log_file"
echo "meta_file=$meta_file"
exit "$status"
