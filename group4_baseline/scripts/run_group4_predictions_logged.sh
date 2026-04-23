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

for p in "$python_bin" "$pred_script"; do
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

default_metrics_1="$repo_root/artifacts/peft_smoke/lora_lora-qv_target-qv_rows-64_seed-42_metrics.json"
default_metrics_2="$repo_root/artifacts/peft_smoke/selective_ft_lora-na_target-qv_rows-64_seed-42_metrics.json"
default_metrics_3="$repo_root/artifacts/peft_smoke/lora_lora-all_weights_target-all_rows-64_seed-42_metrics.json"

declare -a pred_args
if [[ "$#" -eq 0 ]]; then
  pred_args=(
    --metrics-json
    "$default_metrics_1"
    "$default_metrics_2"
    "$default_metrics_3"
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
