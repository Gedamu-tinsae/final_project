#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
project_root="$(cd "$repo_root/.." && pwd)"
log_root="$project_root/logs/runs"
mkdir -p "$log_root"

ts="$(date +%Y%m%d_%H%M%S)"
run_tag="group4_full_smoke"
log_file="$log_root/${ts}_${run_tag}.log"
meta_file="$log_root/${ts}_${run_tag}.meta.txt"

executor="${EXECUTOR:-local}"  # local (TPU/default) | cloudexe
gpuspec="${GPUSPEC:-H100x1}"
python_bin="$project_root/group1_baseline/.venv/bin/python"
workflow_script="$repo_root/scripts/run_group4_workflow.py"
peft_script="$repo_root/scripts/run_group4_peft_smoke.py"
config_rel="configs/workflow_paths_subset_10000.json"

max_rows="${MAX_ROWS:-10000}"
batch_size="${BATCH_SIZE:-1}"
epochs="${EPOCHS:-1}"
plan_overwrite="${PLAN_OVERWRITE:-0}"  # 0 keeps existing plan/registry for resume safety
allow_overwrite_outputs="${ALLOW_OVERWRITE_OUTPUTS:-0}"

for p in "$python_bin" "$workflow_script" "$peft_script"; do
  if [[ ! -e "$p" ]]; then
    echo "Missing required path: $p" >&2
    exit 1
  fi
done

# Make JAX CUDA libs discoverable at runtime (prevents cuSPARSE fallback-to-CPU).
export LD_LIBRARY_PATH="$("$python_bin" - <<'PY'
import site, glob, os
libs = []
for r in site.getsitepackages():
    libs += glob.glob(os.path.join(r, "nvidia", "*", "lib"))
print(":".join(libs))
PY
):${LD_LIBRARY_PATH:-}"

{
  echo "timestamp=$ts"
  echo "run_tag=$run_tag"
  echo "executor=$executor"
  echo "gpuspec=$gpuspec"
  echo "repo_root=$repo_root"
  echo "project_root=$project_root"
  echo "python_bin=$python_bin"
  echo "workflow_script=$workflow_script"
  echo "peft_script=$peft_script"
  echo "config_rel=$config_rel"
  echo "max_rows=$max_rows"
  echo "batch_size=$batch_size"
  echo "epochs=$epochs"
  echo "plan_overwrite=$plan_overwrite"
  echo "allow_overwrite_outputs=$allow_overwrite_outputs"
  echo "log_file=$log_file"
} > "$meta_file"

run_logged() {
  local label="$1"
  shift
  echo "==== ${label} ====" | tee -a "$log_file"
  set +e
  if [[ "$executor" == "cloudexe" ]]; then
    cloudexe --gpuspec "$gpuspec" -- "$python_bin" "$@" 2>&1 | tee -a "$log_file"
  elif [[ "$executor" == "local" ]]; then
    "$python_bin" "$@" 2>&1 | tee -a "$log_file"
  else
    echo "Unknown EXECUTOR: $executor" | tee -a "$log_file"
    local status=2
    set -e
    echo "${label}_exit_code=${status}" >> "$meta_file"
    echo "FAILED: ${label} (exit ${status})"
    echo "exit_code=${status}" >> "$meta_file"
    exit "$status"
  fi
  local status=${PIPESTATUS[0]}
  set -e
  echo "${label}_exit_code=${status}" >> "$meta_file"
  if [[ $status -ne 0 ]]; then
    echo "FAILED: ${label} (exit ${status})"
    echo "exit_code=${status}" >> "$meta_file"
    exit "$status"
  fi
}

# Stage orchestration (plan + registry)
workflow_args=( "$workflow_script" --config "$config_rel" --stages 1,2,3 )
if [[ "$plan_overwrite" == "1" ]]; then
  workflow_args+=( --overwrite )
fi
run_logged "workflow_stage123" "${workflow_args[@]}"

# LoRA smoke
peft_lora_args=( "$peft_script" \
  --method lora \
  --config "$config_rel" \
  --lora-variant qv \
  --target-modules qv \
  --max-rows "$max_rows" \
  --max-rows-guard 10000 \
  --batch-size "$batch_size" \
  --epochs "$epochs" \
  --append-manual-results )
if [[ "$allow_overwrite_outputs" == "1" ]]; then
  peft_lora_args+=( --overwrite )
fi
run_logged "peft_lora_smoke" "${peft_lora_args[@]}"

# Selective FT smoke
peft_selective_args=( "$peft_script" \
  --method selective_ft \
  --config "$config_rel" \
  --target-modules qv \
  --selection-strategy magnitude \
  --budget-pct 1.0 \
  --max-rows "$max_rows" \
  --max-rows-guard 10000 \
  --batch-size "$batch_size" \
  --epochs "$epochs" \
  --append-manual-results )
if [[ "$allow_overwrite_outputs" == "1" ]]; then
  peft_selective_args+=( --overwrite )
fi
run_logged "peft_selective_smoke" "${peft_selective_args[@]}"

# Summary (if manual results file now has rows)
run_logged "workflow_stage4" "$workflow_script" --config "$config_rel" --stages 4 --overwrite

echo "exit_code=0" >> "$meta_file"
echo "log_file=$log_file"
echo "meta_file=$meta_file"
