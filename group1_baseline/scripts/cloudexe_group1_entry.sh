#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
workflow_script="$repo_root/scripts/run_baseline_workflow.py"
default_config_rel="configs/workflow_paths_subset_10000.json"

if [[ ! -x "$python_bin" ]]; then
  echo "Missing python executable: $python_bin" >&2
  exit 1
fi

if [[ ! -f "$workflow_script" ]]; then
  echo "Missing workflow script: $workflow_script" >&2
  exit 1
fi

cuda_libs="$("$python_bin" - <<'PY'
import glob
import os
import site

libs = []
for root in site.getsitepackages():
    libs.extend(glob.glob(os.path.join(root, "nvidia", "*", "lib")))
print(":".join(libs))
PY
)"
export LD_LIBRARY_PATH="${cuda_libs}:${LD_LIBRARY_PATH:-}"

backend="$("$python_bin" - <<'PY'
import jax
print(jax.default_backend())
PY
)"

echo "JAX backend (cloudexe entry): $backend"
if [[ "${REQUIRE_GPU:-1}" == "1" && "$backend" != "gpu" ]]; then
  echo "GPU required but JAX backend is '$backend'. Set REQUIRE_GPU=0 to bypass." >&2
  exit 2
fi

declare -a workflow_args
if [[ "$#" -eq 0 ]]; then
  workflow_args=(--config "$default_config_rel" --max-rows-guard 10000)
else
  workflow_args=("$@")
fi

exec "$python_bin" "$workflow_script" "${workflow_args[@]}"
