#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
project_root="$(cd "$repo_root/.." && pwd)"

RUN_STAGE4_EXPERIMENTS=1 \
RUN_STAGE5_EXPERIMENTS=1 \
"$repo_root/scripts/run_group2_tpu_logged.sh" "$@"

