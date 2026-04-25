#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
project_root="$(cd "$repo_root/.." && pwd)"
log_root="$project_root/logs/runs"
mkdir -p "$log_root"

ts="$(date +%Y%m%d_%H%M%S)"
run_tag="group4_relora"
log_file="$log_root/${ts}_${run_tag}.log"
meta_file="$log_root/${ts}_${run_tag}.meta.txt"

# Execution controls
executor="${EXECUTOR:-local}"           # local (TPU/default) | cloudexe
gpuspec="${GPUSPEC:-H100x1}"
python_bin="$project_root/group1_baseline/.venv/bin/python"
workflow_script="$repo_root/scripts/run_group4_workflow.py"
pred_script="$repo_root/scripts/run_group4_smoke_predictions.py"
eval_script="$repo_root/scripts/run_group4_eval.py"
config_rel="${CONFIG_REL:-configs/workflow_paths_subset_10000_relora.json}"
allow_shared_config="${ALLOW_SHARED_CONFIG:-0}"

# Train defaults
max_rows="${MAX_ROWS:-10000}"
batch_size="${BATCH_SIZE:-1}"
epochs="${EPOCHS:-1}"
learning_rate="${LEARNING_RATE:-1e-5}"
dtype="${DTYPE:-bfloat16}"
seed="${SEED:-42}"
val_every_steps="${VAL_EVERY_STEPS:-200}"
val_max_batches="${VAL_MAX_BATCHES:-0}"

# ReLoRA-only execution controls (resume-safe by default)
plan_overwrite="${PLAN_OVERWRITE:-0}"  # 0 keeps existing plan/registry so resume works.
max_experiments="${MAX_EXPERIMENTS:-0}"  # 0 = all selected experiments
plan_experiment_ids="${PLAN_EXPERIMENT_IDS:-}"  # comma list of experiment_id values
plan_target_modules="${PLAN_TARGET_MODULES:-}"  # comma list: qv,all
plan_lora_ranks="${PLAN_LORA_RANKS:-}"  # comma list: 4,8,16
plan_relora_merge_freqs="${PLAN_RELORA_MERGE_FREQS:-500}"  # comma list: 250,500,...
allow_overwrite_experiment_outputs="${ALLOW_OVERWRITE_EXPERIMENT_OUTPUTS:-0}"
plan_retries="${PLAN_RETRIES:-2}"
plan_retry_sleep_sec="${PLAN_RETRY_SLEEP_SEC:-20}"
run_train_mode="${RUN_TRAIN_MODE:-auto}"  # auto|1|0 ; auto => train only when EVAL_MODE=none

# Shared summary safety (default: do not touch shared summary files).
refresh_shared_summary="${REFRESH_SHARED_SUMMARY:-0}"
summary_overwrite="${SUMMARY_OVERWRITE:-0}"

# ReLoRA evaluation controls (dedicated eval dir under /eval/relora)
eval_mode="${EVAL_MODE:-none}"  # none | human_pack | human_aggregate | api_judge
relora_eval_out_dir_default="$repo_root/data/processed/subsets/subset_10000_seed42/eval/relora"
relora_eval_out_dir="${RELORA_EVAL_OUT_DIR:-$relora_eval_out_dir_default}"
relora_pred_json="${RELORA_PRED_JSON:-$relora_eval_out_dir/relora_prediction_samples.json}"
relora_pred_md="${RELORA_PRED_MD:-$relora_eval_out_dir/relora_prediction_samples.md}"
relora_generations_jsonl="${RELORA_GENERATIONS_JSONL:-$relora_eval_out_dir/relora_generations_filled.jsonl}"
baseline_method="${BASELINE_METHOD:-baseline}"
candidate_methods="${CANDIDATE_METHODS:-}"
max_requests="${MAX_REQUESTS:-0}"
human_results_jsonl="${HUMAN_RESULTS_JSONL:-$relora_eval_out_dir/human_results_filled.jsonl}"
openai_model="${OPENAI_MODEL:-gpt-4.1-mini}"
openai_api_key="${OPENAI_API_KEY:-}"
update_results_manual="${UPDATE_RESULTS_MANUAL:-0}"
regen_eval_inputs="${REGEN_EVAL_INPUTS:-0}"  # 1 => regenerate predictions/generations even for aggregate

for p in "$python_bin" "$workflow_script" "$pred_script" "$eval_script"; do
  if [[ ! -e "$p" ]]; then
    echo "Missing required path: $p" >&2
    exit 1
  fi
done

if [[ "$allow_shared_config" != "1" ]]; then
  case "$config_rel" in
    *relora*)
      ;;
    *)
      echo "Refusing non-ReLoRA config: $config_rel" >&2
      echo "Use CONFIG_REL=configs/workflow_paths_subset_10000_relora.json or set ALLOW_SHARED_CONFIG=1 intentionally." >&2
      exit 2
      ;;
  esac
fi

{
  echo "timestamp=$ts"
  echo "run_tag=$run_tag"
  echo "executor=$executor"
  echo "gpuspec=$gpuspec"
  echo "repo_root=$repo_root"
  echo "project_root=$project_root"
  echo "python_bin=$python_bin"
  echo "workflow_script=$workflow_script"
  echo "config_rel=$config_rel"
  echo "allow_shared_config=$allow_shared_config"
  echo "max_rows=$max_rows"
  echo "batch_size=$batch_size"
  echo "epochs=$epochs"
  echo "learning_rate=$learning_rate"
  echo "dtype=$dtype"
  echo "seed=$seed"
  echo "val_every_steps=$val_every_steps"
  echo "val_max_batches=$val_max_batches"
  echo "plan_overwrite=$plan_overwrite"
  echo "max_experiments=$max_experiments"
  echo "plan_experiment_ids=$plan_experiment_ids"
  echo "plan_methods=relora"
  echo "plan_target_modules=$plan_target_modules"
  echo "plan_lora_ranks=$plan_lora_ranks"
  echo "plan_relora_merge_freqs=$plan_relora_merge_freqs"
  echo "allow_overwrite_experiment_outputs=$allow_overwrite_experiment_outputs"
  echo "plan_retries=$plan_retries"
  echo "plan_retry_sleep_sec=$plan_retry_sleep_sec"
  echo "run_train_mode=$run_train_mode"
  echo "refresh_shared_summary=$refresh_shared_summary"
  echo "summary_overwrite=$summary_overwrite"
  echo "eval_mode=$eval_mode"
  echo "relora_eval_out_dir=$relora_eval_out_dir"
  echo "relora_pred_json=$relora_pred_json"
  echo "relora_pred_md=$relora_pred_md"
  echo "relora_generations_jsonl=$relora_generations_jsonl"
  echo "baseline_method=$baseline_method"
  echo "candidate_methods=$candidate_methods"
  echo "max_requests=$max_requests"
  echo "human_results_jsonl=$human_results_jsonl"
  echo "openai_model=$openai_model"
  echo "update_results_manual=$update_results_manual"
  echo "regen_eval_inputs=$regen_eval_inputs"
  echo "log_file=$log_file"
} > "$meta_file"

run_logged() {
  local label="$1"
  shift
  echo "==== ${label} ====" | tee -a "$log_file"
  set +e
  if [[ "$executor" == "cloudexe" ]]; then
    cloudexe --gpuspec "$gpuspec" -- "$python_bin" "$@" 2>&1 | tee -a "$log_file"
    local status=${PIPESTATUS[0]}
  elif [[ "$executor" == "local" ]]; then
    "$python_bin" "$@" 2>&1 | tee -a "$log_file"
    local status=${PIPESTATUS[0]}
  else
    echo "Unknown EXECUTOR: $executor" | tee -a "$log_file"
    local status=2
  fi
  set -e
  echo "${label}_exit_code=${status}" >> "$meta_file"
  if [[ $status -ne 0 ]]; then
    echo "FAILED: ${label} (exit ${status})" | tee -a "$log_file"
    echo "exit_code=${status}" >> "$meta_file"
    exit "$status"
  fi
}

# Decide whether to run training stage.
run_train=0
case "$run_train_mode" in
  1|true|TRUE|yes|YES) run_train=1 ;;
  0|false|FALSE|no|NO) run_train=0 ;;
  auto|AUTO|"")
    if [[ "$eval_mode" == "none" ]]; then
      run_train=1
    else
      run_train=0
    fi
    ;;
  *)
    echo "Invalid RUN_TRAIN_MODE: $run_train_mode (expected auto|1|0)" | tee -a "$log_file"
    echo "exit_code=2" >> "$meta_file"
    exit 2
    ;;
esac

if [[ "$run_train" == "1" ]]; then
  # 1) Workflow stage1+2+3 and execute plan, filtered to ReLoRA only.
  workflow_args=(
    "$workflow_script"
    --config "$config_rel"
    --stages 1,2,3
    --execute-plan
    --max-experiments "$max_experiments"
    --max-rows "$max_rows"
    --batch-size "$batch_size"
    --epochs "$epochs"
    --learning-rate "$learning_rate"
    --dtype "$dtype"
    --seed "$seed"
    --val-every-steps "$val_every_steps"
    --val-max-batches "$val_max_batches"
    --plan-retries "$plan_retries"
    --plan-retry-sleep-sec "$plan_retry_sleep_sec"
    --plan-methods "relora"
  )
  if [[ "$plan_overwrite" == "1" ]]; then
    workflow_args+=(--overwrite)
  fi
  if [[ "$allow_overwrite_experiment_outputs" == "1" ]]; then
    workflow_args+=(--allow-overwrite-experiment-outputs)
  fi
  if [[ -n "$plan_experiment_ids" ]]; then
    workflow_args+=(--plan-experiment-ids "$plan_experiment_ids")
  fi
  if [[ -n "$plan_target_modules" ]]; then
    workflow_args+=(--plan-target-modules "$plan_target_modules")
  fi
  if [[ -n "$plan_lora_ranks" ]]; then
    workflow_args+=(--plan-lora-ranks "$plan_lora_ranks")
  fi
  if [[ -n "$plan_relora_merge_freqs" ]]; then
    workflow_args+=(--plan-relora-merge-freqs "$plan_relora_merge_freqs")
  fi
  run_logged "workflow_stage123_execute_relora_plan" "${workflow_args[@]}"
else
  echo "Skipping training stage (RUN_TRAIN_MODE=$run_train_mode, EVAL_MODE=$eval_mode)." | tee -a "$log_file"
fi

# 2) Optional: refresh shared summary (off by default to avoid touching existing shared summaries).
if [[ "$refresh_shared_summary" == "1" ]]; then
  stage4_args=( "$workflow_script" --config "$config_rel" --stages 4 )
  if [[ "$summary_overwrite" == "1" ]]; then
    stage4_args+=( --overwrite )
  fi
  run_logged "workflow_stage4_shared_summary" "${stage4_args[@]}"
else
  echo "Skipping shared summary refresh (REFRESH_SHARED_SUMMARY=0)." | tee -a "$log_file"
fi

# 3) Optional ReLoRA eval pipeline in dedicated eval dir.
mkdir -p "$relora_eval_out_dir"

need_eval_inputs=0
case "$eval_mode" in
  human_pack|api_judge) need_eval_inputs=1 ;;
  human_aggregate)
    if [[ "$regen_eval_inputs" == "1" ]]; then
      need_eval_inputs=1
    else
      need_eval_inputs=0
    fi
    ;;
  none) need_eval_inputs=0 ;;
esac

if [[ "$need_eval_inputs" == "1" ]]; then
  shopt -s nullglob
  metrics_matches=( "$repo_root"/artifacts/relora_smoke/g4_relora_*_metrics.json )
  shopt -u nullglob
  if [[ ${#metrics_matches[@]} -eq 0 ]]; then
    echo "No ReLoRA metrics files found under $repo_root/artifacts/relora_smoke/" | tee -a "$log_file"
    echo "exit_code=3" >> "$meta_file"
    exit 3
  fi
  run_logged "relora_predictions" \
    "$pred_script" \
    --metrics-json "$repo_root"/artifacts/relora_smoke/g4_relora_*_metrics.json \
    --max-samples "$max_rows" \
    --method-id-field experiment_id \
    --include-group1-baseline \
    --baseline-method-name "$baseline_method" \
    --output-json "$relora_pred_json" \
    --output-md "$relora_pred_md"

  run_logged "relora_generations_jsonl" -c "
import json
from pathlib import Path
src = Path(r'$relora_pred_json')
dst = Path(r'$relora_generations_jsonl')
obj = json.loads(src.read_text(encoding='utf-8'))
rows = []
for run in obj.get('results', []):
    method = str(run.get('method_id') or run.get('experiment_id') or run.get('run_id') or '').strip()
    if not method:
        continue
    for s in run.get('samples', []):
        out = (s.get('pred_text') or '').strip()
        if not out:
            continue
        sample_id = str(s.get('sample_id') or f\"s_{int(s.get('sample_index', 0)):04d}\")
        rows.append({
            'sample_id': sample_id,
            'prompt': str(s.get('vision_path', '')),
            'method': method,
            'output': out,
        })
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open('w', encoding='utf-8') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\\n')
print('wrote', dst, 'rows=', len(rows))
"
else
  echo "Skipping prediction/generation rebuild for eval_mode=$eval_mode (REGEN_EVAL_INPUTS=$regen_eval_inputs)." | tee -a "$log_file"
fi

case "$eval_mode" in
  none)
    echo "Skipping eval (EVAL_MODE=none)." | tee -a "$log_file"
    ;;
  human_pack)
    run_logged "relora_eval_human_pack" \
      "$eval_script" \
      --config "$config_rel" \
      --mode human_pack \
      --output-dir "$relora_eval_out_dir" \
      --generations-jsonl "$relora_generations_jsonl" \
      --baseline-method "$baseline_method" \
      --candidate-methods "$candidate_methods" \
      --max-requests "$max_requests"
    ;;
  human_aggregate)
    if [[ ! -f "$relora_generations_jsonl" ]]; then
      echo "Missing generations file for aggregation: $relora_generations_jsonl" | tee -a "$log_file"
      echo "exit_code=4" >> "$meta_file"
      exit 4
    fi
    if [[ ! -f "$human_results_jsonl" ]]; then
      echo "Missing human results file for aggregation: $human_results_jsonl" | tee -a "$log_file"
      echo "exit_code=4" >> "$meta_file"
      exit 4
    fi
    agg_args=(
      "$eval_script"
      --config "$config_rel"
      --mode human_aggregate
      --output-dir "$relora_eval_out_dir"
      --generations-jsonl "$relora_generations_jsonl"
      --baseline-method "$baseline_method"
      --candidate-methods "$candidate_methods"
      --human-results-jsonl "$human_results_jsonl"
      --max-requests "$max_requests"
    )
    if [[ "$update_results_manual" == "1" ]]; then
      agg_args+=( --update-results-manual )
    fi
    run_logged "relora_eval_human_aggregate" \
      "${agg_args[@]}"
    ;;
  api_judge)
    api_args=(
      "$eval_script"
      --config "$config_rel"
      --mode api_judge
      --output-dir "$relora_eval_out_dir"
      --generations-jsonl "$relora_generations_jsonl"
      --baseline-method "$baseline_method"
      --candidate-methods "$candidate_methods"
      --max-requests "$max_requests"
      --openai-model "$openai_model"
      --openai-api-key "$openai_api_key"
    )
    if [[ "$update_results_manual" == "1" ]]; then
      api_args+=( --update-results-manual )
    fi
    run_logged "relora_eval_api_judge" \
      "${api_args[@]}"
    ;;
  *)
    echo "Unknown EVAL_MODE: $eval_mode" | tee -a "$log_file"
    echo "exit_code=2" >> "$meta_file"
    exit 2
    ;;
esac

echo "exit_code=0" >> "$meta_file"
echo "log_file=$log_file"
echo "meta_file=$meta_file"
