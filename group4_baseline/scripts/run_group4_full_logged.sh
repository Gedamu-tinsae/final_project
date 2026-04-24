#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
project_root="$(cd "$repo_root/.." && pwd)"
log_root="$project_root/logs/runs"
mkdir -p "$log_root"

ts="$(date +%Y%m%d_%H%M%S)"
run_tag="group4_full"
log_file="$log_root/${ts}_${run_tag}.log"
meta_file="$log_root/${ts}_${run_tag}.meta.txt"

# Execution controls
executor="${EXECUTOR:-local}"           # local (TPU/default) | cloudexe
gpuspec="${GPUSPEC:-H100x1}"
python_bin="$project_root/group1_baseline/.venv/bin/python"
workflow_script="$repo_root/scripts/run_group4_workflow.py"
peft_script="$repo_root/scripts/run_group4_peft_smoke.py"
eval_script="$repo_root/scripts/run_group4_eval.py"
config_rel="${CONFIG_REL:-configs/workflow_paths_subset_10000.json}"

# Train defaults
max_rows="${MAX_ROWS:-10000}"
batch_size="${BATCH_SIZE:-1}"
epochs="${EPOCHS:-1}"
learning_rate="${LEARNING_RATE:-1e-5}"
dtype="${DTYPE:-bfloat16}"
seed="${SEED:-42}"
val_every_steps="${VAL_EVERY_STEPS:-200}"
val_max_batches="${VAL_MAX_BATCHES:-0}"

# Which training methods to run
run_lora_qv="${RUN_LORA_QV:-1}"
run_lora_all_weights="${RUN_LORA_ALL_WEIGHTS:-1}"
run_selective_qv="${RUN_SELECTIVE_QV:-1}"

# Evaluation controls
# EVAL_MODE: none | template | human_pack | human_aggregate | api_judge
eval_mode="${EVAL_MODE:-none}"
baseline_method="${BASELINE_METHOD:-baseline}"
candidate_methods="${CANDIDATE_METHODS:-}"
template_methods="${TEMPLATE_METHODS:-}"
template_max_samples="${TEMPLATE_MAX_SAMPLES:-200}"
max_requests="${MAX_REQUESTS:-0}"
openai_model="${OPENAI_MODEL:-gpt-4.1-mini}"
openai_api_key="${OPENAI_API_KEY:-}"

eval_out_dir_default="$repo_root/data/processed/subsets/subset_10000_seed42/eval"
eval_out_dir="${EVAL_OUT_DIR:-$eval_out_dir_default}"
generations_jsonl_default="$eval_out_dir/group4_generations_template.jsonl"
generations_jsonl="${GENERATIONS_JSONL:-$generations_jsonl_default}"
human_results_jsonl="${HUMAN_RESULTS_JSONL:-$eval_out_dir/human_results_filled.jsonl}"

for p in "$python_bin" "$workflow_script" "$peft_script" "$eval_script"; do
  if [[ ! -e "$p" ]]; then
    echo "Missing required path: $p" >&2
    exit 1
  fi
done

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
  echo "eval_script=$eval_script"
  echo "config_rel=$config_rel"
  echo "max_rows=$max_rows"
  echo "batch_size=$batch_size"
  echo "epochs=$epochs"
  echo "learning_rate=$learning_rate"
  echo "dtype=$dtype"
  echo "seed=$seed"
  echo "val_every_steps=$val_every_steps"
  echo "val_max_batches=$val_max_batches"
  echo "run_lora_qv=$run_lora_qv"
  echo "run_lora_all_weights=$run_lora_all_weights"
  echo "run_selective_qv=$run_selective_qv"
  echo "eval_mode=$eval_mode"
  echo "baseline_method=$baseline_method"
  echo "candidate_methods=$candidate_methods"
  echo "template_methods=$template_methods"
  echo "template_max_samples=$template_max_samples"
  echo "max_requests=$max_requests"
  echo "openai_model=$openai_model"
  echo "eval_out_dir=$eval_out_dir"
  echo "generations_jsonl=$generations_jsonl"
  echo "human_results_jsonl=$human_results_jsonl"
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

# 1) Workflow preflight/plan/registry
run_logged "workflow_stage123" \
  "$workflow_script" \
  --config "$config_rel" \
  --stages 1,2,3 \
  --overwrite

# 2) Training runs (method comparisons)
if [[ "$run_lora_qv" == "1" ]]; then
  run_logged "peft_lora_qv" \
    "$peft_script" \
    --config "$config_rel" \
    --method lora \
    --lora-variant qv \
    --target-modules qv \
    --max-rows "$max_rows" \
    --max-rows-guard 10000 \
    --batch-size "$batch_size" \
    --epochs "$epochs" \
    --learning-rate "$learning_rate" \
    --dtype "$dtype" \
    --seed "$seed" \
    --val-every-steps "$val_every_steps" \
    --val-max-batches "$val_max_batches" \
    --append-manual-results \
    --overwrite
fi

if [[ "$run_lora_all_weights" == "1" ]]; then
  run_logged "peft_lora_all_weights" \
    "$peft_script" \
    --config "$config_rel" \
    --method lora \
    --lora-variant all_weights \
    --target-modules all \
    --max-rows "$max_rows" \
    --max-rows-guard 10000 \
    --batch-size "$batch_size" \
    --epochs "$epochs" \
    --learning-rate "$learning_rate" \
    --dtype "$dtype" \
    --seed "$seed" \
    --val-every-steps "$val_every_steps" \
    --val-max-batches "$val_max_batches" \
    --append-manual-results \
    --overwrite
fi

if [[ "$run_selective_qv" == "1" ]]; then
  run_logged "peft_selective_ft_qv" \
    "$peft_script" \
    --config "$config_rel" \
    --method selective_ft \
    --target-modules qv \
    --selection-strategy magnitude \
    --budget-pct 1.0 \
    --max-rows "$max_rows" \
    --max-rows-guard 10000 \
    --batch-size "$batch_size" \
    --epochs "$epochs" \
    --learning-rate "$learning_rate" \
    --dtype "$dtype" \
    --seed "$seed" \
    --val-every-steps "$val_every_steps" \
    --val-max-batches "$val_max_batches" \
    --append-manual-results \
    --overwrite
fi

# 3) Summary + method comparison charts
run_logged "workflow_stage4_pre_eval" \
  "$workflow_script" \
  --config "$config_rel" \
  --stages 4 \
  --overwrite

# 4) Optional evaluation pipeline
case "$eval_mode" in
  none)
    echo "Skipping evaluation (EVAL_MODE=none)" | tee -a "$log_file"
    ;;
  template)
    run_logged "eval_template" \
      "$eval_script" \
      --config "$config_rel" \
      --mode template \
      --output-dir "$eval_out_dir" \
      --template-methods "$template_methods" \
      --template-max-samples "$template_max_samples"
    ;;
  human_pack)
    run_logged "eval_human_pack" \
      "$eval_script" \
      --config "$config_rel" \
      --mode human_pack \
      --output-dir "$eval_out_dir" \
      --generations-jsonl "$generations_jsonl" \
      --baseline-method "$baseline_method" \
      --candidate-methods "$candidate_methods" \
      --max-requests "$max_requests"
    ;;
  human_aggregate)
    run_logged "eval_human_aggregate" \
      "$eval_script" \
      --config "$config_rel" \
      --mode human_aggregate \
      --output-dir "$eval_out_dir" \
      --generations-jsonl "$generations_jsonl" \
      --baseline-method "$baseline_method" \
      --candidate-methods "$candidate_methods" \
      --human-results-jsonl "$human_results_jsonl" \
      --max-requests "$max_requests" \
      --update-results-manual
    run_logged "workflow_stage4_post_eval" \
      "$workflow_script" \
      --config "$config_rel" \
      --stages 4 \
      --overwrite
    ;;
  api_judge)
    run_logged "eval_api_judge" \
      "$eval_script" \
      --config "$config_rel" \
      --mode api_judge \
      --output-dir "$eval_out_dir" \
      --generations-jsonl "$generations_jsonl" \
      --baseline-method "$baseline_method" \
      --candidate-methods "$candidate_methods" \
      --max-requests "$max_requests" \
      --openai-model "$openai_model" \
      --openai-api-key "$openai_api_key" \
      --update-results-manual
    run_logged "workflow_stage4_post_eval" \
      "$workflow_script" \
      --config "$config_rel" \
      --stages 4 \
      --overwrite
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
