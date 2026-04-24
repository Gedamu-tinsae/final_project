# Group4 Scripts Map

Use this to avoid confusion. These are the current script roles.

## Primary scripts (use these)

1. `scripts/run_group4_workflow.py`
- Orchestration stages:
  - Stage 1 preflight
  - Stage 2 experiment plan
  - Stage 3 registry (+ optional `--execute-plan`)
  - Stage 4 summary + comparison charts

2. `scripts/run_group4_peft_smoke.py`
- Actual Group4 training runner (LoRA / selective-FT).
- Writes model artifacts + metrics + curves.

3. `scripts/run_group4_eval.py`
- Modular evaluation entrypoint.
- Modes:
  - `template`: build generations template JSONL
  - `human_pack`: build blind pairwise pack
  - `human_aggregate`: aggregate human judgments
  - `api_judge`: API-based judging (OpenAI-compatible)

## Convenience runner scripts (optional wrappers)

These call the primary scripts with preset arguments:

- `scripts/run_group4_full_logged.sh`
- `scripts/run_group4_subset10k_tpu_logged.sh`
- `scripts/run_group4_full_smoke_logged.sh`
- `scripts/run_group4_lora_smoke_logged.sh`
- `scripts/run_group4_lora_all_weights_smoke_logged.sh`
- `scripts/run_group4_selective_smoke_logged.sh`
- `scripts/run_group4_predictions_logged.sh`

Use these when you want one-command runs with logs/meta files.

## Legacy/specialized utility

- `scripts/run_group4_smoke_predictions.py`
  - Quick prediction comparisons from smoke checkpoints.
  - Useful for qualitative checks, not the main workflow.

## Suggested minimal flow now

1. Train methods:
- `run_group4_peft_smoke.py` (or wrapper `.sh`)

2. Build summary:
- `run_group4_workflow.py --stages 4 --overwrite`

3. Build eval template:
- `run_group4_eval.py --mode template ...`

4. Human/API judging:
- `run_group4_eval.py --mode human_pack ...`
- `run_group4_eval.py --mode human_aggregate ... --update-results-manual`
  or
- `run_group4_eval.py --mode api_judge ... --update-results-manual`

5. Re-run summary after eval update:
- `run_group4_workflow.py --stages 4 --overwrite`

## One-command master runner

Use:
- `scripts/run_group4_full_logged.sh`

It now includes:
1. Workflow stages `1,2,3`
2. Training runs:
   - LoRA qv
   - LoRA all_weights
   - Selective-FT qv
3. Workflow stage `4` summary + method comparison charts
4. Optional eval stage (`EVAL_MODE`)
5. Stage 4 re-summary after eval update (for `human_aggregate`/`api_judge`)

Common env controls:
- `EXECUTOR=cloudexe|local` (default `cloudexe`)
- `GPUSPEC=H100x1`
- `MAX_ROWS`, `BATCH_SIZE`, `EPOCHS`
- `RUN_LORA_QV=1`, `RUN_LORA_ALL_WEIGHTS=1`, `RUN_SELECTIVE_QV=1`
- `EVAL_MODE=none|template|human_pack|human_aggregate|api_judge`
