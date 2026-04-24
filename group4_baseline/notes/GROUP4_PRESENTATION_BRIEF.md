# Group 4 Presentation Brief (PEFT Track)

## 1) What Group 4 Does
- Goal: compare parameter-efficient adaptation strategies on top of Group 1 baseline artifacts.
- Methods compared:
  - `LoRA` (q/v adapter tuning)
  - `Selective FT` (update only a small targeted subset of base model weights)
- We track:
  - quality proxies (`smoke_loss_first`, `smoke_loss_last`, `val_loss`, `val_token_accuracy`, `val_perplexity`, `win_rate_vs_baseline`)
  - compute/efficiency (`trainable_params_millions`, `wall_time_sec`, `steps_per_sec`, GPU stats)

## 2) Dependencies (Inputs from Other Groups)
- Required from Group 1:
  - Stage-1/Stage-2 manifests
  - `projector_stage1.pkl`
  - `projector_stage2.pkl`
  - `llama_stage2.pkl`
- Optional from Group 2:
  - `engine_comparison_summary.json` (used in broader reporting flow, not required for PEFT smoke training execution)

## 3) Group 4 Workflow
- Script: `scripts/run_group4_full_smoke_logged.sh`
- Stages executed:
  1. `run_group4_workflow.py --stages 1,2,3 --overwrite`
  2. LoRA smoke run (`run_group4_peft_smoke.py`)
  3. Selective FT smoke run (`run_group4_peft_smoke.py`)
  4. Summary stage (`run_group4_workflow.py --stages 4 --overwrite`)

## 4) Experimental Setup (Current Smoke Runs)
- Hardware: `cloudexe` GPU (`H100x1`)
- Shared config for smoke:
  - `smoke_rows=64`
  - `batch_size=1`
  - `epochs=1`
  - `learning_rate=1e-5`
  - `dtype=bfloat16`
- LoRA config:
  - method=`lora`, variant=`qv`, target_modules=`qv`
- Selective FT config:
  - method=`selective_ft`, target_modules=`qv`, strategy=`magnitude`, budget_pct=`1.0`

## 5) Current Results Snapshot
Source files:
- `artifacts/peft_smoke/lora_lora-qv_target-qv_rows-64_seed-42_metrics.json`
- `artifacts/peft_smoke/selective_ft_lora-na_target-qv_rows-64_seed-42_metrics.json`

### LoRA (qv)
- trainable params: `2.424832M`
- loss first/last: `3.9412 -> 4.1503`
- val_loss: `2.9617`
- wall time: `209.26s`
- steps/sec: `0.2437`
- GPU util avg/max: `2.41% / 83%`

### Selective FT (qv, 1%)
- trainable params: `5.767168M`
- loss first/last: `4.0808 -> 3.3573`
- val_loss / token_accuracy / perplexity: available on reruns with current script
- wall time: `213.30s`
- steps/sec: `0.3000`
- GPU util avg/max: `0.25% / 13%`

## 6) Efficiency Comparison (Key Slide)
- LoRA vs Selective FT trainable parameters:
  - LoRA: `2.424832M`
  - Selective FT: `5.767168M`
  - LoRA uses about **58% fewer trainable parameters**.

## 7) Alignment/Performance Story (What to Say)
- At smoke scale, both methods run end-to-end and produce reproducible metrics/artifacts.
- LoRA shows stronger parameter efficiency.
- Selective FT showed improving training loss over the smoke run.
- Final alignment/performance claim should use:
  - validation loss + token accuracy + perplexity from both methods
  - automatic `win_rate_vs_baseline` from current script (or pairwise eval if you add external judging)

## 8) Artifacts You Can Point To
- `group4_baseline/artifacts/peft_smoke/*_metrics.json`
- `group4_baseline/artifacts/peft_smoke/*_projector.pkl`
- `group4_baseline/artifacts/peft_smoke/*_llama.pkl`
- `group4_baseline/data/processed/group4_experiment_plan.json`
- `group4_baseline/data/processed/group4_run_registry.json`
- `group4_baseline/data/processed/group4_results_manual.json`

## 9) Recommended Final Step Before Presentation
- Re-run LoRA + selective FT once with current script and confirm metrics are written in both:
  - each method metrics JSON
  - `group4_results_manual.json`
- Then run Stage 4 summary to generate final ranking markdown/json.
