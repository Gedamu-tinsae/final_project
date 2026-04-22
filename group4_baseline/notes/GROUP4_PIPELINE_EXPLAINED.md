# Group 4 Pipeline Explained (What Is Actually Training?)

## Short answer
- `run_group4_workflow.py` is orchestration only (planning/checking/summarizing).
- `run_group4_peft_smoke.py` does real training updates (LoRA or selective fine-tuning), but on a small budget by default.

So yes: despite the word "smoke", it is still actual gradient-based training.  
"Smoke" here means **small/quick run**, not "fake run".

## Why this design exists
Group 4 needs two things:
1. Reproducible experiment management (plan, registry, summary).
2. Actual PEFT model updates and metrics collection.

Those are split into two scripts on purpose.

## Script responsibilities

### 1) `scripts/run_group4_workflow.py`
What it does:
- Stage 1: preflight checks (upstream artifacts exist).
- Stage 2: generate experiment plan JSON.
- Stage 3: generate run registry JSON.
- Stage 4: summarize results (ranking), if results file exists.

What it does **not** do:
- no model loading for training
- no optimizer steps
- no GPU telemetry collection
- no LoRA/selective-FT parameter updates

### 2) `scripts/run_group4_peft_smoke.py`
What it does:
- Loads model + projector state.
- Runs actual training steps with optimizer updates.
- Supports:
  - `--method lora`
  - `--method selective_ft`
- Writes run metrics including:
  - trainable parameter counts
  - loss progression (`loss_first`, `loss_last`)
  - runtime/throughput (`wall_time_sec`, `steps_per_sec`, `samples_per_sec`)
  - GPU stats (`gpu_stats`: util/mem/power)

Why it says "smoke":
- It uses reduced defaults (`--max-rows`, low epochs/batch), so you can validate setup quickly.

## End-to-end Group 4 execution order

1. Build/check experiment metadata:
```bash
python scripts/run_group4_workflow.py --stages 1,2,3
```

2. Run at least one LoRA training run:
```bash
python scripts/run_group4_peft_smoke.py --method lora --lora-variant qv --target-modules qv --max-rows 64 --epochs 1 --batch-size 1 --append-manual-results
```

3. Run at least one selective-FT training run:
```bash
python scripts/run_group4_peft_smoke.py --method selective_ft --target-modules qv --selection-strategy magnitude --budget-pct 1.0 --max-rows 64 --epochs 1 --batch-size 1 --append-manual-results
```

4. Add evaluation-side metrics (`val_loss`, `win_rate_vs_baseline`) to manual results.

5. Generate final comparison summary:
```bash
python scripts/run_group4_workflow.py --stages 4
```

## Inputs and outputs

### Inputs (from upstream groups)
- Group 1 artifacts (manifest + projector/model states).
- Group 2 comparison summary (for dependency checks / alignment).

### Outputs (Group 4)
- Planning:
  - `data/processed/group4_experiment_plan.json`
  - `data/processed/group4_run_registry.json`
- Training run artifacts:
  - `artifacts/peft_smoke/*_projector.pkl`
  - `artifacts/peft_smoke/*_llama.pkl`
  - `artifacts/peft_smoke/*_metrics.json`
- Summary:
  - `data/processed/group4_results_summary.json`
  - `data/processed/group4_results_summary.md`

## Presentation mapping
- Efficiency: `trainable_params_millions`
- Quality: `loss_last` (plus `val_loss`, `win_rate_vs_baseline` after eval)
- Compute cost: `wall_time_sec`, `steps_per_sec`, `gpu_stats`
