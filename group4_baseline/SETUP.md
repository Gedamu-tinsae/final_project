# Group 4 Baseline Setup (Parameter-Efficient Tuning Track)

Group 4 builds on top of Group 1 and Group 2 artifacts.
The Group4 PEFT backbones are now vendored under:
- `src/group4_backbones/`

## 1) Use Group1 environment

```bash
cd /root/final_project/group1_baseline
source .venv/bin/activate
```

Group 4 scripts are pure orchestration and can run in the same `.venv`.

## 2) Run Group4 preflight

```bash
cd /root/final_project/group4_baseline
python scripts/check_group4_inputs.py
```

This verifies required upstream artifacts exist:
- Group1 manifests + stage states
- Group2 engine summary

## 3) Build Group4 plan/registry (safe, no training)

```bash
python scripts/run_group4_workflow.py --stages 1,2,3
```

Outputs:
- `data/processed/group4_experiment_plan.json`
- `data/processed/group4_run_registry.json`

By default, existing files are reused (`overwrite=False`).

## 4) CloudExe usage

```bash
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group4_baseline/scripts/run_group4_workflow.py --stages 1,2,3
```

## 5) Run PEFT smoke experiments (actual Group4 implementation)

LoRA (partner qv variant):

```bash
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group4_baseline/scripts/run_group4_peft_smoke.py \
  --method lora --lora-variant qv --target-modules qv --max-rows 64 --batch-size 1 --epochs 1 --append-manual-results
```

LoRA (partner all-weights variant):

```bash
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group4_baseline/scripts/run_group4_peft_smoke.py \
  --method lora --lora-variant all_weights --target-modules all --max-rows 64 --batch-size 1 --epochs 1 --append-manual-results
```

Selective fine-tuning baseline:

```bash
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group4_baseline/scripts/run_group4_peft_smoke.py \
  --method selective_ft --target-modules qv --selection-strategy magnitude --budget-pct 1.0 --max-rows 64 --batch-size 1 --epochs 1 --append-manual-results
```

Outputs from each run:
- `artifacts/peft_smoke/*_metrics.json`
- optional append into `data/processed/group4_results_manual.json`

Key presentation metrics now logged per run:
- `trainable_params_total`, `trainable_params_millions`
- `loss_first`, `loss_last`
- `wall_time_sec`, `steps_per_sec`, `samples_per_sec`
- `gpu_stats` (avg/max GPU util, avg/max memory used, avg/max power)

## 6) LoRA vs Selective-FT comparison flow (for slides)

Run at least one LoRA config and one selective-FT config, then compare:
1. Efficiency: `trainable_params_millions`
2. Quality: `loss_last` (plus `val_loss` / `win_rate_vs_baseline` when available)
3. Compute cost: `wall_time_sec`, `steps_per_sec`, `gpu_stats`

Suggested smoke pair:
- LoRA: `--method lora --lora-variant qv --target-modules qv`
- Selective FT: `--method selective_ft --target-modules qv --selection-strategy magnitude --budget-pct 1.0`

Important:
- `run_group4_peft_smoke.py` logs training-side metrics automatically.
- `val_loss` and `win_rate_vs_baseline` are not auto-produced by the smoke runner; fill them after evaluation.

## 7) After training runs are executed

Populate:
- `data/processed/group4_results_manual.json`

Format:

```json
{
  "results": [
    {
      "experiment_id": "g4_lora_001",
      "method": "lora",
      "win_rate_vs_baseline": 0.54,
      "val_loss": 3.12,
      "trainable_params_millions": 12.3
    }
  ]
}
```

Then summarize:

```bash
python scripts/run_group4_workflow.py --stages 4
```

Outputs:
- `data/processed/group4_results_summary.json`
- `data/processed/group4_results_summary.md`
