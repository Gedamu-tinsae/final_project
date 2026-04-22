# Group 4 Baseline Setup (Parameter-Efficient Tuning Track)

Group 4 builds on top of Group 1 and Group 2 artifacts.

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

## 6) After training runs are executed

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
