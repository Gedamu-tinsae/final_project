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
- `val_loss`
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
- `run_group4_peft_smoke.py` now logs `val_loss` automatically from a held-out split in the smoke subset.
- `win_rate_vs_baseline` still requires pairwise eval and should be filled after that eval step.

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

## 9) Cross-run comparison report + figures

After running Group1/Group2/Group4 workflows, generate one presentation-ready comparison pack:

```bash
cd /root/final_project
python common/generate_comparison_report.py --outputs-root outputs --run-name presentation
```

Generated:
- `outputs/comparison/<run_id>/comparison_table.csv`
- `outputs/comparison/<run_id>/comparison_table.md`
- `outputs/comparison/<run_id>/comparison_figures/fig_group_comparison.png`
- `outputs/comparison/<run_id>/comparison_figures/fig_method_comparison_bar.png`
- `outputs/comparison/<run_id>/comparison_figures/fig_trainable_params_bar.png`
- `outputs/comparison/<run_id>/comparison_figures/fig_overview_dashboard.png`
- `outputs/REPORT_INDEX.md`

## 10) Group4 run output contract

Each Group4 run writes:
- `outputs/group4/<run_id>/...`
- stage contract dirs under `outputs/group4/stage<k>/<run_id>/...`

Always present:
- `run_config.json`, `stage_meta.json`, `timing.json`
- `resource_usage.csv`, `stdout.log`, `artifacts_manifest.json`
- `plots_data/stage_timing.csv`, `plots_data/resource_usage.csv`
- `fig_throughput_steps_per_sec.png`, `fig_memory_usage.png`, `fig_overview_dashboard.png`

PEFT smoke runs additionally write:
- `peft/train_history.csv`, `peft/val_history.csv`
- `peft/metrics.json`
- `peft/fig_loss_train_vs_step.png`
- `peft/fig_loss_val_vs_epoch_or_evalstep.png`

## 8) Logging runs with tee

Use the wrapper script (recommended). It handles:
- log file creation
- meta file creation
- stage-by-stage logging for workflow + PEFT runs
- comparison report generation at end
- correct exit-code handling

```bash
cd /root/final_project/group4_baseline
./scripts/run_group4_full_logged.sh
```

TPU subset convenience script:

```bash
cd ~/final_project/group4_baseline
source ../group1_baseline/.venv/bin/activate
./scripts/run_group4_subset10k_tpu_logged.sh
```

Optional GPU override:

```bash
GPUSPEC=EUNH100x1 ./scripts/run_group4_full_logged.sh
```

Logs are written to:
- `/root/final_project/logs/runs/*_group4_full.log`
- `/root/final_project/logs/runs/*_group4_full.meta.txt`

Monitor the latest log:

```bash
tail -f /root/final_project/logs/runs/*_group4_full.log
```

### tmux variant

1. Start tmux:

```bash
tmux new -s g4_full
```

2. Inside tmux, run:

```bash
cd /root/final_project/group4_baseline
./scripts/run_group4_full_logged.sh
```

3. Detach without stopping: `Ctrl+b`, then `d`

4. Reattach later:

```bash
tmux attach -t g4_full
```

5. Monitor log from another shell:

```bash
tail -f /root/final_project/logs/runs/*_group4_full.log
```
