# COSI 159A Final Project

End-to-end multimodal fine-tuning workflow with a Group4 focus on parameter-efficient tuning (LoRA, selective FT, ReLoRA), built on upstream Group1/Group2 artifacts.

## Repository Layout
- `code/group1_baseline/`  
  Baseline multimodal pipeline (data prep, manifests, staged training artifacts).
- `code/group2_baseline/`  
  Variant/engine comparison workflow and reporting artifacts.
- `code/group4_baseline/`  
  PEFT workflows: LoRA, selective FT, ReLoRA, evaluation pipelines.
- `Documentation/presentation/final_class_presentation/`  
  Final slides, report draft, artifact index, runbooks.
- `legacy/`  
  Historical notebooks/experiments retained for reference.
- `outputs/`  
  Run outputs (metrics, logs, plots) written by workflows.

## Project Goal
Compare parameter-efficient methods on a fixed multimodal baseline under constrained compute:
1. Quality (`win_rate_vs_baseline`, `val_loss`)
2. Efficiency (`trainable_params_millions`)
3. Compute (`wall_time_sec`, throughput/resource logs)

## Core Dependency Order
1. Run/verify Group1 artifacts
2. Run/verify required Group2 summary artifact
3. Run Group4 PEFT workflows and evaluation

## Quick Start (TPU shell)
```bash
cd ~/final_project
source code/group1_baseline/.venv/bin/activate
```

### Group1
```bash
cd ~/final_project/code/group1_baseline
./scripts/run_group1_subset10k_tpu_logged.sh
```

### Group2
```bash
cd ~/final_project/code/group2_baseline
./scripts/run_group2_tpu_logged.sh
```

### Group4 (LoRA/SFT)
```bash
cd ~/final_project/code/group4_baseline
./scripts/run_group4_full_logged.sh
```

### Group4 (ReLoRA)
```bash
cd ~/final_project/code/group4_baseline
./scripts/run_group4_relora_logged.sh
```

## Key Artifacts
- Group4 final comparison:
  - `.../group4_baseline/data/processed/subsets/subset_10000_seed42/final_comparison/.../group4_3way_comparison.{md,csv,json}`
- Human eval summaries:
  - `.../eval/human_eval_summary.json`
  - `.../eval/relora/human_eval_summary.json`

For full path-indexed references, see:
- `Documentation/presentation/final_class_presentation/00_index/artifacts_index.md`

## Final Presentation and Report
- Slides content:
  - `Documentation/presentation/final_class_presentation/COSI Final slides.md`
- Final report draft:
  - `Documentation/presentation/final_class_presentation/FINAL_REPORT_DRAFT.md`

## Notes
- Main experimental subset: `subset_10000_seed42`
- Typical Group4 defaults: `epochs=1`, `batch_size=1`, `learning_rate=1e-5`, `dtype=bfloat16`
- ReLoRA track is isolated in separate config/output roots to avoid overwrite with LoRA/SFT outputs.
