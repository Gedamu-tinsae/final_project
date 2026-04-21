# Group2 Data Directory Guide

## Overview

Group2 data outputs are centered on Stage 2 variant comparison experiments.

- `data/raw/`
  - Reuses COCO images from Group1 baseline (`train2017/*.jpg`).
- `data/processed/stage2_instruction/`
  - Main Group2 experiment artifacts (variant datasets, splits, manifests, summaries).
- `data/processed/stage2_features/`
  - Precomputed CLIP `.npy` features used by Group2 Stage 2 manifests/training.

## `data/processed/stage2_instruction/`

This is the core Group2 output directory.

### Global control files

- `shared_quality_pool.json`
  - Selected common image IDs used across all variants for fair comparison.
- `shared_split.json`
  - Deterministic train/val split for those selected image IDs.
- `dataset_quality_diagnostics.json`
  - Per-variant statistics (row counts, task distribution, prompt/response length stats, duplicates).
- `qualitative_comparison_samples.json`
  - Aligned examples used for manual side-by-side response inspection.
- `prompt_alignment_audit.json`
  - Checks whether variants use the same instruction text per aligned sample.

### Engine comparison outputs

- `engine_comparison_summary.json`
  - Ranking summary of variant training/eval outcomes.
- `baseline_relative_comparison.json`
  - Comparison of each candidate variant against baseline (`gemma` by default).
- `report_figures/`
  - Plots/tables generated from comparison results.

### Pairwise and heldout evaluation outputs

- `heldout_eval_pack.json`
  - Held-out aligned samples for evaluation.
- `pairwise_judge_requests.json`
  - Pairwise A/B prompts for human or model judging.
- `pairwise_judge_results_template.json`
  - Empty template to fill judgment outcomes.
- `pairwise_judge_results_filled.json`
  - Completed judge outcomes (if produced).
- `pairwise_judge_summary.json`
  - Aggregated summary of pairwise judgments.

### Quantity ablation outputs

- `quantity_ablation/`
  - Generated quantity-specific variant datasets.
- `quantity_registration_status.json`
  - Tracks which quantity variants were registered into standard variant folders.
- `quantity_prep_status.json`
  - Tracks tokenization/feature/manifest preparation for quantity variants.
- `quantity_results.json`
  - Training/eval outcomes for quantity experiments.
- `quantity_results_summary.json`
  - Final quantity ranking + best quantity choice.

### Variant subfolders (example: `gemma/`, `qwen/`, `llama/`)

Each variant directory under `stage2_instruction/` commonly contains:

- `stage2_dataset.jsonl`
  - Full Stage 2 samples for that generator variant.
- `stage2_train.jsonl`
  - Train subset from shared split.
- `stage2_val.jsonl`
  - Validation subset from shared split.
- `stage2_tokenized_train.json`
  - Tokenized train records.
- `stage2_tokenized_val.json`
  - Tokenized val records.
- `stage2_tokenized_full.json`
  - Tokenized full dataset (optional, depending on run path).
- `stage2_manifest_train.json`
  - Train manifest linking tokenized text to CLIP feature file paths.
- `stage2_manifest_val.json`
  - Val manifest linking tokenized text to CLIP feature file paths.
- `stage2_manifest_full.json`
  - Full manifest (optional).
- `metadata.json`
  - Extra metadata for quantity-generated variants.

## `data/processed/stage2_features/`

- Contains CLIP feature tensors as `.npy` files (usually by image filename stem).
- These are reused across variants/splits to avoid recomputing vision features.
- Missing feature files cause manifest rows to be dropped or training to fail for those samples.

## Reusing Group1 CLIP Features (Avoid Expensive Recompute)

Group2 is configured to reuse Group1 CLIP embeddings first, then compute only missing features.

- Config key: `reuse_clip_feature_roots` in `configs/workflow_paths.json`
- Default points to:
  - `${PROJECT_ROOT}/../group1_baseline/data/processed/clip_embeddings`

Behavior in Stage 2 prep:

- If feature exists in Group2 `clip_feature_root`: skip compute.
- Else if feature exists in any `reuse_clip_feature_roots`: reuse that path in manifests (no recompute).
- Else: compute feature and save under Group2 `clip_feature_root`.

## Safety Behavior (Overwrite Guard)

Group2 pipeline functions are guarded to avoid expensive accidental overwrite:

- If output exists and `overwrite=False`, steps return a `skipped_existing` mode.
- To regenerate intentionally, rerun with `overwrite=True` (or delete target outputs first).

This matches the Group1 safety style: no silent destructive regeneration.
