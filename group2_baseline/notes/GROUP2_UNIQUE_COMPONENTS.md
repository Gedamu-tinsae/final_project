# Group2 Unique Components

Group2 builds on Group1 baseline and adds stage2 comparison experiments.

## Reused From Group1

- COCO acquisition and stage1 data prep
- CLIP helpers and feature extraction primitives
- Core projector + stage1/stage2 training internals
- Llama loader and config patterns

## Added In Group2 Baseline

- `src/group2_stage2/audit.py`
  - Validates each variant dataset and checks image-set consistency.
- `src/group2_stage2/splits.py`
  - Builds `shared_quality_pool.json` and `shared_split.json`.
  - Creates deterministic `stage2_train.jsonl` and `stage2_val.jsonl` per variant.
- `src/group2_stage2/tokenization.py`
  - Serializes stage2 chat-style samples into supervised token sequences.
- `src/group2_stage2/features.py`
  - Precomputes stage2 CLIP features with shared cache paths.
- `src/group2_stage2/manifests.py`
  - Builds train/val/full stage2 manifests.
- `src/group2_stage2/pipeline.py`
  - One orchestration entrypoint for tokenization + features + manifests.
- `src/group2_stage2/quality_eval.py`
  - Dataset quality diagnostics
  - Qualitative sample pack generation
  - Pairwise judge request generation
- `src/group2_stage2/training_orchestration.py`
  - Stage2 collator/minibatch iterator
  - Stage2 train/eval orchestration helpers
  - Stage2 snapshot save/load
- `src/group2_stage2/experiment_tracking.py`
  - Variant selection and result persistence
  - Prompt alignment audit
  - Engine ranking summary
  - Baseline-relative comparison
- `src/group2_stage2/quantity_ablation.py`
  - Quantity plan derivation from quality summary
  - Quantity dataset creation/register/prepare/run/summarize
- `src/group2_stage2/evaluation_pack.py`
  - Held-out validation evaluation pack generation
- `src/group2_stage2/reporting.py`
  - Engine comparison figures and markdown table
