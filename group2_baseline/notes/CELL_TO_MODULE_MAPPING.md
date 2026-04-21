# Group2 Cell-to-Module Mapping

Notebook source: `legacy/LLaVA_Public_Group2.ipynb` (94 cells)

## Reused Group1 baseline logic

- Cells 0-57: baseline setup, stage1 pipeline, core stage2 training internals
- Reused from `code/group1_baseline/src/*`

## Group2-specific modularized logic

- Cells 58-62 (variant audit, shared quality pool, train/val split):
  - `src/group2_stage2/audit.py`
  - `src/group2_stage2/splits.py`

- Cells 60, 63-65 (stage2 serialization/tokenization, features, manifests, prep loop):
  - `src/group2_stage2/tokenization.py`
  - `src/group2_stage2/features.py`
  - `src/group2_stage2/manifests.py`
  - `src/group2_stage2/pipeline.py`

- Cells 66-76 (stage2 collate/training/eval orchestration, reset snapshots, experiment runner):
  - `src/group2_stage2/training_orchestration.py`

- Cells 77-82, 91 (quality diagnostics, result tracking, prompt audit, engine comparison, baseline-relative comparison):
  - `src/group2_stage2/quality_eval.py`
  - `src/group2_stage2/experiment_tracking.py`

- Cells 82-87 (quantity plan/build/register/prepare/run/summarize):
  - `src/group2_stage2/quantity_ablation.py`

- Cell 90 (plots + markdown report table):
  - `src/group2_stage2/reporting.py`

- Cells 92-93 (heldout eval pack + pairwise judge requests):
  - `src/group2_stage2/evaluation_pack.py`
  - `src/group2_stage2/quality_eval.py` (`build_pairwise_judge_requests`)

## Helper script

- `scripts/run_group2_nonmodel.py` covers early non-model flow:
  - audit
  - shared pool
  - deterministic split generation

