# Group 2 Baseline Setup

This baseline reuses the Group 1 environment and core training modules.

## 1) Environment

From `code/group1_baseline`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use that same `.venv` for Group 2.

## 2) Paths

Update:

- `code/group2_baseline/configs/workflow_paths.json`

Set:

- `project_root`
- `stage2_root`
- `image_root`
- `clip_feature_root`

## 3) Notebook

Open:

- `code/group2_baseline/notebooks/LLaVA_Group2_Workflow.ipynb`

Run stage-by-stage. The notebook is orchestration only; heavy logic is in `src/group2_stage2`.

## 4) GPU-safe execution mode (until TPU is ready)

Use conservative settings first:

1. In Group1, verify backend:
   - `python scripts/check_accelerator.py`
2. In Group2 notebook Stage 2 prep cell:
   - keep `OVERWRITE_STAGE2_PREP = False`
   - prepare only one variant first (for example baseline only)
   - prepare only `val` split first when testing
3. In Stage 4/5:
   - keep run toggles `False` until inputs/manifests look correct
   - run one variant experiment at a time

This avoids accidental full recompute/training spend on GPU.
