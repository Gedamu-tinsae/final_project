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

## 5) Script workflow (notebook-equivalent orchestration)

Use CloudExe style as the default execution path:

```bash
# All safe orchestration stages
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group2_baseline/scripts/run_group2_workflow.py --stages all

# Stage 1-3 only
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group2_baseline/scripts/run_group2_workflow.py --stages 1,2,3

# Stage 2 prep for all variants + train/val splits
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group2_baseline/scripts/run_group2_workflow.py --stages 2 --stage2-variants all --stage2-splits train,val

# Quantity stage with input preparation
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group2_baseline/scripts/run_group2_workflow.py --stages 5 --stage5-prepare-inputs
```


By default, `overwrite=False`, so existing expensive artifacts are reused/skipped.

### Recommended smoke check before Group 4 work

CloudExe style (recommended):

```bash
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group2_baseline/scripts/run_group2_workflow.py \
  --stages 1,2,3,6 --stage2-variants baseline --stage2-splits val
```

What this does:
- `--stages 1,2,3,6`: runs audit/split, stage2 prep, quality artifacts, and heldout/pairwise generation
- `--stage2-variants baseline`: only baseline variant (fast + safe)
- `--stage2-splits val`: only validation split prep (smaller than train+val)

### Stage 4/5 sanity checks (CloudExe style)

```bash
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group2_baseline/scripts/run_group2_workflow.py --stages 4
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group2_baseline/scripts/run_group2_workflow.py --stages 5
```

Expected before full experiments:
- Stage 4 may report missing variants in `all_results_manual.json`
- Stage 5 may report missing `engine_comparison_summary.json`

These are acceptable gating messages; traceback exceptions are the real failures.
