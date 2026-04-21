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

