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

Update (default subset run):

- `code/group2_baseline/configs/workflow_paths_subset_10000.json`

Optional full-data config:

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

Fallback direct-server style (same behavior):

```bash
cd /root/final_project/group2_baseline
source /root/final_project/group1_baseline/.venv/bin/activate
python scripts/run_group2_workflow.py --stages all
```

By default, `overwrite=False`, so existing expensive artifacts are reused/skipped.
Group2 scripts now default to `workflow_paths_subset_10000.json`.

### Recommended smoke check before Group 4 work

CloudExe style (recommended):

```bash
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group2_baseline/scripts/run_group2_workflow.py \
  --stages 1,2,3,6 --stage2-variants baseline --stage2-splits val
```

Equivalent direct-server style:

```bash
cd /root/final_project/group2_baseline
source /root/final_project/group1_baseline/.venv/bin/activate
python scripts/run_group2_workflow.py --stages 1,2,3,6 --stage2-variants baseline --stage2-splits val
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

## 6) Logging runs with tee

Use the wrapper script (recommended). It handles:
- log file creation
- meta file creation
- correct `cloudexe | tee` exit-code handling
- no fragile shell variables required

```bash
cd /root/final_project/group2_baseline
./scripts/run_group2_full_logged.sh
```

Optional args:

```bash
./scripts/run_group2_full_logged.sh --stages 1,2,3,6 --stage2-variants baseline --stage2-splits val --overwrite
```

Optional GPU override:

```bash
GPUSPEC=EUNH100x1 ./scripts/run_group2_full_logged.sh
```

Logs are written to:
- `/root/final_project/logs/runs/*_group2_full.log`
- `/root/final_project/logs/runs/*_group2_full.meta.txt`

Monitor the latest log:

```bash
tail -f /root/final_project/logs/runs/*_group2_full.log
```

TPU subset convenience script:

```bash
cd ~/final_project/group2_baseline
source ../group1_baseline/.venv/bin/activate
./scripts/run_group2_tpu_logged.sh
```

Before workflow stages, the runner now auto-executes:

```bash
python scripts/create_stage2_subset_profile.py \
  --config configs/workflow_paths_subset_10000.json \
  --rows 10000 \
  --seed 42 \
  --overwrite
```

This trims each variant `stage2_dataset.jsonl` to deterministic 10k and writes:
- `data/processed/subsets/subset_10000_seed42/stage2_instruction/subset_manifest.json`

So downstream Group2 stages always consume 10k subset inputs.

Default TPU sequence runs:
1. `--stages 1,2,3,6 --stage2-variants baseline --stage2-splits val --overwrite`
2. `--stages 4 --overwrite`
3. `--stages 5 --stage5-prepare-inputs --overwrite`

### tmux variant

1. Start tmux:

```bash
tmux new -s g2_full
```

2. Inside tmux, run:

```bash
cd /root/final_project/group2_baseline
./scripts/run_group2_full_logged.sh
```

3. Detach without stopping: `Ctrl+b`, then `d`

4. Reattach later:

```bash
tmux attach -t g2_full
```

5. Monitor log from another shell:

```bash
tail -f /root/final_project/logs/runs/*_group2_full.log
```

## 7) Standard metrics/figure outputs (all stages)

Each run now writes to:
- `outputs/group2/<run_id>/...`
- plus stage contract directories: `outputs/group2/stage<k>/<run_id>/...`

Core files per run/stage:
- `run_config.json`
- `stage_meta.json`
- `timing.json`
- `resource_usage.csv`
- `stdout.log`
- `artifacts_manifest.json`

Plot data and figures:
- `plots_data/stage_timing.csv`
- `plots_data/resource_usage.csv`
- `fig_stage_timing.png`
- `fig_throughput_steps_per_sec.png`
- `fig_memory_usage.png`
- `fig_overview_dashboard.png`

## 8) Cross-run comparison pack (with Group1/Group4)

```bash
cd /root/final_project
python common/generate_comparison_report.py --outputs-root outputs --run-name presentation
```

This creates:
- `outputs/comparison/<run_id>/comparison_table.csv`
- `outputs/comparison/<run_id>/comparison_table.md`
- `outputs/comparison/<run_id>/comparison_figures/*.png`
- `outputs/REPORT_INDEX.md`
