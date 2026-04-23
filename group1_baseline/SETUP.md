# Group 1 Baseline Setup

## 1) Create venv

```bash
python -m venv .venv
```

Remote server (recommended for TPU path):

```bash
python3.11 -m venv .venv
```

## 2) Activate

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

## 3) Choose install profile

### Option A: Core code only (`src/` modules)

```bash
pip install --upgrade pip
pip install -r requirements-core.txt
```

### Option B: Core + notebook storytelling workflow

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` includes:
- `requirements-core.txt`
- `requirements-notebook.txt`

### Option C: Full TPU stack (core + notebook + tunix/qwix)

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-tpu.txt
```

### Option D: GPU-safe path (recommended until TPU is ready)

Install regular requirements first:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Then verify accelerator backend:

```bash
python scripts/check_accelerator.py
```

If backend is `cpu`, GPU JAX is not installed yet (safe to continue smoke on CPU, but slower).

## 4) Run environment preflight check

From `code/group1_baseline`:

```bash
python scripts/check_env.py --profile core
python scripts/check_env.py --profile notebook
python scripts/check_env.py --profile tpu
```

Or all at once:

```bash
python scripts/check_env.py --profile all
```

Run this before notebook stages.

## 5) VS Code / Pylance

Select the `.venv` interpreter for this folder, otherwise Pylance will show
"import could not be resolved" even if the code is valid.

## 6) VS Code Notebook (Jupyter) Setup

From `code/group1_baseline` (with `.venv` activated), install and register kernel:

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name group1-baseline --display-name "Python (group1-baseline)"
```

In VS Code:

1. `Ctrl+Shift+P` -> `Python: Select Interpreter` -> choose this folder's `.venv`.
2. Open `notebooks/LLaVA_Baseline_Workflow.ipynb`.
3. Click `Select Kernel` (top-right) -> choose `Python (group1-baseline)` (or `.venv`).
4. Run cells top-to-bottom by stage.

## 7) `.env` and Config Files

Use:
- `.env` for secrets (example: `HF_TOKEN`)
- `configs/workflow_paths.json` for paths/artifact locations

`.env` example:

```bash
HF_TOKEN=your_hf_token_here
```

Quick token verification:

```bash
cat .env
echo $HF_TOKEN
```

The workflow notebook loads both automatically in Stage 0.

## 8) Safe GPU Smoke Run (no full expensive run)

From `code/group1_baseline`:

```bash
python scripts/run_tpu_smoke.py --max-rows 64 --stage1-batch-size 1 --stage2-batch-size 1 --dtype float32
```

Notes:
- Keep `--overwrite` off unless you intentionally want to regenerate artifacts.
- Use small `--max-rows` first to validate pipeline wiring before full runs.

## 9) Logging full runs with tee

Use the wrapper script (recommended). It handles:
- log file creation
- meta file creation
- correct `cloudexe | tee` exit-code handling
- GPU backend verification inside CloudExe entry (`REQUIRE_GPU=1` by default)
- no fragile shell variables required

```bash
cd /root/final_project/group1_baseline
./scripts/run_group1_full_logged.sh
```

Optional args:

```bash
./scripts/run_group1_full_logged.sh --overwrite --train none
./scripts/run_group1_full_logged.sh --overwrite --train stage1
./scripts/run_group1_full_logged.sh --overwrite --train stage2
./scripts/run_group1_full_logged.sh --overwrite --train both --smoke --smoke-rows 128
```

Optional GPU override:

```bash
GPUSPEC=EUNH100x1 ./scripts/run_group1_full_logged.sh
```

If you need to allow CPU fallback temporarily:

```bash
REQUIRE_GPU=0 ./scripts/run_group1_full_logged.sh
```

Logs are written to:
- `/root/final_project/logs/runs/*_group1_full.log`
- `/root/final_project/logs/runs/*_group1_full.meta.txt`

Monitor the latest log:

```bash
tail -f /root/final_project/logs/runs/*_group1_full.log
```

### tmux variant

1. Start tmux:

```bash
tmux new -s g1_full
```

2. Inside tmux, run:

```bash
cd /root/final_project/group1_baseline
./scripts/run_group1_full_logged.sh
```

3. Detach without stopping: `Ctrl+b`, then `d`

4. Reattach later:

```bash
tmux attach -t g1_full
```

5. Monitor log from another shell:

```bash
tail -f /root/final_project/logs/runs/*_group1_full.log
```

## 10) Stage 5-7 training-only logged scripts

When Stage 1-4 are already complete, use these instead of full rerun.

### A) Full training continuation (no mesh)

```bash
cd /root/final_project/group1_baseline
./scripts/run_group1_train_nomesh_logged.sh
```

Default args:
- `--train both --no-mesh`

### B) Small-budget training continuation (recommended for fast iteration)

```bash
cd /root/final_project/group1_baseline
./scripts/run_group1_train_smoke_nomesh_logged.sh
```

Default args:
- `--train both --no-mesh --smoke --smoke-rows 256 --batch-size 1 --epochs 1`

Override example:

```bash
./scripts/run_group1_train_smoke_nomesh_logged.sh --train both --no-mesh --smoke --smoke-rows 512 --batch-size 1 --epochs 1
```

Logs/meta for both scripts:
- `/root/final_project/logs/runs/*_group1_train_nomesh.log`
- `/root/final_project/logs/runs/*_group1_train_nomesh.meta.txt`
- `/root/final_project/logs/runs/*_group1_train_smoke_nomesh.log`
- `/root/final_project/logs/runs/*_group1_train_smoke_nomesh.meta.txt`

## 11) Standard metrics/figure outputs (all stages)

Each run now writes to:
- `outputs/group1/<run_id>/...`
- plus stage contract directories: `outputs/group1/stage<k>/<run_id>/...`

Core files:
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

Training stages also write:
- `stage5/train_history.csv`, `stage5/val_history.csv`
- `stage6/train_history.csv`, `stage6/val_history.csv`
- `stage5/fig_loss_train_vs_step.png`, `stage5/fig_loss_val_vs_epoch_or_evalstep.png`
- `stage6/fig_loss_train_vs_step.png`, `stage6/fig_loss_val_vs_epoch_or_evalstep.png`
