# Group 1 Baseline Setup

## 1) Create venv

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
