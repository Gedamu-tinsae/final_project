# TPU Run Setup (Simple Steps)

This guide is for running the project on a TPU VM by connecting from your **local computer**.

## 1) Keep one source of truth

Use one repo copy as canonical (recommended: your latest remote-server repo), then:
- commit changes
- push to GitHub
- pull on TPU VM

Do not manually copy random files between machines once Git is set.

## 2) SSH from local to TPU VM

Make sure TPU host exists in your local SSH config, then connect:

```bash
ssh <your-tpu-host-alias>
```

## 3) Clone or pull repo on TPU VM

If first time:

```bash
git clone <your-repo-url> final_project
cd final_project/group1_baseline
```

If repo already exists:

```bash
cd /path/to/final_project/group1_baseline
git pull
```

## 4) Create venv on TPU VM

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

## 5) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-tpu.txt
```

## 6) Set Hugging Face token

Create `.env`:

```bash
HF_TOKEN=your_token_here
```

Load it and login:

```bash
set -a
source .env
set +a
hf auth login --token "$HF_TOKEN"
hf auth whoami
```

## 7) Preflight checks

```bash
python scripts/check_env.py --profile core
python scripts/check_env.py --profile notebook
python scripts/check_env.py --profile tpu
```

## 8) Run pipeline

Use notebook or scripts on TPU VM. For smoke tests, keep small manifests/batch sizes.

Recommended order:
1. Stage 0 (env/config)
2. Stage 1 data prep (or reuse existing)
3. Stage 2 tokenization
4. Stage 3 features (can be long)
5. Stage 4 manifests
6. Stage 4.5 model bootstrap
7. Stage 5 training smoke run
8. Stage 6 training smoke run

## 9) Save code changes correctly

When you edit code/notebook logic on TPU VM:
- commit and push code changes
- do **not** commit large data/models/artifacts
- keep `.gitignore` protecting `data/`, `artifacts/`, `.env`, `.venv`

## 10) Quick rule

Run compute where hardware lives:
- TPU jobs run on TPU VM
- local machine is for editing/SSH control
