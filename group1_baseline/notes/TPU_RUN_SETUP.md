# TPU Run Setup (Step-by-Step + Fixes Inline)

This is the practical flow that worked for our setup (`tunix-vm`).

## 1) Use one source of truth for code

- Keep one canonical repo copy.
- Push changes there.
- Pull on TPU host before running.

If this fails:
- If local/remote drift happens, compare and sync before running.

## 2) Authenticate gcloud correctly (local machine)

```bash
gcloud auth login
gcloud auth list
gcloud config set account <correct-email>
gcloud config set project computer-vision-491521
```

If this fails:
- `invalid_grant: Account has been deleted` -> run `gcloud auth login` again and set correct account.
- `Required ... permission` -> wrong account/project. Switch account/project to one that owns the VM.

## 3) SSH into `tunix-vm` from local

```bash
ssh tunix-vm
```

If this fails:
- `Permission denied (publickey)`:
  1. Print your public key:
     ```bash
     cat "$HOME/.ssh/gke-kiya-key.pub"
     ```
  2. Add/re-add it in **Compute Engine -> Metadata -> SSH Keys**.
  3. Make sure username matches (`kiya`).
  4. Retry `ssh tunix-vm`.
- `Could not resolve hostname ...`:
  - Make sure hostname is entered on one line (no accidental line break).

Note:
- In this class setup, `tunix-vm` lands on a GKE/COS node (`gke-tunix-pathways-...`).
- SSH command runs are okay; full VS Code dev experience can be fragile there.

## 4) Get repository on TPU host

First time:

```bash
git clone <repo-url> final_project
cd final_project/group1_baseline
```

Existing repo:

```bash
cd ~/final_project/group1_baseline
git pull
```

If this fails:
- `Password authentication is not supported for Git operations`:
  - Option A: use HTTPS + PAT token (not GitHub password).
  - Option B: set up SSH key for GitHub and clone with `git@github.com:...`.
  - Option C (your working fallback): temporarily set repo to public, clone/download, then switch back to private.

## 5) Create fresh virtual environment on TPU host

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

If this fails:
- If venv seems broken, recreate:
  ```bash
  deactivate 2>/dev/null || true
  rm -rf .venv
  python3.11 -m venv .venv
  source .venv/bin/activate
  ```

## 6) Install dependencies

Use `python -m pip` on this host:

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-tpu.txt
```

If this fails:
- `.venv/bin/pip: Permission denied`:
  - Recreate venv (Step 5) and use `python -m pip` only.
- `chmod u+x .venv/bin/*` fails with `Read-only file system`:
  - expected on this COS/GKE host; do not rely on executable scripts in `.venv/bin/`.
  - use `python -m ...` and direct Python commands instead of shell wrappers.

## 7) Configure Hugging Face token

Create `.env`:

```bash
HF_TOKEN=your_token_here
```

Load token:

```bash
set -a
source .env
set +a
```

Working method on this host (recommended):

```bash
python -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'])"
python -c "from huggingface_hub import whoami; print(whoami())"
```

If this fails:
- `hf auth ...` gives `Permission denied`:
  - skip `hf` CLI entirely on this host and use the Python commands above.
- `Not logged in`:
  - ensure `.env` was sourced (`echo $HF_TOKEN` should not be empty).

## 8) Preflight checks

```bash
python scripts/check_env.py --profile core
python scripts/check_env.py --profile notebook
python scripts/check_env.py --profile tpu
```

If this fails:
- missing imports -> install missing requirements in active `.venv`.

Known-good output pattern:
- `whoami()` returns your HF account JSON.
- all three profiles print `All required imports are available.`

## 9) Run pipeline (smoke first, then full)

Recommended order:
1. Stage 0 (env/config)
2. Stage 1 data prep
3. Stage 2 tokenization
4. Stage 3 feature extraction
5. Stage 4 manifests
6. Stage 4.5 model bootstrap
7. Stage 5 smoke run
8. Stage 6 smoke run

If this fails:
- Stage 5 `FileNotFoundError` for clip embeddings:
  - Stage 3 incomplete or manifest built before features finished.
  - use smoke manifest first, then full run later.

Important for this host (`tunix-vm` / GKE COS):
- Jupyter/notebook may fail even after install with errors like:
  - `..._zmq...so: file not located on exec mount`
- Cause: `/home` on this host is `noexec`, and pyzmq native extension cannot load there.
- Practical decision: run stages via Python scripts/commands on this host, not notebook UI.

Script-first smoke flow (works on this host):
```bash
python scripts/run_tpu_smoke.py --overwrite
```

This script handles:
- smoke manifest creation (stage1 + stage2)
- model artifact ensure/load (stage 4.5 equivalent)
- stage 5 smoke run
- stage 6 smoke run

## 10) Save changes safely

- Commit/push code and notebook logic only.
- Do not commit large artifacts/data/models.
- Keep `.gitignore` covering:
  - `.env`, `.venv`, `data/raw`, `data/processed`, `data/models`, `artifacts`.

Quick rule:
- Edit/control from local.
- Run heavy compute on TPU host.
