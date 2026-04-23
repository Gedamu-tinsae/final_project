# TPU Setup (Python 3.11) + 10K Subset Runs

This note is the TPU-specific setup/run guide for this repo.

It covers:
- Python 3.11 venv on TPU
- correct dependency install for TPU (not CloudEXE CUDA stack)
- deterministic 10K subset flow
- logged runs for Group1, Group2, Group4

## 1) Connect to TPU and go to repo

```bash
ssh tpu-v6e-node
cd ~/final_project
```

## 2) Create Group1 venv with Python 3.11

System `python` on TPU may be 3.10. Use project venv with 3.11.

```bash
cd ~/final_project/group1_baseline
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

Expected: `Python 3.11.x`

## 3) Install dependencies (TPU path)

```bash
cd ~/final_project/group1_baseline
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-tpu.txt

python scripts/check_env.py --profile core
python scripts/check_env.py --profile tpu
```

Important:
- Do **not** install CloudEXE CUDA-specific packages here (`jax[cuda12]`, `nvidia-*`).
- TPU env and CloudEXE GPU env are intentionally different.

## 4) Ensure run scripts are executable

```bash
cd ~/final_project
chmod +x group1_baseline/scripts/run_group1_subset10k_tpu_logged.sh
chmod +x group2_baseline/scripts/run_group2_tpu_logged.sh
chmod +x group4_baseline/scripts/run_group4_subset10k_tpu_logged.sh
chmod +x run_all_groups_subset10k_tpu_logged.sh
```

## 5) Run Group1 on deterministic 10K subset (logged)

```bash
cd ~/final_project/group1_baseline
source .venv/bin/activate
./scripts/run_group1_subset10k_tpu_logged.sh
```

What this does:
1. Creates subset profile (10,000 rows, seed 42)
2. Writes subset config: `configs/workflow_paths_subset_10000.json`
3. Runs Group1 workflow on subset with training enabled
4. Logs runtime/memory via `/usr/bin/time -v`

## 6) Run Group2 (logged)

```bash
cd ~/final_project/group2_baseline
source ../group1_baseline/.venv/bin/activate
./scripts/run_group2_tpu_logged.sh
```

Default sequence:
- stage `1,2,3,6` (baseline + val split)
- then stage `4`
- then stage `5` with `--stage5-prepare-inputs`

## 7) Run Group4 on same 10K subset (logged)

```bash
cd ~/final_project/group4_baseline
source ../group1_baseline/.venv/bin/activate
./scripts/run_group4_subset10k_tpu_logged.sh
```

This runs:
- Group4 workflow prep stages
- LoRA experiment (`max_rows=10000`)
- selective FT experiment (`max_rows=10000`)
- summary stage
- cross-run comparison report generation

## 8) One-command run for all groups

```bash
cd ~/final_project
source group1_baseline/.venv/bin/activate
./run_all_groups_subset10k_tpu_logged.sh
```

This runs Group1 -> Group2 -> Group4 in order and writes a combined orchestrator log.

## 9) Logs and where to look

All logs:
- `~/final_project/logs/runs/`

Example:

```bash
ls -1 ~/final_project/logs/runs | tail -n 20
tail -f ~/final_project/logs/runs/<timestamp>_<tag>.log
```

Meta files include exit code and command context:
- `*_group1_subset10k_tpu.meta.txt`
- `*_group2_tpu.meta.txt`
- `*_group4_subset10k_tpu.meta.txt`
- `*_all_groups_subset10k_tpu.meta.txt`

## 10) Common mistakes

1. Running with system Python 3.10 instead of `.venv` Python 3.11.
2. Installing CloudEXE GPU CUDA deps on TPU.
3. Running scripts without activating `group1_baseline/.venv`.
4. Comparing results from different subset sizes.

## 11) One-line quick start

```bash
cd ~/final_project/group1_baseline && source .venv/bin/activate && ./scripts/run_group1_subset10k_tpu_logged.sh
```
