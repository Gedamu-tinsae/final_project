# TPU Provisioning Guide (What Works)

This guide is the clean, working flow based on what we actually hit.

## What TPU provisioning means

- You do **not** immediately get a TPU VM when you request one.
- You create a **queued resource** first.
- While queue state is `WAITING_FOR_RESOURCES`, the TPU node does not exist yet.
- You can SSH only after the TPU VM appears and is `READY`.

---

## 1) Local machine setup (PowerShell)

Run on local Windows PowerShell:

```powershell
gcloud auth login
gcloud config set account tinsae.kiya.gedamu@gmail.com
gcloud config set project computer-vision-491521
gcloud services enable tpu.googleapis.com
gcloud components install alpha
```

Notes:
- `gcloud components install alpha --quiet` can fail on Windows bundled Python in non-interactive mode.
- Use interactive install (`gcloud components install alpha`).

---

## 2) Create TPU queued resource

Use one line in PowerShell:

```powershell
gcloud alpha compute tpus queued-resources create tpu-builder-queue --zone=us-east5-a --accelerator-type=v5p-8 --runtime-version=v2-alpha-tpuv5 --node-id=my-tpu-node --provisioning-model=flex-start --max-run-duration=4h --valid-until-duration=4h
```

If queue already exists:

```powershell
gcloud alpha compute tpus queued-resources delete tpu-builder-queue --zone=us-east5-a
```

Then recreate.

PowerShell line-break rule:
- Bash uses `\`
- PowerShell uses backtick `` ` ``
- safest: always run as one line

---

## 3) Check provisioning status

```powershell
gcloud alpha compute tpus queued-resources describe tpu-builder-queue --zone=us-east5-a
gcloud compute tpus tpu-vm list --zone=us-east5-a
```

Interpretation:
- `WAITING_FOR_RESOURCES` = accepted but no capacity yet (normal)
- `ACTIVE` + node appears in `tpu-vm list` = ready to SSH
- `SUSPENDED` = not running/paused; recreate queue if needed

---

## 4) SSH only after node exists

```powershell
gcloud compute tpus tpu-vm ssh my-tpu-node --zone=us-east5-a
```

If you get `NOT_FOUND`:
- node is not created yet
- keep polling queue status

---

## 5) Important environment difference we found

- `tunix-vm` (your old SSH alias) lands on a `gke-tunix-pathways...` COS node.
- That host had:
  - `noexec` mounts (`/home`, `/tmp`, `/var`)
  - missing native libs for JAX (`libstdc++.so.6`)
- Result: notebook and direct JAX Python execution failed there.

Conclusion:
- Use real TPU VM node (`my-tpu-node`) when ready.
- Don’t rely on the COS/GKE node shell for normal Python ML runtime.

---

## 6) Common errors and exact fixes

### Error: `Password authentication is not supported for Git operations`
- GitHub HTTPS password is blocked.
- Use PAT, GitHub SSH key, or temporary public clone workaround.

### Error: `Permission denied (publickey)` for `ssh tunix-vm`
- Re-add local public key to Compute Engine metadata SSH keys.

### Error: `ModuleNotFoundError: No module named 'src'`
- Run scripts with:
  ```bash
  PYTHONPATH=. python scripts/run_tpu_smoke.py --overwrite
  ```

### Error: `..._zmq...so: file not located on exec mount`
- Jupyter on noexec mount issue on COS node; use script-based runs.

### Error: `libstdc++.so.6` missing
- COS host missing runtime libs for JAX; use proper TPU VM runtime.

---

## 7) Minimal command checklist

```powershell
gcloud auth login
gcloud config set project computer-vision-491521
gcloud components install alpha
gcloud alpha compute tpus queued-resources create tpu-builder-queue --zone=us-east5-a --accelerator-type=v5p-8 --runtime-version=v2-alpha-tpuv5 --node-id=my-tpu-node --provisioning-model=flex-start --max-run-duration=4h --valid-until-duration=4h
gcloud alpha compute tpus queued-resources describe tpu-builder-queue --zone=us-east5-a
gcloud compute tpus tpu-vm list --zone=us-east5-a
gcloud compute tpus tpu-vm ssh my-tpu-node --zone=us-east5-a
```
