# Enable GPU for JAX on CloudExe

If `nvidia-smi` shows a GPU but `python scripts/check_accelerator.py` reports `backend: cpu`, your venv has CPU-only `jaxlib`.

## 1) Activate project venv

```bash
cd /root/final_project/group1_baseline
source .venv/bin/activate
```

## 2) Reinstall JAX with CUDA support

```bash
python -m pip install --upgrade pip
python -m pip uninstall -y jax jaxlib
python -m pip install --upgrade "jax[cuda12]"
```

## 3) Verify backend

```bash
python - <<'PY'
import jax, jaxlib
print("jax", jax.__version__)
print("jaxlib", jaxlib.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
PY
```

Expected: `backend` should be `gpu`.

## Troubleshooting: `Unable to load cuSPARSE`

If you see errors like:

- `Unable to load cuSPARSE. Is it installed?`
- `operation cusparseGetProperty(...) failed`

then JAX CUDA plugin is present but CUDA runtime libraries are not discoverable.

### A) Install CUDA runtime wheels in the venv

```bash
python -m pip install --upgrade "jax[cuda12]" \
  nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cusparse-cu12 nvidia-cudnn-cu12
```

### B) Export NVIDIA pip library paths

```bash
export LD_LIBRARY_PATH="$(python - <<'PY'
import site,glob,os
roots=site.getsitepackages()
libs=[]
for r in roots:
    libs += glob.glob(os.path.join(r, "nvidia", "*", "lib"))
print(":".join(libs))
PY
):$LD_LIBRARY_PATH"
```

### C) Re-check accelerator

```bash
python scripts/check_accelerator.py
```

If backend is still `cpu`, your host image may not provide compatible CUDA runtime linkage for this JAX build.
In that case use CloudExe GPU-job execution (same pattern as HW4):

```bash
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group1_baseline/scripts/check_accelerator.py
```

## 4) Run safe smoke first

```bash
python scripts/run_tpu_smoke.py --max-rows 64 --stage1-batch-size 1 --stage2-batch-size 1 --dtype float32
```

Keep `--overwrite` off unless you intentionally want regeneration.

## Notes

- HW4-style `cloudexe --gpuspec ... -- <cmd>` is for requesting a specific GPU job environment (for example H100).
- On this server, GPU can already be present; the missing piece is usually CUDA-enabled `jaxlib` in the venv.

## Recommended for this project: CloudExe H100 wrapper

For heavy model load/training, use CloudExe GPU allocation directly (H100), not the small local server GPU.

### 1) Confirm allocated GPU

```bash
cloudexe --gpuspec EUNH100x1 -- /usr/bin/nvidia-smi
```

### 2) Run accelerator check inside allocation

```bash
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group1_baseline/scripts/check_accelerator.py
```

### 3) Run Group1 smoke inside allocation

```bash
cloudexe --gpuspec EUNH100x1 -- /root/final_project/group1_baseline/.venv/bin/python /root/final_project/group1_baseline/scripts/run_tpu_smoke.py \
  --max-rows 64 --stage1-batch-size 1 --stage2-batch-size 1 --dtype bfloat16
```

This avoids RTX 3050 VRAM limits and is the preferred route until TPU is ready.
