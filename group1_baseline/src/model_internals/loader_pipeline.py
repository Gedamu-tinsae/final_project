"""Model bootstrap helpers used by the workflow notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from . import model as llama_model_lib
from . import params as llama_params


def ensure_llama_artifacts(
    *,
    repo_id: str,
    local_dir: Path,
    force_download: bool = False,
) -> dict[str, Any]:
    """Download required LLaMA files once and reuse cached/local copies."""
    local_dir = local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    mode = "generated"
    if local_dir.exists() and any(local_dir.iterdir()) and not force_download:
        mode = "skipped_existing"
    else:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "model.safetensors",
                "model.safetensors.index.json",
            ],
        )

    return {
        "mode": mode,
        "repo_id": repo_id,
        "local_dir": str(local_dir),
    }


def _mesh_for_current_devices():
    num_devices = len(jax.devices())
    if num_devices == 8:
        mesh_counts = (2, 4)
    elif num_devices == 4:
        mesh_counts = (1, 4)
    elif num_devices == 1:
        mesh_counts = (1, 1)
    else:
        raise ValueError(f"Unsupported number of JAX devices: {num_devices}")
    return jax.make_mesh(mesh_counts, ("fsdp", "tp"))


def load_llama_model_and_tokenizer(
    *,
    local_dir: Path,
    dtype: str = "bfloat16",
    use_mesh: bool = True,
) -> dict[str, Any]:
    """Load LLaMA model + tokenizer from local safetensors."""
    local_dir = local_dir.resolve()
    if not local_dir.exists():
        raise FileNotFoundError(f"LLaMA model dir not found: {local_dir}")

    cfg = llama_model_lib.ModelConfig.llama3p2_1b_instruct()
    model_dtype = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32

    mesh = None
    llama_model = None
    if use_mesh:
        mesh = _mesh_for_current_devices()
        with mesh:
            llama_model = llama_params.create_model_from_safe_tensors(
                file_dir=str(local_dir),
                config=cfg,
                mesh=mesh,
                dtype=model_dtype,
            )
    else:
        llama_model = llama_params.create_model_from_safe_tensors(
            file_dir=str(local_dir),
            config=cfg,
            mesh=None,
            dtype=model_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
    return {
        "llama_model": llama_model,
        "tokenizer": tokenizer,
        "llama_dir": str(local_dir),
        "num_devices": len(jax.devices()),
        "dtype": dtype,
        "mesh_enabled": use_mesh,
    }
