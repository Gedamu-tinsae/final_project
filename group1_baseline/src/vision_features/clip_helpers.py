"""Helpers to download/load the CLIP vision encoder in Flax.

These utilities are used by stage-1 and stage-2 preprocessing so that image
features can be extracted consistently.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from transformers import AutoProcessor, FlaxCLIPVisionModel


@dataclass
class ClipBundle:
    """Container for CLIP model + processor + key metadata."""
    model: FlaxCLIPVisionModel
    processor: AutoProcessor
    hidden_size: int
    image_size: int
    patch_size: int
    model_dir: str


def download_clip_flax(
    repo_id: str = "openai/clip-vit-base-patch32",
    local_dir: str = str(Path.home() / ".cache" / "hf_models" / "clip-vit-base-patch32"),
) -> str:
    """Download required CLIP files from Hugging Face into local_dir."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.makedirs(local_dir, exist_ok=True)

    cmd = [
        "hf", "download", repo_id,
        "--include",
        "flax_model.msgpack",
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "--local-dir", local_dir,
    ]
    subprocess.run(cmd, check=True)
    return local_dir


def load_clip_flax_local(
    local_dir: str = str(Path.home() / ".cache" / "hf_models" / "clip-vit-base-patch32"),
    dtype=jnp.bfloat16,
) -> ClipBundle:
    """Load CLIP processor and Flax model from an existing local directory."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    processor = AutoProcessor.from_pretrained(
        local_dir,
        local_files_only=True,
    )

    model = FlaxCLIPVisionModel.from_pretrained(
        local_dir,
        local_files_only=True,
        dtype=dtype,
    )

    vision_cfg = model.config

    return ClipBundle(
        model=model,
        processor=processor,
        hidden_size=vision_cfg.hidden_size,
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        model_dir=local_dir,
    )


def build_clip_vision_tower(
    local_dir: str = str(Path.home() / ".cache" / "hf_models" / "clip-vit-base-patch32"),
    dtype=jnp.bfloat16,
) -> ClipBundle:
    """Convenience wrapper used by notebook/training code."""
    return load_clip_flax_local(local_dir=local_dir, dtype=dtype)


def create_clip_from_flax_checkpoint(
    local_dir: str = str(Path.home() / ".cache" / "hf_models" / "clip-vit-base-patch32"),
    download_if_missing: bool = True,
    dtype=jnp.bfloat16,
):
    """Download if needed, then load CLIP locally."""
    if download_if_missing and not os.path.exists(os.path.join(local_dir, "flax_model.msgpack")):
        download_clip_flax(local_dir=local_dir)
    return load_clip_flax_local(local_dir=local_dir, dtype=dtype)
