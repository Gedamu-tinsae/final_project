from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
from transformers import AutoProcessor, AutoTokenizer, FlaxCLIPVisionModel


@dataclass
class ClipBundle:
    model: FlaxCLIPVisionModel
    processor: AutoProcessor
    hidden_size: int
    image_size: int
    patch_size: int
    model_dir: str


def _resolve_dtype(dtype: str):
    return jnp.bfloat16 if dtype == "bfloat16" else jnp.float32


def build_clip_bundle(
    *,
    model_dir: str | Path | None = None,
    repo_id: str = "openai/clip-vit-base-patch32",
    dtype: str = "bfloat16",
) -> ClipBundle:
    local_dir = Path(model_dir) if model_dir else (Path.home() / ".cache" / "hf_models" / "clip-vit-base-patch32")
    local_dir = local_dir.resolve()
    local_files_only = local_dir.exists() and any(local_dir.iterdir())

    processor = AutoProcessor.from_pretrained(
        str(local_dir if local_files_only else repo_id),
        local_files_only=local_files_only,
    )
    model_loaded = FlaxCLIPVisionModel.from_pretrained(
        str(local_dir if local_files_only else repo_id),
        local_files_only=local_files_only,
        dtype=_resolve_dtype(dtype),
    )
    # transformers typing can expose this as model or (model, params) tuple.
    model = model_loaded[0] if isinstance(model_loaded, tuple) else model_loaded
    vision_cfg = model.config
    return ClipBundle(
        model=model,
        processor=processor,
        hidden_size=vision_cfg.hidden_size,
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        model_dir=str(local_dir),
    )


def make_clip_feature_fn(clip_bundle: ClipBundle):
    params = cast(Any, clip_bundle.model.params)

    @jax.jit
    def get_features(pixel_values):
        outputs = clip_bundle.model(
            pixel_values=pixel_values,
            params=params,
            output_hidden_states=True,
        )
        return cast(Any, outputs).hidden_states[-2]

    return get_features


def build_tokenizer(
    *,
    tokenizer_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    local_dir: str | Path | None = None,
):
    if local_dir:
        local_path = Path(local_dir)
        if local_path.exists() and any(local_path.iterdir()):
            return AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
    return AutoTokenizer.from_pretrained(tokenizer_id)


def create_stage2_runtime_objects(cfg: dict[str, Any]) -> dict[str, Any]:
    clip_model_dir = cfg.get("clip_model_dir")
    tokenizer_id = cfg.get("tokenizer_id", "meta-llama/Llama-3.2-1B-Instruct")
    llama_local_dir = cfg.get("llama_local_dir")

    clip_bundle = build_clip_bundle(model_dir=clip_model_dir, dtype=cfg.get("clip_dtype", "bfloat16"))
    get_features_compiled = make_clip_feature_fn(clip_bundle)
    tokenizer = build_tokenizer(tokenizer_id=tokenizer_id, local_dir=llama_local_dir)

    return {
        "clip_bundle": clip_bundle,
        "get_features_compiled": get_features_compiled,
        "tokenizer": tokenizer,
    }
