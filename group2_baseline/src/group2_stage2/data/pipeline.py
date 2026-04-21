from __future__ import annotations

from pathlib import Path

from .features import extract_stage2_features
from .manifests import build_stage2_manifest
from .tokenization import tokenize_stage2_variant


def prepare_stage2_variant_splits(
    stage2_root: Path,
    image_root: Path,
    feature_root: Path,
    tokenizer,
    clip_bundle,
    get_features_compiled,
    all_variants: list[str],
    splits: tuple[str, ...] = ("train", "val"),
    overwrite: bool = False,
) -> list[dict]:
    prep_results: list[dict] = []
    for variant in all_variants:
        for split in splits:
            tok = tokenize_stage2_variant(stage2_root, tokenizer, variant, split, overwrite=overwrite)
            feat = extract_stage2_features(
                stage2_root, image_root, feature_root, clip_bundle, get_features_compiled, variant, split, overwrite=overwrite
            )
            manifest = build_stage2_manifest(stage2_root, feature_root, variant, split, overwrite=overwrite)
            prep_results.append({"variant": variant, "split": split, "tokenize": tok, "features": feat, "manifest": manifest})
    return prep_results
