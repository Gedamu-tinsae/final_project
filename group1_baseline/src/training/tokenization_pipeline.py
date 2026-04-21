"""Notebook-thin orchestration for stage tokenization."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from .tokenization import build_tokenized_stage1_dataset, build_tokenized_stage2_dataset


def run_tokenization_pipeline(
    *,
    tokenizer_id: str,
    stage1_input_json: Path,
    stage1_output_json: Path,
    stage2_input_json: Path,
    stage2_output_json: Path,
    stage1_max_len: int = 128,
    stage2_max_len: int = 256,
    overwrite: bool = False,
) -> dict:
    """Run Stage 1/2 tokenization with Group-1-compatible fallback behavior.

    - Stage 1 input is required.
    - Stage 2 falls back to Stage 1 input if stage2_input_json does not exist.
    - Existing outputs are skipped unless overwrite=True.
    """
    if not stage1_input_json.exists():
        raise FileNotFoundError(f"Missing Stage 1 input: {stage1_input_json}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    result = {
        "tokenizer_id": tokenizer_id,
        "stage1_mode": "skipped",
        "stage2_mode": "skipped",
        "stage2_input_used": None,
    }

    if stage1_output_json.exists() and not overwrite:
        result["stage1_mode"] = "skipped_existing"
    else:
        build_tokenized_stage1_dataset(
            tokenizer,
            str(stage1_input_json),
            str(stage1_output_json),
            max_len=stage1_max_len,
            overwrite=overwrite,
        )
        result["stage1_mode"] = "generated"

    stage2_source = stage2_input_json if stage2_input_json.exists() else stage1_input_json
    result["stage2_input_used"] = str(stage2_source)

    if stage2_output_json.exists() and not overwrite:
        result["stage2_mode"] = "skipped_existing"
    else:
        build_tokenized_stage2_dataset(
            tokenizer,
            str(stage2_source),
            str(stage2_output_json),
            max_len=stage2_max_len,
            overwrite=overwrite,
        )
        result["stage2_mode"] = "generated"

    return result
