from __future__ import annotations

import json
from pathlib import Path

from .common import load_jsonl

SEP = "###"
SYSTEM_PROMPT = (
    "A chat between a curious user and an AI assistant. "
    "The assistant gives helpful, detailed, and polite answers."
)


def resolve_stage2_paths(stage2_root: Path, variant: str, data_split: str) -> tuple[Path, Path, Path]:
    variant_dir = stage2_root / variant
    if data_split == "train":
        return variant_dir, variant_dir / "stage2_train.jsonl", variant_dir / "stage2_tokenized_train.json"
    if data_split == "val":
        return variant_dir, variant_dir / "stage2_val.jsonl", variant_dir / "stage2_tokenized_val.json"
    if data_split == "full":
        return variant_dir, variant_dir / "stage2_dataset.jsonl", variant_dir / "stage2_tokenized_full.json"
    raise ValueError(f"Unsupported data_split: {data_split}")


def _append_segment(tokenizer, input_ids: list[int], labels: list[int], text: str, supervise: bool) -> None:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    input_ids.extend(ids)
    labels.extend(ids if supervise else ([-100] * len(ids)))


def serialize_stage2_sample(tokenizer, sample: dict, variant: str, data_split: str, max_len: int = 256) -> dict:
    input_ids: list[int] = []
    labels: list[int] = []
    _append_segment(tokenizer, input_ids, labels, f"{SYSTEM_PROMPT}\n{SEP}\n", supervise=False)

    for turn in sample.get("history", []):
        _append_segment(tokenizer, input_ids, labels, f"USER: {turn['question'].strip()}\n{SEP}\n", supervise=False)
        _append_segment(tokenizer, input_ids, labels, f"ASSISTANT: {turn['answer'].strip()}\n{SEP}\n", supervise=True)

    _append_segment(tokenizer, input_ids, labels, f"USER: {sample['instruction'].strip()}\n{SEP}\n", supervise=False)
    _append_segment(tokenizer, input_ids, labels, f"ASSISTANT: {sample['response'].strip()}\n{SEP}\n", supervise=True)

    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
        labels = labels[-max_len:]

    return {
        "image": sample["image"],
        "image_id": sample.get("image_id"),
        "input_ids": input_ids,
        "labels": labels,
        "task_type": sample.get("task_type", "unknown"),
        "generator_model": sample.get("generator_model", variant),
        "sample_id": sample.get("sample_id"),
        "requested_variant": variant,
        "data_split": data_split,
    }


def tokenize_stage2_variant(
    stage2_root: Path,
    tokenizer,
    variant: str,
    data_split: str = "train",
    max_len: int = 256,
    overwrite: bool = False,
) -> dict:
    variant_dir, src_jsonl, tokenized_json = resolve_stage2_paths(stage2_root, variant, data_split)
    if tokenized_json.exists() and not overwrite:
        return {
            "mode": "skipped_existing",
            "variant": variant,
            "split": data_split,
            "tokenized_json": str(tokenized_json),
        }
    rows = load_jsonl(src_jsonl)
    tokenized = [serialize_stage2_sample(tokenizer, row, variant=variant, data_split=data_split, max_len=max_len) for row in rows]
    tokenized_json.parent.mkdir(parents=True, exist_ok=True)
    tokenized_json.write_text(json.dumps(tokenized, ensure_ascii=False), encoding="utf-8")
    return {"mode": "generated", "variant": variant, "split": data_split, "rows": len(tokenized), "tokenized_json": str(tokenized_json)}
