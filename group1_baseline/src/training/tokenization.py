"""Tokenization utilities extracted from Group 1 notebook logic.

These functions preserve the original behavior:
- Build USER/ASSISTANT text format
- Mask prompt tokens in labels with -100
- Truncate to max_len
"""

import json
import os


def serialize_instruction_sample(tokenizer, sample, max_len=128):
    """Serialize one instruction-response sample into token ids + masked labels."""
    instruction = sample["instruction"]
    response = sample["response"]

    prefix = f"USER: {instruction}\nASSISTANT:"
    full_text = prefix + " " + response

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]

    full_ids = full_ids[:max_len]
    labels = full_ids.copy()

    prefix_len = min(len(prefix_ids), len(labels))
    labels[:prefix_len] = [-100] * prefix_len

    return {
        "image": sample["image"],
        "input_ids": full_ids,
        "labels": labels,
    }


def build_tokenized_dataset(tokenizer, input_json, output_json, max_len=128):
    """Tokenize all rows from input_json and write a tokenized JSON file."""
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []
    for sample in data:
        ex = serialize_instruction_sample(tokenizer, sample, max_len=max_len)
        processed.append(ex)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(processed, f)


def serialize_stage1_sample(tokenizer, sample, max_len=128):
    """Group-1 notebook-compatible name for stage-1 serialization."""
    return serialize_instruction_sample(tokenizer, sample, max_len=max_len)


def serialize_stage2_sample(tokenizer, sample, max_len=256):
    """Stage-2 uses the same serialization format with a larger default max_len."""
    return serialize_instruction_sample(tokenizer, sample, max_len=max_len)


def build_tokenized_stage1_dataset(tokenizer, input_json, output_json, max_len=128):
    """Group-1 notebook-compatible stage-1 dataset builder."""
    return build_tokenized_dataset(tokenizer, input_json, output_json, max_len=max_len)


def build_tokenized_stage2_dataset(tokenizer, input_json, output_json, max_len=256):
    """Group-1 notebook-compatible stage-2 dataset builder."""
    return build_tokenized_dataset(tokenizer, input_json, output_json, max_len=max_len)
