"""Batching/collation utilities from Group 1 notebook."""

import random

import numpy as np
import jax.numpy as jnp


def pad_list(x, length, pad_value):
    return x + [pad_value] * (length - len(x))


def stage1_collate_fn(batch):
    """Load CLIP feature files and pad text sequences for a minibatch."""
    max_text_len = max(len(x["input_ids"]) for x in batch)

    vision_feats = []
    input_ids = []
    labels = []

    for sample in batch:
        vf = np.load(sample["vision_path"], allow_pickle=False)
        # Some saved CLIP files contain bf16 payloads encoded as NumPy void (|V2).
        # Convert them to real numeric float32 before passing to JAX.
        if vf.dtype.kind == "V" and vf.dtype.itemsize == 2:
            vf = np.array(jnp.asarray(vf.view(np.uint16), dtype=jnp.bfloat16), dtype=np.float32)
        elif not np.issubdtype(vf.dtype, np.number):
            raise TypeError(f"Unsupported vision feature dtype: {vf.dtype} in {sample['vision_path']}")
        else:
            vf = vf.astype(np.float32, copy=False)
        vision_feats.append(vf)

        input_ids.append(pad_list(sample["input_ids"], max_text_len, 0))
        labels.append(pad_list(sample["labels"], max_text_len, -100))

    vision_feats = np.stack(vision_feats, axis=0)
    input_ids = np.array(input_ids, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)

    return {
        "vision_feats": vision_feats,
        "input_ids": input_ids,
        "labels": labels,
    }


def iterate_minibatches(data, batch_size):
    """Shuffle then yield collated minibatches."""
    idxs = list(range(len(data)))
    random.shuffle(idxs)

    for i in range(0, len(idxs), batch_size):
        batch = [data[j] for j in idxs[i : i + batch_size]]
        yield stage1_collate_fn(batch)
